#!/usr/bin/env python3
"""
scripts/acp_poller.py — ACP Agent Polling (5-minute interval)

Ported from maiat-indexer/src/acp-indexer.ts

Polls https://acpx.virtuals.io/api/agents (paginated, 25/page) and upserts
agent data into agent_scores table.

Usage (standalone):
    python scripts/acp_poller.py

Environment:
    DATABASE_URL  — Supabase PostgreSQL connection string
"""

import asyncio
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from db.utils import get_db_url

# Bootstrap path so we can import wadjet modules
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PKG_ROOT)

logger = logging.getLogger("wadjet.acp_poller")

# ─── Config ───────────────────────────────────────────────────────────────────

ACP_LIST_URL = "https://acpx.virtuals.io/api/agents"
PAGE_SIZE = 25
MAX_PAGES = 800  # safety cap — covers ~20,000 agents
RATE_LIMIT_DELAY = 0.5  # 500ms between pages

# ─── Trust Score (mirrors acp-indexer.ts computeTrustScore) ──────────────────

def compute_trust_score(agent: dict) -> dict:
    """
    Compute behavioral trust score from ACP agent data.
    Returns a score dict matching agent_scores schema.
    """
    total_jobs = agent.get("successfulJobCount") or 0
    success_rate_raw = agent.get("successRate") or 0
    success_rate = success_rate_raw / 100.0  # API gives 0-100

    buyer_count = agent.get("uniqueBuyerCount") or 0

    completion_rate = success_rate
    payment_rate = min(success_rate * 1.05, 1.0) if success_rate > 0 else 0.0
    expire_rate = max(1 - success_rate - 0.05, 0.0) if success_rate > 0 else 0.5

    # Volume factor: log scale, max 1.0 at 50+ jobs
    volume_factor = min(
        math.log10(total_jobs + 1) / math.log10(51), 1.0
    ) if total_jobs > 0 else 0.0

    # Diversity factor: multiple unique buyers = more trustworthy
    diversity_factor = min(buyer_count / 5.0, 1.0)

    # Trust score 0-100
    score = int(round(
        completion_rate * 40
        + volume_factor * 25
        + diversity_factor * 20
        + payment_rate * 15
    ))
    score = max(0, min(100, score))

    return {
        "wallet_address": agent.get("walletAddress", ""),
        "name": agent.get("name", ""),
        "trust_score": score,
        "completion_rate": round(completion_rate, 4),
        "payment_rate": round(payment_rate, 4),
        "expire_rate": round(expire_rate, 4),
        "total_jobs": total_jobs,
        "success_rate": success_rate_raw,
        "unique_buyers": buyer_count,
        "is_online": bool(agent.get("isOnline", False)),
        "token_address": agent.get("tokenAddress"),
        "data_source": "ACP_BEHAVIORAL",
        "raw_metrics": {
            "successfulJobCount": agent.get("successfulJobCount"),
            "successRate": agent.get("successRate"),
            "uniqueBuyerCount": agent.get("uniqueBuyerCount"),
            "category": agent.get("category"),
            "description": agent.get("description"),
            "agentId": agent.get("id"),
            "name": agent.get("name"),
            "profilePic": agent.get("profilePic"),
            "twitterHandle": agent.get("twitterHandle"),
            "cluster": agent.get("cluster"),
            "offerings": agent.get("offerings"),
            "grossAgenticAmount": agent.get("grossAgenticAmount"),
            "revenue": agent.get("revenue"),
            "transactionCount": agent.get("transactionCount"),
            "rating": agent.get("rating"),
            "tokenAddress": agent.get("tokenAddress"),
            "indexedAt": datetime.now(timezone.utc).isoformat(),
        },
    }


# ─── Fetch helpers ────────────────────────────────────────────────────────────

async def fetch_page(client: httpx.AsyncClient, page: int) -> list[dict]:
    """Fetch one page from ACP API (1-indexed)."""
    params = {
        "pagination[page]": str(page),
        "pagination[pageSize]": str(PAGE_SIZE),
    }
    resp = await client.get(ACP_LIST_URL, params=params, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data") or []


async def fetch_all_agents(verbose: bool = False) -> list[dict]:
    """Paginate through all ACP agents, deduplicating by walletAddress."""
    seen: dict[str, dict] = {}

    async with httpx.AsyncClient() as client:
        for page in range(1, MAX_PAGES + 1):
            try:
                agents = await fetch_page(client, page)
                if not agents:
                    if verbose:
                        logger.info(f"Page {page}: empty — done")
                    break

                new_count = 0
                for a in agents:
                    key = (a.get("walletAddress") or "").lower()
                    if key and key not in seen:
                        seen[key] = a
                        new_count += 1

                if verbose:
                    logger.debug(f"Page {page}: {len(agents)} results, {new_count} new")

                if len(agents) < PAGE_SIZE:
                    break  # last page

            except Exception as e:
                logger.warning(f"Page {page} failed: {e}")
                break

            await asyncio.sleep(RATE_LIMIT_DELAY)

    return list(seen.values())


# ─── DB upsert ────────────────────────────────────────────────────────────────

def upsert_agents(agents_data: list[dict]) -> tuple[int, int]:
    """
    Upsert agent scores into agent_scores table.
    Preserves existing priceData, healthSignals, liquiditySnapshots.
    Returns (written, failed).
    """
    import psycopg2
    import psycopg2.extras

    db_url = get_db_url()
    written = 0
    failed = 0

    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = False
        cur = conn.cursor()

        # Set statement timeout to 30s to avoid hanging on bloated indexes
        cur.execute("SET statement_timeout = '30s'")

        # Batch-fetch existing rawMetrics for these wallets
        wallets = [a["wallet_address"] for a in agents_data if a["wallet_address"]]
        existing_map: dict[str, dict] = {}

        for i in range(0, len(wallets), 100):
            chunk = wallets[i:i + 100]
            cur.execute(
                "SELECT wallet_address, raw_metrics FROM agent_scores WHERE wallet_address = ANY(%s)",
                (chunk,)
            )
            for row in cur.fetchall():
                try:
                    rm = row[1] if isinstance(row[1], dict) else json.loads(row[1] or "{}")
                    existing_map[row[0]] = rm
                except Exception:
                    existing_map[row[0]] = {}

        upsert_sql = """
            INSERT INTO agent_scores (
                wallet_address, name, trust_score, completion_rate, payment_rate,
                expire_rate, total_jobs, success_rate, unique_buyers, is_online,
                token_address, data_source, raw_metrics, updated_at
            ) VALUES (
                %(wallet_address)s, %(name)s, %(trust_score)s, %(completion_rate)s,
                %(payment_rate)s, %(expire_rate)s, %(total_jobs)s, %(success_rate)s,
                %(unique_buyers)s, %(is_online)s, %(token_address)s,
                %(data_source)s, %(raw_metrics)s, NOW()
            )
            ON CONFLICT (wallet_address) DO UPDATE SET
                name            = EXCLUDED.name,
                trust_score     = EXCLUDED.trust_score,
                completion_rate = EXCLUDED.completion_rate,
                payment_rate    = EXCLUDED.payment_rate,
                expire_rate     = EXCLUDED.expire_rate,
                total_jobs      = EXCLUDED.total_jobs,
                success_rate    = EXCLUDED.success_rate,
                unique_buyers   = EXCLUDED.unique_buyers,
                is_online       = EXCLUDED.is_online,
                data_source     = EXCLUDED.data_source,
                raw_metrics     = EXCLUDED.raw_metrics,
                updated_at      = NOW(),
                -- Only update token_address if we have a value (don't clobber existing)
                token_address   = COALESCE(EXCLUDED.token_address, agent_scores.token_address)
        """

        for a in agents_data:
            if not a["wallet_address"]:
                continue
            try:
                # Merge: preserve priceData, healthSignals, liquiditySnapshots
                existing_rm = existing_map.get(a["wallet_address"], {})
                merged_rm = dict(a["raw_metrics"])
                for preserve_key in ("priceData", "healthSignals", "liquiditySnapshots",
                                     "previousCompletionRate"):
                    if preserve_key in existing_rm and preserve_key not in merged_rm:
                        merged_rm[preserve_key] = existing_rm[preserve_key]

                params = dict(a)
                params["raw_metrics"] = json.dumps(merged_rm)

                cur.execute(upsert_sql, params)
                conn.commit()  # commit each agent individually to avoid bulk timeout
                written += 1
            except Exception as e:
                conn.rollback()
                logger.warning(f"Upsert failed for {a['wallet_address'][:10]}…: {e}")
                failed += 1
        cur.close()
        conn.close()

    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        raise

    return written, failed


# ─── Main indexer logic ───────────────────────────────────────────────────────

async def run_acp_index(verbose: bool = False) -> dict:
    """
    Full ACP indexing run.
    Returns summary dict.
    """
    start = time.time()
    logger.info("🔍 ACP Poller: starting index run…")

    agents = await fetch_all_agents(verbose=verbose)
    logger.info(f"Fetched {len(agents)} unique agents from ACP API")

    scores = [compute_trust_score(a) for a in agents]

    # Stats
    with_jobs = [s for s in scores if s["total_jobs"] > 0]
    avg_score = (
        round(sum(s["trust_score"] for s in with_jobs) / len(with_jobs))
        if with_jobs else 0
    )
    high_count = sum(1 for s in scores if s["trust_score"] >= 80)
    med_count = sum(1 for s in scores if 60 <= s["trust_score"] < 80)
    low_count = sum(1 for s in scores if s["trust_score"] < 60)

    logger.info(f"Score distribution — high: {high_count}, med: {med_count}, low: {low_count}, avg: {avg_score}")

    written, failed = 0, 0
    # Split into UPSERT_BATCH_SIZE chunks to avoid statement timeouts on large datasets
    for i in range(0, len(scores), UPSERT_BATCH_SIZE):
        batch = scores[i:i + UPSERT_BATCH_SIZE]
        try:
            w, f = upsert_agents(batch)
            written += w
            failed += f
        except Exception as e:
            logger.error(f"Upsert batch {i//UPSERT_BATCH_SIZE} failed: {e}")
            failed += len(batch)
    logger.info(f"Upsert complete — written: {written}, failed: {failed}")

    duration = round(time.time() - start, 2)
    return {
        "indexed": len(agents),
        "updated": written,
        "failed": failed,
        "duration_seconds": duration,
        "stats": {
            "total_agents": len(agents),
            "agents_with_jobs": len(with_jobs),
            "average_score": avg_score,
            "high_score_count": high_count,
            "medium_score_count": med_count,
            "low_score_count": low_count,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Background loop (used by main.py startup) ────────────────────────────────

async def acp_poll_loop(interval: int = 300) -> None:
    """
    Continuous background loop — runs ACP index every `interval` seconds.
    Designed to be launched as asyncio.create_task().
    """
    logger.info(f"ACP poll loop started (interval={interval}s)")
    while True:
        try:
            result = await run_acp_index()
            logger.info(
                f"ACP poll done — indexed={result['indexed']} "
                f"updated={result['updated']} failed={result['failed']}"
            )
        except Exception as e:
            logger.error(f"ACP poll loop error: {e}", exc_info=True)
        await asyncio.sleep(interval)


# ─── Standalone entrypoint ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = asyncio.run(run_acp_index(verbose=True))
    print("\n📊 Result:", json.dumps(result, indent=2))
