#!/usr/bin/env python3
"""
scripts/virtuals_sync.py — Virtuals Protocol Token Sync

Ported from maiat-indexer/src/virtuals-sync.ts

Fetches all agents from https://api.virtuals.io/api/virtuals (paginated),
cross-matches with agent_scores by name or sentient wallet address to fill
token_address + token_symbol.

Also creates new entries for Virtuals agents not already in ACP data.

Usage (standalone):
    python scripts/virtuals_sync.py

Environment:
    DATABASE_URL   — Supabase PostgreSQL connection string
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

# Bootstrap path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PKG_ROOT)

logger = logging.getLogger("wadjet.virtuals_sync")

# ─── Config ───────────────────────────────────────────────────────────────────

VIRTUALS_API = "https://api.virtuals.io/api/virtuals"
PAGE_SIZE = 25
DELAY_BETWEEN_PAGES = 0.5  # 500ms


# ─── Fetch helpers ────────────────────────────────────────────────────────────

async def fetch_page(client: httpx.AsyncClient, page: int) -> tuple[list[dict], int]:
    """
    Fetch one page from Virtuals API.
    Returns (agents, total_pages).
    """
    resp = await client.get(
        VIRTUALS_API,
        params={"page": str(page), "limit": str(PAGE_SIZE)},
        timeout=15.0,
    )
    resp.raise_for_status()
    data = resp.json()

    agents = data.get("data") or []
    pagination = (data.get("meta") or {}).get("pagination") or {}
    total_pages = int(pagination.get("pageCount") or 0)
    return agents, total_pages


async def fetch_all_virtuals() -> list[dict]:
    """Paginate through all Virtuals Protocol agents."""
    all_agents = []

    async with httpx.AsyncClient() as client:
        # Get total pages on first call
        try:
            agents, total_pages = await fetch_page(client, 1)
            all_agents.extend(agents)
        except Exception as e:
            logger.error(f"Virtuals API first page failed: {e}")
            return []

        logger.info(f"Virtuals API: {total_pages} pages to fetch")

        for page in range(2, total_pages + 1):
            try:
                agents, _ = await fetch_page(client, page)
                if not agents:
                    break
                all_agents.extend(agents)

                if page % 50 == 0:
                    logger.info(f"  … page {page}/{total_pages}, {len(all_agents)} agents")

            except Exception as e:
                logger.warning(f"Page {page} failed: {e}")

            await asyncio.sleep(DELAY_BETWEEN_PAGES)

    logger.info(f"Fetched {len(all_agents)} Virtuals agents total")
    return all_agents


# ─── DB operations ────────────────────────────────────────────────────────────

def get_existing_agents() -> tuple[dict[str, str], set[str]]:
    """
    Return:
      name_to_wallet: {lowercase_name: wallet_address}
      wallets_with_token: set of lowercase wallet_addresses that already have token_address
    """
    import psycopg2
    import psycopg2.extras

    db_url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(db_url)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT wallet_address, token_address, raw_metrics FROM agent_scores")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    name_to_wallet: dict[str, str] = {}
    wallets_with_token: set[str] = set()

    for row in rows:
        wallet = row["wallet_address"]
        rm = row["raw_metrics"] or {}
        if isinstance(rm, str):
            try:
                rm = json.loads(rm)
            except Exception:
                rm = {}

        name = rm.get("name")
        if name:
            name_to_wallet[name.lower().strip()] = wallet

        if row["token_address"]:
            wallets_with_token.add(wallet.lower())

    return name_to_wallet, wallets_with_token


def update_agent_token(wallet_address: str, token_address: str, token_symbol: Optional[str]) -> None:
    """Set token_address and token_symbol on an existing agent_scores row."""
    import psycopg2

    db_url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("""
        UPDATE agent_scores
        SET token_address = %s,
            token_symbol  = %s,
            updated_at    = NOW()
        WHERE wallet_address = %s
    """, (token_address, token_symbol, wallet_address))
    conn.commit()
    cur.close()
    conn.close()


def upsert_virtuals_agent(agent: dict) -> None:
    """
    Insert a new agent_scores row for a Virtuals agent not in ACP.
    Uses sentientWalletAddress if available, else 'virtuals-{id}' as placeholder.
    """
    import psycopg2

    db_url = os.environ["DATABASE_URL"]
    wallet = agent.get("sentientWalletAddress") or f"virtuals-{agent.get('id', 'unknown')}"
    token_address = agent.get("tokenAddress") or agent.get("token_address")
    if not token_address:
        return

    raw_metrics = {
        "name": agent.get("name"),
        "description": agent.get("description"),
        "category": agent.get("category"),
        "tokenAddress": token_address,
        "symbol": agent.get("symbol"),
        "status": agent.get("status"),
        "lpAddress": agent.get("lpAddress"),
        "daoAddress": agent.get("daoAddress"),
        "virtualId": agent.get("virtualId"),
        "indexedAt": datetime.now(timezone.utc).isoformat(),
        "source": "virtuals-protocol",
    }

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO agent_scores (
            wallet_address, name, trust_score, completion_rate, payment_rate,
            expire_rate, total_jobs, data_source, token_address, token_symbol,
            raw_metrics, updated_at
        ) VALUES (
            %s, %s, 0, 0, 0, 0, 0, 'VIRTUALS_PROTOCOL', %s, %s, %s::jsonb, NOW()
        )
        ON CONFLICT (wallet_address) DO UPDATE SET
            token_address = COALESCE(EXCLUDED.token_address, agent_scores.token_address),
            token_symbol  = COALESCE(EXCLUDED.token_symbol, agent_scores.token_symbol),
            updated_at    = NOW()
    """, (
        wallet,
        agent.get("name"),
        token_address,
        agent.get("symbol"),
        json.dumps(raw_metrics),
    ))
    conn.commit()
    cur.close()
    conn.close()


# ─── Main sync logic ──────────────────────────────────────────────────────────

async def run_virtuals_sync(verbose: bool = False) -> dict:
    """
    Full Virtuals token sync run.
    Returns summary dict.
    """
    start = time.time()
    logger.info("🔄 Virtuals Sync: starting…")

    result = {
        "total_virtuals": 0,
        "matched_by_name": 0,
        "new_agents": 0,
        "updated_token": 0,
        "failed": 0,
    }

    # Load existing DB state
    try:
        name_to_wallet, wallets_with_token = get_existing_agents()
        logger.info(
            f"DB: {len(name_to_wallet)} named agents, "
            f"{len(wallets_with_token)} with token"
        )
    except Exception as e:
        logger.error(f"Failed to load existing agents: {e}")
        return result

    # Fetch all Virtuals agents
    virtuals_agents = await fetch_all_virtuals()
    result["total_virtuals"] = len(virtuals_agents)

    for agent in virtuals_agents:
        token_address = agent.get("tokenAddress") or agent.get("token_address")
        if not token_address:
            continue

        name = (agent.get("name") or "").lower().strip()
        match_wallet = name_to_wallet.get(name)

        if match_wallet:
            # Matched by name — update token if not set
            if match_wallet.lower() not in wallets_with_token:
                try:
                    update_agent_token(
                        wallet_address=match_wallet,
                        token_address=token_address,
                        token_symbol=agent.get("symbol"),
                    )
                    result["updated_token"] += 1
                    wallets_with_token.add(match_wallet.lower())
                    if verbose:
                        logger.debug(
                            f"Updated token for {name!r}: {token_address[:12]}…"
                        )
                except Exception as e:
                    logger.warning(f"Token update failed for {name!r}: {e}")
                    result["failed"] += 1
            result["matched_by_name"] += 1
            continue

        # No name match — create new entry
        try:
            upsert_virtuals_agent(agent)
            result["new_agents"] += 1
        except Exception as e:
            logger.warning(f"Upsert failed for Virtuals agent {agent.get('id')}: {e}")
            result["failed"] += 1

    duration = round(time.time() - start, 2)
    result["duration_seconds"] = duration
    result["completed_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        f"Virtuals sync done — total={result['total_virtuals']} "
        f"matched_by_name={result['matched_by_name']} "
        f"new_agents={result['new_agents']} "
        f"updated_token={result['updated_token']} "
        f"failed={result['failed']} "
        f"({duration}s)"
    )
    return result


# ─── Standalone entrypoint ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = asyncio.run(run_virtuals_sync(verbose=True))
    print("\n📊 Result:", json.dumps(result, indent=2, default=str))
