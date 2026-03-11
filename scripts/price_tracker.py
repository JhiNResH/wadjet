#!/usr/bin/env python3
"""
scripts/price_tracker.py — DexScreener Price Tracking

Ported from maiat-indexer/src/dexscreener.ts

Fetches token prices for all agents with token_address in agent_scores,
detects crashes (price_change_24h < -30%), and stores price snapshots.

Batches up to 30 addresses per request (DexScreener limit).
Rate limit: 1 request per 1.5s ≈ 40 req/min (well under 300 req/min limit).

Usage (standalone):
    python scripts/price_tracker.py

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
from db.utils import get_db_url

# Bootstrap path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PKG_ROOT)

logger = logging.getLogger("wadjet.price_tracker")

# ─── Config ───────────────────────────────────────────────────────────────────

DEXSCREENER_BASE = "https://api.dexscreener.com/latest/dex/tokens"
BATCH_SIZE = 30          # DexScreener max per request
DELAY_BETWEEN_BATCHES = 1.5  # seconds — 40 req/min, well under 300/min limit
CRASH_THRESHOLD = -30.0  # percent

# ─── Price fetch helpers ──────────────────────────────────────────────────────

async def fetch_batch(client: httpx.AsyncClient, token_addresses: list[str]) -> dict[str, dict]:
    """
    Fetch price data for a batch of token addresses (max 30).
    Returns {lowercase_address: price_dict}.
    """
    result: dict[str, dict] = {}
    joined = ",".join(token_addresses)

    try:
        resp = await client.get(
            f"{DEXSCREENER_BASE}/{joined}",
            timeout=15.0,
        )
        if resp.status_code != 200:
            logger.warning(f"DexScreener batch returned {resp.status_code}")
            return result

        data = resp.json()
        pairs = data.get("pairs") or []

        # Group by base token, pick highest-liquidity pair
        best_pair: dict[str, dict] = {}
        for pair in pairs:
            addr = (pair.get("baseToken") or {}).get("address", "").lower()
            if not addr:
                continue
            existing = best_pair.get(addr)
            new_liq = (pair.get("liquidity") or {}).get("usd") or 0
            old_liq = (existing.get("liquidity") or {}).get("usd") or 0 if existing else 0
            if not existing or new_liq > old_liq:
                best_pair[addr] = pair

        for addr, pair in best_pair.items():
            price_change = pair.get("priceChange") or {}
            volume = pair.get("volume") or {}
            liquidity = pair.get("liquidity") or {}
            txns = (pair.get("txns") or {}).get("h24") or {}

            result[addr] = {
                "price_usd": float(pair.get("priceUsd") or 0),
                "price_change_24h": float(price_change.get("h24") or 0),
                "price_change_6h": float(price_change.get("h6") or 0),
                "price_change_1h": float(price_change.get("h1") or 0),
                "volume_24h": float(volume.get("h24") or 0),
                "liquidity_usd": float(liquidity.get("usd") or 0),
                "market_cap": float(pair.get("marketCap") or 0),
                "fdv": float(pair.get("fdv") or 0),
                "pair_address": pair.get("pairAddress") or "",
                "dex_id": pair.get("dexId") or "",
                "txns_buys_24h": int(txns.get("buys") or 0),
                "txns_sells_24h": int(txns.get("sells") or 0),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

    except Exception as e:
        logger.warning(f"DexScreener batch fetch failed: {e}")

    return result


async def fetch_all_prices(token_addresses: list[str]) -> dict[str, dict]:
    """
    Fetch price data for all given token addresses.
    Returns {lowercase_address: price_dict}.
    """
    unique = list({a.lower() for a in token_addresses if a})
    logger.info(f"Fetching prices for {len(unique)} tokens…")

    all_prices: dict[str, dict] = {}

    async with httpx.AsyncClient() as client:
        for i in range(0, len(unique), BATCH_SIZE):
            batch = unique[i:i + BATCH_SIZE]
            batch_prices = await fetch_batch(client, batch)
            all_prices.update(batch_prices)

            if i + BATCH_SIZE < len(unique):
                await asyncio.sleep(DELAY_BETWEEN_BATCHES)

    logger.info(f"Got prices for {len(all_prices)}/{len(unique)} tokens")
    return all_prices


# ─── Crash detection ──────────────────────────────────────────────────────────

def detect_crashes(
    price_map: dict[str, dict],
    agent_names: dict[str, str],
) -> list[dict]:
    """
    Return sorted list of crash alerts (price_change_24h < CRASH_THRESHOLD).
    """
    alerts = []
    for token_addr, data in price_map.items():
        change = data.get("price_change_24h", 0)
        if change <= CRASH_THRESHOLD:
            alerts.append({
                "token_address": token_addr,
                "agent_name": agent_names.get(token_addr, token_addr[:10]),
                "price_change_24h": change,
                "current_price": data.get("price_usd", 0),
                "liquidity_usd": data.get("liquidity_usd", 0),
            })
    return sorted(alerts, key=lambda x: x["price_change_24h"])


# ─── DB operations ────────────────────────────────────────────────────────────

def get_agents_with_tokens() -> list[dict]:
    """Fetch all agents from agent_scores that have a token_address."""
    import psycopg2
    import psycopg2.extras

    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT wallet_address, token_address, raw_metrics
        FROM agent_scores
        WHERE token_address IS NOT NULL AND token_address != ''
    """)
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def save_price_snapshot(
    wallet_address: str,
    token_address: str,
    price_data: dict,
    existing_raw_metrics: dict,
) -> None:
    """
    Update agent_scores with new price data and liquidity snapshot history.
    Keeps last 96 snapshots (24h at 15min intervals).
    """
    import psycopg2

    db_url = get_db_url()

    # Build liquidity snapshot history
    existing_snapshots = []
    try:
        raw = existing_raw_metrics or {}
        if isinstance(raw, str):
            raw = json.loads(raw)
        existing_snapshots = raw.get("liquiditySnapshots") or []
        if not isinstance(existing_snapshots, list):
            existing_snapshots = []
    except Exception:
        existing_snapshots = []

    new_snapshot = {
        "liquidity": price_data.get("liquidity_usd", 0),
        "volume24h": price_data.get("volume_24h", 0),
        "priceUsd": price_data.get("price_usd", 0),
        "timestamp": price_data["fetched_at"],
    }
    snapshots = (existing_snapshots + [new_snapshot])[-96:]  # keep last 96

    # Merge into existing raw_metrics
    try:
        raw = existing_raw_metrics or {}
        if isinstance(raw, str):
            raw = json.loads(raw)
    except Exception:
        raw = {}

    raw["priceData"] = {
        "priceUsd": price_data.get("price_usd", 0),
        "volume24h": price_data.get("volume_24h", 0),
        "liquidity": price_data.get("liquidity_usd", 0),
        "priceChange24h": price_data.get("price_change_24h", 0),
        "priceChange6h": price_data.get("price_change_6h", 0),
        "priceChange1h": price_data.get("price_change_1h", 0),
        "fdv": price_data.get("fdv", 0),
        "fetchedAt": price_data["fetched_at"],
    }
    raw["liquiditySnapshots"] = snapshots

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("""
        UPDATE agent_scores
        SET raw_metrics = %s::jsonb, updated_at = NOW()
        WHERE wallet_address = %s
    """, (json.dumps(raw), wallet_address))
    conn.commit()
    cur.close()
    conn.close()


def save_price_data_to_snapshots_table(
    token_address: str,
    wallet_address: Optional[str],
    price_data: dict,
) -> None:
    """
    Optionally store price snapshot in wadjet_daily_snapshots (price_data column).
    Adds price_data JSONB column if it doesn't exist.
    """
    import psycopg2

    db_url = get_db_url()
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        # Ensure price_data column exists
        cur.execute("""
            ALTER TABLE wadjet_daily_snapshots
            ADD COLUMN IF NOT EXISTS price_data JSONB DEFAULT NULL
        """)

        cur.execute("""
            INSERT INTO wadjet_daily_snapshots
                (token_address, wallet_address, snapshot_date, price_data,
                 price_usd, liquidity_usd, volume_24h, market_cap)
            VALUES (%s, %s, CURRENT_DATE, %s::jsonb, %s, %s, %s, %s)
            ON CONFLICT (token_address, snapshot_date) DO UPDATE SET
                price_data   = EXCLUDED.price_data,
                price_usd    = EXCLUDED.price_usd,
                liquidity_usd = EXCLUDED.liquidity_usd,
                volume_24h   = EXCLUDED.volume_24h,
                market_cap   = EXCLUDED.market_cap
        """, (
            token_address.lower(),
            wallet_address,
            json.dumps(price_data),
            price_data.get("price_usd"),
            price_data.get("liquidity_usd"),
            price_data.get("volume_24h"),
            price_data.get("market_cap"),
        ))

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.debug(f"price snapshot table write failed (non-fatal): {e}")


# ─── Main price tracking run ──────────────────────────────────────────────────

async def run_price_tracking() -> dict:
    """
    Full price tracking run.
    Returns summary dict.
    """
    start = time.time()
    logger.info("💰 Price Tracker: starting run…")

    agents = get_agents_with_tokens()
    if not agents:
        logger.info("No agents with token addresses — skipping")
        return {"fetched": 0, "updated": 0, "crashes": 0}

    token_addresses = [a["token_address"] for a in agents if a.get("token_address")]
    agent_names: dict[str, str] = {}
    raw_metrics_map: dict[str, dict] = {}

    for a in agents:
        if a.get("token_address"):
            addr = a["token_address"].lower()
            rm = a.get("raw_metrics") or {}
            if isinstance(rm, str):
                try:
                    rm = json.loads(rm)
                except Exception:
                    rm = {}
            agent_names[addr] = rm.get("name") or a["wallet_address"][:10]
            raw_metrics_map[a["wallet_address"]] = rm

    # Fetch prices
    price_map = await fetch_all_prices(token_addresses)

    # Update DB
    updated = 0
    failed = 0
    for agent in agents:
        token_addr = (agent.get("token_address") or "").lower()
        if not token_addr or token_addr not in price_map:
            continue
        try:
            price_data = price_map[token_addr]
            save_price_snapshot(
                wallet_address=agent["wallet_address"],
                token_address=token_addr,
                price_data=price_data,
                existing_raw_metrics=raw_metrics_map.get(agent["wallet_address"], {}),
            )
            # Also try to write to daily snapshots table
            save_price_data_to_snapshots_table(
                token_address=token_addr,
                wallet_address=agent.get("wallet_address"),
                price_data=price_data,
            )
            updated += 1
        except Exception as e:
            logger.warning(f"Price save failed for {token_addr[:10]}…: {e}")
            failed += 1

    # Detect crashes
    crashes = detect_crashes(price_map, agent_names)
    for crash in crashes[:5]:  # log top 5
        logger.warning(
            f"🚨 CRASH: {crash['agent_name']} {crash['price_change_24h']}% "
            f"(price=${crash['current_price']:.6f}, liq=${crash['liquidity_usd']:,.0f})"
        )

    duration = round(time.time() - start, 2)
    logger.info(
        f"Price tracking done — fetched={len(price_map)}, "
        f"updated={updated}, failed={failed}, crashes={len(crashes)} "
        f"in {duration}s"
    )

    return {
        "fetched": len(price_map),
        "updated": updated,
        "failed": failed,
        "crashes": len(crashes),
        "crash_alerts": crashes,
        "duration_seconds": duration,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Background loop ──────────────────────────────────────────────────────────

async def price_track_loop(interval: int = 900) -> None:
    """
    Continuous background loop — runs price tracking every `interval` seconds.
    Designed to be launched as asyncio.create_task().
    """
    logger.info(f"Price track loop started (interval={interval}s)")
    while True:
        try:
            result = await run_price_tracking()
            logger.info(
                f"Price track done — fetched={result['fetched']} "
                f"updated={result['updated']} crashes={result['crashes']}"
            )
        except Exception as e:
            logger.error(f"Price track loop error: {e}", exc_info=True)
        await asyncio.sleep(interval)


# ─── Standalone entrypoint ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = asyncio.run(run_price_tracking())
    print("\n📊 Result:", json.dumps(result, indent=2, default=str))
