#!/usr/bin/env python3
"""
Wadjet Auto-Outcome Reporter — closes the feedback loop for unreported queries.

Logic:
1. Query `query_logs` where outcome IS NULL and createdAt < NOW() - 7 days
2. For each row, get `target` address
3. Look up current trust_score in `agent_scores` — compare vs original
4. Check DexScreener for token price (if token_address available)
5. Determine outcome:
   - Price drop >80% → scam
   - Price drop >50% → failure
   - Trust score dropped to 0 → failure
   - Still active, stable → success
   - No data → expired
6. Update query_logs SET outcome = <result> WHERE id = <query_id>
7. Print summary

Usage:
    python scripts/auto_outcomes.py

Environment:
    DATABASE_URL — PostgreSQL connection string (required)
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from db.utils import get_db_url

# ─── Bootstrap path ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PKG_ROOT)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("wadjet.auto_outcomes")

# ─── Config ──────────────────────────────────────────────────────────────────
DEXSCREENER_BASE = "https://api.dexscreener.com/latest/dex/tokens"
DEXSCREENER_RATE_LIMIT = 0.2    # 5 req/sec max → sleep 0.2s between calls
LOOKBACK_DAYS = 7               # Only process queries older than this
BATCH_SIZE = 100                # Max rows to process per run

# ─── DB helpers ──────────────────────────────────────────────────────────────

def _get_conn():
    """Return a raw psycopg2 connection (strip pgbouncer param)."""
    import psycopg2
    db_url = get_db_url()
    return psycopg2.connect(db_url)


def fetch_pending_logs(conn) -> list[dict]:
    """
    Fetch query_logs rows where outcome IS NULL and older than LOOKBACK_DAYS.
    NOTE: column names are camelCase (Prisma schema).
    """
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                id,
                target,
                "trustScore",
                "createdAt",
                type,
                metadata
            FROM query_logs
            WHERE
                outcome IS NULL
                AND "createdAt" < NOW() - INTERVAL '{LOOKBACK_DAYS} days'
            ORDER BY "createdAt" ASC
            LIMIT %s
            """,
            (BATCH_SIZE,),
        )
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_agent_score(conn, wallet_address: str) -> Optional[dict]:
    """Fetch current agent_scores row for a given address."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT trust_score, total_jobs, completion_rate, token_address
            FROM agent_scores
            WHERE wallet_address = %s OR token_address = %s
            LIMIT 1
            """,
            (wallet_address, wallet_address),
        )
        row = cur.fetchone()
        if row:
            return {
                "trust_score":     row[0],
                "total_jobs":      row[1],
                "completion_rate": row[2],
                "token_address":   row[3],
            }
    return None


def update_outcome(conn, log_id: str, outcome: str) -> None:
    """SET outcome on a query_logs row."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE query_logs SET outcome = %s WHERE id = %s",
            (outcome, log_id),
        )
    conn.commit()


# ─── DexScreener helpers ─────────────────────────────────────────────────────

_dex_cache: dict[str, Optional[dict]] = {}


def fetch_dexscreener(token_address: str) -> Optional[dict]:
    """
    Fetch DexScreener data for a token address.
    Returns dict with price_usd, price_change_24h, liquidity_usd or None.
    Caches per-address so we don't spam the API.
    Rate-limited to max 5 req/sec.
    """
    addr = token_address.lower()
    if addr in _dex_cache:
        return _dex_cache[addr]

    time.sleep(DEXSCREENER_RATE_LIMIT)   # rate limit

    try:
        resp = requests.get(
            f"{DEXSCREENER_BASE}/{addr}",
            timeout=10,
            headers={"Accept": "application/json"},
        )
        if resp.status_code != 200:
            logger.debug(f"DexScreener {addr[:10]}… HTTP {resp.status_code}")
            _dex_cache[addr] = None
            return None

        data = resp.json()
        pairs = data.get("pairs") or []
        if not pairs:
            _dex_cache[addr] = None
            return None

        # Pick pair with highest liquidity
        pair = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0)

        result = {
            "price_usd":        float(pair.get("priceUsd") or 0),
            "price_change_24h": float((pair.get("priceChange") or {}).get("h24") or 0),
            "price_change_6h":  float((pair.get("priceChange") or {}).get("h6") or 0),
            "liquidity_usd":    float((pair.get("liquidity") or {}).get("usd") or 0),
            "volume_24h":       float((pair.get("volume") or {}).get("h24") or 0),
            "market_cap":       float(pair.get("marketCap") or 0),
            "pair_created_at":  pair.get("pairCreatedAt"),
            "fdv":              float(pair.get("fdv") or 0),
        }
        _dex_cache[addr] = result
        return result

    except Exception as e:
        logger.warning(f"DexScreener fetch failed for {addr[:10]}…: {e}")
        _dex_cache[addr] = None
        return None


# ─── Outcome determination ────────────────────────────────────────────────────

def determine_outcome(
    log: dict,
    agent: Optional[dict],
    dex: Optional[dict],
) -> str:
    """
    Determine the outcome for a query_log row.

    Priority:
      1. DexScreener price drop signals (strongest signal for token health)
      2. Trust score signals (ACP behavioral data)
      3. Fallback to 'expired' (no data available)

    Returns one of: 'scam', 'failure', 'success', 'expired'
    """
    original_trust = log.get("trustScore") or 0

    # ── DexScreener signals ────────────────────────────────────────────────
    if dex:
        price_change_24h = dex.get("price_change_24h", 0) or 0
        liquidity_usd    = dex.get("liquidity_usd", 0) or 0
        price_usd        = dex.get("price_usd", 0) or 0
        volume_24h       = dex.get("volume_24h", 0) or 0

        # Absolute crash: price dropped 80%+ in 24h → scam
        if price_change_24h <= -80:
            logger.info(
                f"  → scam (price_change_24h={price_change_24h:.1f}%)"
            )
            return "scam"

        # Significant dump: price dropped 50%+ → failure
        if price_change_24h <= -50:
            logger.info(
                f"  → failure (price_change_24h={price_change_24h:.1f}%)"
            )
            return "failure"

        # Dead token: effectively zero liquidity and volume
        if liquidity_usd < 100 and volume_24h < 10 and price_usd < 0.0001:
            logger.info(
                f"  → failure (dead token: liq=${liquidity_usd:.2f}, vol=${volume_24h:.2f})"
            )
            return "failure"

        # Still active, reasonable liquidity → success
        if liquidity_usd > 500 and price_change_24h > -30:
            logger.info(
                f"  → success (liq=${liquidity_usd:.0f}, price_change={price_change_24h:.1f}%)"
            )
            return "success"

    # ── Agent score signals ────────────────────────────────────────────────
    if agent:
        current_trust = agent.get("trust_score") or 0
        total_jobs    = agent.get("total_jobs") or 0

        # Trust collapsed to 0 → failure
        if original_trust > 0 and current_trust == 0:
            logger.info(
                f"  → failure (trust_score: {original_trust} → 0)"
            )
            return "failure"

        # Trust dropped dramatically (>60 points) → failure
        trust_drop = original_trust - current_trust
        if trust_drop >= 60 and original_trust > 50:
            logger.info(
                f"  → failure (trust_score dropped {trust_drop} pts: {original_trust} → {current_trust})"
            )
            return "failure"

        # Still active with decent trust → success
        if current_trust >= 50 and total_jobs > 0:
            logger.info(
                f"  → success (trust={current_trust}, jobs={total_jobs})"
            )
            return "success"

    # ── No sufficient data → expired ──────────────────────────────────────
    logger.info("  → expired (insufficient data)")
    return "expired"


# ─── Main function ────────────────────────────────────────────────────────────

def run_auto_outcomes() -> dict:
    """
    Main auto-outcome reporter.

    Returns a summary dict:
      {
        "processed": int,
        "outcomes": {"scam": n, "failure": n, "success": n, "expired": n},
        "errors": int,
        "duration_seconds": float,
      }
    """
    start = time.time()
    logger.info("=" * 60)
    logger.info("WADJET AUTO-OUTCOME REPORTER STARTED")
    logger.info(f"  Lookback: {LOOKBACK_DAYS} days | Batch: {BATCH_SIZE}")
    logger.info("=" * 60)

    outcome_counts: dict[str, int] = {
        "scam":    0,
        "failure": 0,
        "success": 0,
        "expired": 0,
    }
    errors = 0

    conn = _get_conn()
    try:
        # Step 1: fetch pending logs
        pending = fetch_pending_logs(conn)
        logger.info(f"Pending logs to process: {len(pending)}")

        if not pending:
            logger.info("Nothing to process — all caught up!")
            return {
                "processed": 0,
                "outcomes": outcome_counts,
                "errors": 0,
                "duration_seconds": round(time.time() - start, 2),
            }

        for log in pending:
            log_id    = log["id"]
            target    = log.get("target") or ""
            orig_trust = log.get("trustScore") or 0

            logger.info(
                f"Processing [{log_id}] target={target[:12]}… "
                f"type={log.get('type')} trust_at_query={orig_trust}"
            )

            try:
                # Step 2: current agent score
                agent = fetch_agent_score(conn, target) if target else None

                # Step 3: DexScreener lookup (use token_address from agent_scores, or target itself)
                token_addr = None
                if agent and agent.get("token_address"):
                    token_addr = agent["token_address"]
                elif target and target.startswith("0x") and len(target) == 42:
                    token_addr = target

                dex = None
                if token_addr:
                    dex = fetch_dexscreener(token_addr)

                # Step 4: determine outcome
                outcome = determine_outcome(log, agent, dex)

                # Step 5: persist
                update_outcome(conn, log_id, outcome)
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
                logger.info(f"  ✓ SET outcome='{outcome}' for {log_id}")

            except Exception as e:
                logger.error(f"  ✗ Error processing {log_id}: {e}", exc_info=True)
                errors += 1
                # Don't let one row poison the whole batch
                continue

    finally:
        conn.close()

    duration = time.time() - start

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("AUTO-OUTCOME REPORTER COMPLETE")
    logger.info(f"  Processed:       {sum(outcome_counts.values())}")
    logger.info(f"  scam:            {outcome_counts['scam']}")
    logger.info(f"  failure:         {outcome_counts['failure']}")
    logger.info(f"  success:         {outcome_counts['success']}")
    logger.info(f"  expired:         {outcome_counts['expired']}")
    logger.info(f"  errors:          {errors}")
    logger.info(f"  duration:        {duration:.1f}s")
    logger.info("=" * 60)

    return {
        "processed":        sum(outcome_counts.values()),
        "outcomes":         outcome_counts,
        "errors":           errors,
        "duration_seconds": round(duration, 2),
    }


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_auto_outcomes()
    exit_code = 0 if result["errors"] == 0 else 1
    sys.exit(exit_code)
