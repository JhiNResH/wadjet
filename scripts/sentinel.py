#!/usr/bin/env python3
"""
Wadjet Sentinel — Real-time On-chain Monitoring Module (Stage 4)

Two-stage alert system:
  Stage 1 (hourly):    Scan ALL indexed tokens via GoPlus → add to watchlist on delta threshold
  Stage 2 (10-min):    Monitor watchlist items → Alchemy transfer tracking → detect dump/rug patterns

Usage:
    python scripts/sentinel.py --stage1          # Hourly GoPlus scan
    python scripts/sentinel.py --stage2          # High-frequency watchlist check
    python scripts/sentinel.py --stage1 --stage2 # Both

Environment:
    DATABASE_URL        — Supabase PostgreSQL connection string
    ALCHEMY_API_URL     — Alchemy base URL (default: https://base-mainnet.g.alchemy.com/v2/)
    ALCHEMY_API_KEY     — Alchemy API key
    GOPLUS_BATCH_SIZE   — tokens per GoPlus batch (default: 10)

Validation:
    $ELYS creator: 0xe25cbfce47b24b99b9108872263bbf3cf50b86e6
    $ELYS token:   0xb6bdc0f422a901062ecd948c6d0d785acd131ce1
"""

import argparse
import asyncio
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

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
logger = logging.getLogger("wadjet.sentinel")

# ─── Config ──────────────────────────────────────────────────────────────────
GOPLUS_BATCH_SIZE     = int(os.environ.get("GOPLUS_BATCH_SIZE", "10"))
GOPLUS_RATE_LIMIT_RPS = 10            # max 10 req/min → 1 req/6s between batches
GOPLUS_SLEEP_BETWEEN  = 6.5           # seconds between GoPlus batches
ALCHEMY_API_URL       = os.environ.get("ALCHEMY_API_URL", "https://base-mainnet.g.alchemy.com/v2/")
ALCHEMY_API_KEY       = os.environ.get("ALCHEMY_API_KEY", "")
CHAIN_ID              = "8453"

# ─── Stage 1: Delta thresholds (trigger watchlist addition) ──────────────────
THRESHOLDS = {
    "top10_holder_pct_delta":  0.05,   # +5% concentration jump → suspicious
    "creator_percent_delta":   0.03,   # +3% creator accumulation
    "liquidity_delta_pct":    -0.30,   # -30% liquidity drop
    "volume_delta_pct":        5.00,   # 5x volume spike
    "price_delta_pct":        -0.50,   # -50% price crash
}

# ─── Stage 2: Sell pattern thresholds ────────────────────────────────────────
SELL_SIGNAL_PCT    = 0.20   # > 20% of holdings in a single tx
DUMP_WINDOW_HOURS  = 4      # multiple sells within 4h window
CONFIRMED_DUMP_PCT = 0.80   # > 80% sold within 48h
ABANDONMENT_HOURS  = 6      # no tx for 6+ hours after big sell
CONFIRMED_DUMP_HOURS = 48   # 48-hour window for confirmed dump


# ─── DB helpers ──────────────────────────────────────────────────────────────

def _get_db_conn():
    import psycopg2
    from db.supabase_client import DATABASE_URL
    return psycopg2.connect(DATABASE_URL)


def _log_cron(run_id: str, status: str, extra: dict):
    """Write a sentinel run record to cron_logs."""
    try:
        from db.supabase_client import get_cursor
        with get_cursor() as cur:
            cur.execute("""
                INSERT INTO cron_logs (run_id, status, extra, ran_at)
                VALUES (%s, %s, %s::jsonb, NOW())
                ON CONFLICT (run_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    extra  = EXCLUDED.extra
            """, (run_id, status, __import__("json").dumps(extra)))
    except Exception as e:
        logger.warning(f"cron_logs write failed (non-fatal): {e}")


# ─── GoPlus helpers ──────────────────────────────────────────────────────────

async def fetch_goplus_batch(token_addresses: list[str], client: httpx.AsyncClient) -> dict:
    """
    Fetch GoPlus token_security for up to 10 tokens in one call.
    Returns dict keyed by lowercase token_address.
    """
    addresses_str = ",".join(a.lower() for a in token_addresses)
    try:
        resp = await client.get(
            f"https://api.gopluslabs.io/api/v1/token_security/{CHAIN_ID}",
            params={"contract_addresses": addresses_str},
            timeout=15.0,
        )
        if resp.status_code != 200:
            logger.warning(f"GoPlus HTTP {resp.status_code} for batch of {len(token_addresses)}")
            return {}
        data = resp.json()
        return data.get("result") or {}
    except Exception as e:
        logger.warning(f"GoPlus batch fetch failed: {e}")
        return {}


def _goplus_to_snapshot(gp: dict, token_address: str) -> dict:
    """Extract comparable snapshot fields from GoPlus result."""
    top10 = sum(
        float(h.get("percent", 0)) for h in gp.get("holders", [])[:10]
    )
    lp_locked = sum(
        float(h.get("percent", 0)) for h in gp.get("lp_holders", [])
        if h.get("is_locked") == 1
    )
    return {
        "token_address":   token_address.lower(),
        "top10_holder_pct": top10,
        "creator_percent": float(gp.get("creator_percent") or 0),
        "owner_percent":   float(gp.get("owner_percent") or 0),
        "lp_locked_pct":   lp_locked,
        "holder_count":    int(gp.get("holder_count") or 0),
        "creator_address": gp.get("creator_address", ""),
        "is_honeypot":     gp.get("is_honeypot", "0") == "1",
        "sell_tax":        float(gp.get("sell_tax") or 0),
    }


def _compute_delta_severity(deltas: dict) -> tuple[bool, str, str]:
    """
    Given deltas dict, decide if token should be watchlisted.
    Returns: (should_flag, reason, severity)
    """
    reasons = []
    max_sev = "medium"

    if deltas.get("top10_delta", 0) >= THRESHOLDS["top10_holder_pct_delta"]:
        pct = deltas["top10_delta"] * 100
        reasons.append(f"Top10 holder concentration +{pct:.1f}%")
        max_sev = "high"

    if deltas.get("creator_delta", 0) >= THRESHOLDS["creator_percent_delta"]:
        pct = deltas["creator_delta"] * 100
        reasons.append(f"Creator holdings increased +{pct:.1f}%")
        max_sev = "high"

    liq_delta = deltas.get("liquidity_delta_pct", 0)
    if liq_delta <= THRESHOLDS["liquidity_delta_pct"]:
        reasons.append(f"Liquidity dropped {liq_delta*100:.0f}%")
        max_sev = "critical"

    price_delta = deltas.get("price_delta_pct", 0)
    if price_delta <= THRESHOLDS["price_delta_pct"]:
        reasons.append(f"Price crashed {price_delta*100:.0f}%")
        max_sev = "critical"

    vol_delta = deltas.get("volume_delta_pct", 0)
    if vol_delta >= THRESHOLDS["volume_delta_pct"]:
        reasons.append(f"Volume spike {vol_delta:.1f}x")
        max_sev = "high" if max_sev == "medium" else max_sev

    if not reasons:
        return False, "", "medium"

    return True, " | ".join(reasons), max_sev


# ─── Alchemy helpers ─────────────────────────────────────────────────────────

def _alchemy_url() -> str:
    base = ALCHEMY_API_URL.rstrip("/")
    if ALCHEMY_API_KEY and not base.endswith(ALCHEMY_API_KEY):
        return f"{base}/{ALCHEMY_API_KEY}"
    return base


async def fetch_recent_transfers(
    token_address: str,
    wallet_address: str,
    client: httpx.AsyncClient,
    direction: str = "from",  # "from" or "to"
    max_count: int = 50,
) -> list:
    """
    Fetch recent ERC20 transfers from/to a specific wallet for a specific token.

    direction="from" → wallet is SELLING (fromAddress=wallet)
    direction="to"   → wallet is BUYING  (toAddress=wallet)
    """
    url = _alchemy_url()
    payload_params: dict = {
        "contractAddresses": [token_address],
        "category": ["erc20"],
        "order": "desc",
        "maxCount": hex(max_count),
        "withMetadata": True,
    }
    if direction == "from":
        payload_params["fromAddress"] = wallet_address
    else:
        payload_params["toAddress"] = wallet_address

    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "alchemy_getAssetTransfers",
        "params":  [payload_params],
    }
    try:
        resp = await client.post(url, json=payload, timeout=15.0)
        if resp.status_code != 200:
            logger.warning(f"Alchemy transfers HTTP {resp.status_code} for {wallet_address[:10]}...")
            return []
        data = resp.json()
        return (data.get("result") or {}).get("transfers", [])
    except Exception as e:
        logger.warning(f"Alchemy transfers fetch failed: {e}")
        return []


async def fetch_external_transfers_from(
    wallet_address: str,
    client: httpx.AsyncClient,
    max_count: int = 50,
) -> list:
    """
    Fetch external (ETH) transfers FROM a wallet.
    Used to find wallets funded by the creator (connected wallet detection).
    """
    url = _alchemy_url()
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "alchemy_getAssetTransfers",
        "params":  [{
            "fromAddress": wallet_address,
            "category":    ["external"],
            "order":       "desc",
            "maxCount":    hex(max_count),
            "withMetadata": True,
        }],
    }
    try:
        resp = await client.post(url, json=payload, timeout=15.0)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return (data.get("result") or {}).get("transfers", [])
    except Exception as e:
        logger.warning(f"Alchemy external transfers failed: {e}")
        return []


# ─── Connected wallet detection ──────────────────────────────────────────────

async def find_connected_wallets(
    creator_address: str,
    token_address: str,
    client: httpx.AsyncClient,
    lookback_days: int = 30,
) -> list[str]:
    """
    Simple heuristic: find wallets that:
    1. Received ETH from creator_address in last 30 days (funded wallets)
    2. AND have buy transfers for the token (active in the token)

    Returns list of connected wallet addresses (lowercase).
    Phase 2 will do full graph traversal.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    connected = []

    # Step 1: wallets funded by creator
    ext_transfers = await fetch_external_transfers_from(creator_address, client)
    funded_wallets = set()
    for tx in ext_transfers:
        # Filter by cutoff
        ts = tx.get("metadata", {}).get("blockTimestamp")
        if ts:
            try:
                tx_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if tx_time < cutoff:
                    continue
            except Exception:
                pass
        to_addr = tx.get("to", "")
        if to_addr and to_addr.startswith("0x") and len(to_addr) == 42:
            funded_wallets.add(to_addr.lower())

    if not funded_wallets:
        return []

    # Step 2: check if any funded wallet bought the token
    # Fetch "to" transfers for the token (buyers)
    buy_transfers = await fetch_recent_transfers(
        token_address, creator_address, client,
        direction="to", max_count=100
    )
    buyers = {tx.get("to", "").lower() for tx in buy_transfers if tx.get("to")}

    # Intersection = connected wallets that are also buying
    connected = list(funded_wallets & buyers)
    if connected:
        logger.info(
            f"Connected wallets for {creator_address[:10]}...: {connected[:3]}..."
            f"(total {len(connected)})"
        )
    return connected[:5]  # cap at 5 for now


# ─── Sell pattern detection ───────────────────────────────────────────────────

def _parse_tx_time(tx: dict) -> Optional[datetime]:
    """Parse block timestamp from Alchemy transfer metadata."""
    ts = (tx.get("metadata") or {}).get("blockTimestamp")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def detect_sell_patterns(
    transfers: list,
    initial_holdings: float,
) -> dict:
    """
    Analyse ERC20 outbound transfers from a wallet to detect dump patterns.

    Args:
        transfers:        List of Alchemy transfer dicts (already filtered to FROM wallet)
        initial_holdings: Estimated initial token holdings (raw units)

    Returns dict with detected patterns:
        sell_signal       — bool: single tx > 20% of holdings
        dump_pattern      — bool: multiple sells within 4h
        confirmed_dump    — bool: total sold > 80% within 48h
        abandonment       — bool: no tx for 6h+ after big sell
        total_sold        — float: total value sold
        details           — dict: supporting data
    """
    now = datetime.now(timezone.utc)
    window_48h = now - timedelta(hours=CONFIRMED_DUMP_HOURS)
    window_4h  = now - timedelta(hours=DUMP_WINDOW_HOURS)

    patterns = {
        "sell_signal":    False,
        "dump_pattern":   False,
        "confirmed_dump": False,
        "abandonment":    False,
        "total_sold":     0.0,
        "sells_count":    0,
        "last_sell_time": None,
        "details":        {},
    }

    if not transfers:
        return patterns

    sells_48h  = []
    sells_4h   = []
    last_sell  = None

    for tx in transfers:
        value = float(tx.get("value") or 0)
        tx_time = _parse_tx_time(tx)

        if value <= 0:
            continue

        patterns["total_sold"] += value
        patterns["sells_count"] += 1

        if tx_time:
            if tx_time >= window_48h:
                sells_48h.append((tx_time, value))
            if tx_time >= window_4h:
                sells_4h.append((tx_time, value))
            if last_sell is None or tx_time > last_sell:
                last_sell = tx_time

    patterns["last_sell_time"] = last_sell.isoformat() if last_sell else None

    # SELL_SIGNAL: single tx > 20% of holdings
    if initial_holdings > 0:
        for tx in transfers:
            value = float(tx.get("value") or 0)
            if value / initial_holdings > SELL_SIGNAL_PCT:
                patterns["sell_signal"] = True
                patterns["details"]["sell_signal_tx"] = tx.get("hash")
                break

    # DUMP_PATTERN: multiple (≥2) sells within 4h window
    if len(sells_4h) >= 2:
        patterns["dump_pattern"] = True
        patterns["details"]["sells_in_4h"] = len(sells_4h)

    # CONFIRMED_DUMP: total sold > 80% of holdings within 48h
    total_48h = sum(v for _, v in sells_48h)
    if initial_holdings > 0 and total_48h / initial_holdings > CONFIRMED_DUMP_PCT:
        patterns["confirmed_dump"] = True
        patterns["details"]["pct_sold_48h"] = round(total_48h / initial_holdings * 100, 1)

    # ABANDONMENT: no tx in 6h+ after a big sell (sell_signal + silence)
    if patterns["sell_signal"] and last_sell:
        hours_since = (now - last_sell).total_seconds() / 3600
        if hours_since >= ABANDONMENT_HOURS:
            patterns["abandonment"] = True
            patterns["details"]["hours_silent"] = round(hours_since, 1)

    return patterns


# ─── Alert DB functions ───────────────────────────────────────────────────────

def create_alert(
    token_address: str,
    alert_type: str,
    severity: str,
    details: dict,
    wallet_address: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> None:
    """Insert a new alert into wadjet_alerts."""
    import json
    try:
        from db.supabase_client import get_cursor
        with get_cursor() as cur:
            cur.execute("""
                INSERT INTO wadjet_alerts
                    (token_address, wallet_address, agent_name, alert_type, severity, details)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            """, (
                token_address.lower(),
                wallet_address,
                agent_name,
                alert_type,
                severity,
                json.dumps(details),
            ))
        logger.info(
            f"🚨 Alert created: [{severity.upper()}] {alert_type} "
            f"for {token_address[:10]}..."
        )
    except Exception as e:
        logger.error(f"Failed to create alert: {e}")


def update_watchlist_status(token_address: str, status: str, notes: str = "") -> None:
    """Update watchlist item status (e.g., 'confirmed_rug')."""
    try:
        from db.supabase_client import get_cursor
        with get_cursor() as cur:
            cur.execute("""
                UPDATE wadjet_watchlist
                SET status = %s,
                    notes  = %s,
                    last_checked = NOW()
                WHERE token_address = %s
            """, (status, notes, token_address.lower()))
    except Exception as e:
        logger.error(f"Failed to update watchlist status: {e}")


def update_agent_rug_score(token_address: str, rug_score: int = 87) -> None:
    """Elevate rug_score on agent_scores for a confirmed rug token."""
    try:
        from db.supabase_client import get_cursor
        with get_cursor() as cur:
            cur.execute("""
                UPDATE agent_scores
                SET rug_score    = GREATEST(COALESCE(rug_score, 0), %s),
                    last_updated = NOW()
                WHERE token_address = %s
            """, (rug_score, token_address.lower()))
            affected = cur.rowcount
            if affected:
                logger.info(
                    f"Updated rug_score → {rug_score} for {token_address[:10]}... "
                    f"({affected} rows)"
                )
    except Exception as e:
        logger.warning(f"Failed to update agent rug_score (non-fatal): {e}")


# ─── Stage 1: Hourly Scan ─────────────────────────────────────────────────────

async def run_stage1_scan() -> dict:
    """
    Stage 1: Hourly GoPlus scan of all indexed tokens.

    1. Fetch all agents with token_address from agent_scores
    2. Batch-fetch GoPlus token_security (10 req/min rate limit)
    3. Compare with last snapshot → compute deltas
    4. If any threshold triggered → add to wadjet_watchlist
    5. Log to cron_logs
    """
    run_id = f"sentinel-stage1-{uuid.uuid4().hex[:8]}"
    started_at = time.time()

    logger.info(f"=== Stage 1 Scan started (run_id={run_id}) ===")

    # 1. Fetch indexed tokens
    from db.supabase_client import get_cursor, upsert_watchlist_item, get_last_two_snapshots

    tokens = []
    try:
        with get_cursor() as cur:
            cur.execute("""
                SELECT token_address, wallet_address, agent_name
                FROM agent_scores
                WHERE token_address IS NOT NULL
                  AND token_address != ''
                ORDER BY last_updated DESC NULLS LAST
                LIMIT 500
            """)
            tokens = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.error(f"Stage 1: Failed to fetch agent_scores: {e}")
        _log_cron(run_id, "failed", {"error": str(e), "stage": 1})
        return {"status": "failed", "error": str(e)}

    logger.info(f"Stage 1: {len(tokens)} tokens to scan")

    # 2. Batch GoPlus calls with rate limiting
    watchlisted = 0
    errors = []

    token_list = [t["token_address"].lower() for t in tokens]
    token_meta  = {t["token_address"].lower(): t for t in tokens}

    async with httpx.AsyncClient() as client:
        for batch_start in range(0, len(token_list), GOPLUS_BATCH_SIZE):
            batch = token_list[batch_start: batch_start + GOPLUS_BATCH_SIZE]
            logger.info(
                f"Stage 1: GoPlus batch {batch_start // GOPLUS_BATCH_SIZE + 1} "
                f"({len(batch)} tokens)"
            )

            gp_results = await fetch_goplus_batch(batch, client)

            for token_addr in batch:
                gp = gp_results.get(token_addr, {})
                if not gp:
                    continue

                try:
                    current = _goplus_to_snapshot(gp, token_addr)
                    snapshots = get_last_two_snapshots(token_addr)

                    if len(snapshots) < 1:
                        # No history yet — just note, don't flag
                        continue

                    # Compare with most recent snapshot
                    prev = snapshots[0]

                    def _rel_delta(cur_val, prev_val):
                        if prev_val and prev_val != 0:
                            return (cur_val - prev_val) / abs(prev_val)
                        return 0.0

                    deltas = {
                        "top10_delta":          current["top10_holder_pct"] - (prev.get("top10_holder_pct") or 0),
                        "creator_delta":        current["creator_percent"]  - (prev.get("creator_percent") or 0),
                        "liquidity_delta_pct":  _rel_delta(
                            current.get("lp_locked_pct", 0), prev.get("lp_locked_pct", 0)
                        ),
                        "price_delta_pct":      0.0,   # no price from GoPlus; handled by DexScreener in daily_cron
                        "volume_delta_pct":     0.0,
                    }

                    # Immediate honeypot / sell-tax flags bypass delta check
                    if current.get("is_honeypot"):
                        should_flag, reason, sev = True, "GoPlus: HONEYPOT detected", "critical"
                    elif current.get("sell_tax", 0) >= 0.5:
                        should_flag, reason, sev = True, f"GoPlus: sell tax {current['sell_tax']:.0%}", "critical"
                    else:
                        should_flag, reason, sev = _compute_delta_severity(deltas)

                    if should_flag:
                        meta = token_meta.get(token_addr, {})
                        upsert_watchlist_item({
                            "token_address": token_addr,
                            "wallet_address": meta.get("wallet_address") or current.get("creator_address"),
                            "agent_name":    meta.get("agent_name"),
                            "reason":        reason,
                            "severity":      sev,
                        })

                        create_alert(
                            token_address=token_addr,
                            alert_type="watchlist_added",
                            severity="info" if sev == "medium" else sev,
                            details={
                                "reason":  reason,
                                "deltas":  deltas,
                                "goplus_snapshot": current,
                            },
                            wallet_address=meta.get("wallet_address") or current.get("creator_address"),
                            agent_name=meta.get("agent_name"),
                        )

                        watchlisted += 1
                        logger.info(
                            f"Watchlisted [{sev.upper()}] {token_addr[:12]}...: {reason}"
                        )

                except Exception as e:
                    err_msg = f"Stage 1 error for {token_addr[:10]}...: {e}"
                    logger.warning(err_msg)
                    errors.append(err_msg)

            # Rate limit: sleep between batches
            if batch_start + GOPLUS_BATCH_SIZE < len(token_list):
                logger.debug(f"Rate limit sleep {GOPLUS_SLEEP_BETWEEN}s...")
                await asyncio.sleep(GOPLUS_SLEEP_BETWEEN)

    duration = round(time.time() - started_at, 1)
    result = {
        "stage": 1,
        "status":       "ok",
        "tokens_scanned": len(token_list),
        "newly_watchlisted": watchlisted,
        "errors":       len(errors),
        "duration_s":   duration,
        "run_id":       run_id,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }

    _log_cron(run_id, "ok", result)
    logger.info(
        f"=== Stage 1 done: {len(token_list)} scanned, "
        f"{watchlisted} watchlisted, {len(errors)} errors in {duration}s ==="
    )
    return result


# ─── Stage 2: High-frequency Watchlist Check ─────────────────────────────────

async def check_single_watchlist_item(
    item: dict,
    client: httpx.AsyncClient,
) -> dict:
    """
    For one watchlisted token:
    1. Detect connected wallets
    2. Fetch their ERC20 sells for this token
    3. Run pattern detection
    4. If CONFIRMED_DUMP or (DUMP_PATTERN + ABANDONMENT) → escalate
    """
    token_address  = item["token_address"]
    wallet_address = item.get("wallet_address")
    agent_name     = item.get("agent_name")

    result = {
        "token_address": token_address,
        "wallet_address": wallet_address,
        "patterns": {},
        "escalated": False,
        "alert_type": None,
    }

    # Need a wallet to check
    if not wallet_address:
        # Try to find creator_address from GoPlus
        try:
            gp = await fetch_goplus_batch([token_address], client)
            gp_data = gp.get(token_address.lower(), {})
            wallet_address = gp_data.get("creator_address")
        except Exception:
            pass

    if not wallet_address:
        logger.debug(f"Stage 2: No wallet for {token_address[:12]}... — skipping")
        return result

    # Update last_checked
    try:
        from db.supabase_client import get_cursor
        with get_cursor() as cur:
            cur.execute(
                "UPDATE wadjet_watchlist SET last_checked = NOW() WHERE token_address = %s",
                (token_address,)
            )
    except Exception:
        pass

    # Find connected wallets
    connected_wallets = await find_connected_wallets(wallet_address, token_address, client)
    all_wallets_to_check = [wallet_address] + connected_wallets

    # Estimate initial holdings (total tokens received by wallet)
    received_transfers = await fetch_recent_transfers(
        token_address, wallet_address, client,
        direction="to", max_count=50
    )
    initial_holdings = sum(float(tx.get("value") or 0) for tx in received_transfers)
    # Fallback: if no received, assume unknown (use 1.0 relative scale)
    if initial_holdings == 0:
        initial_holdings = 1.0

    # Fetch outbound sells
    all_sells = []
    for wallet in all_wallets_to_check:
        sells = await fetch_recent_transfers(
            token_address, wallet, client,
            direction="from", max_count=50
        )
        all_sells.extend(sells)

    # Deduplicate by tx hash
    seen_hashes = set()
    unique_sells = []
    for tx in all_sells:
        h = tx.get("hash")
        if h and h not in seen_hashes:
            seen_hashes.add(h)
            unique_sells.append(tx)

    patterns = detect_sell_patterns(unique_sells, initial_holdings)
    result["patterns"] = patterns

    # Determine if we need to escalate
    is_confirmed_dump = patterns["confirmed_dump"]
    is_dump_abandonment = patterns["dump_pattern"] and patterns["abandonment"]

    if is_confirmed_dump:
        alert_type = "confirmed_rug"
        severity   = "critical"
        result["escalated"] = True
        result["alert_type"] = alert_type

        # Update watchlist → confirmed_rug
        notes = (
            f"Confirmed rug: {patterns['details'].get('pct_sold_48h', '?')}% "
            f"of holdings sold within 48h. "
            f"Wallet silent {patterns['details'].get('hours_silent', '?')}h."
        )
        update_watchlist_status(token_address, "confirmed_rug", notes)
        update_agent_rug_score(token_address, rug_score=90)
        create_alert(
            token_address=token_address,
            alert_type="confirmed_rug",
            severity="critical",
            details={**patterns, "connected_wallets": connected_wallets},
            wallet_address=wallet_address,
            agent_name=agent_name,
        )
        logger.warning(
            f"🔴 CONFIRMED RUG: {token_address[:12]}... "
            f"wallet={wallet_address[:10]}... "
            f"sold={patterns['details'].get('pct_sold_48h', '?')}%"
        )

    elif is_dump_abandonment:
        alert_type = "dump_pattern"
        severity   = "critical"
        result["escalated"] = True
        result["alert_type"] = alert_type

        notes = (
            f"Dump+Abandonment: {patterns['sells_count']} sells, "
            f"wallet silent {patterns['details'].get('hours_silent', '?')}h."
        )
        update_watchlist_status(token_address, "confirmed_rug", notes)
        update_agent_rug_score(token_address, rug_score=87)
        create_alert(
            token_address=token_address,
            alert_type="dump_pattern",
            severity="critical",
            details={**patterns, "connected_wallets": connected_wallets},
            wallet_address=wallet_address,
            agent_name=agent_name,
        )
        logger.warning(
            f"🟠 DUMP+ABANDONMENT: {token_address[:12]}... "
            f"wallet={wallet_address[:10]}... "
            f"sells={patterns['sells_count']}"
        )

    elif patterns["sell_signal"]:
        # Non-escalating but worth alerting
        create_alert(
            token_address=token_address,
            alert_type="sell_signal",
            severity="warning",
            details=patterns,
            wallet_address=wallet_address,
            agent_name=agent_name,
        )
        result["alert_type"] = "sell_signal"
        logger.info(
            f"⚠️  SELL_SIGNAL: {token_address[:12]}... "
            f"wallet={wallet_address[:10]}..."
        )

    elif patterns["dump_pattern"]:
        create_alert(
            token_address=token_address,
            alert_type="dump_pattern",
            severity="warning",
            details=patterns,
            wallet_address=wallet_address,
            agent_name=agent_name,
        )
        result["alert_type"] = "dump_pattern"

    return result


async def run_stage2_check() -> dict:
    """
    Stage 2: High-frequency watchlist monitoring (every 10 min).

    1. Fetch all active watchlist items
    2. For each: detect sell patterns via Alchemy
    3. Escalate confirmed rugs
    4. Log results
    """
    run_id = f"sentinel-stage2-{uuid.uuid4().hex[:8]}"
    started_at = time.time()

    logger.info(f"=== Stage 2 Check started (run_id={run_id}) ===")

    from db.supabase_client import get_watchlist
    items = get_watchlist(status="active", limit=200)

    logger.info(f"Stage 2: {len(items)} active watchlist items to check")

    if not items:
        result = {
            "stage": 2,
            "status": "ok",
            "watchlist_checked": 0,
            "escalated": 0,
            "run_id": run_id,
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }
        _log_cron(run_id, "ok", result)
        return result

    escalated = 0
    errors = []

    async with httpx.AsyncClient() as client:
        for item in items:
            try:
                check_result = await check_single_watchlist_item(item, client)
                if check_result.get("escalated"):
                    escalated += 1
            except Exception as e:
                err_msg = f"Stage 2 error for {item.get('token_address', '?')[:10]}...: {e}"
                logger.warning(err_msg)
                errors.append(err_msg)

            # Small sleep between items to avoid Alchemy rate limits
            await asyncio.sleep(0.5)

    duration = round(time.time() - started_at, 1)
    result = {
        "stage": 2,
        "status":            "ok",
        "watchlist_checked": len(items),
        "escalated":         escalated,
        "errors":            len(errors),
        "duration_s":        duration,
        "run_id":            run_id,
        "completed_at":      datetime.utcnow().isoformat() + "Z",
    }

    _log_cron(run_id, "ok", result)
    logger.info(
        f"=== Stage 2 done: {len(items)} checked, "
        f"{escalated} escalated, {len(errors)} errors in {duration}s ==="
    )
    return result


# ─── Entrypoint ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Wadjet Sentinel — on-chain monitoring")
    parser.add_argument("--stage1", action="store_true", help="Run Stage 1 (hourly GoPlus scan)")
    parser.add_argument("--stage2", action="store_true", help="Run Stage 2 (watchlist check)")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate $ELYS token: 0xb6bdc0f422a901062ecd948c6d0d785acd131ce1",
    )
    args = parser.parse_args()

    if args.validate:
        logger.info("=== Validation Mode: $ELYS token ===")
        ELYS_CREATOR = "0xe25cbfce47b24b99b9108872263bbf3cf50b86e6"
        ELYS_TOKEN   = "0xb6bdc0f422a901062ecd948c6d0d785acd131ce1"

        async with httpx.AsyncClient() as client:
            logger.info("Fetching GoPlus data for $ELYS...")
            gp = await fetch_goplus_batch([ELYS_TOKEN], client)
            gp_data = gp.get(ELYS_TOKEN, {})
            if gp_data:
                snap = _goplus_to_snapshot(gp_data, ELYS_TOKEN)
                logger.info(f"GoPlus snapshot: {snap}")
            else:
                logger.warning("No GoPlus data for $ELYS (may be unlisted)")

            logger.info(f"Fetching transfers for creator: {ELYS_CREATOR}")
            sells = await fetch_recent_transfers(ELYS_TOKEN, ELYS_CREATOR, client, direction="from")
            logger.info(f"Found {len(sells)} outbound transfers from creator")

            if sells:
                patterns = detect_sell_patterns(sells, initial_holdings=1_000_000)
                logger.info(f"Sell patterns: {patterns}")

            connected = await find_connected_wallets(ELYS_CREATOR, ELYS_TOKEN, client)
            logger.info(f"Connected wallets: {connected}")

        return

    if not args.stage1 and not args.stage2:
        parser.print_help()
        sys.exit(1)

    results = []

    if args.stage1:
        r1 = await run_stage1_scan()
        results.append(r1)

    if args.stage2:
        r2 = await run_stage2_check()
        results.append(r2)

    # Summary
    for r in results:
        logger.info(f"Result Stage {r.get('stage', '?')}: {r}")


if __name__ == "__main__":
    asyncio.run(main())
