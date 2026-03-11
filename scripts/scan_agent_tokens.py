"""
Wadjet Phase 3 — Agent Token Scanner
======================================
Scans all 2,752 agent tokens through the v2 model.
Outputs ranked list of riskiest tokens.
Stores results in Supabase `wadjet_agent_token_risks` table.

Usage:
  python scripts/scan_agent_tokens.py [--limit N] [--dry-run]

Options:
  --limit N    Only scan first N tokens (default: all)
  --dry-run    Compute predictions but don't write to DB
  --cache-only Use cached DexScreener/GoPlus data only (no new API calls)
"""

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from db.utils import get_db_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scan_agent_tokens")

SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR.parent / "data"
MODELS_DIR  = SCRIPT_DIR.parent / "models"
CACHE_FILE  = DATA_DIR / "dex_cache.json"

DB_URL = get_db_url()
ALCHEMY_KEY  = "okgmVpKT-5iqER0g5yjyn"
ALCHEMY_BASE = f"https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}"
DEXSCREENER_URL = "https://api.dexscreener.com/latest/dex/tokens/{}"
GOPLUS_URL   = "https://api.gopluslabs.io/api/v1/token_security/8453"
THRESHOLD    = 0.35

# Feature order MUST match training
V1_FEATURES = [
    "holder_concentration",
    "liquidity_lock_ratio",
    "creator_tx_pattern",
    "buy_sell_ratio",
    "contract_similarity_score",
    "fund_flow_pattern",
    "price_change_24h",
    "liquidity_usd",
    "volume_24h",
    "total_jobs",
    "completion_rate",
    "trust_score",
    "age_days",
    "lp_drain_rate",
    "deployer_age_days",
    "token_supply_concentration",
    "renounced_ownership",
    "verified_contract",
    "social_presence_score",
    "audit_score",
]

VIRTUALS_FEATURES = [
    "bonding_curve_position",
    "lp_locked_pct",
    "creator_other_tokens",
    "creator_wallet_age",
    "holder_count_norm",
    "top10_holder_pct",
    "acp_job_count",
    "acp_completion_rate",
    "acp_trust_score",
    "token_age_days_norm",
    "volume_to_mcap_ratio",
    "price_volatility_7d",
    "social_presence",
    "is_honeypot",
    "is_mintable",
    "hidden_owner",
    "slippage_modifiable",
    "buy_tax",
    "sell_tax",
    "creator_percent",
    "owner_percent",
    "has_dex_data",
    "is_virtuals_token",
]

ALL_FEATURES = V1_FEATURES + VIRTUALS_FEATURES


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))

def load_model():
    """Load v2 model; fall back to v1 if v2 not found."""
    v2_path = MODELS_DIR / "wadjet_xgb_v2_agent.joblib"
    v1_path = MODELS_DIR / "wadjet_xgb.joblib"
    if v2_path.exists():
        model = joblib.load(v2_path)
        logger.info(f"Loaded V2 agent model from {v2_path}")
        return model, "v2"
    elif v1_path.exists():
        model = joblib.load(v1_path)
        logger.warning(f"V2 model not found — using V1 fallback from {v1_path}")
        return model, "v1"
    else:
        logger.error("No model found. Run training first.")
        sys.exit(1)

def fetch_agents_from_db(limit: int = 0) -> list[dict]:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    query = """
        SELECT wallet_address, token_address, token_symbol,
               trust_score, completion_rate, total_jobs, last_updated
        FROM agent_scores
        WHERE token_address IS NOT NULL
        ORDER BY trust_score DESC
    """
    if limit > 0:
        query += f" LIMIT {limit}"
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "wallet_address": r[0],
            "token_address": r[1].lower() if r[1] else None,
            "token_symbol": r[2],
            "trust_score": r[3],
            "completion_rate": r[4],
            "total_jobs": r[5],
            "last_updated": str(r[6]),
        }
        for r in rows if r[1]
    ]

def fetch_dex_data(token_addr: str, cache: dict, cache_only: bool = False) -> dict:
    key = f"dex:{token_addr}"
    cached = cache.get(key)
    if cached and (time.time() - cached.get("_ts", 0)) < 86400:
        return cached
    if cache_only:
        return {}

    url = DEXSCREENER_URL.format(token_addr)
    try:
        resp = httpx.get(url, timeout=15.0)
        if resp.status_code == 429:
            logger.warning("DexScreener rate limit — sleeping 15s")
            time.sleep(15)
            resp = httpx.get(url, timeout=15.0)
        if resp.status_code != 200:
            return {}

        data = resp.json()
        pairs = data.get("pairs") or []
        if not pairs:
            cache[key] = {"_ts": time.time(), "no_data": True}
            return {}

        pair = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0)
        result = {
            "_ts": time.time(),
            "price_usd": float(pair.get("priceUsd") or 0),
            "price_change_24h": float((pair.get("priceChange") or {}).get("h24") or 0),
            "price_change_6h": float((pair.get("priceChange") or {}).get("h6") or 0),
            "volume_24h": float((pair.get("volume") or {}).get("h24") or 0),
            "liquidity_usd": float((pair.get("liquidity") or {}).get("usd") or 0),
            "market_cap": float(pair.get("marketCap") or 0),
            "fdv": float(pair.get("fdv") or 0),
            "pair_created_at": pair.get("pairCreatedAt"),
            "txns_24h_buys": int((pair.get("txns") or {}).get("h24", {}).get("buys") or 0),
            "txns_24h_sells": int((pair.get("txns") or {}).get("h24", {}).get("sells") or 0),
            "info_websites": [w.get("url","") for w in (pair.get("info") or {}).get("websites", [])],
            "info_socials": [s.get("url","") for s in (pair.get("info") or {}).get("socials", [])],
            "base_token": pair.get("baseToken", {}).get("symbol", ""),
            "dex_id": pair.get("dexId", ""),
        }
        cache[key] = result
        return result
    except Exception as e:
        logger.debug(f"DexScreener error {token_addr}: {e}")
        return {}

def fetch_goplus_data(token_addr: str, cache: dict, cache_only: bool = False) -> dict:
    key = f"gp:{token_addr}"
    cached = cache.get(key)
    if cached and (time.time() - cached.get("_ts", 0)) < 86400:
        return cached
    if cache_only:
        return {}

    try:
        resp = httpx.get(GOPLUS_URL, params={"contract_addresses": token_addr}, timeout=15.0)
        time.sleep(0.25)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        rd = (data.get("result") or {}).get(token_addr.lower(), {})
        if not rd:
            return {}

        lp_holders = rd.get("lp_holders", [])
        lp_locked = sum(
            float(h.get("percent", 0)) for h in lp_holders if h.get("is_locked") == 1
        ) if lp_holders else 0.0

        result = {
            "_ts": time.time(),
            "is_honeypot": rd.get("is_honeypot", "0") == "1",
            "is_mintable": rd.get("is_mintable", "0") == "1",
            "hidden_owner": rd.get("hidden_owner", "0") == "1",
            "can_take_back_ownership": rd.get("can_take_back_ownership", "0") == "1",
            "slippage_modifiable": rd.get("slippage_modifiable", "0") == "1",
            "is_open_source": rd.get("is_open_source", "0") == "1",
            "buy_tax": float(rd.get("buy_tax") or 0),
            "sell_tax": float(rd.get("sell_tax") or 0),
            "holder_count": int(rd.get("holder_count") or 0),
            "top10_holder_pct": sum(
                float(h.get("percent", 0)) for h in rd.get("holders", [])[:10]
            ),
            "lp_locked_pct": lp_locked,
            "creator_address": rd.get("creator_address", ""),
            "creator_percent": float(rd.get("creator_percent") or 0),
            "owner_percent": float(rd.get("owner_percent") or 0),
        }
        cache[key] = result
        return result
    except Exception as e:
        logger.debug(f"GoPlus error {token_addr}: {e}")
        return {}

def build_feature_vector(agent: dict, dex: dict, gp: dict) -> np.ndarray:
    """Build unified feature vector matching ALL_FEATURES order."""
    # Token age
    created_at = dex.get("pair_created_at")
    token_age_days = 0.0
    if created_at:
        try:
            age_ms = time.time() * 1000 - float(created_at)
            token_age_days = max(0, age_ms / (1000 * 86400))
        except Exception:
            pass

    # Basic signals
    buys  = dex.get("txns_24h_buys", 0)
    sells = dex.get("txns_24h_sells", 0)
    buy_sell_ratio = buys / (buys + sells) if (buys + sells) > 0 else 0.5

    volume_24 = dex.get("volume_24h", 0)
    mcap      = dex.get("market_cap", 0)
    liquidity = dex.get("liquidity_usd", 0)
    price24h  = dex.get("price_change_24h", 0)

    vol_to_mcap = min(volume_24 / mcap, 10.0) if mcap > 0 else 0.0
    price_vol   = abs(price24h) / 100.0

    # Social presence
    websites = dex.get("info_websites", [])
    socials  = dex.get("info_socials", [])
    social_presence = min(1.0, (len(websites) + len(socials)) / 3.0)

    # Normalizations
    lp_locked    = min(gp.get("lp_locked_pct", 0.0), 1.0)
    top10        = min(gp.get("top10_holder_pct", 0.0), 1.0)
    holder_count = gp.get("holder_count", 0)

    trust_score = agent.get("trust_score", 0)
    total_jobs  = agent.get("total_jobs", 0)
    comp_rate   = agent.get("completion_rate", 0)

    acp_trust_norm = trust_score / 100.0
    log_jobs = math.log1p(total_jobs) / math.log1p(100000)

    holder_count_norm    = math.log1p(holder_count) / math.log1p(100000)
    token_age_days_norm  = math.log1p(token_age_days) / math.log1p(3650)
    liquidity_norm       = math.log1p(liquidity) / math.log1p(1_000_000)
    volume_norm          = math.log1p(volume_24) / math.log1p(1_000_000)
    price_change_norm    = max(-1.0, min(1.0, price24h / 100.0))

    # Creator signals
    creator_percent = gp.get("creator_percent", 0)
    owner_percent   = gp.get("owner_percent", 0)
    is_open_source  = 1 if gp.get("is_open_source") else 0
    is_mintable     = 1 if gp.get("is_mintable") else 0
    hidden_owner    = 1 if gp.get("hidden_owner") else 0
    slippage_mod    = 1 if gp.get("slippage_modifiable") else 0
    buy_tax         = min(gp.get("buy_tax", 0), 1.0)
    sell_tax        = min(gp.get("sell_tax", 0), 1.0)
    is_honeypot     = 1 if gp.get("is_honeypot") else 0

    has_dex = 1 if dex and not dex.get("no_data") else 0

    v1 = [
        # holder_concentration
        top10,
        # liquidity_lock_ratio
        lp_locked,
        # creator_tx_pattern (new wallet = high risk)
        0.5,
        # buy_sell_ratio
        buy_sell_ratio,
        # contract_similarity_score
        0.3,
        # fund_flow_pattern
        0.0,
        # price_change_24h
        price_change_norm,
        # liquidity_usd
        liquidity_norm,
        # volume_24h
        volume_norm,
        # total_jobs
        log_jobs,
        # completion_rate
        min(comp_rate, 1.0),
        # trust_score
        acp_trust_norm,
        # age_days
        token_age_days_norm,
        # lp_drain_rate
        0.0,
        # deployer_age_days (unknown = 0.5)
        0.5,
        # token_supply_concentration
        max(creator_percent, owner_percent),
        # renounced_ownership
        0,
        # verified_contract
        is_open_source,
        # social_presence_score
        social_presence,
        # audit_score
        0.0,
    ]

    v2_virtuals = [
        # bonding_curve_position
        0.0,
        # lp_locked_pct
        lp_locked,
        # creator_other_tokens
        0.0,
        # creator_wallet_age
        0.5,
        # holder_count_norm
        holder_count_norm,
        # top10_holder_pct
        top10,
        # acp_job_count
        log_jobs,
        # acp_completion_rate
        min(comp_rate, 1.0),
        # acp_trust_score
        acp_trust_norm,
        # token_age_days_norm
        token_age_days_norm,
        # volume_to_mcap_ratio
        vol_to_mcap,
        # price_volatility_7d
        price_vol,
        # social_presence
        social_presence,
        # is_honeypot
        float(is_honeypot),
        # is_mintable
        float(is_mintable),
        # hidden_owner
        float(hidden_owner),
        # slippage_modifiable
        float(slippage_mod),
        # buy_tax
        buy_tax,
        # sell_tax
        sell_tax,
        # creator_percent
        creator_percent,
        # owner_percent
        owner_percent,
        # has_dex_data
        float(has_dex),
        # is_virtuals_token
        1.0,
    ]

    return np.array(v1 + v2_virtuals, dtype=np.float32)

def compute_risk_signals(dex: dict, gp: dict, agent: dict) -> list[str]:
    """Generate human-readable risk signal list."""
    signals = []

    if gp.get("is_honeypot"):
        signals.append("HONEYPOT")
    if gp.get("sell_tax", 0) >= 0.5:
        signals.append(f"HIGH_SELL_TAX_{gp['sell_tax']:.0%}")
    if gp.get("hidden_owner"):
        signals.append("HIDDEN_OWNER")
    if gp.get("slippage_modifiable"):
        signals.append("SLIPPAGE_MODIFIABLE")

    top10 = gp.get("top10_holder_pct", 0)
    if top10 > 0.8:
        signals.append(f"CONCENTRATED_{top10:.0%}")

    volume_24 = dex.get("volume_24h", 0)
    liquidity  = dex.get("liquidity_usd", 0)
    price24h   = dex.get("price_change_24h", 0)
    mcap       = dex.get("market_cap", 0)

    if price24h <= -50:
        signals.append(f"PRICE_DUMP_{price24h:.0f}%")
    if liquidity < 500 and mcap > 0:
        signals.append(f"LOW_LIQUIDITY_${liquidity:.0f}")

    created_at = dex.get("pair_created_at")
    token_age_days = 0
    if created_at:
        try:
            token_age_days = (time.time() * 1000 - float(created_at)) / (1000 * 86400)
        except Exception:
            pass

    if token_age_days > 14 and volume_24 == 0:
        signals.append("DEAD_VOLUME")

    trust_score = agent.get("trust_score", 0)
    total_jobs  = agent.get("total_jobs", 0)
    if trust_score == 0 and total_jobs == 0:
        signals.append("GHOST_AGENT")

    vol_to_mcap = volume_24 / mcap if mcap > 0 else 0
    if vol_to_mcap > 5.0:
        signals.append(f"WASH_TRADING_{vol_to_mcap:.1f}x")

    return signals

def ensure_db_table():
    """Create wadjet_agent_token_risks table if not exists."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS wadjet_agent_token_risks (
            token_address       TEXT PRIMARY KEY,
            wallet_address      TEXT,
            token_symbol        TEXT,
            risk_score          FLOAT NOT NULL,
            risk_label          INTEGER NOT NULL,
            model_version       TEXT,
            risk_signals        JSONB DEFAULT '[]',
            dex_data            JSONB DEFAULT '{}',
            goplus_data         JSONB DEFAULT '{}',
            liquidity_usd       FLOAT,
            volume_24h          FLOAT,
            price_change_24h    FLOAT,
            market_cap          FLOAT,
            holder_count        INTEGER,
            top10_holder_pct    FLOAT,
            acp_trust_score     INTEGER,
            acp_total_jobs      INTEGER,
            acp_completion_rate FLOAT,
            token_age_days      FLOAT,
            scanned_at          TIMESTAMP DEFAULT NOW(),
            updated_at          TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS wadjet_risks_score_idx 
            ON wadjet_agent_token_risks(risk_score DESC);
        CREATE INDEX IF NOT EXISTS wadjet_risks_label_idx 
            ON wadjet_agent_token_risks(risk_label);
    """)
    conn.commit()
    cur.close()
    conn.close()
    logger.info("✅ wadjet_agent_token_risks table ready")

def upsert_risk_records(records: list[dict]):
    """Bulk upsert risk records to Supabase."""
    if not records:
        return

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    upsert_sql = """
        INSERT INTO wadjet_agent_token_risks (
            token_address, wallet_address, token_symbol,
            risk_score, risk_label, model_version,
            risk_signals, dex_data, goplus_data,
            liquidity_usd, volume_24h, price_change_24h, market_cap,
            holder_count, top10_holder_pct,
            acp_trust_score, acp_total_jobs, acp_completion_rate,
            token_age_days, scanned_at, updated_at
        ) VALUES %s
        ON CONFLICT (token_address) DO UPDATE SET
            risk_score          = EXCLUDED.risk_score,
            risk_label          = EXCLUDED.risk_label,
            model_version       = EXCLUDED.model_version,
            risk_signals        = EXCLUDED.risk_signals,
            dex_data            = EXCLUDED.dex_data,
            goplus_data         = EXCLUDED.goplus_data,
            liquidity_usd       = EXCLUDED.liquidity_usd,
            volume_24h          = EXCLUDED.volume_24h,
            price_change_24h    = EXCLUDED.price_change_24h,
            market_cap          = EXCLUDED.market_cap,
            holder_count        = EXCLUDED.holder_count,
            top10_holder_pct    = EXCLUDED.top10_holder_pct,
            acp_trust_score     = EXCLUDED.acp_trust_score,
            acp_total_jobs      = EXCLUDED.acp_total_jobs,
            acp_completion_rate = EXCLUDED.acp_completion_rate,
            token_age_days      = EXCLUDED.token_age_days,
            updated_at          = NOW()
    """

    now = datetime.now(timezone.utc)
    values = []
    for r in records:
        token_age = 0.0
        created_at = r.get("dex_data", {}).get("pair_created_at")
        if created_at:
            try:
                token_age = (time.time() * 1000 - float(created_at)) / (1000 * 86400)
            except Exception:
                pass

        values.append((
            r["token_address"],
            r.get("wallet_address", ""),
            r.get("token_symbol", ""),
            float(r["risk_score"]),
            int(r["risk_label"]),
            r.get("model_version", "v2"),
            json.dumps(r.get("risk_signals", [])),
            json.dumps({k: v for k, v in r.get("dex_data", {}).items() if not k.startswith("_")}),
            json.dumps({k: v for k, v in r.get("goplus_data", {}).items() if not k.startswith("_")}),
            float(r.get("dex_data", {}).get("liquidity_usd", 0)),
            float(r.get("dex_data", {}).get("volume_24h", 0)),
            float(r.get("dex_data", {}).get("price_change_24h", 0)),
            float(r.get("dex_data", {}).get("market_cap", 0)),
            int(r.get("goplus_data", {}).get("holder_count", 0)),
            float(r.get("goplus_data", {}).get("top10_holder_pct", 0)),
            int(r.get("agent", {}).get("trust_score", 0)),
            int(r.get("agent", {}).get("total_jobs", 0)),
            float(r.get("agent", {}).get("completion_rate", 0)),
            float(token_age),
            now,
            now,
        ))

    psycopg2.extras.execute_values(cur, upsert_sql, values, page_size=100)
    conn.commit()
    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Scan Virtuals agent tokens for rug risk")
    parser.add_argument("--limit", type=int, default=0, help="Max tokens to scan (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--cache-only", action="store_true", help="Use cached data only")
    parser.add_argument("--min-score", type=float, default=0.0, help="Only store tokens above this risk score")
    args = parser.parse_args()

    logger.info("=== Wadjet Agent Token Scanner ===")

    # Load model
    model, model_version = load_model()

    # Check model feature count
    n_model_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else len(V1_FEATURES)
    use_v2_features = n_model_features >= len(ALL_FEATURES)
    logger.info(f"Model expects {n_model_features} features | Using {'V2 (all)' if use_v2_features else 'V1 (fallback)'} features")

    # Ensure DB table
    if not args.dry_run:
        ensure_db_table()

    # Load cache
    cache = load_cache()

    # Fetch agents
    agents = fetch_agents_from_db(args.limit)
    logger.info(f"Scanning {len(agents)} agent tokens...")

    results = []
    batch = []
    skipped = 0
    scanned = 0

    for i, agent in enumerate(agents):
        token_addr = agent["token_address"]

        if (i + 1) % 25 == 0:
            logger.info(f"Progress: {i+1}/{len(agents)} | Scanned: {scanned} | Skipped: {skipped}")
            save_cache(cache)
            if not args.dry_run and batch:
                upsert_risk_records(batch)
                batch = []

        # Fetch data
        dex = fetch_dex_data(token_addr, cache, args.cache_only)
        if not args.cache_only:
            time.sleep(2.0)

        if not dex or dex.get("no_data"):
            skipped += 1
            continue

        gp = fetch_goplus_data(token_addr, cache, args.cache_only)
        if not args.cache_only:
            time.sleep(0.25)

        # Build feature vector
        try:
            features = build_feature_vector(agent, dex, gp)
            if not use_v2_features:
                features = features[:len(V1_FEATURES)]
            risk_prob = float(model.predict_proba(features.reshape(1, -1))[0, 1])
            risk_label = int(risk_prob >= THRESHOLD)
        except Exception as e:
            logger.debug(f"Prediction error for {token_addr}: {e}")
            skipped += 1
            continue

        if risk_prob < args.min_score and not risk_label:
            scanned += 1
            continue

        risk_signals = compute_risk_signals(dex, gp, agent)

        record = {
            "token_address":  token_addr,
            "wallet_address": agent["wallet_address"],
            "token_symbol":   agent.get("token_symbol") or dex.get("base_token", ""),
            "risk_score":     risk_prob,
            "risk_label":     risk_label,
            "model_version":  model_version,
            "risk_signals":   risk_signals,
            "dex_data":       dex,
            "goplus_data":    gp,
            "agent":          agent,
        }
        results.append(record)
        batch.append(record)
        scanned += 1

    # Final batch write
    save_cache(cache)
    if not args.dry_run and batch:
        upsert_risk_records(batch)

    # ─── Report ───────────────────────────────────────────────────────────
    logger.info(f"\n=== Scan Complete ===")
    logger.info(f"Total agents:     {len(agents)}")
    logger.info(f"Scanned:          {scanned}")
    logger.info(f"Skipped:          {skipped}")
    logger.info(f"Results stored:   {len(results)}")
    logger.info(f"High risk (>=0.7): {sum(1 for r in results if r['risk_score'] >= 0.7)}")

    if results:
        df = pd.DataFrame([
            {
                "token": r.get("token_symbol") or r["token_address"][:12],
                "risk_score": f"{r['risk_score']:.3f}",
                "risk": "🔴 RUG" if r["risk_label"] else "🟢 OK",
                "signals": ", ".join(r["risk_signals"][:3]),
                "liquidity": f"${r['dex_data'].get('liquidity_usd', 0):,.0f}",
                "trust": r["agent"].get("trust_score", 0),
            }
            for r in sorted(results, key=lambda x: x["risk_score"], reverse=True)[:30]
        ])
        print("\n🏆 TOP 30 RISKIEST AGENT TOKENS:")
        print(df.to_string(index=False))

    return results


if __name__ == "__main__":
    main()
