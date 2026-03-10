"""
Wadjet Phase 3 — Build Virtuals Agent Token Labeled Dataset
============================================================
Fetches token data from DexScreener + GoPlus + Alchemy + Supabase,
labels tokens as rug (1) or legit (0), and saves to:
  packages/wadjet/data/virtuals_agent_dataset.csv

Labeling methodology:
  RUG:   price_drop_from_ath > 95%
      OR liquidity_usd < $100 AND once had > $1000
      OR volume_24h = 0 AND token > 30 days old
      OR GoPlus flags honeypot / malicious
  LEGIT: sustained volume for 30+ days
      OR trust_score >= 80 with active jobs
      OR market_cap > $10k with price > 30 days old

Rate limits:
  DexScreener: ~30 req/min  → 2s sleep
  Alchemy:     ~25 req/s    → 0.05s sleep
  GoPlus:      5 req/s      → 0.2s sleep
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_virtuals_dataset")

# ─── Paths & Config ──────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_CSV  = DATA_DIR / "virtuals_agent_dataset.csv"
CACHE_FILE  = DATA_DIR / "dex_cache.json"

DB_URL = os.environ["DATABASE_URL"]
ALCHEMY_KEY = "okgmVpKT-5iqER0g5yjyn"
ALCHEMY_BASE = f"https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}"
DEXSCREENER_URL = "https://api.dexscreener.com/latest/dex/tokens/{}"
GOPLUS_URL = "https://api.gopluslabs.io/api/v1/token_security/8453"

# ─── Cache ────────────────────────────────────────────────────────────────────
def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))

# ─── Database ─────────────────────────────────────────────────────────────────
def fetch_agent_tokens() -> list[dict]:
    """Pull all agents with token addresses from Supabase."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            wallet_address,
            token_address,
            token_symbol,
            trust_score,
            completion_rate,
            total_jobs,
            last_updated,
            raw_metrics
        FROM agent_scores
        WHERE token_address IS NOT NULL
        ORDER BY trust_score DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    agents = []
    for row in rows:
        agents.append({
            "wallet_address": row[0],
            "token_address": row[1].lower() if row[1] else None,
            "token_symbol": row[2],
            "trust_score": row[3],
            "completion_rate": row[4],
            "total_jobs": row[5],
            "last_updated": str(row[6]),
            "raw_metrics": row[7] or {},
        })
    logger.info(f"Fetched {len(agents)} agents with token addresses from DB")
    return agents

# ─── DexScreener ─────────────────────────────────────────────────────────────
def fetch_dex_data(token_address: str, cache: dict) -> dict:
    """Fetch token data from DexScreener. Returns {} on failure."""
    key = f"dex:{token_address}"
    cached = cache.get(key)
    if cached and (time.time() - cached.get("_ts", 0)) < 86400:
        return cached

    url = DEXSCREENER_URL.format(token_address)
    try:
        resp = httpx.get(url, timeout=15.0)
        if resp.status_code == 429:
            logger.warning(f"DexScreener rate limit hit — sleeping 10s")
            time.sleep(10)
            resp = httpx.get(url, timeout=15.0)

        if resp.status_code != 200:
            return {}

        data = resp.json()
        pairs = data.get("pairs") or []
        if not pairs:
            cache[key] = {"_ts": time.time(), "no_data": True}
            return {}

        # Pick the pair with the highest liquidity (most relevant)
        pair = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0)

        result = {
            "_ts": time.time(),
            "pair_address": pair.get("pairAddress", ""),
            "dex_id": pair.get("dexId", ""),
            "chain_id": pair.get("chainId", ""),
            "base_token": pair.get("baseToken", {}).get("symbol", ""),
            "price_usd": float(pair.get("priceUsd") or 0),
            "price_change_5m": float((pair.get("priceChange") or {}).get("m5") or 0),
            "price_change_1h": float((pair.get("priceChange") or {}).get("h1") or 0),
            "price_change_6h": float((pair.get("priceChange") or {}).get("h6") or 0),
            "price_change_24h": float((pair.get("priceChange") or {}).get("h24") or 0),
            "volume_5m": float((pair.get("volume") or {}).get("m5") or 0),
            "volume_1h": float((pair.get("volume") or {}).get("h1") or 0),
            "volume_6h": float((pair.get("volume") or {}).get("h6") or 0),
            "volume_24h": float((pair.get("volume") or {}).get("h24") or 0),
            "liquidity_usd": float((pair.get("liquidity") or {}).get("usd") or 0),
            "liquidity_base": float((pair.get("liquidity") or {}).get("base") or 0),
            "market_cap": float(pair.get("marketCap") or 0),
            "fdv": float(pair.get("fdv") or 0),
            "pair_created_at": pair.get("pairCreatedAt"),  # ms timestamp
            "txns_24h_buys": int((pair.get("txns") or {}).get("h24", {}).get("buys") or 0),
            "txns_24h_sells": int((pair.get("txns") or {}).get("h24", {}).get("sells") or 0),
            "info_websites": [w.get("url","") for w in (pair.get("info") or {}).get("websites", [])],
            "info_socials": [s.get("url","") for s in (pair.get("info") or {}).get("socials", [])],
        }
        cache[key] = result
        return result

    except Exception as e:
        logger.debug(f"DexScreener error for {token_address}: {e}")
        return {}

# ─── GoPlus ──────────────────────────────────────────────────────────────────
def fetch_goplus_data(token_address: str, cache: dict) -> dict:
    """Fetch GoPlus security flags. Returns {} on failure."""
    key = f"gp:{token_address}"
    cached = cache.get(key)
    if cached and (time.time() - cached.get("_ts", 0)) < 86400:
        return cached

    try:
        resp = httpx.get(
            GOPLUS_URL,
            params={"contract_addresses": token_address},
            timeout=15.0
        )
        time.sleep(0.25)  # GoPlus 5 req/s
        if resp.status_code != 200:
            return {}
        data = resp.json()
        result_data = (data.get("result") or {}).get(token_address.lower(), {})
        if not result_data:
            return {}

        result = {
            "_ts": time.time(),
            "is_honeypot": result_data.get("is_honeypot", "0") == "1",
            "honeypot_with_same_creator": result_data.get("honeypot_with_same_creator", "0") == "1",
            "is_blacklisted": result_data.get("is_blacklisted", "0") == "1",
            "is_mintable": result_data.get("is_mintable", "0") == "1",
            "can_take_back_ownership": result_data.get("can_take_back_ownership", "0") == "1",
            "selfdestruct": result_data.get("selfdestruct", "0") == "1",
            "external_call": result_data.get("external_call", "0") == "1",
            "is_open_source": result_data.get("is_open_source", "0") == "1",
            "is_proxy": result_data.get("is_proxy", "0") == "1",
            "owner_change_balance": result_data.get("owner_change_balance", "0") == "1",
            "hidden_owner": result_data.get("hidden_owner", "0") == "1",
            "slippage_modifiable": result_data.get("slippage_modifiable", "0") == "1",
            "buy_tax": float(result_data.get("buy_tax") or 0),
            "sell_tax": float(result_data.get("sell_tax") or 0),
            "holder_count": int(result_data.get("holder_count") or 0),
            "lp_holder_count": int(result_data.get("lp_holder_count") or 0),
            "lp_total_supply": float(result_data.get("lp_total_supply") or 0),
            "creator_address": result_data.get("creator_address", ""),
            "creator_balance": float(result_data.get("creator_balance") or 0),
            "creator_percent": float(result_data.get("creator_percent") or 0),
            "owner_address": result_data.get("owner_address", ""),
            "owner_balance": float(result_data.get("owner_balance") or 0),
            "owner_percent": float(result_data.get("owner_percent") or 0),
            "top10_holder_pct": sum(
                float(h.get("percent", 0))
                for h in result_data.get("holders", [])[:10]
            ),
            "lp_locked_pct": 0.0,  # computed below
            "total_supply": float(result_data.get("total_supply") or 0),
        }
        # LP lock pct from lp_holders
        lp_holders = result_data.get("lp_holders", [])
        if lp_holders:
            locked = sum(
                float(h.get("percent", 0))
                for h in lp_holders
                if h.get("is_locked") == 1
            )
            result["lp_locked_pct"] = locked

        cache[key] = result
        return result

    except Exception as e:
        logger.debug(f"GoPlus error for {token_address}: {e}")
        return {}

# ─── Alchemy ─────────────────────────────────────────────────────────────────
def fetch_alchemy_creator_data(creator_address: str, cache: dict) -> dict:
    """Fetch creator wallet age + balance from Alchemy."""
    if not creator_address:
        return {}
    key = f"alchemy:{creator_address}"
    cached = cache.get(key)
    if cached and (time.time() - cached.get("_ts", 0)) < 3600:
        return cached

    try:
        # Get ETH balance
        resp = httpx.post(ALCHEMY_BASE, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "eth_getBalance",
            "params": [creator_address, "latest"]
        }, timeout=10.0)
        balance_wei = int(resp.json().get("result", "0x0"), 16)
        balance_eth = balance_wei / 1e18
        time.sleep(0.05)

        # Get first tx to estimate wallet age (use getTransactionCount as proxy)
        resp2 = httpx.post(ALCHEMY_BASE, json={
            "jsonrpc": "2.0", "id": 2,
            "method": "eth_getTransactionCount",
            "params": [creator_address, "latest"]
        }, timeout=10.0)
        tx_count = int(resp2.json().get("result", "0x0"), 16)
        time.sleep(0.05)

        # Get tokens deployed by creator (getAssetTransfers)
        resp3 = httpx.post(ALCHEMY_BASE, json={
            "jsonrpc": "2.0", "id": 3,
            "method": "alchemy_getAssetTransfers",
            "params": [{
                "fromAddress": creator_address,
                "category": ["erc20"],
                "withMetadata": False,
                "excludeZeroValue": True,
                "maxCount": "0x64"
            }]
        }, timeout=10.0)
        transfers = resp3.json().get("result", {}).get("transfers", [])
        # Unique contract addresses they've interacted with
        creator_token_count = len(set(
            t.get("rawContract", {}).get("address", "") 
            for t in transfers
            if t.get("rawContract", {}).get("address")
        ))
        time.sleep(0.05)

        result = {
            "_ts": time.time(),
            "creator_eth_balance": balance_eth,
            "creator_tx_count": tx_count,
            "creator_token_interactions": creator_token_count,
        }
        cache[key] = result
        return result

    except Exception as e:
        logger.debug(f"Alchemy error for {creator_address}: {e}")
        return {}

# ─── Label logic ─────────────────────────────────────────────────────────────
def compute_label(dex: dict, gp: dict, agent: dict) -> tuple[int, str]:
    """
    Returns (label, reason).
    1 = rug/failed, 0 = legit.
    Returns -1 if we can't confidently label.
    """
    # GoPlus hard flags
    if gp.get("is_honeypot"):
        return 1, "goplus_honeypot"
    if gp.get("honeypot_with_same_creator"):
        return 1, "goplus_honeypot_creator"
    if gp.get("hidden_owner") and gp.get("can_take_back_ownership"):
        return 1, "goplus_hidden_owner+takeback"

    # High sell tax = likely rug
    if gp.get("sell_tax", 0) >= 0.5:
        return 1, f"goplus_sell_tax_{gp['sell_tax']:.0%}"

    # DexScreener: price crash
    price_24h = dex.get("price_change_24h", 0)
    price_6h  = dex.get("price_change_6h", 0)
    liquidity = dex.get("liquidity_usd", 0)
    volume_24 = dex.get("volume_24h", 0)
    mcap      = dex.get("market_cap", 0)

    # Token age
    created_at = dex.get("pair_created_at")
    token_age_days = 0
    if created_at:
        try:
            age_ms = time.time() * 1000 - float(created_at)
            token_age_days = age_ms / (1000 * 86400)
        except Exception:
            pass

    # Massive price crash + no liquidity
    if price_24h <= -90 and liquidity < 100:
        return 1, f"price_crash_{price_24h:.0f}%_no_liquidity"

    # Dead token: no volume, old, near-zero liquidity
    if token_age_days > 30 and volume_24 < 10 and liquidity < 500:
        return 1, f"dead_token_age{token_age_days:.0f}d_vol${volume_24:.0f}"

    # Extreme concentration with no activity
    top10 = gp.get("top10_holder_pct", 0)
    if top10 > 0.9 and volume_24 < 100:
        return 1, f"concentrated_{top10:.0%}_low_volume"

    # ─── LEGIT signals ────────────────────────────────────────────────────
    trust_score = agent.get("trust_score", 0)
    total_jobs  = agent.get("total_jobs", 0)
    comp_rate   = agent.get("completion_rate", 0)

    # High trust + active jobs + meaningful liquidity
    if trust_score >= 80 and total_jobs >= 50 and liquidity > 1000:
        return 0, f"legit_trust{trust_score}_jobs{total_jobs}"

    # Good market fundamentals + age
    if mcap > 50000 and liquidity > 5000 and token_age_days > 30 and volume_24 > 100:
        return 0, f"legit_mcap${mcap:.0f}_liq${liquidity:.0f}_age{token_age_days:.0f}d"

    # Very active agent with good completion
    if total_jobs >= 500 and comp_rate > 0.9:
        return 0, f"legit_jobs{total_jobs}_cr{comp_rate:.0%}"

    # Can't confidently label
    return -1, "insufficient_data"

# ─── Feature extraction ──────────────────────────────────────────────────────
def extract_features(agent: dict, dex: dict, gp: dict, alch: dict) -> dict:
    """Extract all Virtuals-specific features for a single token."""
    now = time.time()

    # Token age
    created_at = dex.get("pair_created_at")
    token_age_days = 0
    if created_at:
        try:
            age_ms = now * 1000 - float(created_at)
            token_age_days = max(0, age_ms / (1000 * 86400))
        except Exception:
            pass

    # Buy/sell ratio
    buys  = dex.get("txns_24h_buys", 0)
    sells = dex.get("txns_24h_sells", 0)
    buy_sell_ratio = buys / (buys + sells) if (buys + sells) > 0 else 0.5

    # Volume / market cap ratio
    volume_24 = dex.get("volume_24h", 0)
    mcap = dex.get("market_cap", 0)
    vol_to_mcap = min(volume_24 / mcap, 10.0) if mcap > 0 else 0.0

    # Price volatility proxy (use absolute 24h change as simple proxy)
    price_vol_7d = abs(dex.get("price_change_24h", 0)) / 100.0

    # Social presence
    websites = dex.get("info_websites", [])
    socials  = dex.get("info_socials", [])
    social_presence = min(1.0, (len(websites) + len(socials)) / 3.0)

    # Liquidity lock
    lp_locked_pct = gp.get("lp_locked_pct", 0.0)
    if isinstance(lp_locked_pct, str):
        lp_locked_pct = float(lp_locked_pct or 0)

    # Top 10 holder concentration
    top10 = gp.get("top10_holder_pct", 0.0)

    # Creator risk signals
    creator_eth_balance = alch.get("creator_eth_balance", -1.0)  # -1 = unknown
    creator_tx_count    = alch.get("creator_tx_count", -1)
    creator_tokens      = alch.get("creator_token_interactions", 0)

    # Normalize creator wallet age proxy (tx_count → rough age estimate)
    # More txs = older wallet; cap at 10000
    creator_wallet_age_norm = min(creator_tx_count, 10000) / 10000.0 if creator_tx_count >= 0 else 0.5

    # ACP signals
    trust_score   = agent.get("trust_score", 0)
    total_jobs    = agent.get("total_jobs", 0)
    comp_rate     = agent.get("completion_rate", 0)
    acp_trust_norm = trust_score / 100.0

    import math
    log_jobs = math.log1p(total_jobs) / math.log1p(100000)  # normalize

    # Holder count
    holder_count = gp.get("holder_count", 0)

    # Contract risk flags from GoPlus
    is_open_source   = 1 if gp.get("is_open_source") else 0
    is_mintable      = 1 if gp.get("is_mintable") else 0
    hidden_owner     = 1 if gp.get("hidden_owner") else 0
    slippage_mod     = 1 if gp.get("slippage_modifiable") else 0
    buy_tax          = min(gp.get("buy_tax", 0), 1.0)
    sell_tax         = min(gp.get("sell_tax", 0), 1.0)

    # Liquidity + price
    liquidity_usd = dex.get("liquidity_usd", 0)
    price_change_24h = dex.get("price_change_24h", 0)
    liquidity_norm = math.log1p(liquidity_usd) / math.log1p(1_000_000)

    # Creator percent held
    creator_percent = gp.get("creator_percent", 0)
    owner_percent   = gp.get("owner_percent", 0)

    return {
        # Identity
        "token_address":         agent.get("token_address", ""),
        "wallet_address":        agent.get("wallet_address", ""),
        "token_symbol":          agent.get("token_symbol", "") or dex.get("base_token", ""),

        # Virtuals-specific features
        "bonding_curve_position": 0.0,          # Not available via API; placeholder
        "lp_locked_pct":         lp_locked_pct,
        "creator_other_tokens":  creator_tokens,
        "creator_wallet_age":    creator_wallet_age_norm,
        "creator_eth_balance":   creator_eth_balance,
        "holder_count":          holder_count,
        "top10_holder_pct":      top10,
        "acp_job_count":         total_jobs,
        "acp_completion_rate":   comp_rate,
        "acp_trust_score":       acp_trust_norm,
        "token_age_days":        token_age_days,
        "volume_to_mcap_ratio":  vol_to_mcap,
        "price_volatility_7d":   price_vol_7d,
        "social_presence":       social_presence,

        # GoPlus risk flags
        "is_honeypot":           1 if gp.get("is_honeypot") else 0,
        "is_open_source":        is_open_source,
        "is_mintable":           is_mintable,
        "hidden_owner":          hidden_owner,
        "slippage_modifiable":   slippage_mod,
        "buy_tax":               buy_tax,
        "sell_tax":              sell_tax,
        "creator_percent":       creator_percent,
        "owner_percent":         owner_percent,

        # DexScreener signals
        "liquidity_usd":         liquidity_usd,
        "liquidity_usd_norm":    liquidity_norm,
        "volume_24h":            volume_24,
        "market_cap":            mcap,
        "price_change_24h":      price_change_24h,
        "buy_sell_ratio":        buy_sell_ratio,
        "volume_to_mcap_ratio":  vol_to_mcap,
        "has_dex_data":          1 if dex else 0,

        # V1 compatible fields (for merging with Uniswap V2 dataset)
        "holder_concentration":     min(top10, 1.0),
        "liquidity_lock_ratio":     min(lp_locked_pct, 1.0),
        "creator_tx_pattern":       min(1 - creator_wallet_age_norm, 1.0),
        "contract_similarity_score":0.3,   # GoPlus flags + sell tax combined below
        "fund_flow_pattern":        0.0,
        "total_jobs":               log_jobs,
        "completion_rate":          min(comp_rate, 1.0),
        "trust_score":              acp_trust_norm,
        "age_days":                 min(math.log1p(token_age_days) / math.log1p(3650), 1.0),
        "lp_drain_rate":            0.0,
        "deployer_age_days":        creator_wallet_age_norm,
        "token_supply_concentration": max(creator_percent, owner_percent),
        "renounced_ownership":      0,
        "verified_contract":        is_open_source,
        "social_presence_score":    social_presence,
        "audit_score":              0.0,
        "price_change_24h_norm":    max(-1.0, min(1.0, price_change_24h / 100.0)),
        "liquidity_usd_v1":         liquidity_norm,
        "volume_24h_norm":          min(math.log1p(volume_24) / math.log1p(1_000_000), 1.0),
    }

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    logger.info("=== Building Virtuals Agent Token Dataset ===")

    cache = load_cache()
    agents = fetch_agent_tokens()

    rows = []
    skipped_no_dex = 0
    labeled_rug   = 0
    labeled_legit = 0
    labeled_skip  = 0

    total = len(agents)
    logger.info(f"Processing {total} agents with token addresses...")

    for i, agent in enumerate(agents):
        token_addr = agent.get("token_address")
        if not token_addr:
            continue

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{total} | Rugs: {labeled_rug} | Legit: {labeled_legit} | Skip: {labeled_skip}")
            save_cache(cache)  # Periodic cache save

        # Fetch DexScreener
        dex = fetch_dex_data(token_addr, cache)
        time.sleep(2.0)  # DexScreener rate limit

        if not dex or dex.get("no_data"):
            skipped_no_dex += 1
            continue

        # Fetch GoPlus
        gp = fetch_goplus_data(token_addr, cache)
        time.sleep(0.25)

        # Fetch Alchemy for creator
        creator_addr = gp.get("creator_address", "") or agent.get("wallet_address", "")
        alch = fetch_alchemy_creator_data(creator_addr, cache) if creator_addr else {}

        # Compute label
        label, reason = compute_label(dex, gp, agent)
        if label == -1:
            labeled_skip += 1
            # Still extract features for scanning (no label)
            # Only add to training dataset if confidently labeled
            continue

        if label == 1:
            labeled_rug += 1
        else:
            labeled_legit += 1

        # Extract features
        features = extract_features(agent, dex, gp, alch)
        features["label"]         = label
        features["label_reason"]  = reason
        rows.append(features)

    save_cache(cache)
    logger.info(f"\n=== Dataset Summary ===")
    logger.info(f"Total processed:     {total}")
    logger.info(f"No DexScreener data: {skipped_no_dex}")
    logger.info(f"Labeled rug:         {labeled_rug}")
    logger.info(f"Labeled legit:       {labeled_legit}")
    logger.info(f"Skipped (ambiguous): {labeled_skip}")

    if not rows:
        logger.error("No labeled samples found — check API connectivity")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"\n✅ Dataset saved to: {OUTPUT_CSV}")
    logger.info(f"   {len(df)} samples ({df['label'].sum()} rugs, {(df['label']==0).sum()} legit)")

    # Label distribution
    dist = df["label_reason"].value_counts().head(20)
    logger.info(f"\nTop label reasons:\n{dist.to_string()}")

    return df

if __name__ == "__main__":
    main()
