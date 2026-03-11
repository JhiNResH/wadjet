"""
Fast Virtuals Dataset Builder — Uses existing cache + DB data only.
No new API calls. Uses DexScreener/GoPlus cache from previous run.
Also creates DB-only labels for tokens with strong ACP signals.

Run after build_virtuals_dataset.py has collected some cache data.
"""
import json
import math
import time
import logging
from pathlib import Path

import pandas as pd
import psycopg2
from db.utils import get_db_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_from_cache")

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
CACHE_FILE = DATA_DIR / "dex_cache.json"
OUTPUT_CSV = DATA_DIR / "virtuals_agent_dataset.csv"

DB_URL = get_db_url()


def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def fetch_all_agents():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT wallet_address, token_address, token_symbol,
               trust_score, completion_rate, total_jobs, last_updated
        FROM agent_scores
        WHERE token_address IS NOT NULL
        ORDER BY trust_score DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {
        row[1].lower(): {
            "wallet_address": row[0],
            "token_address": row[1].lower(),
            "token_symbol": row[2],
            "trust_score": row[3],
            "completion_rate": row[4],
            "total_jobs": row[5],
        }
        for row in rows if row[1]
    }


def compute_label(dex, gp, agent) -> tuple[int, str]:
    """Label using strict quality criteria."""
    # GoPlus hard flags
    if gp.get("is_honeypot"):
        return 1, "goplus_honeypot"
    if gp.get("sell_tax", 0) >= 0.5:
        return 1, f"high_sell_tax_{gp['sell_tax']:.0%}"
    if gp.get("hidden_owner") and gp.get("can_take_back_ownership"):
        return 1, "hidden_owner_takeback"

    liquidity = dex.get("liquidity_usd", 0)
    volume_24 = dex.get("volume_24h", 0)
    price24h  = dex.get("price_change_24h", 0)
    mcap      = dex.get("market_cap", 0)

    created_at = dex.get("pair_created_at")
    token_age_days = 0.0
    if created_at:
        try:
            token_age_days = (time.time() * 1000 - float(created_at)) / (1000 * 86400)
        except Exception:
            pass

    top10 = gp.get("top10_holder_pct", 0)
    trust_score = agent.get("trust_score", 0)
    total_jobs  = agent.get("total_jobs", 0)
    comp_rate   = agent.get("completion_rate", 0)

    # === RUG LABELS ===
    # Dead token: old, no volume, minimal liquidity
    if token_age_days > 30 and volume_24 < 10 and liquidity < 500:
        return 1, f"dead_token_age{token_age_days:.0f}d_vol${volume_24:.0f}_liq${liquidity:.0f}"
    
    # Price crash + no liquidity
    if price24h <= -90 and liquidity < 100:
        return 1, f"crash_{price24h:.0f}%_liq${liquidity:.0f}"
    
    # Very high concentration + essentially no volume
    if top10 > 0.95 and volume_24 < 50:
        return 1, f"extreme_concentration_{top10:.0%}_vol${volume_24:.0f}"

    # === LEGIT LABELS ===
    # Good ACP with meaningful liquidity
    if trust_score >= 85 and total_jobs >= 100 and liquidity > 2000:
        return 0, f"legit_trust{trust_score}_jobs{total_jobs}_liq${liquidity:.0f}"
    
    # Good market fundamentals
    if mcap > 100000 and liquidity > 10000 and token_age_days > 20 and volume_24 > 500:
        return 0, f"legit_mcap${mcap:.0f}_liq${liquidity:.0f}_age{token_age_days:.0f}d"
    
    # Very high trust + active
    if trust_score >= 95 and total_jobs >= 500 and comp_rate > 0.95:
        return 0, f"legit_high_trust_{trust_score}_jobs{total_jobs}"

    # Strong volume relative to mcap (active trading) + age
    if token_age_days > 14 and volume_24 > 1000 and liquidity > 5000:
        return 0, f"legit_active_vol${volume_24:.0f}_liq${liquidity:.0f}"

    return -1, "ambiguous"


def extract_features(agent, dex, gp):
    now = time.time()
    created_at = dex.get("pair_created_at")
    token_age_days = 0.0
    if created_at:
        try:
            token_age_days = max(0, (now * 1000 - float(created_at)) / (1000 * 86400))
        except Exception:
            pass

    buys  = dex.get("txns_24h_buys", 0)
    sells = dex.get("txns_24h_sells", 0)
    buy_sell_ratio = buys / (buys + sells) if (buys + sells) > 0 else 0.5
    
    volume_24 = dex.get("volume_24h", 0)
    mcap      = dex.get("market_cap", 0)
    liquidity = dex.get("liquidity_usd", 0)
    price24h  = dex.get("price_change_24h", 0)
    
    vol_to_mcap = min(volume_24 / mcap, 10.0) if mcap > 0 else 0.0
    price_vol   = abs(price24h) / 100.0
    
    websites = dex.get("info_websites", []) or dex.get("websites", [])
    socials  = dex.get("info_socials", []) or dex.get("socials", [])
    social_presence = min(1.0, (len(websites) + len(socials)) / 3.0)
    
    lp_locked    = min(gp.get("lp_locked_pct", 0.0), 1.0)
    top10        = min(gp.get("top10_holder_pct", 0.0), 1.0)
    holder_count = gp.get("holder_count", 0)
    is_open_src  = 1 if gp.get("is_open_source") else 0
    is_mintable  = 1 if gp.get("is_mintable") else 0
    hidden_owner = 1 if gp.get("hidden_owner") else 0
    slippage_mod = 1 if gp.get("slippage_modifiable") else 0
    buy_tax      = min(gp.get("buy_tax", 0), 1.0)
    sell_tax     = min(gp.get("sell_tax", 0), 1.0)
    is_honeypot  = 1 if gp.get("is_honeypot") else 0
    creator_pct  = gp.get("creator_percent", 0)
    owner_pct    = gp.get("owner_percent", 0)
    
    trust_score = agent.get("trust_score", 0)
    total_jobs  = agent.get("total_jobs", 0)
    comp_rate   = agent.get("completion_rate", 0)
    
    acp_trust_norm     = min(trust_score / 100.0, 1.0)
    log_jobs           = math.log1p(total_jobs) / math.log1p(100000)
    holder_count_norm  = math.log1p(holder_count) / math.log1p(100000)
    token_age_days_norm = math.log1p(token_age_days) / math.log1p(3650)
    liquidity_norm     = math.log1p(liquidity) / math.log1p(1_000_000)
    volume_norm        = math.log1p(volume_24) / math.log1p(1_000_000)
    price_norm         = max(-1.0, min(1.0, price24h / 100.0))
    
    return {
        # Identity
        "token_address":   agent.get("token_address", ""),
        "wallet_address":  agent.get("wallet_address", ""),
        "token_symbol":    agent.get("token_symbol") or dex.get("base_token", ""),
        
        # === V1-compatible features ===
        "holder_concentration":        top10,
        "liquidity_lock_ratio":        lp_locked,
        "creator_tx_pattern":          0.5,
        "buy_sell_ratio":              buy_sell_ratio,
        "contract_similarity_score":   0.3,
        "fund_flow_pattern":           0.0,
        "price_change_24h":            price_norm,
        "liquidity_usd":               liquidity_norm,
        "volume_24h":                  volume_norm,
        "total_jobs":                  log_jobs,
        "completion_rate":             min(comp_rate, 1.0),
        "trust_score":                 acp_trust_norm,
        "age_days":                    token_age_days_norm,
        "lp_drain_rate":               0.0,
        "deployer_age_days":           0.5,
        "token_supply_concentration":  max(creator_pct, owner_pct),
        "renounced_ownership":         0,
        "verified_contract":           is_open_src,
        "social_presence_score":       social_presence,
        "audit_score":                 0.0,
        
        # === Virtuals-specific features ===
        "bonding_curve_position":   0.0,
        "lp_locked_pct":            lp_locked,
        "creator_other_tokens":     0.0,
        "creator_wallet_age":       0.5,
        "holder_count_norm":        holder_count_norm,
        "top10_holder_pct":         top10,
        "acp_job_count":            log_jobs,
        "acp_completion_rate":      min(comp_rate, 1.0),
        "acp_trust_score":          acp_trust_norm,
        "token_age_days_norm":      token_age_days_norm,
        "volume_to_mcap_ratio":     vol_to_mcap,
        "price_volatility_7d":      price_vol,
        "social_presence":          social_presence,
        "is_honeypot":              float(is_honeypot),
        "is_mintable":              float(is_mintable),
        "hidden_owner":             float(hidden_owner),
        "slippage_modifiable":      float(slippage_mod),
        "buy_tax":                  buy_tax,
        "sell_tax":                 sell_tax,
        "creator_percent":          creator_pct,
        "owner_percent":            owner_pct,
        "has_dex_data":             1.0,
        "is_virtuals_token":        1.0,
        
        # Raw values for reference
        "raw_liquidity_usd":        liquidity,
        "raw_volume_24h":           volume_24,
        "raw_market_cap":           mcap,
        "raw_price_change_24h":     price24h,
        "raw_token_age_days":       token_age_days,
        "raw_trust_score":          trust_score,
        "raw_total_jobs":           total_jobs,
        "raw_holder_count":         holder_count,
        "raw_top10_holder_pct":     top10 * 100,
    }


def main():
    logger.info("=== Building Virtuals Dataset from Cache ===")
    
    cache = load_cache()
    agents = fetch_all_agents()
    
    logger.info(f"Cache entries: {len(cache)}")
    logger.info(f"Agents in DB: {len(agents)}")
    
    rows = []
    labeled_rug = 0
    labeled_legit = 0
    skipped = 0
    
    for token_addr, agent in agents.items():
        dex_key = f"dex:{token_addr}"
        gp_key  = f"gp:{token_addr}"
        
        dex = cache.get(dex_key, {})
        gp  = cache.get(gp_key, {})
        
        if not dex or dex.get("no_data"):
            continue
        
        # Compute label
        label, reason = compute_label(dex, gp, agent)
        if label == -1:
            skipped += 1
            continue
        
        if label == 1:
            labeled_rug += 1
        else:
            labeled_legit += 1
        
        features = extract_features(agent, dex, gp)
        features["label"]        = label
        features["label_reason"] = reason
        features["data_source"]  = "virtuals_agent"
        rows.append(features)
    
    logger.info(f"\n=== Results ===")
    logger.info(f"Labeled rug:   {labeled_rug}")
    logger.info(f"Labeled legit: {labeled_legit}")
    logger.info(f"Skipped:       {skipped}")
    
    if not rows:
        logger.warning("No labeled samples — cache may be empty. Run build_virtuals_dataset.py first.")
        return None
    
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"\n✅ Dataset saved: {OUTPUT_CSV}")
    logger.info(f"   {len(df)} samples ({df['label'].sum()} rugs, {(df['label']==0).sum()} legit)")
    
    # Show samples
    logger.info("\nLabeled samples:")
    for _, row in df.iterrows():
        label_str = "🔴 RUG" if row["label"] else "🟢 LEGIT"
        sym = row.get("token_symbol") or row["token_address"][:12]
        logger.info(f"  {label_str} | {sym:<20} | trust={row['raw_trust_score']:3.0f} | jobs={row['raw_total_jobs']:5.0f} | liq=${row['raw_liquidity_usd']:,.0f} | reason: {row['label_reason']}")
    
    return df


if __name__ == "__main__":
    main()
