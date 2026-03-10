"""
Augment Virtuals dataset with synthetic rug examples based on known patterns.
Virtuals-specific rug patterns (different from Uniswap V2):
  1. Ghost agents: launched token, never developed, no ACP jobs
  2. Pump-and-dump: extreme price dump after initial pump
  3. Pre-graduation squeeze: extreme concentration on bonding curve
  4. Post-graduation abandonment: graduated but zero activity
  5. Honeypot/high-tax: GoPlus-flagged tokens

Also adds more legit samples from DB-only ACP signals
(agents with very high trust + active jobs — even without DEX data).
"""
import math
import logging
import random
import pandas as pd
import psycopg2
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("augment")

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
INPUT_CSV  = DATA_DIR / "virtuals_agent_dataset.csv"
OUTPUT_CSV = DATA_DIR / "virtuals_agent_dataset.csv"

DB_URL = os.environ["DATABASE_URL"]

random.seed(42)


def empty_virtuals_row():
    """Base empty Virtuals token row."""
    return {
        # V1 features
        "holder_concentration":       0.0,
        "liquidity_lock_ratio":       0.0,
        "creator_tx_pattern":         0.5,
        "buy_sell_ratio":             0.5,
        "contract_similarity_score":  0.3,
        "fund_flow_pattern":          0.0,
        "price_change_24h":           0.0,
        "liquidity_usd":              0.0,
        "volume_24h":                 0.0,
        "total_jobs":                 0.0,
        "completion_rate":            0.0,
        "trust_score":                0.0,
        "age_days":                   0.0,
        "lp_drain_rate":              0.0,
        "deployer_age_days":          0.5,
        "token_supply_concentration": 0.0,
        "renounced_ownership":        0,
        "verified_contract":          0,
        "social_presence_score":      0.0,
        "audit_score":                0.0,
        # V2 Virtuals features
        "bonding_curve_position":     0.0,
        "lp_locked_pct":              0.0,
        "creator_other_tokens":       0.0,
        "creator_wallet_age":         0.5,
        "holder_count_norm":          0.0,
        "top10_holder_pct":           0.0,
        "acp_job_count":              0.0,
        "acp_completion_rate":        0.0,
        "acp_trust_score":            0.0,
        "token_age_days_norm":        0.0,
        "volume_to_mcap_ratio":       0.0,
        "price_volatility_7d":        0.0,
        "social_presence":            0.0,
        "is_honeypot":                0.0,
        "is_mintable":                0.0,
        "hidden_owner":               0.0,
        "slippage_modifiable":        0.0,
        "buy_tax":                    0.0,
        "sell_tax":                   0.0,
        "creator_percent":            0.0,
        "owner_percent":              0.0,
        "has_dex_data":               1.0,
        "is_virtuals_token":          1.0,
        # Raw
        "raw_liquidity_usd":          0.0,
        "raw_volume_24h":             0.0,
        "raw_market_cap":             0.0,
        "raw_price_change_24h":       0.0,
        "raw_token_age_days":         0.0,
        "raw_trust_score":            0,
        "raw_total_jobs":             0,
        "raw_holder_count":           0,
        "raw_top10_holder_pct":       0.0,
        # Identity
        "token_address":   "",
        "wallet_address":  "",
        "token_symbol":    "",
        "data_source":     "virtuals_agent_synthetic",
    }


def norm_log(val, cap):
    return math.log1p(max(0, val)) / math.log1p(cap)


def make_ghost_agent_rug(n=15):
    """Ghost agents: token launched, no ACP jobs, no activity, old token."""
    rows = []
    for i in range(n):
        age_days  = random.uniform(30, 180)   # Old tokens
        liquidity = random.uniform(0, 300)     # Near-zero liquidity
        volume    = random.uniform(0, 10)      # Dead volume
        price_ch  = random.uniform(-95, -70)   # Price crashed
        top10     = random.uniform(0.85, 0.99) # Concentrated

        row = empty_virtuals_row()
        row.update({
            "holder_concentration":       min(top10, 1.0),
            "liquidity_usd":              norm_log(liquidity, 1_000_000),
            "volume_24h":                 norm_log(volume, 1_000_000),
            "price_change_24h":           max(-1.0, price_ch / 100.0),
            "age_days":                   norm_log(age_days, 3650),
            "token_supply_concentration": random.uniform(0.3, 0.7),
            "social_presence_score":      random.uniform(0, 0.2),
            "top10_holder_pct":           min(top10, 1.0),
            "acp_job_count":              0.0,
            "acp_completion_rate":        0.0,
            "acp_trust_score":            0.0,
            "token_age_days_norm":        norm_log(age_days, 3650),
            "volume_to_mcap_ratio":       0.0,
            "price_volatility_7d":        abs(price_ch) / 100.0,
            "social_presence":            random.uniform(0, 0.2),
            "raw_liquidity_usd":          liquidity,
            "raw_volume_24h":             volume,
            "raw_price_change_24h":       price_ch,
            "raw_token_age_days":         age_days,
            "raw_trust_score":            0,
            "raw_total_jobs":             0,
            "raw_top10_holder_pct":       top10 * 100,
            "token_symbol":               f"GHOST{i}",
            "label":                      1,
            "label_reason":               "synthetic_ghost_agent_no_activity",
        })
        rows.append(row)
    return rows


def make_pump_dump_rug(n=10):
    """Pump-and-dump: launched recently, big price pump then crash."""
    rows = []
    for i in range(n):
        age_days  = random.uniform(1, 14)       # Very new
        price_ch  = random.uniform(-95, -80)    # Massive dump
        top10     = random.uniform(0.75, 0.95)  # Concentrated
        vol_mcap  = random.uniform(3.0, 10.0)   # High wash trading

        row = empty_virtuals_row()
        row.update({
            "holder_concentration":   min(top10, 1.0),
            "price_change_24h":       max(-1.0, price_ch / 100.0),
            "age_days":               norm_log(age_days, 3650),
            "buy_sell_ratio":         random.uniform(0.1, 0.25),  # Mostly sells
            "top10_holder_pct":       min(top10, 1.0),
            "acp_job_count":          0.0,
            "acp_completion_rate":    0.0,
            "acp_trust_score":        0.0,
            "token_age_days_norm":    norm_log(age_days, 3650),
            "volume_to_mcap_ratio":   vol_mcap,
            "price_volatility_7d":    abs(price_ch) / 100.0,
            "social_presence_score":  random.uniform(0, 0.3),
            "social_presence":        random.uniform(0, 0.3),
            "raw_price_change_24h":   price_ch,
            "raw_token_age_days":     age_days,
            "raw_trust_score":        0,
            "raw_total_jobs":         0,
            "raw_top10_holder_pct":   top10 * 100,
            "token_symbol":           f"PUMP{i}",
            "label":                  1,
            "label_reason":           "synthetic_pump_dump_new_token",
        })
        rows.append(row)
    return rows


def make_honeypot_rug(n=8):
    """Honeypot/high-tax tokens — can't sell."""
    rows = []
    for i in range(n):
        sell_tax = random.uniform(0.5, 0.99)
        age_days = random.uniform(5, 60)
        top10    = random.uniform(0.7, 0.95)

        row = empty_virtuals_row()
        row.update({
            "holder_concentration":       min(top10, 1.0),
            "is_honeypot":                1.0 if sell_tax > 0.9 else 0.0,
            "slippage_modifiable":        1.0,
            "sell_tax":                   sell_tax,
            "buy_tax":                    random.uniform(0, 0.1),
            "top10_holder_pct":           min(top10, 1.0),
            "token_supply_concentration": random.uniform(0.4, 0.8),
            "acp_trust_score":            0.0,
            "token_age_days_norm":        norm_log(age_days, 3650),
            "raw_token_age_days":         age_days,
            "raw_trust_score":            0,
            "raw_top10_holder_pct":       top10 * 100,
            "token_symbol":               f"HONEY{i}",
            "label":                      1,
            "label_reason":               f"synthetic_honeypot_sell_tax_{sell_tax:.0%}",
        })
        rows.append(row)
    return rows


def make_abandoned_graduated_rug(n=10):
    """Graduated (LP locked) but completely abandoned — no jobs, no volume, old."""
    rows = []
    for i in range(n):
        age_days  = random.uniform(45, 300)  # Old
        liquidity = random.uniform(100, 2000) # Some LP locked (graduated), but illiquid
        volume    = random.uniform(0, 5)      # Dead volume

        row = empty_virtuals_row()
        row.update({
            "liquidity_usd":          norm_log(liquidity, 1_000_000),
            "liquidity_lock_ratio":   random.uniform(0.9, 1.0),  # LP locked (graduated)
            "lp_locked_pct":          random.uniform(0.9, 1.0),
            "volume_24h":             norm_log(volume, 1_000_000),
            "age_days":               norm_log(age_days, 3650),
            "token_age_days_norm":    norm_log(age_days, 3650),
            "acp_job_count":          0.0,
            "acp_completion_rate":    0.0,
            "acp_trust_score":        0.0,
            "total_jobs":             0.0,
            "volume_to_mcap_ratio":   0.0,
            "social_presence_score":  random.uniform(0, 0.2),
            "social_presence":        random.uniform(0, 0.2),
            "raw_liquidity_usd":      liquidity,
            "raw_volume_24h":         volume,
            "raw_token_age_days":     age_days,
            "raw_trust_score":        random.randint(0, 10),
            "raw_total_jobs":         0,
            "token_symbol":           f"DEAD{i}",
            "label":                  1,
            "label_reason":           "synthetic_abandoned_graduated_agent",
        })
        rows.append(row)
    return rows


def fetch_more_legit_from_db(n=30):
    """Fetch agents with very high trust + jobs from DB (no DEX data needed)."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT wallet_address, token_address, token_symbol,
               trust_score, completion_rate, total_jobs
        FROM agent_scores
        WHERE token_address IS NOT NULL
          AND trust_score >= 90
          AND total_jobs >= 200
          AND completion_rate > 0.9
        ORDER BY trust_score DESC, total_jobs DESC
        LIMIT %s
    """, (n * 3,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    synthetic_rows = []
    for row in rows[:n]:
        trust_score  = row[3]
        comp_rate    = row[4]
        total_jobs   = row[5]

        # These agents have strong ACP signals → legit
        log_jobs = norm_log(total_jobs, 100000)
        acp_norm = trust_score / 100.0

        r = empty_virtuals_row()
        r.update({
            "total_jobs":         log_jobs,
            "completion_rate":    min(comp_rate, 1.0),
            "trust_score":        acp_norm,
            "acp_job_count":      log_jobs,
            "acp_completion_rate":min(comp_rate, 1.0),
            "acp_trust_score":    acp_norm,
            # Plausible DEX signals for active agent
            "liquidity_usd":      norm_log(random.uniform(5000, 100000), 1_000_000),
            "volume_24h":         norm_log(random.uniform(100, 10000), 1_000_000),
            "age_days":           norm_log(random.uniform(30, 300), 3650),
            "token_age_days_norm":norm_log(random.uniform(30, 300), 3650),
            "buy_sell_ratio":     random.uniform(0.4, 0.65),
            "social_presence_score": random.uniform(0.3, 1.0),
            "social_presence":    random.uniform(0.3, 1.0),
            "has_dex_data":       0.0,  # No DEX data
            "raw_trust_score":    trust_score,
            "raw_total_jobs":     total_jobs,
            "token_address":      row[1].lower() if row[1] else "",
            "wallet_address":     row[0],
            "token_symbol":       row[2] or "",
            "data_source":        "virtuals_agent_db_only",
            "label":              0,
            "label_reason":       f"db_legit_trust{trust_score}_jobs{total_jobs}",
        })
        synthetic_rows.append(r)

    logger.info(f"Fetched {len(synthetic_rows)} high-trust legit agents from DB")
    return synthetic_rows


def main():
    logger.info("=== Augmenting Virtuals Dataset ===")

    # Load existing labeled data
    existing = pd.DataFrame()
    if INPUT_CSV.exists():
        existing = pd.read_csv(INPUT_CSV)
        logger.info(f"Existing dataset: {len(existing)} samples (rugs: {existing['label'].sum()}, legit: {(existing['label']==0).sum()})")

    # Generate synthetic rug examples
    rug_rows = []
    rug_rows.extend(make_ghost_agent_rug(15))
    rug_rows.extend(make_pump_dump_rug(10))
    rug_rows.extend(make_honeypot_rug(8))
    rug_rows.extend(make_abandoned_graduated_rug(10))
    logger.info(f"Generated {len(rug_rows)} synthetic rug examples")

    # Fetch more legit from DB
    legit_db_rows = fetch_more_legit_from_db(30)

    # Combine
    all_new = rug_rows + legit_db_rows
    new_df = pd.DataFrame(all_new)

    if not existing.empty:
        # Ensure columns match
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    # Fill NaN
    combined = combined.fillna(0)

    # De-duplicate by token_address only for real tokens (non-empty addresses)
    real = combined[combined["token_address"].str.len() > 5]
    synth = combined[combined["token_address"].str.len() <= 5]
    real = real.drop_duplicates(subset=["token_address"], keep="first")
    combined = pd.concat([real, synth], ignore_index=True)

    combined.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"\n✅ Augmented dataset saved: {OUTPUT_CSV}")
    logger.info(f"   Total: {len(combined)} samples")
    logger.info(f"   Rugs:  {combined['label'].sum()}")
    logger.info(f"   Legit: {(combined['label']==0).sum()}")

    logger.info("\nLabel reason breakdown:")
    for reason, count in combined["label_reason"].value_counts().items():
        label = combined[combined["label_reason"]==reason]["label"].iloc[0]
        icon = "🔴" if label == 1 else "🟢"
        logger.info(f"  {icon} {count:3d}x {reason}")

    return combined


if __name__ == "__main__":
    main()
