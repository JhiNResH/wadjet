"""
Wadjet — Real Rug Pull Dataset Builder
=======================================
Sources:
  1. kangmyoungseok/RugPull-Prediction-AI (18,296 Uniswap V2 tokens, labeled True/False)
     URL: https://github.com/kangmyoungseok/RugPull-Prediction-AI/raw/main/Dataset_v1.9.csv
     18,296 tokens, 16,462 rug pulls + 1,834 legit (labeled by LP removal detection)

Maps 18-col source features → our 20-col schema.

Feature Mapping:
  holder_concentration        ← clip(lp_creator_holding_ratio, 0, 1)
  liquidity_lock_ratio        ← clip(lp_lock_ratio, 0, 1)
  creator_tx_pattern          ← log_norm(mint_count_per_week)
  buy_sell_ratio              ← swap_in / (swap_in + swap_out)
  contract_similarity_score   ← log_norm(number_of_token_creation_of_creator)
  fund_flow_pattern           ← clip(mint_ratio + burn_ratio * 5, 0, 1)
  price_change_24h            ← clip(lp_std / (lp_avg + 1e-9), 0, 1)
  liquidity_usd               ← lp_avg / 100.0  (already normalized 0-100)
  volume_24h                  ← log_norm(swap_in_per_week + swap_out_per_week)
  total_jobs                  ← 0.0  (Maiat-specific, NA)
  completion_rate             ← 0.5  (Maiat-specific, NA)
  trust_score                 ← composite(lock_ratio, creator_ratio, creator_tx)
  age_days                    ← swap_mean_period  (proxy for token longevity)
  lp_drain_rate               ← clip(lp_std / (lp_avg + 1e-9), 0, 1)  (LP volatility)
  deployer_age_days           ← 1 - log_norm(number_of_token_creation_of_creator)
  token_supply_concentration  ← clip(token_creator_holding_ratio_norm, 0, 1)
  renounced_ownership         ← 0  (NA)
  verified_contract           ← 0  (NA)
  social_presence_score       ← 0.3  (NA, neutral)
  audit_score                 ← 0.0  (NA)
"""

import sys
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SOURCE_URL = "https://raw.githubusercontent.com/kangmyoungseok/RugPull-Prediction-AI/main/Dataset_v1.9.csv"
SOURCE_CACHE = DATA_DIR / "kangmyoungseok_rugpull_v1.9.csv"
OUTPUT_PATH = DATA_DIR / "rug_pull_dataset_real.csv"

FEATURES = [
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


def log_norm(series: pd.Series, cap_percentile: float = 0.99) -> pd.Series:
    """Log-normalize a heavy-tailed series to [0, 1]."""
    cap = series.quantile(cap_percentile)
    if cap == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    log_vals = np.log1p(series.clip(lower=0))
    log_cap = np.log1p(cap)
    return (log_vals / log_cap).clip(0, 1)


def download_source(url: str, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        print(f"  ✓ Using cached: {cache_path.name}")
    else:
        print(f"  ⬇  Downloading {url}")
        urllib.request.urlretrieve(url, cache_path)
        print(f"  ✓ Saved: {cache_path}")
    return pd.read_csv(cache_path)


def map_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map kangmyoungseok features → Wadjet 20-feature schema."""
    out = pd.DataFrame(index=df.index)

    # ── Clip noisy columns to plausible physical bounds ───────────────────
    lp_creator = df["lp_creator_holding_ratio"].clip(0, 1)
    lp_lock    = df["lp_lock_ratio"].clip(0, 1)
    lp_avg     = df["lp_avg"].clip(0.001, 100)
    lp_std     = df["lp_std"].clip(0, 50)
    token_cr   = df["token_creator_holding_ratio"].clip(-500, 6250)

    # ── Normalize token_creator_holding_ratio (very heavy tail) ──────────
    # 90th pct clip → covers normal creator holdings up to ~1.0 (100%)
    token_cr_norm = log_norm(token_cr.clip(lower=0))

    # ── LP volatility: std/avg = coefficient of variation ─────────────────
    lp_cv = (lp_std / lp_avg).clip(0, 1)  # >1 means fully drained

    # ── Log-normed counts ─────────────────────────────────────────────────
    mint_norm    = log_norm(df["mint_count_per_week"])
    n_contracts  = log_norm(df["number_of_token_creation_of_creator"])
    vol_norm     = log_norm(df["swap_in_per_week"] + df["swap_out_per_week"])

    # ── Buy/sell ratio: fraction of swaps that are buys ───────────────────
    swap_in      = df["swap_in_per_week"].clip(lower=0)
    swap_out     = df["swap_out_per_week"].clip(lower=0)
    buy_ratio    = (swap_in / (swap_in + swap_out + 1e-9)).clip(0, 1)

    # ── Fund flow: fraction of non-swap activity (minting + burning) ──────
    fund_flow = (df["mint_ratio"] + df["burn_ratio"] * 5).clip(0, 1)

    # ── Trust score composite (higher = safer) ────────────────────────────
    trust = (
        0.40 * lp_lock                       # locked liquidity
        + 0.35 * (1 - lp_creator)            # creator doesn't own all LP
        + 0.25 * (1 - token_cr_norm)         # creator doesn't hold all supply
    ).clip(0, 1)

    # ── Map to schema ─────────────────────────────────────────────────────
    out["holder_concentration"]        = lp_creator          # LP creator holding → holder concentration
    out["liquidity_lock_ratio"]        = lp_lock
    out["creator_tx_pattern"]          = mint_norm
    out["buy_sell_ratio"]              = buy_ratio
    out["contract_similarity_score"]   = n_contracts         # many contracts = serial scammer
    out["fund_flow_pattern"]           = fund_flow
    out["price_change_24h"]            = lp_cv               # LP CoV as price volatility proxy
    out["liquidity_usd"]               = (lp_avg / 100.0)   # already normalized 0-100
    out["volume_24h"]                  = vol_norm
    out["total_jobs"]                  = 0.0                 # Maiat-specific, NA
    out["completion_rate"]             = 0.5                 # Maiat-specific, NA
    out["trust_score"]                 = trust
    out["age_days"]                    = df["swap_mean_period"].clip(0, 1)  # proxy for age
    out["lp_drain_rate"]               = lp_cv               # LP CoV = drain proxy
    out["deployer_age_days"]           = (1 - n_contracts)   # fewer contracts = newer deployer
    out["token_supply_concentration"]  = token_cr_norm
    out["renounced_ownership"]         = 0
    out["verified_contract"]           = 0
    out["social_presence_score"]       = 0.3
    out["audit_score"]                 = 0.0
    out["label"]                       = df["Label"].astype(bool).astype(int)

    return out


def main():
    print("=" * 60)
    print("🔮 Wadjet — Real Dataset Builder")
    print("=" * 60)

    print("\n[1] Downloading source dataset…")
    raw = download_source(SOURCE_URL, SOURCE_CACHE)
    print(f"    Raw shape: {raw.shape}")
    print(f"    Labels: {raw['Label'].value_counts().to_dict()}")

    print("\n[2] Mapping features…")
    mapped = map_features(raw)

    # Sanity check ranges
    print("\n[3] Range check (all should be 0-1):")
    for col in FEATURES:
        if col in ("renounced_ownership", "verified_contract"):
            continue
        col_min = mapped[col].min()
        col_max = mapped[col].max()
        status = "✓" if col_min >= -0.01 and col_max <= 1.01 else "⚠️"
        print(f"  {status} {col:<40} [{col_min:.3f}, {col_max:.3f}]")

    print(f"\n[4] Output label distribution:")
    print(f"    Rug (1): {mapped['label'].sum()}")
    print(f"    Legit (0): {(mapped['label'] == 0).sum()}")
    print(f"    Total: {len(mapped)}")

    # Drop any NaN rows
    before = len(mapped)
    mapped = mapped.dropna()
    after = len(mapped)
    if before != after:
        print(f"\n  ⚠️  Dropped {before - after} rows with NaN values")

    mapped.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved real dataset: {OUTPUT_PATH}")
    print(f"   {len(mapped)} samples, {mapped.columns.tolist()}")

    return mapped


if __name__ == "__main__":
    main()
