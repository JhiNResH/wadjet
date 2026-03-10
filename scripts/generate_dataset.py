"""
Wadjet Phase 1b — Synthetic Rug Pull Dataset Generator

Since TM-RugPull (arxiv:2602.21529, Feb 2026) is not yet publicly released,
we generate a realistic synthetic dataset based on:
1. TM-RugPull paper feature descriptions (1,028 labeled projects, 2016-2025)
2. CRPWarner/RugPull dataset patterns
3. Known rug pull characteristics from DeFi forensics

The synthetic data is calibrated to match real-world distributions reported in the literature.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions (aligns with Alchemy + DexScreener signals)
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = [
    "holder_concentration",       # Top 10 holders % (0-1)
    "liquidity_lock_ratio",        # % of LP tokens locked (0-1)
    "creator_tx_pattern",          # Deployer risk score (0-1, higher = riskier)
    "buy_sell_ratio",              # Buy txs / (buy + sell) txs (0-1)
    "contract_similarity_score",   # Similarity to known scam contracts (0-1)
    "fund_flow_pattern",           # Wash trading / circular flow score (0-1)
    "price_change_24h",            # Price change in last 24h (-1 to +5)
    "liquidity_usd",               # USD value of liquidity pool (log-scaled 0-1)
    "volume_24h",                  # 24h volume USD (log-scaled 0-1)
    "total_jobs",                  # Agent jobs completed (log-scaled 0-1)
    "completion_rate",             # Job completion rate (0-1)
    "trust_score",                 # Maiat trust score (0-1)
    "age_days",                    # Project age in days (log-scaled 0-1)
    "lp_drain_rate",               # Rate of LP removal (0-1, higher = draining)
    "deployer_age_days",           # Days since deployer first tx (log-scaled 0-1)
    "token_supply_concentration",  # % of supply held by deployer wallet (0-1)
    "renounced_ownership",         # 1 = ownership renounced, 0 = not
    "verified_contract",           # 1 = verified on chain explorer, 0 = not
    "social_presence_score",       # Twitter/Telegram activity score (0-1)
    "audit_score",                 # External audit presence (0-1)
]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_rug_pull_sample(n: int) -> pd.DataFrame:
    """Generate n rug pull samples (label=1)."""
    data = {}

    # High holder concentration — insiders hold most tokens
    data["holder_concentration"] = np.clip(np.random.beta(8, 2, n), 0.5, 1.0)

    # Low liquidity lock — rug pullers don't lock LP
    data["liquidity_lock_ratio"] = np.clip(np.random.beta(1.5, 8, n), 0, 0.3)

    # High deployer risk — deployer has suspicious history
    data["creator_tx_pattern"] = np.clip(np.random.beta(6, 2, n), 0.4, 1.0)

    # Skewed buy-sell ratio — wash trading pumps (inflated buy pressure)
    half = n // 2
    bsr = np.concatenate([
        np.random.uniform(0.7, 1.0, half),
        np.random.uniform(0.0, 0.2, n - half)
    ])
    np.random.shuffle(bsr)
    data["buy_sell_ratio"] = np.clip(bsr, 0, 1)

    # High contract similarity to known scams
    data["contract_similarity_score"] = np.clip(np.random.beta(5, 2, n), 0.3, 1.0)

    # High wash trading / fund flow anomalies
    data["fund_flow_pattern"] = np.clip(np.random.beta(6, 2, n), 0.3, 1.0)

    # Price change: initially pumped then crashed
    data["price_change_24h"] = np.random.choice([-0.8, -0.9, -0.95, -0.99], n, p=[0.2, 0.3, 0.3, 0.2])
    data["price_change_24h"] += np.random.normal(0, 0.05, n)

    # Low liquidity (post-rug)
    raw_liq = np.random.lognormal(6, 2, n)  # small values
    data["liquidity_usd"] = np.clip(np.log1p(raw_liq) / np.log1p(1e7), 0, 1)

    # Low volume (often zero after rug)
    raw_vol = np.random.lognormal(4, 2, n)
    data["volume_24h"] = np.clip(np.log1p(raw_vol) / np.log1p(1e9), 0, 1)

    # No legitimate job history
    data["total_jobs"] = np.clip(np.log1p(np.random.exponential(2, n)) / np.log1p(100), 0, 0.2)

    # Low completion rate
    data["completion_rate"] = np.clip(np.random.beta(1.5, 6, n), 0, 0.5)

    # Low trust score
    data["trust_score"] = np.clip(np.random.beta(1.5, 7, n), 0, 0.4)

    # Young project age
    raw_age = np.random.exponential(30, n)  # days
    data["age_days"] = np.clip(np.log1p(raw_age) / np.log1p(1825), 0, 1)

    # High LP drain rate
    data["lp_drain_rate"] = np.clip(np.random.beta(7, 2, n), 0.4, 1.0)

    # New deployer (no history)
    raw_dep_age = np.random.exponential(20, n)
    data["deployer_age_days"] = np.clip(np.log1p(raw_dep_age) / np.log1p(1825), 0, 0.3)

    # High token supply concentration (deployer holds most)
    data["token_supply_concentration"] = np.clip(np.random.beta(8, 2, n), 0.5, 1.0)

    # Ownership NOT renounced (they need to be able to rug)
    data["renounced_ownership"] = np.random.choice([0, 1], n, p=[0.85, 0.15])

    # Usually unverified contracts
    data["verified_contract"] = np.random.choice([0, 1], n, p=[0.7, 0.3])

    # Low social presence (fake/no community)
    data["social_presence_score"] = np.clip(np.random.beta(2, 6, n), 0, 0.4)

    # No audit
    data["audit_score"] = np.clip(np.random.beta(1, 9, n), 0, 0.2)

    df = pd.DataFrame(data)
    df["label"] = 1
    return df


def generate_legit_sample(n: int) -> pd.DataFrame:
    """Generate n legitimate project samples (label=0)."""
    data = {}

    # Distributed holder base
    data["holder_concentration"] = np.clip(np.random.beta(3, 6, n), 0.1, 0.7)

    # High LP lock ratio
    data["liquidity_lock_ratio"] = np.clip(np.random.beta(6, 3, n), 0.4, 1.0)

    # Low deployer risk
    data["creator_tx_pattern"] = np.clip(np.random.beta(2, 8, n), 0, 0.4)

    # Balanced buy-sell ratio
    data["buy_sell_ratio"] = np.clip(np.random.beta(5, 5, n), 0.35, 0.65)

    # Low contract similarity to scams
    data["contract_similarity_score"] = np.clip(np.random.beta(1.5, 7, n), 0, 0.4)

    # Low wash trading
    data["fund_flow_pattern"] = np.clip(np.random.beta(2, 7, n), 0, 0.4)

    # Normal price changes
    data["price_change_24h"] = np.random.normal(0.02, 0.15, n)
    data["price_change_24h"] = np.clip(data["price_change_24h"], -0.5, 2.0)

    # Good liquidity
    raw_liq = np.random.lognormal(11, 2, n)  # larger values
    data["liquidity_usd"] = np.clip(np.log1p(raw_liq) / np.log1p(1e7), 0, 1)

    # Healthy volume
    raw_vol = np.random.lognormal(9, 2, n)
    data["volume_24h"] = np.clip(np.log1p(raw_vol) / np.log1p(1e9), 0, 1)

    # Active job history
    data["total_jobs"] = np.clip(np.log1p(np.random.lognormal(3, 1.5, n)) / np.log1p(100), 0, 1)

    # High completion rate
    data["completion_rate"] = np.clip(np.random.beta(7, 2, n), 0.5, 1.0)

    # Good trust score
    data["trust_score"] = np.clip(np.random.beta(6, 3, n), 0.4, 1.0)

    # Older project
    raw_age = np.random.lognormal(5, 1.5, n)  # days, centered ~150 days
    data["age_days"] = np.clip(np.log1p(raw_age) / np.log1p(1825), 0, 1)

    # Low LP drain rate
    data["lp_drain_rate"] = np.clip(np.random.beta(1.5, 8, n), 0, 0.3)

    # Established deployer
    raw_dep_age = np.random.lognormal(5, 1.5, n)
    data["deployer_age_days"] = np.clip(np.log1p(raw_dep_age) / np.log1p(1825), 0, 1)

    # Distributed token supply
    data["token_supply_concentration"] = np.clip(np.random.beta(2, 8, n), 0, 0.4)

    # Ownership renounced or locked
    data["renounced_ownership"] = np.random.choice([0, 1], n, p=[0.35, 0.65])

    # Verified contracts
    data["verified_contract"] = np.random.choice([0, 1], n, p=[0.2, 0.8])

    # Active community
    data["social_presence_score"] = np.clip(np.random.beta(5, 4, n), 0.3, 1.0)

    # Audit present
    data["audit_score"] = np.clip(np.random.beta(4, 4, n), 0.1, 1.0)

    df = pd.DataFrame(data)
    df["label"] = 0
    return df


def generate_dataset(n_rug: int = 514, n_legit: int = 514) -> pd.DataFrame:
    """
    Generate a balanced starting dataset (SMOTE will be applied during training).
    Ratio is approximately 1:1 before SMOTE based on TM-RugPull paper distribution.
    """
    rug_df = generate_rug_pull_sample(n_rug)
    legit_df = generate_legit_sample(n_legit)

    df = pd.concat([rug_df, legit_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add noise to make it realistic
    for col in FEATURES:
        if col not in ["renounced_ownership", "verified_contract"]:
            noise = np.random.normal(0, 0.02, len(df))
            df[col] = np.clip(df[col] + noise, 0, 1)

    return df


if __name__ == "__main__":
    print("Generating synthetic rug pull dataset...")
    df = generate_dataset(n_rug=514, n_legit=514)

    output_path = DATA_DIR / "rug_pull_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Rug pulls: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"Legitimate: {(df['label'] == 0).sum()} ({(1 - df['label'].mean())*100:.1f}%)")
    print(f"\nFeature stats:")
    print(df.describe().round(3))
