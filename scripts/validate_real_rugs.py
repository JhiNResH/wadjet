"""
Wadjet — Manual Validation Against Known Rug Pulls + Legit Tokens
=================================================================
Tests the retrained XGBoost model against well-documented cases.

Known Rug Pulls (label=1):
  1. Squid Game Token (SQUID) — Oct 2021, ETH — ~$3.38M stolen
  2. AnubisDAO (ANKH) — Oct 2021, ETH — ~$60M stolen  
  3. Frosties NFT/FLOKI — multiple rugpulls
  4. Meerkat Finance — BSC — $31M
  5. TurtleDex — BSC — $2.5M

Known Legit Tokens (label=0):
  1. Uniswap (UNI) — long-lived DEX token
  2. Chainlink (LINK) — oracle token
  3. Aave (AAVE) — lending protocol
  4. Compound (COMP) — DeFi bluechip
  5. SushiSwap (SUSHI) — DEX (had drama but not a rug)

These are manually constructed feature vectors based on known on-chain characteristics.
Features are approximations since we don't have historical Uniswap V2 data for all of these.
"""

import sys
import json
from pathlib import Path
import numpy as np
import joblib

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR.parent / "models"

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

# ── Known Rug Pulls ─────────────────────────────────────────────────────────
# Feature values derived from on-chain behavior profiles
KNOWN_RUGS = [
    {
        "name": "Squid Game Token (SQUID)",
        "date": "2021-11-01",
        "chain": "ETH",
        "loss_usd": 3_380_000,
        "source": "rekt.news, CoinMarketCap",
        # Characteristics: creator held all supply, no lock, buy-only (sell disabled!)
        # LP drained instantly after pump
        "features": {
            "holder_concentration": 0.90,   # creator held vast majority of LP
            "liquidity_lock_ratio": 0.0,    # no lock
            "creator_tx_pattern": 0.85,     # heavy minting activity
            "buy_sell_ratio": 0.98,         # sell was disabled — only buys possible
            "contract_similarity_score": 0.70,
            "fund_flow_pattern": 0.80,
            "price_change_24h": 0.95,       # 45,000% pump before rug
            "liquidity_usd": 0.40,
            "volume_24h": 0.80,
            "total_jobs": 0.0,
            "completion_rate": 0.5,
            "trust_score": 0.05,
            "age_days": 0.05,               # very new token
            "lp_drain_rate": 0.95,          # LP completely removed
            "deployer_age_days": 0.10,
            "token_supply_concentration": 0.90,
            "renounced_ownership": 0,
            "verified_contract": 0,
            "social_presence_score": 0.90,  # huge social hype
            "audit_score": 0.0,
        },
    },
    {
        "name": "AnubisDAO (ANKH)",
        "date": "2021-10-28",
        "chain": "ETH",
        "loss_usd": 60_000_000,
        "source": "rekt.news",
        # 20 hours after launch, all ETH drained from LP (573 ETH)
        "features": {
            "holder_concentration": 0.85,
            "liquidity_lock_ratio": 0.0,    # zero lock — drained immediately
            "creator_tx_pattern": 0.65,
            "buy_sell_ratio": 0.40,
            "contract_similarity_score": 0.55,
            "fund_flow_pattern": 0.75,
            "price_change_24h": 0.98,       # went to near-zero
            "liquidity_usd": 0.70,
            "volume_24h": 0.85,
            "total_jobs": 0.0,
            "completion_rate": 0.5,
            "trust_score": 0.04,
            "age_days": 0.02,               # only 20 hours old
            "lp_drain_rate": 0.98,          # LP completely drained
            "deployer_age_days": 0.05,
            "token_supply_concentration": 0.80,
            "renounced_ownership": 0,
            "verified_contract": 0,
            "social_presence_score": 0.75,
            "audit_score": 0.0,
        },
    },
    {
        "name": "Meerkat Finance",
        "date": "2021-03-04",
        "chain": "BSC",
        "loss_usd": 31_000_000,
        "source": "rekt.news — 'Meerkat Finance'",
        "features": {
            "holder_concentration": 0.75,
            "liquidity_lock_ratio": 0.02,
            "creator_tx_pattern": 0.70,
            "buy_sell_ratio": 0.20,
            "contract_similarity_score": 0.60,
            "fund_flow_pattern": 0.85,
            "price_change_24h": 0.95,
            "liquidity_usd": 0.80,
            "volume_24h": 0.90,
            "total_jobs": 0.0,
            "completion_rate": 0.5,
            "trust_score": 0.06,
            "age_days": 0.01,               # 1 day old
            "lp_drain_rate": 0.99,
            "deployer_age_days": 0.08,
            "token_supply_concentration": 0.78,
            "renounced_ownership": 0,
            "verified_contract": 0,
            "social_presence_score": 0.65,
            "audit_score": 0.0,
        },
    },
    {
        "name": "TurtleDex (TTDX)",
        "date": "2021-03-17",
        "chain": "BSC",
        "loss_usd": 2_500_000,
        "source": "rekt.news — 'TurtleDex'",
        "features": {
            "holder_concentration": 0.80,
            "liquidity_lock_ratio": 0.0,
            "creator_tx_pattern": 0.75,
            "buy_sell_ratio": 0.30,
            "contract_similarity_score": 0.65,
            "fund_flow_pattern": 0.78,
            "price_change_24h": 0.90,
            "liquidity_usd": 0.55,
            "volume_24h": 0.70,
            "total_jobs": 0.0,
            "completion_rate": 0.5,
            "trust_score": 0.07,
            "age_days": 0.04,
            "lp_drain_rate": 0.97,
            "deployer_age_days": 0.12,
            "token_supply_concentration": 0.82,
            "renounced_ownership": 0,
            "verified_contract": 0,
            "social_presence_score": 0.50,
            "audit_score": 0.0,
        },
    },
    {
        "name": "Snowdog (SDOG) — OHM fork rug",
        "date": "2021-11-24",
        "chain": "AVAX",
        "loss_usd": 30_000_000,
        "source": "rekt.news — 'Snowdog'",
        "features": {
            "holder_concentration": 0.88,
            "liquidity_lock_ratio": 0.05,
            "creator_tx_pattern": 0.78,
            "buy_sell_ratio": 0.35,
            "contract_similarity_score": 0.72,
            "fund_flow_pattern": 0.82,
            "price_change_24h": 0.93,
            "liquidity_usd": 0.65,
            "volume_24h": 0.88,
            "total_jobs": 0.0,
            "completion_rate": 0.5,
            "trust_score": 0.05,
            "age_days": 0.03,
            "lp_drain_rate": 0.96,
            "deployer_age_days": 0.15,
            "token_supply_concentration": 0.85,
            "renounced_ownership": 0,
            "verified_contract": 0,
            "social_presence_score": 0.80,
            "audit_score": 0.0,
        },
    },
]

# ── Known Legit Tokens ──────────────────────────────────────────────────────
KNOWN_LEGIT = [
    {
        "name": "Uniswap (UNI)",
        "chain": "ETH",
        "notes": "Governance token, 4yr vesting, community treasury",
        "features": {
            "holder_concentration": 0.15,   # well distributed
            "liquidity_lock_ratio": 0.85,   # protocol-controlled LP
            "creator_tx_pattern": 0.15,
            "buy_sell_ratio": 0.52,
            "contract_similarity_score": 0.10,
            "fund_flow_pattern": 0.05,
            "price_change_24h": 0.15,
            "liquidity_usd": 0.98,
            "volume_24h": 0.97,
            "total_jobs": 0.0,
            "completion_rate": 0.5,
            "trust_score": 0.88,
            "age_days": 0.95,
            "lp_drain_rate": 0.02,
            "deployer_age_days": 0.90,
            "token_supply_concentration": 0.12,
            "renounced_ownership": 1,
            "verified_contract": 1,
            "social_presence_score": 0.95,
            "audit_score": 0.95,
        },
    },
    {
        "name": "Chainlink (LINK)",
        "chain": "ETH",
        "notes": "Oracle infrastructure, active development since 2017",
        "features": {
            "holder_concentration": 0.20,
            "liquidity_lock_ratio": 0.80,
            "creator_tx_pattern": 0.10,
            "buy_sell_ratio": 0.50,
            "contract_similarity_score": 0.05,
            "fund_flow_pattern": 0.03,
            "price_change_24h": 0.20,
            "liquidity_usd": 0.99,
            "volume_24h": 0.96,
            "total_jobs": 0.0,
            "completion_rate": 0.5,
            "trust_score": 0.90,
            "age_days": 0.99,
            "lp_drain_rate": 0.01,
            "deployer_age_days": 0.99,
            "token_supply_concentration": 0.18,
            "renounced_ownership": 1,
            "verified_contract": 1,
            "social_presence_score": 0.90,
            "audit_score": 0.90,
        },
    },
    {
        "name": "Aave (AAVE)",
        "chain": "ETH",
        "notes": "Lending protocol, multiple audits, DAO governance",
        "features": {
            "holder_concentration": 0.18,
            "liquidity_lock_ratio": 0.82,
            "creator_tx_pattern": 0.12,
            "buy_sell_ratio": 0.53,
            "contract_similarity_score": 0.08,
            "fund_flow_pattern": 0.04,
            "price_change_24h": 0.18,
            "liquidity_usd": 0.97,
            "volume_24h": 0.94,
            "total_jobs": 0.0,
            "completion_rate": 0.5,
            "trust_score": 0.92,
            "age_days": 0.97,
            "lp_drain_rate": 0.01,
            "deployer_age_days": 0.96,
            "token_supply_concentration": 0.15,
            "renounced_ownership": 1,
            "verified_contract": 1,
            "social_presence_score": 0.88,
            "audit_score": 0.98,
        },
    },
    {
        "name": "Compound (COMP)",
        "chain": "ETH",
        "notes": "DeFi lending governance token, Gauntlet-audited",
        "features": {
            "holder_concentration": 0.22,
            "liquidity_lock_ratio": 0.78,
            "creator_tx_pattern": 0.14,
            "buy_sell_ratio": 0.51,
            "contract_similarity_score": 0.07,
            "fund_flow_pattern": 0.04,
            "price_change_24h": 0.22,
            "liquidity_usd": 0.95,
            "volume_24h": 0.92,
            "total_jobs": 0.0,
            "completion_rate": 0.5,
            "trust_score": 0.89,
            "age_days": 0.94,
            "lp_drain_rate": 0.02,
            "deployer_age_days": 0.93,
            "token_supply_concentration": 0.20,
            "renounced_ownership": 1,
            "verified_contract": 1,
            "social_presence_score": 0.85,
            "audit_score": 0.95,
        },
    },
    {
        "name": "SushiSwap (SUSHI)",
        "chain": "ETH",
        "notes": "DEX fork of Uniswap, Chef Nomi drama resolved, still live",
        "features": {
            "holder_concentration": 0.30,   # slightly higher (early concentration)
            "liquidity_lock_ratio": 0.65,
            "creator_tx_pattern": 0.20,
            "buy_sell_ratio": 0.49,
            "contract_similarity_score": 0.15,
            "fund_flow_pattern": 0.08,
            "price_change_24h": 0.30,
            "liquidity_usd": 0.90,
            "volume_24h": 0.89,
            "total_jobs": 0.0,
            "completion_rate": 0.5,
            "trust_score": 0.72,
            "age_days": 0.90,
            "lp_drain_rate": 0.05,
            "deployer_age_days": 0.85,
            "token_supply_concentration": 0.25,
            "renounced_ownership": 0,
            "verified_contract": 1,
            "social_presence_score": 0.80,
            "audit_score": 0.75,
        },
    },
]


def predict_one(model, feat_dict: dict, threshold: float = 0.35) -> tuple[float, bool]:
    vec = np.array([[feat_dict[f] for f in FEATURES]], dtype=np.float32)
    prob = float(model.predict_proba(vec)[0][1])
    return prob, prob >= threshold


def main():
    print("=" * 70)
    print("🔮 Wadjet — Manual Validation (Known Rugs + Legit Tokens)")
    print("=" * 70)

    model_path = MODELS_DIR / "wadjet_xgb.joblib"
    if not model_path.exists():
        print("❌ Model not found. Run train_model.py first.")
        sys.exit(1)

    model = joblib.load(model_path)
    THRESHOLD = 0.35

    results = {
        "true_positives": [],   # correctly identified rugs
        "false_negatives": [],  # missed rugs
        "true_negatives": [],   # correctly identified legit
        "false_positives": [],  # wrongly flagged legit
    }

    print("\n─── KNOWN RUG PULLS (expected: flagged as rug) ───────────────────")
    print(f"{'Token':<35} {'Prob':>6} {'Predict':>9} {'Correct':>8}")
    print("-" * 65)

    for case in KNOWN_RUGS:
        prob, is_rug = predict_one(model, case["features"], THRESHOLD)
        correct = "✅" if is_rug else "❌ MISS"
        category = "true_positives" if is_rug else "false_negatives"
        results[category].append(case["name"])
        print(f"{case['name']:<35} {prob:>5.1%} {'RUG':>9} {correct:>8}")

    print("\n─── KNOWN LEGIT TOKENS (expected: NOT flagged) ───────────────────")
    print(f"{'Token':<35} {'Prob':>6} {'Predict':>9} {'Correct':>8}")
    print("-" * 65)

    for case in KNOWN_LEGIT:
        prob, is_rug = predict_one(model, case["features"], THRESHOLD)
        correct = "✅" if not is_rug else "❌ FP"
        category = "true_negatives" if not is_rug else "false_positives"
        results[category].append(case["name"])
        print(f"{case['name']:<35} {prob:>5.1%} {'LEGIT' if not is_rug else 'RUG':>9} {correct:>8}")

    print("\n─── SUMMARY ──────────────────────────────────────────────────────")
    n_rugs = len(KNOWN_RUGS)
    n_legit = len(KNOWN_LEGIT)
    tp = len(results["true_positives"])
    fn = len(results["false_negatives"])
    tn = len(results["true_negatives"])
    fp = len(results["false_positives"])

    print(f"  Rug detection:  {tp}/{n_rugs} correctly flagged  (recall {tp/n_rugs:.0%})")
    print(f"  Legit accuracy: {tn}/{n_legit} correctly passed  (specificity {tn/n_legit:.0%})")

    if fn:
        print(f"\n  ⚠️  Missed rugs: {results['false_negatives']}")
    if fp:
        print(f"  ⚠️  False alarms: {results['false_positives']}")

    if fn == 0 and fp == 0:
        print("\n  🎯 Perfect validation — no misses, no false alarms!")

    # Save validation report
    report = {
        "validation_date": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "model_version": "1.1.0",
        "threshold": THRESHOLD,
        "results": results,
        "cases_tested": {
            "rugs": [c["name"] for c in KNOWN_RUGS],
            "legit": [c["name"] for c in KNOWN_LEGIT],
        },
        "summary": {
            "rug_recall": tp / n_rugs,
            "legit_specificity": tn / n_legit,
            "overall": (tp + tn) / (n_rugs + n_legit),
        }
    }

    report_path = MODELS_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Validation report saved: {report_path}")

    return report


if __name__ == "__main__":
    main()
