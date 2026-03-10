"""
Wadjet Phase 3 — Validation Script
====================================
Validates the V2 agent model against known Virtuals tokens.
Checks accuracy on our labeled dataset and cached real tokens.
"""
import json
import math
import time
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_v2")

SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR.parent / "data"
MODELS_DIR  = SCRIPT_DIR.parent / "models"
CACHE_FILE  = DATA_DIR / "dex_cache.json"

THRESHOLD = 0.35

V1_FEATURES = [
    "holder_concentration", "liquidity_lock_ratio", "creator_tx_pattern",
    "buy_sell_ratio", "contract_similarity_score", "fund_flow_pattern",
    "price_change_24h", "liquidity_usd", "volume_24h", "total_jobs",
    "completion_rate", "trust_score", "age_days", "lp_drain_rate",
    "deployer_age_days", "token_supply_concentration", "renounced_ownership",
    "verified_contract", "social_presence_score", "audit_score",
]
VIRTUALS_FEATURES = [
    "bonding_curve_position", "lp_locked_pct", "creator_other_tokens",
    "creator_wallet_age", "holder_count_norm", "top10_holder_pct",
    "acp_job_count", "acp_completion_rate", "acp_trust_score",
    "token_age_days_norm", "volume_to_mcap_ratio", "price_volatility_7d",
    "social_presence", "is_honeypot", "is_mintable", "hidden_owner",
    "slippage_modifiable", "buy_tax", "sell_tax", "creator_percent",
    "owner_percent", "has_dex_data", "is_virtuals_token",
]
ALL_FEATURES = V1_FEATURES + VIRTUALS_FEATURES


def load_model():
    v2_path = MODELS_DIR / "wadjet_xgb_v2_agent.joblib"
    v1_path = MODELS_DIR / "wadjet_xgb.joblib"
    if v2_path.exists():
        return joblib.load(v2_path), "V2"
    return joblib.load(v1_path), "V1"


def predict_row(model, row: dict, n_features: int) -> tuple[float, int]:
    X = np.array([row.get(f, 0.0) for f in ALL_FEATURES[:n_features]], dtype=np.float32)
    prob = float(model.predict_proba(X.reshape(1, -1))[0, 1])
    return prob, int(prob >= THRESHOLD)


def main():
    model, model_tag = load_model()
    n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 20
    logger.info(f"Loaded {model_tag} model, expects {n_features} features")

    # Load labeled dataset
    dataset_path = DATA_DIR / "virtuals_agent_dataset.csv"
    if not dataset_path.exists():
        logger.error("No labeled dataset found")
        return

    df = pd.read_csv(dataset_path)
    real_df = df[df["data_source"] == "virtuals_agent"]  # Real data only (no synthetic)
    logger.info(f"Real Virtuals samples: {len(real_df)} (rugs: {real_df['label'].sum()}, legit: {(real_df['label']==0).sum()})")

    print("\n" + "=" * 80)
    print(f"VALIDATION: REAL VIRTUALS TOKENS ({model_tag} model)")
    print("=" * 80)
    print(f"{'Token':<16} {'True':>6} {'Pred':>6} {'Score':>8} {'Risk':<10} {'Verdict':<10} Reason")
    print("-" * 80)

    correct = 0
    false_pos = 0
    false_neg = 0

    for _, row in real_df.iterrows():
        features = {f: row.get(f, 0.0) for f in ALL_FEATURES}
        prob, pred = predict_row(model, features, n_features)

        true_label = int(row["label"])
        sym = str(row.get("token_symbol") or row["token_address"][:12])

        if pred == true_label:
            correct += 1
            verdict = "✅ CORRECT"
        elif pred == 1 and true_label == 0:
            false_pos += 1
            verdict = "❌ FALSE+  "
        else:
            false_neg += 1
            verdict = "❌ FALSE-  "

        true_str = "🔴 RUG" if true_label else "🟢 OK"
        pred_str = "🔴 RUG" if pred else "🟢 OK"
        print(f"{sym:<16} {true_str:>8} {pred_str:>8} {prob:8.3f} {verdict:<10} {str(row.get('label_reason',''))[:30]}")

    total = len(real_df)
    acc = correct / total if total > 0 else 0
    print("\n" + "=" * 80)
    print(f"RESULTS ON REAL VIRTUALS DATA:")
    print(f"  Total samples:     {total}")
    print(f"  Correct:           {correct} ({acc:.0%})")
    print(f"  False Positives:   {false_pos} (flagged legit as rug)")
    print(f"  False Negatives:   {false_neg} (missed rug)")
    print("=" * 80)

    # Specifically test the cache data tokens
    print("\n" + "=" * 80)
    print("KNOWN TOKEN SPOT-CHECK (from DexScreener cache)")
    print("=" * 80)

    cache = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())

    # Known patterns to test
    known_tests = [
        # (description, features_dict, expected_label)
        (
            "WAYE - 99% holder concentration",
            {
                "holder_concentration": 0.99,
                "top10_holder_pct": 0.99,
                "volume_24h": math.log1p(28) / math.log1p(1_000_000),
                "trust_score": 0.99,
                "acp_trust_score": 0.99,
                "is_virtuals_token": 1.0,
                "has_dex_data": 1.0,
                "acp_job_count": math.log1p(183) / math.log1p(100000),
            },
            1  # Expected: RUG
        ),
        (
            "ETHY - 100 trust, 1.1M jobs",
            {
                "trust_score": 1.0,
                "completion_rate": 0.99,
                "total_jobs": math.log1p(1138440) / math.log1p(100000),
                "acp_trust_score": 1.0,
                "acp_completion_rate": 0.99,
                "acp_job_count": math.log1p(1138440) / math.log1p(100000),
                "liquidity_usd": math.log1p(247265) / math.log1p(1_000_000),
                "is_virtuals_token": 1.0,
                "has_dex_data": 1.0,
                "holder_concentration": 0.0,
            },
            0  # Expected: LEGIT
        ),
        (
            "Ghost agent - no jobs, old token, no volume",
            {
                "acp_job_count": 0.0,
                "acp_trust_score": 0.0,
                "trust_score": 0.0,
                "total_jobs": 0.0,
                "volume_24h": 0.0,
                "liquidity_usd": math.log1p(50) / math.log1p(1_000_000),
                "age_days": math.log1p(90) / math.log1p(3650),
                "token_age_days_norm": math.log1p(90) / math.log1p(3650),
                "is_virtuals_token": 1.0,
                "has_dex_data": 1.0,
            },
            1  # Expected: RUG
        ),
        (
            "Honeypot - 90% sell tax",
            {
                "sell_tax": 0.90,
                "is_honeypot": 1.0,
                "slippage_modifiable": 1.0,
                "holder_concentration": 0.85,
                "top10_holder_pct": 0.85,
                "is_virtuals_token": 1.0,
            },
            1  # Expected: RUG
        ),
        (
            "AXR - 100 trust, 127K jobs, $162K liquidity",
            {
                "trust_score": 1.0,
                "completion_rate": 0.99,
                "total_jobs": math.log1p(127212) / math.log1p(100000),
                "acp_trust_score": 1.0,
                "acp_job_count": math.log1p(127212) / math.log1p(100000),
                "liquidity_usd": math.log1p(161846) / math.log1p(1_000_000),
                "is_virtuals_token": 1.0,
                "has_dex_data": 1.0,
            },
            0  # Expected: LEGIT
        ),
    ]

    correct_known = 0
    for desc, feat_dict, expected in known_tests:
        features = {f: 0.0 for f in ALL_FEATURES}
        features.update(feat_dict)
        prob, pred = predict_row(model, features, n_features)

        verdict = "✅" if pred == expected else "❌"
        true_str = "🔴 RUG" if expected else "🟢 OK "
        pred_str = "🔴 RUG" if pred else "🟢 OK "
        print(f"  {verdict} {desc}")
        print(f"     Expected: {true_str} | Predicted: {pred_str} | Score: {prob:.3f}")
        if pred == expected:
            correct_known += 1

    print(f"\n  Spot-check accuracy: {correct_known}/{len(known_tests)} = {correct_known/len(known_tests):.0%}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Real data accuracy: {acc:.0%} ({correct}/{total})")
    print(f"Known pattern accuracy: {correct_known/len(known_tests):.0%}")
    print(f"\nNote: Model is primarily trained on V1 (Uniswap V2) data.")
    print(f"As more Virtuals samples are collected, retrain for better accuracy.")


if __name__ == "__main__":
    main()
