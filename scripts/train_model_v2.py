"""
Wadjet Phase 3 — XGBoost v2 Agent-Enhanced Training Script
===========================================================
Trains a new XGBoost model that combines:
  1. Original 18K Uniswap V2 rug detection data (generic DeFi patterns)
  2. New Virtuals agent token data (agent-specific features)

New Virtuals-specific features are set to 0/NaN for Uniswap V2 samples.
Outputs:
  - models/wadjet_xgb_v2_agent.joblib  — production model
  - models/model_v2_metadata.json      — metrics + feature importance
"""

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
MODELS_DIR = SCRIPT_DIR.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ─── Feature Sets ─────────────────────────────────────────────────────────────

# V1 features (present in BOTH Uniswap V2 and Virtuals datasets)
V1_FEATURES_CORE = [
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

# New V1.2 features — added after original 20 (backward-compat: old model uses :20)
V1_NEW_FEATURES = [
    "data_completeness",  # Fraction of V1 core features with real values (1.0 = all real)
    "is_ghost_agent",     # 1 if total_jobs==0 AND completion_rate==0 AND token exists
]

V1_FEATURES = V1_FEATURES_CORE + V1_NEW_FEATURES

# V2 features — Virtuals/agent-specific (set to 0 for Uniswap V2 samples)
VIRTUALS_FEATURES = [
    "bonding_curve_position",
    "lp_locked_pct",
    "creator_other_tokens",
    "creator_wallet_age",
    "holder_count_norm",       # log-normalized
    "top10_holder_pct",
    "acp_job_count",
    "acp_completion_rate",
    "acp_trust_score",
    "token_age_days_norm",     # log-normalized
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
    "is_virtuals_token",       # 1 for Virtuals tokens, 0 for Uniswap V2
    # Dynamic delta features (0.0 for all existing training data — no history)
    "holder_concentration_delta_1d",
    "liquidity_delta_1d",
    "volume_delta_1d",
    "price_delta_1d",
    "creator_percent_delta_1d",
]

ALL_FEATURES = V1_FEATURES + VIRTUALS_FEATURES


def load_v1_dataset() -> pd.DataFrame:
    """Load the original Uniswap V2 training dataset."""
    real_path = DATA_DIR / "rug_pull_dataset_real.csv"
    synth_path = DATA_DIR / "rug_pull_dataset.csv"

    if real_path.exists():
        df = pd.read_csv(real_path)
        source = "real"
    elif synth_path.exists():
        df = pd.read_csv(synth_path)
        source = "synthetic"
    else:
        print("❌ No v1 dataset found. Run build_real_dataset.py first.")
        sys.exit(1)

    print(f"📊 V1 dataset: {source}, {len(df)} samples")
    print(f"   V1 label distribution: {df['label'].value_counts().to_dict()}")

    # Rename v1 columns to match unified schema
    col_map = {
        "price_change_24h": "price_change_24h",
        "liquidity_usd":    "liquidity_usd",
        "volume_24h":       "volume_24h",
    }

    # ── New V1.2 features ─────────────────────────────────────────────────
    # data_completeness: real datasets have all features known → 1.0
    df["data_completeness"] = 1.0

    # is_ghost_agent: 1 if no ACP jobs AND no completion rate
    df["is_ghost_agent"] = 0.0
    ghost_mask = (df["total_jobs"] == 0.0) & (df["completion_rate"] == 0.0)
    df.loc[ghost_mask, "is_ghost_agent"] = 1.0

    # ── Append known rugs validation set ──────────────────────────────────
    known_rugs_path = DATA_DIR / "known_rugs_validation.csv"
    if known_rugs_path.exists():
        known_df_raw = pd.read_csv(known_rugs_path)
        print(f"\n⚠️  Loading known rugs: {len(known_df_raw)} entries")
        known_rows = []
        for _, row in known_df_raw.iterrows():
            known_rows.append({
                "holder_concentration": 0.9, "liquidity_lock_ratio": 0.0,
                "creator_tx_pattern": 0.8, "buy_sell_ratio": 0.2,
                "contract_similarity_score": 0.7, "fund_flow_pattern": 0.8,
                "price_change_24h": -0.98, "liquidity_usd": 0.01,
                "volume_24h": 0.0, "total_jobs": 0.0,
                "completion_rate": 0.0, "trust_score": 0.0,
                "age_days": 0.05, "lp_drain_rate": 0.9,
                "deployer_age_days": 0.1, "token_supply_concentration": 0.9,
                "renounced_ownership": 0, "verified_contract": 0,
                "social_presence_score": 0.0, "audit_score": 0.0,
                "data_completeness": 1.0, "is_ghost_agent": 1.0,
                "label": 1,
            })
        known_rug_df = pd.DataFrame(known_rows)
        # Add Virtuals features (zeros for these known rugs)
        for feat in VIRTUALS_FEATURES:
            if feat not in known_rug_df.columns:
                known_rug_df[feat] = 0.0
        known_rug_df["is_virtuals_token"] = 0.0
        known_rug_df["data_source"] = "known_rug"
        df = pd.concat([df, known_rug_df], ignore_index=True)
        print(f"   After known rugs: {len(df)} total V1 samples")

    # Add Virtuals-specific columns (all zeros for V1 data, including new delta features)
    for feat in VIRTUALS_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    df["is_virtuals_token"] = df.get("is_virtuals_token", pd.Series(0.0, index=df.index))
    df["data_source"] = df.get("data_source", pd.Series("uniswap_v2", index=df.index))

    return df[V1_FEATURES + VIRTUALS_FEATURES + ["label", "data_source"]]


def load_v2_virtuals_dataset() -> pd.DataFrame:
    """Load the Virtuals agent token dataset."""
    path = DATA_DIR / "virtuals_agent_dataset.csv"
    if not path.exists():
        print("⚠️  Virtuals dataset not found. Run build_virtuals_dataset.py first.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"📊 Virtuals dataset: {len(df)} samples")
    print(f"   Virtuals label distribution: {df['label'].value_counts().to_dict()}")

    import math

    # Map Virtuals CSV columns to unified feature names
    if "holder_count" in df.columns:
        df["holder_count_norm"] = df["holder_count"].apply(
            lambda x: math.log1p(max(0, x)) / math.log1p(100000)
        )
    else:
        df["holder_count_norm"] = 0.0

    if "token_age_days" in df.columns:
        df["token_age_days_norm"] = df["token_age_days"].apply(
            lambda x: math.log1p(max(0, x)) / math.log1p(3650)
        )
    else:
        df["token_age_days_norm"] = 0.0

    # Map v2 DexScreener/normalized cols to v1 compatible names
    col_map = {
        "price_change_24h_norm":  "price_change_24h",
        "liquidity_usd_v1":       "liquidity_usd",
        "volume_24h_norm":        "volume_24h",
        "social_presence":        "social_presence_score",
    }
    for src, dst in col_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # Ensure all V1 features exist
    for feat in V1_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    # ── New V1.2 features for Virtuals data ──────────────────────────────
    # data_completeness: Virtuals data is largely synthetic/estimated → 0.3
    df["data_completeness"] = 0.3

    # is_ghost_agent: 1 if no ACP jobs AND no completion rate
    acp_jobs_col = "acp_job_count" if "acp_job_count" in df.columns else "total_jobs"
    acp_rate_col = "acp_completion_rate" if "acp_completion_rate" in df.columns else "completion_rate"
    df["is_ghost_agent"] = 0.0
    if acp_jobs_col in df.columns and acp_rate_col in df.columns:
        ghost_mask = (df[acp_jobs_col] == 0.0) & (df[acp_rate_col] == 0.0)
        df.loc[ghost_mask, "is_ghost_agent"] = 1.0

    # Ensure all Virtuals features exist (fills missing columns with 0.0)
    for feat in VIRTUALS_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    df["is_virtuals_token"] = 1.0
    df["data_source"] = "virtuals_agent"

    return df[V1_FEATURES + VIRTUALS_FEATURES + ["label", "data_source"]]


def build_xgb_v2_model(scale_pos_weight: float = 1.5):
    """XGBoost v2 with higher capacity for agent-specific patterns."""
    return XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )


def train_and_evaluate(X: np.ndarray, y: np.ndarray, feature_names: list):
    """Train + cross-validate + evaluate on test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (5-fold, Stratified)")
    print("=" * 60)

    model = build_xgb_v2_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_recall  = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall")
    cv_f1      = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    cv_roc     = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_acc     = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    print(f"Accuracy:  {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
    print(f"F1:        {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"ROC AUC:   {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")

    print("\n" + "=" * 60)
    print("FINAL MODEL TRAINING")
    print("=" * 60)

    final_model = build_xgb_v2_model()
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_prob = final_model.predict_proba(X_test)[:, 1]
    THRESHOLD = 0.35
    y_pred = (y_prob >= THRESHOLD).astype(int)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)
    cm        = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nTest Set Results (threshold={THRESHOLD}):")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}  ← (1 - false_negative_rate)")
    print(f"F1 Score:   {f1:.4f}")
    print(f"ROC AUC:    {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    print(f"False Negative Rate: {fn/(fn+tp):.4f} (missed rugs)")

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Top 20)")
    print("=" * 60)
    importance = final_model.feature_importances_
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:20]:
        bar = "█" * int(imp * 100)
        print(f"  {feat:<40} {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("VIRTUALS-SPECIFIC FEATURE IMPORTANCE")
    print("=" * 60)
    virtuals_imp = [(f, i) for f, i in feat_imp if f in VIRTUALS_FEATURES]
    for feat, imp in virtuals_imp[:15]:
        bar = "█" * int(imp * 100)
        marker = " ⭐" if imp > 0.01 else ""
        print(f"  {feat:<40} {imp:.4f} {bar}{marker}")

    print(classification_report(y_test, y_pred, target_names=["Legit", "Rug Pull"]))

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "confusion_matrix": cm.tolist(),
        "cv_accuracy_mean": float(cv_acc.mean()),
        "cv_recall_mean": float(cv_recall.mean()),
        "cv_f1_mean": float(cv_f1.mean()),
        "prediction_threshold": THRESHOLD,
        "feature_importance": {f: float(i) for f, i in feat_imp},
        "virtuals_feature_importance": {f: float(i) for f, i in virtuals_imp},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    return final_model, metrics


def compare_with_v1(metrics_v2: dict):
    """Load v1 metrics and print comparison table."""
    v1_meta_path = MODELS_DIR / "model_metadata.json"
    if not v1_meta_path.exists():
        print("\n⚠️  V1 metadata not found — skipping comparison")
        return

    with open(v1_meta_path) as f:
        v1_meta = json.load(f)

    v1_m = v1_meta.get("metrics", {})
    v2_m = metrics_v2

    print("\n" + "=" * 60)
    print("MODEL COMPARISON: V1 (generic) vs V2 (agent-enhanced)")
    print("=" * 60)
    for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "false_negative_rate"]:
        v1_val = v1_m.get(metric, 0)
        v2_val = v2_m.get(metric, 0)
        delta  = v2_val - v1_val
        arrow  = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {metric:<25} V1: {v1_val:.4f}  V2: {v2_val:.4f}  {arrow} {abs(delta):.4f}")


def main():
    print("🔮 Wadjet XGBoost v2 — Agent-Enhanced Rug Detection Training")
    print("=" * 60)

    # Load both datasets
    df_v1 = load_v1_dataset()
    df_v2 = load_v2_virtuals_dataset()

    if df_v2.empty:
        print("\n⚠️  Training on V1 data only (no Virtuals dataset)")
        df_combined = df_v1
    else:
        print(f"\n📊 Combining datasets:")
        print(f"   V1 (Uniswap V2): {len(df_v1)} samples")
        print(f"   V2 (Virtuals):   {len(df_v2)} samples")
        df_combined = pd.concat([df_v1, df_v2], ignore_index=True)

    print(f"   Combined total:  {len(df_combined)} samples")
    print(f"   Label distribution: {df_combined['label'].value_counts().to_dict()}")
    print(f"   Virtuals samples: {(df_combined['data_source'] == 'virtuals_agent').sum()}")

    # Prepare features
    df_combined = df_combined.fillna(0.0)
    X = df_combined[ALL_FEATURES].values.astype(np.float32)
    y = df_combined["label"].values.astype(int)

    # SMOTE to balance classes
    print(f"\nApplying SMOTE for class balancing...")
    try:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"After SMOTE: {len(X_resampled)} samples (rugs: {y_resampled.sum()}, legit: {(y_resampled==0).sum()})")
    except Exception as e:
        print(f"⚠️  SMOTE failed ({e}), using original data")
        X_resampled, y_resampled = X, y

    # Train
    model, metrics = train_and_evaluate(X_resampled, y_resampled, ALL_FEATURES)

    # Compare with V1
    compare_with_v1(metrics)

    # Save model
    model_path = MODELS_DIR / "wadjet_xgb_v2_agent.joblib"
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved: {model_path}")

    # Save JSON for ONNX/portable use
    model.save_model(str(MODELS_DIR / "wadjet_xgb_v2_agent.json"))
    print(f"✅ Model JSON saved: {MODELS_DIR / 'wadjet_xgb_v2_agent.json'}")

    # Save metadata
    metadata = {
        "model_version": "2.2.0",  # v2.2 = delta features (holder_conc, liquidity, volume, price, creator)
        "model_type": "agent_enhanced",
        "training_date": datetime.utcnow().isoformat() + "Z",
        "algorithm": "XGBoost",
        "n_features": len(ALL_FEATURES),
        "features": ALL_FEATURES,
        "v1_features": V1_FEATURES,
        "virtuals_features": VIRTUALS_FEATURES,
        "metrics": metrics,
        "dataset": {
            "v1_samples": len(df_v1),
            "v2_samples": len(df_v2) if not df_v2.empty else 0,
            "combined_samples": len(df_combined),
            "samples_after_smote": len(X_resampled),
        },
        "deployment": {
            "prediction_threshold": metrics["prediction_threshold"],
            "use_endpoint": "POST /predict/agent",
            "note": "Use /predict/agent for Virtuals tokens, /predict for generic DeFi"
        }
    }

    meta_path = MODELS_DIR / "model_v2_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved: {meta_path}")

    print("\n" + "=" * 60)
    print("🎯 V2 TRAINING COMPLETE")
    print(f"   Accuracy:  {metrics['accuracy']:.1%}")
    print(f"   Recall:    {metrics['recall']:.1%}")
    print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"   False Neg: {metrics['false_negative_rate']:.1%} (missed rugs)")
    print("=" * 60)

    return metadata


if __name__ == "__main__":
    main()
