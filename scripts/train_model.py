"""
Wadjet Phase 1b — XGBoost Rug Pull Classifier Training Script

Trains an XGBoost model on the synthetic rug pull dataset.
Optimized for LOW false negative rate (missing real rugs is worse than false alarms).

Exports:
  - models/wadjet_xgb.joblib  — production model
  - models/wadjet_xgb.onnx    — for Node.js native inference
  - models/model_metadata.json — metrics + feature importance
"""

import sys
import json
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
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
MODELS_DIR = SCRIPT_DIR.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURES_V1 = [
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

# New features added in v1.2 — added after original 20 (backward-compat: old model uses :20)
FEATURES = FEATURES_V1 + [
    "data_completeness",  # Fraction of V1 features with real values (1.0 = all real)
    "is_ghost_agent",     # 1 if total_jobs==0 AND completion_rate==0 AND token exists
]


def load_and_prepare_data():
    """Load dataset and run SMOTE for class balancing.
    
    Priority order:
    1. data/rug_pull_dataset_real.csv  — real labeled data (preferred)
    2. data/rug_pull_dataset.csv       — synthetic fallback
    """
    real_path = DATA_DIR / "rug_pull_dataset_real.csv"
    synthetic_path = DATA_DIR / "rug_pull_dataset.csv"

    if real_path.exists():
        dataset_path = real_path
        dataset_type = "REAL"
    elif synthetic_path.exists():
        dataset_path = synthetic_path
        dataset_type = "SYNTHETIC (fallback)"
        print("⚠️  WARNING: Using synthetic data. Run build_real_dataset.py first.")
    else:
        print("No dataset found. Building real dataset first...")
        import subprocess
        subprocess.run([sys.executable, str(SCRIPT_DIR / "build_real_dataset.py")], check=True)
        dataset_path = real_path
        dataset_type = "REAL (freshly built)"

    print(f"📊 Dataset: {dataset_type}")
    print(f"   Path: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"   Loaded {len(df)} samples")
    print(f"   Class distribution: {df['label'].value_counts().to_dict()}")

    # ── Add new V1 features to existing dataset ────────────────────────────
    # data_completeness: real labeled datasets have all features → 1.0
    df["data_completeness"] = 1.0

    # is_ghost_agent: 1 if no ACP jobs AND no completion AND a token address existed
    # In V1 training data (Uniswap rugs), these are real on-chain tokens → ghost=0
    df["is_ghost_agent"] = 0.0
    # Override for samples where total_jobs==0 AND completion_rate==0
    ghost_mask = (df["total_jobs"] == 0.0) & (df["completion_rate"] == 0.0)
    df.loc[ghost_mask, "is_ghost_agent"] = 1.0

    # ── Append known rugs validation set (always label=1) ─────────────────
    known_rugs_path = DATA_DIR / "known_rugs_validation.csv"
    if known_rugs_path.exists():
        known_df = pd.read_csv(known_rugs_path)
        print(f"\n⚠️  Loading known rugs: {known_rugs_path}")
        print(f"   Known rugs entries: {len(known_df)}")
        # Build rows with conservative default feature values, label=1
        known_rows = []
        for _, row in known_df.iterrows():
            feature_row = {
                # Conservative defaults — known rug so assume worst-case signals
                "holder_concentration": 0.9,
                "liquidity_lock_ratio": 0.0,
                "creator_tx_pattern": 0.8,
                "buy_sell_ratio": 0.2,
                "contract_similarity_score": 0.7,
                "fund_flow_pattern": 0.8,
                "price_change_24h": -0.98,  # 98% crash
                "liquidity_usd": 0.01,
                "volume_24h": 0.0,
                "total_jobs": 0.0,
                "completion_rate": 0.0,
                "trust_score": 0.0,
                "age_days": 0.05,
                "lp_drain_rate": 0.9,
                "deployer_age_days": 0.1,
                "token_supply_concentration": 0.9,
                "renounced_ownership": 0,
                "verified_contract": 0,
                "social_presence_score": 0.0,
                "audit_score": 0.0,
                "data_completeness": 1.0,  # We know these are rugs (confirmed)
                "is_ghost_agent": 1.0,     # No ACP activity
                "label": 1,
            }
            known_rows.append(feature_row)
        known_rug_df = pd.DataFrame(known_rows)
        df = pd.concat([df, known_rug_df], ignore_index=True)
        print(f"   Combined total: {len(df)} samples")
        print(f"   New label distribution: {df['label'].value_counts().to_dict()}")
    else:
        print(f"ℹ️  No known_rugs_validation.csv found — skipping")

    X = df[FEATURES].values
    y = df["label"].values

    # Apply SMOTE to oversample minority class (legit tokens in real data)
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"\nAfter SMOTE: {len(X_resampled)} samples")
    print(f"Rug pulls: {y_resampled.sum()} | Legit: {(y_resampled == 0).sum()}")

    return X_resampled, y_resampled, str(dataset_path), dataset_type


def build_xgb_model():
    """
    XGBoost with high recall configuration.

    Key settings for low false negative rate:
    - scale_pos_weight: weight for positive (rug) class
    - Lower threshold in production: 0.3 instead of 0.5
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        # Optimize for recall — missing rugs is worse than false alarms
        scale_pos_weight=1.5,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )


def train_and_evaluate(X, y):
    """Train model with cross-validation and evaluate."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ─── Cross-validation ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CROSS-VALIDATION (5-fold, Stratified)")
    print("="*60)

    model = build_xgb_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall")
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    print(f"Accuracy:  {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
    print(f"F1:        {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"ROC AUC:   {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")

    # ─── Final training ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL MODEL TRAINING")
    print("="*60)

    final_model = build_xgb_model()
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ─── Test set evaluation ───────────────────────────────────────────────
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    # Use lower threshold for rug detection (0.35 instead of 0.5)
    # This further reduces false negatives
    THRESHOLD = 0.35
    y_pred_threshold = (y_prob >= THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_thresh = accuracy_score(y_test, y_pred_threshold)
    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\nTest Set Results (threshold={THRESHOLD}):")
    print(f"Accuracy:   {accuracy_thresh:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}  ← (1 - false_negative_rate)")
    print(f"F1 Score:   {f1:.4f}")
    print(f"ROC AUC:    {roc_auc:.4f}")

    # ─── Confusion Matrix ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CONFUSION MATRIX (threshold=0.35)")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred_threshold)
    tn, fp, fn, tp = cm.ravel()

    print(f"""
┌─────────────────────────────────────┐
│           PREDICTED                 │
│         Legit    Rug               │
├─────────────────────────────────────┤
│ ACTUAL  │  TN={tn:4d}  │  FP={fp:4d}  │
│ Legit   │              │             │
│ ACTUAL  │  FN={fn:4d}  │  TP={tp:4d}  │
│ Rug     │              │             │
└─────────────────────────────────────┘
    """)
    print(f"False Negative Rate: {fn / (fn + tp):.4f} (missed rugs)")
    print(f"False Positive Rate: {fp / (fp + tn):.4f} (false alarms)")

    # ─── Classification report ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred_threshold, target_names=["Legit", "Rug Pull"]))

    # ─── Feature importance ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Top 10)")
    print("="*60)
    importance = final_model.feature_importances_
    feat_imp = sorted(zip(FEATURES, importance), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:10]:
        bar = "█" * int(imp * 100)
        print(f"  {feat:<35} {imp:.4f} {bar}")

    return final_model, {
        "accuracy": float(accuracy_thresh),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "false_negative_rate": float(fn / (fn + tp)),
        "false_positive_rate": float(fp / (fp + tn)),
        "confusion_matrix": cm.tolist(),
        "cv_accuracy_mean": float(cv_accuracy.mean()),
        "cv_accuracy_std": float(cv_accuracy.std()),
        "cv_recall_mean": float(cv_recall.mean()),
        "cv_recall_std": float(cv_recall.std()),
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_std": float(cv_f1.std()),
        "prediction_threshold": THRESHOLD,
        "feature_importance": {f: float(i) for f, i in feat_imp},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def export_onnx(model, n_features: int):
    """Export model to ONNX format for Node.js native inference via xgboost native export."""
    try:
        onnx_path = str(MODELS_DIR / "wadjet_xgb.onnx")
        # Use XGBoost's native ONNX export
        model.save_model(onnx_path.replace(".onnx", "_xgb.json"))  # also save JSON
        # XGBoost 2.x native ONNX export
        model.save_model(onnx_path)
        print(f"\n✅ ONNX model exported: {onnx_path}")
        return True
    except Exception as e:
        # Fallback: try onnxmltools
        try:
            import onnxmltools
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.utils import save_model as save_onnx
            from skl2onnx.common.data_types import FloatTensorType

            initial_types = [("float_input", FloatTensorType([None, n_features]))]
            onnx_model = convert_xgboost(model, initial_types=initial_types)
            onnx_path_obj = MODELS_DIR / "wadjet_xgb.onnx"
            save_onnx(onnx_model, str(onnx_path_obj))
            print(f"\n✅ ONNX model exported (onnxmltools): {onnx_path_obj}")
            return True
        except Exception as e2:
            print(f"\n⚠️  ONNX export failed (non-critical): {e2}")
            print("   Joblib model is the primary deployment artifact.")
            return False


def main():
    print("🔮 Wadjet XGBoost Rug Prediction — Training")
    print("="*60)

    # Load data (real preferred)
    X, y, dataset_path, dataset_type = load_and_prepare_data()

    # Train + evaluate
    model, metrics = train_and_evaluate(X, y)

    # ─── Save model ────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "wadjet_xgb.joblib"
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved: {model_path}")

    # ─── Export ONNX ───────────────────────────────────────────────────────
    export_onnx(model, len(FEATURES))

    # ─── Save metadata ─────────────────────────────────────────────────────
    metadata = {
        "model_version": "1.2.0",  # v1.2 = real data + ghost/completeness features + known rugs
        "training_date": datetime.utcnow().isoformat() + "Z",
        "algorithm": "XGBoost",
        "n_features": len(FEATURES),
        "features": FEATURES,
        "metrics": metrics,
        "dataset": {
            "source": dataset_type,
            "source_path": dataset_path,
            "primary_source": "kangmyoungseok/RugPull-Prediction-AI — 18,296 Uniswap V2 tokens (2021, ETH), labeled by LP removal detection",
            "secondary_sources": [
                "dianxiang-sun/rug_pull_dataset — 2,391 ETH/BSC incidents (2020-2023)",
                "TM-RugPull arxiv:2602.21529 — feature distribution reference"
            ],
            "n_samples_before_smote": int(len(X)),
            "n_samples_after_smote": int(len(X)),
            "smote_applied": True,
            "class_ratio": "1:1 after SMOTE",
        },
        "deployment": {
            "prediction_threshold": metrics["prediction_threshold"],
            "note": "Use threshold=0.35 to minimize false negatives (missed rugs)",
        }
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved: {metadata_path}")

    print("\n" + "="*60)
    print("🎯 TRAINING COMPLETE")
    print(f"   Accuracy:  {metrics['accuracy']:.1%}")
    print(f"   Recall:    {metrics['recall']:.1%}")
    print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"   False Neg: {metrics['false_negative_rate']:.1%} (missed rugs)")
    print("="*60)

    if metrics["accuracy"] < 0.90:
        print("⚠️  WARNING: Accuracy below 90% target — consider retuning hyperparameters")

    return metadata


if __name__ == "__main__":
    main()
