#!/usr/bin/env python3
"""
Train Audio-Visual models (binary + multiclass) using ONLY audio + image features.
Uses GroupKFold on config_folder to prevent data leakage across welding configurations.
These models handle inference when sensor CSV data is unavailable.

Output:
  artifacts/binary_model_av.joblib
  artifacts/multiclass_model_av.joblib
  artifacts/label_encoder_av.joblib
  artifacts/binary_av_metrics.json
  artifacts/multiclass_av_metrics.json
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ARTIFACTS_DIR = "artifacts"
CACHE_PATH = os.path.join(ARTIFACTS_DIR, "feature_cache.npz")


def load_all_data():
    """
    Load ALL training + validation data pooled together.
    Also returns config_folder groups for GroupKFold.
    """
    if not os.path.exists(CACHE_PATH):
        logging.error(f"Feature cache not found: {CACHE_PATH}")
        logging.error("Run: python cache_features.py first")
        sys.exit(1)

    cache = np.load(CACHE_PATH, allow_pickle=True)

    # Pool train + val features
    audio_all = np.concatenate([cache["train_audio"], cache["val_audio"]], axis=0)
    image_all = np.concatenate([cache["train_image"], cache["val_image"]], axis=0)
    y_binary_all = np.concatenate([cache["train_labels"], cache["val_labels"]], axis=0)
    y_codes_all  = np.concatenate([cache["train_label_codes"], cache["val_label_codes"]], axis=0)

    # AV features: audio(120) + image(97) = 217
    X_all = np.concatenate([audio_all, image_all], axis=1)

    # Load split CSVs to get config_folder groups
    train_df = pd.read_csv("train_split.csv")
    val_df   = pd.read_csv("val_split.csv")
    all_df   = pd.concat([train_df, val_df], ignore_index=True)
    groups   = all_df["config_folder"].values  # used for GroupKFold

    n_train = len(cache["train_labels"])
    n_val   = len(cache["val_labels"])
    logging.info(f"Pooled data: {n_train} train + {n_val} val = {len(X_all)} total samples")
    logging.info(f"Unique config groups: {len(set(groups))}")
    logging.info(f"Class balance: good={np.sum(y_binary_all==0)}, defect={np.sum(y_binary_all==1)}")

    return X_all, y_binary_all, y_codes_all, groups


def train_binary_av(X_all, y_all, groups):
    """Train binary defect detector using GroupKFold to prevent data leakage."""
    logging.info("\n=== Training Binary AV Model (GroupKFold, n=5) ===")

    n_good   = np.sum(y_all == 0)
    n_defect = np.sum(y_all == 1)
    scale    = n_good / max(n_defect, 1)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=2,
        scale_pos_weight=scale,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        tree_method="hist",
    )

    # Cross-validated OOF probabilities with group-aware splits
    gkf = GroupKFold(n_splits=5)
    oof_probs = np.zeros(len(y_all))

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y_all, groups)):
        model.fit(X_all[tr_idx], y_all[tr_idx])
        oof_probs[va_idx] = model.predict_proba(X_all[va_idx])[:, 1]
        logging.info(f"  Fold {fold+1}/5 done")

    elapsed = time.time() - t0
    logging.info(f"  OOF cross-validation done in {elapsed:.1f}s")

    # Find best threshold on OOF predictions
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (oof_probs >= t).astype(int)
        f = f1_score(y_all, preds)
        if f > best_f1:
            best_f1 = f
            best_thresh = t

    oof_preds = (oof_probs >= best_thresh).astype(int)

    metrics = {
        "f1":             float(f1_score(y_all, oof_preds)),
        "precision":      float(precision_score(y_all, oof_preds, zero_division=0)),
        "recall":         float(recall_score(y_all, oof_preds, zero_division=0)),
        "roc_auc":        float(roc_auc_score(y_all, oof_probs)),
        "best_threshold": float(best_thresh),
        "features_used":  "audio(120) + image(97) = 217",
        "cv_strategy":    "GroupKFold(5) on config_folder (no data leakage)",
    }

    logging.info(f"  OOF Binary — F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}, Threshold: {best_thresh:.3f}")

    # Refit on ALL data for deployment
    logging.info("  Refitting on full dataset for deployment...")
    model.fit(X_all, y_all)

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "binary_model_av.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "binary_av_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, best_thresh, metrics


def train_multiclass_av(X_all, y_codes_all, groups):
    """Train multiclass defect type classifier using GroupKFold."""
    logging.info("\n=== Training Multiclass AV Model (GroupKFold, n=5) ===")

    # Only defective samples
    defect_mask = y_codes_all != "00"
    X_d      = X_all[defect_mask]
    y_codes_d = y_codes_all[defect_mask]
    groups_d  = groups[defect_mask]

    le = LabelEncoder()
    y_d = le.fit_transform(y_codes_d)

    n_classes = len(le.classes_)
    logging.info(f"  Classes: {list(le.classes_)} ({n_classes} types)")
    logging.info(f"  Total defect samples: {len(X_d)}")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        tree_method="hist",
    )

    # OOF predictions
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y_d), dtype=int)

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_d, y_d, groups_d)):
        model.fit(X_d[tr_idx], y_d[tr_idx])
        oof_preds[va_idx] = model.predict(X_d[va_idx])
        logging.info(f"  Fold {fold+1}/5 done")

    elapsed = time.time() - t0
    logging.info(f"  OOF cross-validation done in {elapsed:.1f}s")

    macro_f1    = float(f1_score(y_d, oof_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_d, oof_preds, average="weighted", zero_division=0))
    report      = classification_report(y_d, oof_preds, target_names=le.classes_, output_dict=True, zero_division=0)

    logging.info(f"  OOF Multiclass — Macro F1: {macro_f1:.4f}")
    logging.info(f"\n{classification_report(y_d, oof_preds, target_names=le.classes_, zero_division=0)}")

    metrics = {
        "macro_f1":   macro_f1,
        "weighted_f1": weighted_f1,
        "per_class":  {cls: {"f1": report[cls]["f1-score"], "precision": report[cls]["precision"],
                             "recall": report[cls]["recall"]}
                       for cls in le.classes_},
        "features_used": "audio(120) + image(97) = 217",
        "cv_strategy":   "GroupKFold(5) on config_folder (no data leakage)",
    }

    # Refit on all defect data
    logging.info("  Refitting on full defect dataset for deployment...")
    model.fit(X_d, y_d)

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "multiclass_model_av.joblib"))
    joblib.dump(le,    os.path.join(ARTIFACTS_DIR, "label_encoder_av.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "multiclass_av_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, le, metrics


def main():
    logging.info("=" * 60)
    logging.info("  Audio-Visual Model Training (GroupKFold — No Leakage)")
    logging.info("=" * 60)

    X_all, y_binary_all, y_codes_all, groups = load_all_data()

    bin_model, threshold, bin_metrics = train_binary_av(X_all, y_binary_all, groups)
    multi_model, le, multi_metrics    = train_multiclass_av(X_all, y_codes_all, groups)

    binary_f1    = bin_metrics["f1"]
    type_macro_f1 = multi_metrics["macro_f1"]
    final_score  = 0.6 * binary_f1 + 0.4 * type_macro_f1

    logging.info("\n" + "=" * 60)
    logging.info("  AUDIO-VISUAL MODEL RESULTS (GroupKFold OOF)")
    logging.info(f"  Binary F1:       {binary_f1:.4f}")
    logging.info(f"  Type Macro F1:   {type_macro_f1:.4f}")
    logging.info(f"  Final Score:     {final_score:.4f} ({final_score*100:.1f}%)")
    logging.info("  (No data leakage — GroupKFold by config_folder)")
    logging.info("=" * 60)

    with open(os.path.join(ARTIFACTS_DIR, "pipeline_av_metrics.json"), "w") as f:
        json.dump({
            "binary_f1":      binary_f1,
            "type_macro_f1":  type_macro_f1,
            "final_score":    final_score,
            "best_threshold": threshold,
            "features":       "audio(120) + image(97) = 217 dims",
            "cv_strategy":    "GroupKFold(5) on config_folder",
        }, f, indent=2)


if __name__ == "__main__":
    main()
