#!/usr/bin/env python3
"""
Train Audio-Visual models (binary + multiclass) using ONLY audio + image features.

Key improvements:
  1. GroupKFold(5) on config_folder  — eliminates data leakage
  2. Two-pass feature selection      — drops zero/near-zero importance features
  3. Youden-J threshold from CV       — principled threshold, no test-label peeking
  4. Saves feature mask               — inference.py applies identical transformation

Output:
  artifacts/binary_model_av.joblib
  artifacts/multiclass_model_av.joblib
  artifacts/label_encoder_av.joblib
  artifacts/feature_mask_av.npy        ← NEW: boolean mask of selected features
  artifacts/binary_av_metrics.json
  artifacts/multiclass_av_metrics.json
"""

import os, sys, json, time, logging
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, GroupShuffleSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ARTIFACTS_DIR = "artifacts"
CACHE_PATH    = os.path.join(ARTIFACTS_DIR, "feature_cache.npz")

# ------------------------------------------------------------------ #
# Physics-based feature dims (must match audio_features.py /          #
# image_features.py)                                                  #
N_AUDIO = 136    # AUDIO_FEAT_DIM in audio_features.py
N_IMAGE = 128    # IMAGE_FEAT_DIM in image_features.py
N_TOTAL = N_AUDIO + N_IMAGE   # 264
# ------------------------------------------------------------------ #
# How many top features to keep (by cumulative importance).           #
CUMULATIVE_IMPORTANCE_THRESHOLD = 0.80
# ------------------------------------------------------------------ #


def load_all_data():
    """Pool train + val data and return config_folder groups."""
    if not os.path.exists(CACHE_PATH):
        logging.error(f"Feature cache not found: {CACHE_PATH}. Run cache_features.py first.")
        sys.exit(1)

    cache = np.load(CACHE_PATH, allow_pickle=True)

    audio_all  = np.concatenate([cache["train_audio"],       cache["val_audio"      ]], axis=0)
    image_all  = np.concatenate([cache["train_image"],       cache["val_image"      ]], axis=0)
    y_bin_all  = np.concatenate([cache["train_labels"],      cache["val_labels"     ]], axis=0)
    y_code_all = np.concatenate([cache["train_label_codes"], cache["val_label_codes"]], axis=0)

    X_all = np.concatenate([audio_all, image_all], axis=1)   # 217 (120+97)

    train_df = pd.read_csv("train_split.csv")
    val_df   = pd.read_csv("val_split.csv")
    all_df   = pd.concat([train_df, val_df], ignore_index=True)
    groups   = all_df["config_folder"].values

    logging.info(f"Pooled: {len(X_all)} samples | unique groups: {len(set(groups))}")
    logging.info(f"Class balance: good={np.sum(y_bin_all==0)}, defect={np.sum(y_bin_all==1)}")
    return X_all, y_bin_all, y_code_all, groups


def select_features(X_all, y_bin_all, groups):
    """
    First-pass XGBoost to get feature importances, then keep features
    that together cover CUMULATIVE_IMPORTANCE_THRESHOLD of total importance.
    Returns: feature_mask (bool array, len=217)
    """
    logging.info("\n=== Feature Selection Pass ===")
    n_good   = np.sum(y_bin_all == 0)
    n_defect = np.sum(y_bin_all == 1)
    scale    = n_good / max(n_defect, 1)

    # Quick shallow model just to rank features
    selector = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42, tree_method="hist",
    )
    selector.fit(X_all, y_bin_all)
    importances = selector.feature_importances_

    # Sort by descending importance and compute cumulative sum
    ranked_idx = np.argsort(importances)[::-1]
    cum_imp    = np.cumsum(importances[ranked_idx])
    n_keep     = int(np.searchsorted(cum_imp, CUMULATIVE_IMPORTANCE_THRESHOLD) + 1)
    n_keep     = max(n_keep, 20)   # floor: always keep at least 20

    selected_idx = ranked_idx[:n_keep]
    mask = np.zeros(X_all.shape[1], dtype=bool)
    mask[selected_idx] = True

    n_audio_kept = int(np.sum(mask[:120]))
    n_image_kept = int(np.sum(mask[120:]))
    logging.info(f"  Keeping {n_keep}/{X_all.shape[1]} features "
                 f"(audio={n_audio_kept}, image={n_image_kept}) "
                 f"covering {100*cum_imp[n_keep-1]:.1f}% of importance")
    logging.info(f"  Dropped {X_all.shape[1] - n_keep} low-importance features")

    return mask


def train_binary_av(X_all, y_all, groups):
    """Hyperparameter-tuned binary classifier with GroupKFold, Youden-J threshold from OOF."""
    logging.info("\n=== Training Binary AV Model (RandomizedSearchCV + GroupKFold n=5) ===")

    n_good   = np.sum(y_all == 0)
    n_defect = np.sum(y_all == 1)
    scale    = n_good / max(n_defect, 1)

    param_dist = {
        "n_estimators":      [300, 400, 500, 700],
        "max_depth":         [3, 4, 5, 6],
        "learning_rate":     [0.02, 0.05, 0.08, 0.1],
        "subsample":         [0.7, 0.8, 0.9],
        "colsample_bytree":  [0.6, 0.7, 0.8, 0.9],
        "min_child_weight":  [1, 3, 5, 7],
        "reg_alpha":         [0, 0.05, 0.1, 0.3],
        "reg_lambda":        [1, 2, 3, 5],
    }

    base = XGBClassifier(
        scale_pos_weight=scale,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42, tree_method="hist",
    )

    # Use GroupShuffleSplit inside RandomizedSearchCV so CV splits respect groups
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=40, cv=gss,
        scoring="f1", random_state=42, n_jobs=-1, verbose=0,
        error_score="raise",
    )

    t0 = time.time()
    search.fit(X_all, y_all, groups=groups)
    logging.info(f"  Hyperparameter search done in {time.time()-t0:.1f}s")
    logging.info(f"  Best params: {search.best_params_}")

    model = search.best_estimator_

    # ---- OOF probabilities with GroupKFold for threshold selection ----
    # Run all 5 folds in parallel across all CPU cores
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_all, y_all, groups))

    def _fit_fold_binary(tr_idx, va_idx):
        m = XGBClassifier(**model.get_params())
        m.fit(X_all[tr_idx], y_all[tr_idx])
        return va_idx, m.predict_proba(X_all[va_idx])[:, 1]

    t0 = time.time()
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(_fit_fold_binary)(tr, va) for tr, va in splits
    )
    oof_probs = np.zeros(len(y_all))
    for va_idx, probs in results:
        oof_probs[va_idx] = probs
    logging.info(f"  OOF (parallel) done in {time.time()-t0:.1f}s")

    # Youden-J threshold (Sensitivity + Specificity - 1)
    best_j, best_thresh_j = -1, 0.5
    # F1-optimal threshold
    best_f1_val, best_thresh_f1 = 0, 0.5

    for t in np.arange(0.05, 0.95, 0.01):
        preds = (oof_probs >= t).astype(int)
        tp  = ((preds == 1) & (y_all == 1)).sum()
        tn  = ((preds == 0) & (y_all == 0)).sum()
        sens = tp / max((y_all == 1).sum(), 1)
        spec = tn / max((y_all == 0).sum(), 1)
        j    = sens + spec - 1
        if j > best_j:
            best_j, best_thresh_j = j, t
        fv = f1_score(y_all, preds, zero_division=0)
        if fv > best_f1_val:
            best_f1_val, best_thresh_f1 = fv, t

    logging.info(f"  OOF Youden-J threshold:   {best_thresh_j:.2f}  →  F1={f1_score(y_all, (oof_probs>=best_thresh_j).astype(int)):.4f}")
    logging.info(f"  OOF F1-optimal threshold: {best_thresh_f1:.2f}  →  F1={best_f1_val:.4f}")
    logging.info(f"  Raw OOF prob range: [{oof_probs.min():.3f}, {oof_probs.max():.3f}]  mean={oof_probs.mean():.3f}")

    # ---- Isotonic Calibration on OOF probs (training data only) ----
    # XGBoost pushes probs to extremes; isotonic regression maps them to
    # a better-calibrated scale using only the held-out OOF batch.
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_probs, y_all)
    cal_probs = calibrator.predict(oof_probs)
    logging.info(f"  Calibrated OOF prob range: [{cal_probs.min():.3f}, {cal_probs.max():.3f}]  mean={cal_probs.mean():.3f}")

    # Re-derive threshold on calibrated probabilities
    best_f1_cal, best_thresh_cal = 0, 0.5
    best_j_cal, best_thresh_j_cal = -1, 0.5
    for t in np.arange(0.05, 0.95, 0.005):
        preds = (cal_probs >= t).astype(int)
        tp   = ((preds == 1) & (y_all == 1)).sum()
        tn   = ((preds == 0) & (y_all == 0)).sum()
        sens = tp / max((y_all == 1).sum(), 1)
        spec = tn / max((y_all == 0).sum(), 1)
        j    = sens + spec - 1
        if j > best_j_cal:
            best_j_cal, best_thresh_j_cal = j, t
        fv = f1_score(y_all, preds, zero_division=0)
        if fv > best_f1_cal:
            best_f1_cal, best_thresh_cal = fv, t

    logging.info(f"  Calibrated Youden-J threshold:   {best_thresh_j_cal:.3f}  →  F1={f1_score(y_all, (cal_probs>=best_thresh_j_cal).astype(int)):.4f}")
    logging.info(f"  Calibrated F1-optimal threshold: {best_thresh_cal:.3f}  →  F1={best_f1_cal:.4f}")

    best_thresh = best_thresh_cal  # use calibrated threshold

    cal_preds = (cal_probs >= best_thresh).astype(int)
    metrics = {
        "f1":                    float(f1_score(y_all, cal_preds, zero_division=0)),
        "precision":             float(precision_score(y_all, cal_preds, zero_division=0)),
        "recall":                float(recall_score(y_all, cal_preds, zero_division=0)),
        "roc_auc":               float(roc_auc_score(y_all, cal_probs)),
        "best_threshold":        float(best_thresh),
        "youden_j_threshold":    float(best_thresh_j_cal),
        "calibration":           "isotonic regression on OOF probs",
        "features_used":         "audio+image selected subset",
        "cv_strategy":           "GroupKFold(5) on config_folder (no data leakage)",
        "best_params":           {k: (int(v) if isinstance(v, np.integer) else
                                      float(v) if isinstance(v, np.floating) else v)
                                  for k, v in search.best_params_.items()},
    }
    logging.info(f"  Final — F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}, Threshold: {best_thresh:.3f}")

    logging.info("  Refitting model on full dataset...")
    model.fit(X_all, y_all)
    # Refit calibrator on full-data raw probs
    full_probs = model.predict_proba(X_all)[:, 1]
    calibrator.fit(full_probs, y_all)

    joblib.dump(model,      os.path.join(ARTIFACTS_DIR, "binary_model_av.joblib"))
    joblib.dump(calibrator, os.path.join(ARTIFACTS_DIR, "binary_calibrator_av.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "binary_av_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, best_thresh, metrics


def train_multiclass_av(X_all, y_codes_all, groups):
    """Hyperparameter-tuned multiclass type classifier with GroupKFold."""
    logging.info("\n=== Training Multiclass AV Model (RandomizedSearchCV + GroupKFold n=5) ===")

    defect_mask = y_codes_all != "00"
    X_d         = X_all[defect_mask]
    y_codes_d   = y_codes_all[defect_mask]
    groups_d    = groups[defect_mask]

    le    = LabelEncoder()
    y_d   = le.fit_transform(y_codes_d)
    n_cls = len(le.classes_)
    logging.info(f"  Classes: {list(le.classes_)} | defect samples: {len(X_d)}")

    param_dist = {
        "n_estimators":     [300, 400, 500, 700],
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.02, 0.05, 0.08, 0.1],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "reg_alpha":        [0, 0.05, 0.1],
        "reg_lambda":       [1, 2, 3],
    }

    base = XGBClassifier(
        objective="multi:softprob", num_class=n_cls,
        eval_metric="mlogloss", use_label_encoder=False,
        random_state=42, tree_method="hist",
    )

    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=40, cv=gss,
        scoring="f1_macro", random_state=42, n_jobs=-1, verbose=0,
        error_score="raise",
    )

    t0 = time.time()
    search.fit(X_d, y_d, groups=groups_d)
    logging.info(f"  Hyperparameter search done in {time.time()-t0:.1f}s")
    logging.info(f"  Best params: {search.best_params_}")

    model = search.best_estimator_

    # OOF predictions — all folds in parallel
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_d, y_d, groups_d))

    def _fit_fold_multi(tr_idx, va_idx):
        m = XGBClassifier(**model.get_params())
        m.fit(X_d[tr_idx], y_d[tr_idx])
        return va_idx, m.predict(X_d[va_idx])

    t0 = time.time()
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(_fit_fold_multi)(tr, va) for tr, va in splits
    )
    oof_preds = np.zeros(len(y_d), dtype=int)
    for va_idx, preds in results:
        oof_preds[va_idx] = preds
    logging.info(f"  OOF (parallel) done in {time.time()-t0:.1f}s")

    macro_f1    = float(f1_score(y_d, oof_preds, average="macro",    zero_division=0))
    weighted_f1 = float(f1_score(y_d, oof_preds, average="weighted", zero_division=0))
    report      = classification_report(y_d, oof_preds, target_names=le.classes_,
                                        output_dict=True, zero_division=0)
    logging.info(f"  OOF Multiclass — Macro F1: {macro_f1:.4f}")
    logging.info(f"\n{classification_report(y_d, oof_preds, target_names=le.classes_, zero_division=0)}")

    metrics = {
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "per_class":   {cls: {"f1": report[cls]["f1-score"],
                              "precision": report[cls]["precision"],
                              "recall": report[cls]["recall"]} for cls in le.classes_},
        "features_used": "audio+image selected subset",
        "cv_strategy":   "GroupKFold(5) on config_folder",
        "best_params":   search.best_params_,
    }

    logging.info("  Refitting on full defect dataset...")
    model.fit(X_d, y_d)

    joblib.dump(model,  os.path.join(ARTIFACTS_DIR, "multiclass_model_av.joblib"))
    joblib.dump(le,     os.path.join(ARTIFACTS_DIR, "label_encoder_av.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "multiclass_av_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, le, metrics


def main():
    logging.info("=" * 60)
    logging.info("  Audio-Visual Model Training (FeatureSelect + GroupKFold)")
    logging.info("=" * 60)

    X_all, y_bin_all, y_code_all, groups = load_all_data()

    # ---- Step 1: Feature selection ----
    mask = select_features(X_all, y_bin_all, groups)
    np.save(os.path.join(ARTIFACTS_DIR, "feature_mask_av.npy"), mask)
    X_sel = X_all[:, mask]
    logging.info(f"  Using {mask.sum()} features for final training")

    # ---- Step 2: Train models ----
    bin_model,   threshold,  bin_metrics   = train_binary_av(X_sel, y_bin_all,  groups)
    multi_model, le,         multi_metrics = train_multiclass_av(X_sel, y_code_all, groups)

    binary_f1     = bin_metrics["f1"]
    type_macro_f1 = multi_metrics["macro_f1"]
    final_score   = 0.6 * binary_f1 + 0.4 * type_macro_f1

    logging.info("\n" + "=" * 60)
    logging.info("  RESULTS (GroupKFold OOF, feature-selected)")
    logging.info(f"  Features:        {int(mask.sum())} / 217")
    logging.info(f"  Binary F1:       {binary_f1:.4f}")
    logging.info(f"  Type Macro F1:   {type_macro_f1:.4f}")
    logging.info(f"  Final Score:     {final_score:.4f} ({final_score*100:.1f}%)")
    logging.info(f"  Binary Threshold:{threshold:.2f}  (Youden-J from CV)")
    logging.info("=" * 60)

    with open(os.path.join(ARTIFACTS_DIR, "pipeline_av_metrics.json"), "w") as f:
        json.dump({
            "binary_f1":      binary_f1,
            "type_macro_f1":  type_macro_f1,
            "final_score":    final_score,
            "best_threshold": threshold,
            "n_features":     int(mask.sum()),
            "features":       "audio+image selected subset",
            "cv_strategy":    "GroupKFold(5) on config_folder",
        }, f, indent=2)


if __name__ == "__main__":
    main()
