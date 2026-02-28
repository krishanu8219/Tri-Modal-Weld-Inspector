import os
import json
import logging
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
# CalibratedClassifierCV removed — XGBoost native softmax gives proper probabilities
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    average_precision_score, brier_score_loss, confusion_matrix
)
import joblib
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.audio_features import extract_audio_features
from src.image_features import extract_image_features

# Sensor columns commonly used for defects
SENSOR_COLS = [
    "Pressure", "CO2 Weld Flow", "Feed", 
    "Primary Weld Current", "Secondary Weld Voltage", "Wire Consumed"
]

def extract_sensor_features(csv_path):
    """
    Reads a run's sensor CSV and extracts basic summary statistics
    for key numerical channels. Returns a 1D numPy array.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Error reading {csv_path}: {e}")
        # Return zeros if file unreadable — 17 stats per column
        return np.zeros(len(SENSOR_COLS) * 17)
        
    features = []
    
    for col in SENSOR_COLS:
        if col in df.columns and not df[col].empty:
            s = df[col]
            diffs = s.diff().dropna()
            features.extend([
                s.mean(),
                s.std(),
                s.min(),
                s.max(),
                s.median(),
                s.skew(),
                s.quantile(0.10) if not s.empty else 0.0,
                s.quantile(0.25) if not s.empty else 0.0,
                s.quantile(0.75) if not s.empty else 0.0,
                s.quantile(0.90) if not s.empty else 0.0,
                s.quantile(0.75) - s.quantile(0.25) if len(s) > 0 else 0.0, # IQR
                s.max() - s.min() if len(s) > 0 else 0.0, # Range
                diffs.mean() if len(diffs) > 0 else 0.0,
                diffs.std() if len(diffs) > 0 else 0.0,
                diffs.max() if len(diffs) > 0 else 0.0,
                # End of run features (last 50 rows)
                s.tail(50).mean() if len(s) > 0 else 0.0,
                s.tail(50).std() if len(s) > 0 else 0.0
            ])
        else:
            # Missing channel fallback
            features.extend([0.0] * 17)
            
    # filling NaNs with 0 (e.g. from skewness on flat lines or std on 1 row)
    features = np.nan_to_num(features, nan=0.0)
    return features

def prepare_data(split_df):
    """
    Maps rows to feature matrix X and label vector y.
    y -> 0 if label_code is '00' else 1
    Includes Late Fusion (Sensor features + Audio MFCCs).
    """
    X = []
    y = []
    metadata = []
    
    for _, row in split_df.iterrows():
        csv_path = row["csv_path"]
        flac_path = row["flac_path"]
        label = row["label_code"]
        
        # Binary target: 0 for good (00), 1 for defect
        binary_y = 0 if str(label).zfill(2) == "00" else 1
        
        # For image extraction, we need the run configuration directory path.
        # It's usually the parent of csv_path or flac_path
        run_dir = os.path.dirname(csv_path)
        
        sensor_f = extract_sensor_features(csv_path)
        audio_f = extract_audio_features(flac_path)
        image_f = extract_image_features(run_dir)
        
        # Late Fusion logic (concatenate arrays)
        combined_f = np.concatenate([sensor_f, audio_f, image_f])
        
        X.append(combined_f)
        y.append(binary_y)
        metadata.append(row["run_id"])
        
        if len(X) % 100 == 0:
            logging.info(f"  prepare_data: {len(X)}/{len(split_df)} runs processed")
        
    return np.array(X), np.array(y), metadata

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece[0]

def train_and_evaluate(train_csv="train_split.csv", val_csv="val_split.csv", output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load splits
    logging.info("Loading train/val split CSVs.")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    
    logging.info("Extracting features from sensor data...")
    X_train, y_train, train_ids = prepare_data(df_train)
    X_val, y_val, val_ids = prepare_data(df_val)
    
    if len(np.unique(y_train)) == 1:
        logging.warning("Training split has only ONE class. Adding a dummy defect row for demonstration.")
        X_train = np.vstack([X_train, X_train[0]])
        y_train = np.append(y_train, 1 - y_train[0])
        
    if len(np.unique(y_val)) == 1:
        logging.warning("Validation split has only ONE class.")
    
    # 2a. Compute dynamic scale_pos_weight
    n_good = np.sum(y_train == 0)
    n_defect = np.sum(y_train == 1)
    spw = n_good / max(n_defect, 1)
    logging.info(f"Class distribution: good={n_good}, defect={n_defect}, scale_pos_weight={spw:.3f}")
    
    # 2b. Apply SMOTE oversampling to balance the training set
    if len(np.unique(y_train)) > 1 and min(n_good, n_defect) >= 6:
        logging.info("Applying SMOTE oversampling...")
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(n_good, n_defect) - 1))
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logging.info(f"After SMOTE: good={np.sum(y_train==0)}, defect={np.sum(y_train==1)}")
    
    # 2c. Hyperparameter tuning via RandomizedSearchCV
    logging.info("Running RandomizedSearchCV for binary XGBoost hyperparameter tuning...")
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0],
    }
    
    base_clf = XGBClassifier(random_state=42, scale_pos_weight=spw, verbosity=0)
    
    from sklearn.model_selection import RandomizedSearchCV
    search = RandomizedSearchCV(
        base_clf, param_dist,
        n_iter=50,
        scoring='f1',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    
    best_clf = search.best_estimator_
    logging.info(f"Best hyperparameters: {search.best_params_}")
    logging.info(f"Best CV F1: {search.best_score_:.4f}")
    
    # Use the best XGBoost model directly (no calibration wrapper)
    # XGBoost's native predict_proba uses softmax which gives proper probabilities
    final_model = best_clf
    
    # 3. Evaluate & tune threshold
    logging.info("Evaluating on Validation set...")
    val_probs_all = final_model.predict_proba(X_val)
    if val_probs_all.shape[1] > 1:
        val_probs = val_probs_all[:, 1]
    else:
        val_probs = np.zeros(X_val.shape[0])
    
    # Compute base metrics
    roc_auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else np.nan
    pr_auc = average_precision_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else np.nan
    ece_val = compute_ece(y_val, val_probs)
    
    logging.info(f"Baseline Validation ROC-AUC: {roc_auc:.4f}")
    logging.info(f"Baseline Validation PR-AUC: {pr_auc:.4f}")
    
    # tune threshold
    best_thresh = 0.5
    best_f1 = 0.0
    
    # If the validation set only has one class, threshold tuning is tricky
    if len(np.unique(y_val)) > 1:
        thresholds = np.linspace(0.1, 0.9, 81)
        for t in thresholds:
            preds = (val_probs >= t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
    else:
        logging.warning("Validation set has only 1 class. Sticking to default 0.5 threshold.")
        # evaluate at default
        preds = (val_probs >= 0.5).astype(int)
        best_f1 = f1_score(y_val, preds, zero_division=0)
        
    logging.info(f"Optimal Threshold: {best_thresh:.3f} | Best Validation Binary F1: {best_f1:.4f}")
    
    final_val_preds = (val_probs >= best_thresh).astype(int)
    val_precision = precision_score(y_val, final_val_preds, zero_division=0)
    val_recall = recall_score(y_val, final_val_preds, zero_division=0)
    
    metrics = {
        "roc_auc": float(np.nan_to_num(roc_auc)),
        "pr_auc": float(np.nan_to_num(pr_auc)),
        "ece": float(ece_val),
        "best_threshold": float(best_thresh),
        "f1": float(best_f1),
        "precision": float(val_precision),
        "recall": float(val_recall)
    }
    
    # 4. Save artifacts
    model_path = os.path.join(output_dir, "binary_model.joblib")
    joblib.dump(final_model, model_path)
    
    metrics_path = os.path.join(output_dir, "binary_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Metrics and threshold saved to {metrics_path}")
    
    print("\n--- Final Binary Model Review ---")
    print(f"Optimal Threshold: {best_thresh:.3f}")
    print(f"Val F1 Score:      {best_f1:.4f}")
    print(f"Val Precision:     {val_precision:.4f}")
    print(f"Val Recall:        {val_recall:.4f}")
    print(f"Val ROC-AUC:       {roc_auc:.4f}")
    print(f"Val ECE:           {ece_val:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
