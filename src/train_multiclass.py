import os
import json
import logging
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
# CalibratedClassifierCV removed — XGBoost native softmax gives proper probabilities
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from imblearn.over_sampling import SMOTE

from src.train_binary import extract_sensor_features
from src.audio_features import extract_audio_features
from src.image_features import extract_image_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def prepare_multiclass_data(split_df):
    """
    Maps rows to feature matrix X and string label vector y.
    Includes Late Fusion (Sensor features + Audio MFCCs).
    """
    X = []
    y = []
    metadata = []
    
    for _, row in split_df.iterrows():
        csv_path = row["csv_path"]
        flac_path = row["flac_path"]
        label = str(row["label_code"]).zfill(2)
        
        run_dir = os.path.dirname(csv_path)
        
        sensor_f = extract_sensor_features(csv_path)
        audio_f = extract_audio_features(flac_path)
        image_f = extract_image_features(run_dir)
        
        # Late Fusion logic
        combined_f = np.concatenate([sensor_f, audio_f, image_f])
        
        X.append(combined_f)
        y.append(label)
        metadata.append(row["run_id"])
        
    return np.array(X), np.array(y), metadata

def train_multiclass_and_evaluate(train_csv="train_split.csv", val_csv="val_split.csv", output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load splits
    logging.info("Loading train/val split CSVs.")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    
    logging.info("Extracting features from sensor data...")
    X_train, y_train, _ = prepare_multiclass_data(df_train)
    X_val, y_val, _ = prepare_multiclass_data(df_val)
    
    # Edge case strictly for demonstration: If sampleData doesn't have all classes,
    # generate dummy rows for missing defect types to let sklearn run.
    unique_train_classes = np.unique(y_train)
    if len(unique_train_classes) < 2:
        logging.warning("Training split has < 2 classes. Adding dummy defect labels for demonstration.")
        # Replicate first row with 5 other fake defects
        dummy_classes = ["01", "02", "06", "07", "08", "11"]
        for c in dummy_classes:
            if c not in unique_train_classes:
                X_train = np.vstack([X_train, X_train[0]])
                y_train = np.append(y_train, c)
                
    if len(np.unique(y_val)) < 2:
        logging.warning("Validation split has < 2 classes.")
    
    # XGBoost requires integer labels 0 to n_classes-1
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    
    logging.info("Applying SMOTE oversampling for class balance...")
    # SMOTE requires at least k_neighbors+1 samples per class
    min_class_count = min(np.bincount(y_train_enc))
    if min_class_count >= 6:
        smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))
        X_train, y_train_enc = smote.fit_resample(X_train, y_train_enc)
        logging.info(f"After SMOTE: {len(X_train)} samples, distribution: {dict(zip(*np.unique(y_train_enc, return_counts=True)))}")
    else:
        logging.warning(f"Smallest class has only {min_class_count} samples — skipping SMOTE, using sample_weight instead.")
    
    # Compute balanced sample weights
    sample_weights = compute_sample_weight('balanced', y_train_enc)
    
    # Hyperparameter tuning via RandomizedSearchCV
    logging.info("Running RandomizedSearchCV for multiclass XGBoost hyperparameter tuning...")
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
    
    base_clf = XGBClassifier(random_state=42, use_label_encoder=False, verbosity=0)
    
    from sklearn.model_selection import RandomizedSearchCV
    search = RandomizedSearchCV(
        base_clf, param_dist,
        n_iter=50,
        scoring='f1_macro',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train_enc, sample_weight=sample_weights)
    
    best_clf = search.best_estimator_
    logging.info(f"Best hyperparameters: {search.best_params_}")
    logging.info(f"Best CV Macro F1: {search.best_score_:.4f}")
    
    # Use the best XGBoost model directly (no calibration wrapper)
    final_model = best_clf
        
    # 3. Evaluate multi-class
    logging.info("Evaluating on Validation set...")
    val_preds_enc = final_model.predict(X_val)
    val_preds = le.inverse_transform(val_preds_enc)
    
    # Use Macro F1 to treat each class equally despite heavy imbalance
    macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)
    macro_precision = precision_score(y_val, val_preds, average="macro", zero_division=0)
    macro_recall = recall_score(y_val, val_preds, average="macro", zero_division=0)
    
    logging.info(f"Validation Macro F1: {macro_f1:.4f}")
    
    # We dump class mapping and metrics
    metrics = {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "classes": list(map(str, final_model.classes_))
    }
    
    # 4. Save artifacts
    model_path = os.path.join(output_dir, "multiclass_model.joblib")
    joblib.dump(final_model, model_path)
    
    metrics_path = os.path.join(output_dir, "multiclass_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save LabelEncoder to decode later
    le_path = os.path.join(output_dir, "label_encoder.joblib")
    joblib.dump(le, le_path)
    
    logging.info(f"Multi-class Model saved to {model_path}")
    logging.info(f"LabelEncoder saved to {le_path}")
    logging.info(f"Metrics saved to {metrics_path}")
    
    print("\n--- Final Multi-class Model Review ---")
    print(f"Val Macro F1:       {macro_f1:.4f}")
    if len(np.unique(y_val)) > 1:
        print("\nClassification Report:")
        print(classification_report(y_val, val_preds, zero_division=0))

if __name__ == "__main__":
    train_multiclass_and_evaluate()
