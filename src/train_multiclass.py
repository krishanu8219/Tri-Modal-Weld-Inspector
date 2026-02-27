import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import joblib

from src.train_binary import extract_sensor_features
from src.audio_features import extract_audio_features

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
        
        sensor_f = extract_sensor_features(csv_path)
        audio_f = extract_audio_features(flac_path)
        
        # Late Fusion logic
        combined_f = np.concatenate([sensor_f, audio_f])
        
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
    
    # 2. Train baseline model with class weights
    logging.info("Training Multi-Class Random Forest Classifier with balanced weights...")
    # 'balanced_subsample' computes weights based on bootstrap sample for each tree
    base_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced_subsample")
    base_clf.fit(X_train, y_train)
    
    logging.info("Calibrating multi-class probabilities...")
    try:
        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=min(5, len(y_train)))
        calibrated_clf.fit(X_train, y_train)
    except Exception as e:
        logging.warning(f"Failed to calibrate using Sigmoid 5-fold CV: {e}. Falling back to uncalibrated classifier.")
        calibrated_clf = base_clf
        
    # 3. Evaluate multi-class
    logging.info("Evaluating on Validation set...")
    val_preds = calibrated_clf.predict(X_val)
    
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
        "classes": list(map(str, calibrated_clf.classes_))
    }
    
    # 4. Save artifacts
    model_path = os.path.join(output_dir, "multiclass_model.joblib")
    joblib.dump(calibrated_clf, model_path)
    
    metrics_path = os.path.join(output_dir, "multiclass_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logging.info(f"Multi-class Model saved to {model_path}")
    logging.info(f"Metrics saved to {metrics_path}")
    
    print("\n--- Final Multi-class Model Review ---")
    print(f"Val Macro F1:       {macro_f1:.4f}")
    if len(np.unique(y_val)) > 1:
        print("\nClassification Report:")
        print(classification_report(y_val, val_preds, zero_division=0))

if __name__ == "__main__":
    train_multiclass_and_evaluate()
