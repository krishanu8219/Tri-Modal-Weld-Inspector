import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib

from src.train_binary import prepare_data
from src.train_multiclass import prepare_multiclass_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def tune_pipeline_threshold(val_csv="val_split.csv", artifacts_dir="artifacts"):
    df_val = pd.read_csv(val_csv)
    
    # Extract features (same for both models)
    logging.info("Extracting features from validation set...")
    X_val, y_val_bin, val_ids = prepare_data(df_val)
    _, y_val_str, _ = prepare_multiclass_data(df_val)
    
    bin_model_path = os.path.join(artifacts_dir, "binary_model.joblib")
    multi_model_path = os.path.join(artifacts_dir, "multiclass_model.joblib")
    le_path = os.path.join(artifacts_dir, "label_encoder.joblib")
    
    if not os.path.exists(bin_model_path) or not os.path.exists(multi_model_path) or not os.path.exists(le_path):
        logging.error("Models or LabelEncoder not found. Train binary and multiclass models first.")
        return
        
    binary_model = joblib.load(bin_model_path)
    multiclass_model = joblib.load(multi_model_path)
    le = joblib.load(le_path)
    
    # Get probabilities
    if binary_model.classes_.shape[0] > 1:
        bin_probs = binary_model.predict_proba(X_val)[:, 1]
    else:
        bin_probs = np.zeros(X_val.shape[0])
        
    multi_probs = multiclass_model.predict_proba(X_val)
    classes = le.inverse_transform(multiclass_model.classes_)
    
    # Vectorized fallback logic for multiclass (same as infer_run)
    best_multi_idx = np.argmax(multi_probs, axis=1)
    multi_preds_raw = classes[best_multi_idx]
    
    # If binary says defect but multiclass says 00, fallback to 2nd highest defect class
    fallback_preds = []
    for i in range(len(X_val)):
        if multi_preds_raw[i] == "00" and len(classes) > 1:
            sorted_indices = np.argsort(multi_probs[i])[::-1]
            for idx in sorted_indices:
                if classes[idx] != "00":
                    fallback_preds.append(classes[idx])
                    break
        else:
            fallback_preds.append(multi_preds_raw[i])
            
    fallback_preds = np.array(fallback_preds)
    
    best_thresh = 0.5
    best_final_score = 0.0
    best_metrics = {}
    best_final_preds = None

    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        # 1. Binary Gate
        bin_preds = (bin_probs >= t).astype(int)
        
        # 2. Final pipeline prediction
        final_preds = np.where(bin_preds == 0, "00", fallback_preds)
        
        # 3. Calculate metrics
        final_bin_preds = (final_preds != "00").astype(int)
        binary_f1 = f1_score(y_val_bin, final_bin_preds, zero_division=0)
        type_macro_f1 = f1_score(y_val_str, final_preds, average="macro", zero_division=0)
        
        # Combined Score
        final_score = 0.6 * binary_f1 + 0.4 * type_macro_f1
        
        if final_score > best_final_score:
            best_final_score = final_score
            best_thresh = t
            best_metrics = {
                "binary_f1": binary_f1,
                "type_macro_f1": type_macro_f1,
                "final_score": final_score
            }
            best_final_preds = final_preds
            
    logging.info(f"Optimization Complete.")
    logging.info(f"Optimal End-to-End Threshold: {best_thresh:.3f}")
    logging.info(f"Validation Combined Score: {best_metrics['final_score']:.4f}")
    logging.info(f"Validation Binary F1: {best_metrics['binary_f1']:.4f}")
    logging.info(f"Validation Type Macro-F1: {best_metrics['type_macro_f1']:.4f}")
    
    if len(np.unique(y_val_str)) > 1:
        print("\n--- Final Pipeline Classification Report ---")
        print(classification_report(y_val_str, best_final_preds, zero_division=0))
        
        print("\n--- Confusion Matrix ---")
        labels = np.unique(np.concatenate([y_val_str, best_final_preds]))
        cm = confusion_matrix(y_val_str, best_final_preds, labels=labels)
        df_cm = pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
        print(df_cm)
    
    # Save the optimal threshold to pipeline_metrics.json
    metrics_path = os.path.join(artifacts_dir, "pipeline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "best_pipeline_threshold": float(best_thresh),
            "final_score": float(best_metrics['final_score']),
            "binary_f1": float(best_metrics['binary_f1']),
            "type_macro_f1": float(best_metrics['type_macro_f1'])
        }, f, indent=4)
    logging.info(f"Saved optimal pipeline threshold to {metrics_path}")

if __name__ == "__main__":
    tune_pipeline_threshold()
