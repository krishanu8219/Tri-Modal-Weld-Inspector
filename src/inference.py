import os
import json
import logging
import numpy as np
import pandas as pd
import joblib

# Re-use extract_sensor_features from train_binary
from src.train_binary import extract_sensor_features
from src.audio_features import extract_audio_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class DefectClassifierPipeline:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir
        self.binary_model = None
        self.binary_threshold = 0.5
        self.multiclass_model = None
        
        self._load_models()
        
    def _load_models(self):
        bin_model_path = os.path.join(self.artifacts_dir, "binary_model.joblib")
        bin_metrics_path = os.path.join(self.artifacts_dir, "binary_metrics.json")
        multi_model_path = os.path.join(self.artifacts_dir, "multiclass_model.joblib")
        
        if not os.path.exists(bin_model_path) or not os.path.exists(multi_model_path):
            raise FileNotFoundError("Could not find expected model artifacts. Please run both training scripts first.")
            
        self.binary_model = joblib.load(bin_model_path)
        self.multiclass_model = joblib.load(multi_model_path)
        
        if os.path.exists(bin_metrics_path):
            with open(bin_metrics_path, "r") as f:
                metrics = json.load(f)
                self.binary_threshold = metrics.get("best_threshold", 0.5)
                
    def infer_run(self, csv_path, flac_path):
        """
        Runs the full chained inference logic for a single run CSV and FLAC.
        Returns: dictates {'pred_label_code': str, 'p_defect': float, 'type_confidence': float}
        """
        sensor_features = extract_sensor_features(csv_path)
        audio_features = extract_audio_features(flac_path)
        
        features = np.concatenate([sensor_features, audio_features]).reshape(1, -1)
        
        # 1. Binary prediction
        bin_probs = self.binary_model.predict_proba(features)
        
        # Determine probability of defect
        if bin_probs.shape[1] > 1:
            p_defect = bin_probs[0, 1]
        else:
            # edge case handling if base model saw 1 class
            p_defect = 0.0
            
        if p_defect < self.binary_threshold:
            return {
                "pred_label_code": "00",
                "p_defect": float(p_defect),
                "type_confidence": float(1.0 - p_defect)
            }
            
        # 2. Multi-class prediction
        multi_probs = self.multiclass_model.predict_proba(features)
        classes = self.multiclass_model.classes_
        
        # Choose highest probability
        best_idx = np.argmax(multi_probs[0])
        pred_label_code = classes[best_idx]
        type_confidence = multi_probs[0, best_idx]
        
        # Edge case: If the multiclass model somehow predicted "00" despite binary saying defect, 
        # let's fallback to the second highest defect class if strictly enforcing "defect vs non-defect" cascading.
        # The spec implies if binary says defect, we predict a defect type. 
        if pred_label_code == "00" and len(classes) > 1:
            sorted_indices = np.argsort(multi_probs[0])[::-1]
            for idx in sorted_indices:
                if classes[idx] != "00":
                    pred_label_code = classes[idx]
                    type_confidence = multi_probs[0, idx]
                    break
        
        return {
            "pred_label_code": str(pred_label_code).zfill(2),
            "p_defect": float(p_defect),
            "type_confidence": float(type_confidence)
        }

def test_inference_pipeline(val_split_csv="val_split.csv"):
    if not os.path.exists(val_split_csv):
        logging.warning(f"Could not find {val_split_csv} for testing.")
        return
        
    df = pd.read_csv(val_split_csv)
    pipeline = DefectClassifierPipeline()
    
    print("\n--- Chained Inference Results (Sample) ---")
    results = []
    
    # Process up to 5 runs as a smoke test
    sample_df = df.head(5)
    for _, row in sample_df.iterrows():
        csv_path = row["csv_path"]
        flac_path = row["flac_path"]
        ground_truth = str(row["label_code"]).zfill(2)
        res = pipeline.infer_run(csv_path, flac_path)
        
        print(f"Run {row['run_id']} | True: {ground_truth} | Pred: {res['pred_label_code']} | P(Defect): {res['p_defect']:.3f} | Conf: {res['type_confidence']:.3f}")
        
if __name__ == "__main__":
    test_inference_pipeline()
    print("Inference script executed successfully.") 
