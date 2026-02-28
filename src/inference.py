import os
import json
import logging
import numpy as np
import pandas as pd
import joblib

# Re-use extract_sensor_features from train_binary
from src.train_binary import extract_sensor_features
from src.audio_features import extract_audio_features
from src.image_features import extract_image_features

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
        pipeline_metrics_path = os.path.join(self.artifacts_dir, "pipeline_metrics.json")
        multi_model_path = os.path.join(self.artifacts_dir, "multiclass_model.joblib")
        le_path = os.path.join(self.artifacts_dir, "label_encoder.joblib")
        
        if not os.path.exists(bin_model_path) or not os.path.exists(multi_model_path) or not os.path.exists(le_path):
            raise FileNotFoundError("Could not find expected model artifacts. Please run both training scripts first.")
            
        self.binary_model = joblib.load(bin_model_path)
        self.multiclass_model = joblib.load(multi_model_path)
        self.le = joblib.load(le_path)
        
        if os.path.exists(pipeline_metrics_path):
            with open(pipeline_metrics_path, "r") as f:
                metrics = json.load(f)
                self.binary_threshold = metrics.get("best_pipeline_threshold", 0.5)
        elif os.path.exists(bin_metrics_path):
            with open(bin_metrics_path, "r") as f:
                metrics = json.load(f)
                self.binary_threshold = metrics.get("best_threshold", 0.5)
                
    def infer_run(self, csv_path, flac_path):
        """
        Runs the full chained inference logic for a single run CSV and FLAC.
        Returns: dictates {'pred_label_code': str, 'p_defect': float, 'type_confidence': float}
        """
        run_dir = os.path.dirname(csv_path)
        
        sensor_f = extract_sensor_features(csv_path)
        audio_f = extract_audio_features(flac_path)
        image_f = extract_image_features(run_dir)
        
        features = np.concatenate([sensor_f, audio_f, image_f])
        
        # Defensive: ensure feature vector is exactly 319 dims (102 sensor + 120 audio + 97 image)
        expected_dim = 319
        if len(features) != expected_dim:
            logging.warning(
                f"Feature dimension mismatch: got {len(features)} (sensor={len(sensor_f)}, "
                f"audio={len(audio_f)}, image={len(image_f)}), expected {expected_dim}. "
                f"Padding/truncating to {expected_dim}."
            )
            if len(features) < expected_dim:
                features = np.concatenate([features, np.zeros(expected_dim - len(features))])
            else:
                features = features[:expected_dim]
        
        features = features.reshape(1, -1)
        
        # Use base XGBoost estimator (not wrapped CalibratedClassifier) for probabilities
        # CalibratedClassifierCV crushes probs to exact 0/1, hiding meaningful variation
        def get_base_proba(model, X):
            """Extract predict_proba from the underlying XGBoost, bypassing calibration."""
            try:
                base = model.calibrated_classifiers_[0].estimator
                return base.predict_proba(X)
            except (AttributeError, IndexError):
                return model.predict_proba(X)
        
        # 1. Binary prediction — use base estimator for real probabilities
        bin_probs = get_base_proba(self.binary_model, features)
        
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
                "type_confidence": None
            }
            
        # 2. Multi-class prediction — use base estimator 
        raw_multi_probs = self.multiclass_model.predict_proba(features)
        multi_probs = get_base_proba(self.multiclass_model, features) if hasattr(self.multiclass_model, 'calibrated_classifiers_') else raw_multi_probs
        classes = self.le.inverse_transform(self.multiclass_model.classes_)
        
        # Sort by descending probability
        sorted_indices = np.argsort(multi_probs[0])[::-1]
        
        # Find best non-"00" class (since binary already said defect)
        pred_label_code = None
        best_idx = None
        for idx in sorted_indices:
            if classes[idx] != "00":
                pred_label_code = classes[idx]
                best_idx = idx
                break
        
        # Fallback if all classes are "00" somehow
        if pred_label_code is None:
            pred_label_code = classes[sorted_indices[0]]
            best_idx = sorted_indices[0]
        
        top1_prob = float(multi_probs[0, best_idx])
        
        # type_confidence = raw multiclass probability of the predicted class
        # This reflects how sure the model is about THIS specific defect type
        # e.g. 0.68 → 68% it's "Burn Through" vs other defect types
        type_confidence = top1_prob
        
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
