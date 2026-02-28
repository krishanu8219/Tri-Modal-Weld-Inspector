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
        # Audio-Visual fallback models (for when CSV is missing)
        self.binary_model_av = None
        self.binary_threshold_av = 0.5
        self.multiclass_model_av = None
        self.le_av = None
        self.has_av_models = False
        
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
        
        # Load Audio-Visual fallback models if available
        av_bin_path = os.path.join(self.artifacts_dir, "binary_model_av.joblib")
        av_multi_path = os.path.join(self.artifacts_dir, "multiclass_model_av.joblib")
        av_le_path = os.path.join(self.artifacts_dir, "label_encoder_av.joblib")
        av_metrics_path = os.path.join(self.artifacts_dir, "binary_av_metrics.json")
        
        if os.path.exists(av_bin_path) and os.path.exists(av_multi_path):
            self.binary_model_av = joblib.load(av_bin_path)
            self.multiclass_model_av = joblib.load(av_multi_path)
            if os.path.exists(av_le_path):
                self.le_av = joblib.load(av_le_path)
            else:
                self.le_av = self.le
            if os.path.exists(av_metrics_path):
                with open(av_metrics_path, "r") as f:
                    self.binary_threshold_av = json.load(f).get("best_threshold", 0.5)
            self.has_av_models = True
            logging.info("Audio-Visual fallback models loaded (for no-CSV inference)")
                
    def infer_run(self, csv_path, flac_path):
        """
        Runs inference for a single sample.
        Auto-detects if CSV exists:
          - If CSV exists: uses full tri-modal model (319 features)
          - If no CSV: uses audio-visual fallback model (217 features)
        """
        run_dir = os.path.dirname(csv_path) if csv_path else os.path.dirname(flac_path)
        csv_exists = os.path.exists(csv_path) if csv_path else False
        
        audio_f = extract_audio_features(flac_path)
        image_f = extract_image_features(run_dir)
        
        # ---- AUDIO-VISUAL MODE (no CSV) ----
        if not csv_exists and self.has_av_models:
            features_av = np.concatenate([audio_f, image_f])
            
            expected_dim = 217  # 120 audio + 97 image
            if len(features_av) != expected_dim:
                if len(features_av) < expected_dim:
                    features_av = np.concatenate([features_av, np.zeros(expected_dim - len(features_av))])
                else:
                    features_av = features_av[:expected_dim]
            
            features_av = features_av.reshape(1, -1)
            
            # Binary
            bin_probs = self.binary_model_av.predict_proba(features_av)
            p_defect = float(bin_probs[0, 1]) if bin_probs.shape[1] > 1 else 0.0
            
            if p_defect < self.binary_threshold_av:
                return {"pred_label_code": "00", "p_defect": p_defect, "type_confidence": None}
            
            # Multiclass
            multi_probs = self.multiclass_model_av.predict_proba(features_av)
            classes = self.le_av.inverse_transform(self.multiclass_model_av.classes_)
            sorted_indices = np.argsort(multi_probs[0])[::-1]
            
            pred_label_code = None
            best_idx = None
            for idx in sorted_indices:
                if classes[idx] != "00":
                    pred_label_code = classes[idx]
                    best_idx = idx
                    break
            
            if pred_label_code is None:
                pred_label_code = classes[sorted_indices[0]]
                best_idx = sorted_indices[0]
            
            return {
                "pred_label_code": str(pred_label_code).zfill(2),
                "p_defect": float(p_defect),
                "type_confidence": float(multi_probs[0, best_idx])
            }
        
        # ---- TRI-MODAL MODE (with CSV) ----
        sensor_f = extract_sensor_features(csv_path) if csv_exists else np.zeros(102)
        features = np.concatenate([sensor_f, audio_f, image_f])
        
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
        
        # 1. Binary prediction
        bin_probs = self.binary_model.predict_proba(features)
        p_defect = float(bin_probs[0, 1]) if bin_probs.shape[1] > 1 else 0.0
            
        if p_defect < self.binary_threshold:
            return {
                "pred_label_code": "00",
                "p_defect": float(p_defect),
                "type_confidence": None
            }
            
        # 2. Multi-class prediction
        multi_probs = self.multiclass_model.predict_proba(features)
        classes = self.le.inverse_transform(self.multiclass_model.classes_)
        
        sorted_indices = np.argsort(multi_probs[0])[::-1]
        
        pred_label_code = None
        best_idx = None
        for idx in sorted_indices:
            if classes[idx] != "00":
                pred_label_code = classes[idx]
                best_idx = idx
                break
        
        if pred_label_code is None:
            pred_label_code = classes[sorted_indices[0]]
            best_idx = sorted_indices[0]
        
        top1_prob = float(multi_probs[0, best_idx])
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
