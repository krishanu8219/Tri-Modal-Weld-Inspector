"""
SHAP explainability for the Tri-Modal Weld Defect Classifier.
Uses TreeExplainer (fast, exact for XGBoost) to explain individual predictions.
Groups SHAP values by modality: Sensor / Audio / Image.
"""

import os
import logging
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Feature layout:  102 sensor  |  120 audio  |  97 image
SENSOR_DIM = 102
AUDIO_DIM = 120
IMAGE_DIM = 97

SENSOR_RANGE = (0, SENSOR_DIM)
AUDIO_RANGE = (SENSOR_DIM, SENSOR_DIM + AUDIO_DIM)
IMAGE_RANGE = (SENSOR_DIM + AUDIO_DIM, SENSOR_DIM + AUDIO_DIM + IMAGE_DIM)

SENSOR_COLS = [
    "Pressure", "CO2 Weld Flow", "Feed", 
    "Primary Weld Current", "Secondary Weld Voltage", "Wire Consumed"
]

# 17 stats per sensor column
SENSOR_STAT_NAMES = [
    "mean", "std", "min", "max", "median", "skew",
    "q10", "q25", "q75", "q90", "iqr", "range",
    "diff_mean", "diff_std", "diff_max", "tail_mean", "tail_std"
]

# Lazy-loaded SHAP explainers
_binary_explainer = None
_multi_explainer = None


def _unwrap_model(model):
    """Extract base XGBoost estimator from CalibratedClassifierCV if present."""
    if hasattr(model, 'calibrated_classifiers_'):
        return model.calibrated_classifiers_[0].estimator
    return model


def _get_explainers(artifacts_dir="artifacts"):
    """Lazy-load SHAP TreeExplainers for both models."""
    global _binary_explainer, _multi_explainer
    if _binary_explainer is None:
        import shap
        bin_model = _unwrap_model(joblib.load(os.path.join(artifacts_dir, "binary_model.joblib")))
        multi_model = _unwrap_model(joblib.load(os.path.join(artifacts_dir, "multiclass_model.joblib")))
        _binary_explainer = shap.TreeExplainer(bin_model)
        _multi_explainer = shap.TreeExplainer(multi_model)
    return _binary_explainer, _multi_explainer


def explain_prediction(features, pred_label_code, artifacts_dir="artifacts"):
    """
    Explain a single prediction using SHAP values.
    
    Args:
        features: 1D numpy array of 798 features
        pred_label_code: string like "02" — the predicted defect code
        artifacts_dir: path to model artifacts
        
    Returns:
        dict with modality contributions and top features
    """
    import shap
    
    features_2d = features.reshape(1, -1)
    bin_explainer, multi_explainer = _get_explainers(artifacts_dir)
    
    # Binary SHAP values — explains P(defect)
    bin_shap = bin_explainer.shap_values(features_2d)
    if isinstance(bin_shap, list):
        # For binary: shap_values returns [class0_shap, class1_shap]
        bin_shap = bin_shap[1]  # defect class
    bin_shap = bin_shap.flatten()
    
    # Group SHAP values by modality (sum of absolute contributions)
    sensor_shap = float(np.sum(np.abs(bin_shap[SENSOR_RANGE[0]:SENSOR_RANGE[1]])))
    audio_shap = float(np.sum(np.abs(bin_shap[AUDIO_RANGE[0]:AUDIO_RANGE[1]])))
    image_shap = float(np.sum(np.abs(bin_shap[IMAGE_RANGE[0]:IMAGE_RANGE[1]])))
    
    total_shap = sensor_shap + audio_shap + image_shap
    if total_shap > 0:
        sensor_pct = sensor_shap / total_shap
        audio_pct = audio_shap / total_shap
        image_pct = image_shap / total_shap
    else:
        sensor_pct = audio_pct = image_pct = 1/3

    # Direction: positive SHAP = pushes toward defect
    sensor_direction = float(np.sum(bin_shap[SENSOR_RANGE[0]:SENSOR_RANGE[1]]))
    audio_direction = float(np.sum(bin_shap[AUDIO_RANGE[0]:AUDIO_RANGE[1]]))
    image_direction = float(np.sum(bin_shap[IMAGE_RANGE[0]:IMAGE_RANGE[1]]))
    
    # Top 5 sensor features by |SHAP value|
    sensor_shap_values = bin_shap[SENSOR_RANGE[0]:SENSOR_RANGE[1]]
    top_sensor_indices = np.argsort(np.abs(sensor_shap_values))[::-1][:5]
    
    top_features = []
    for idx in top_sensor_indices:
        col_idx = idx // 17
        stat_idx = idx % 17
        col_name = SENSOR_COLS[col_idx] if col_idx < len(SENSOR_COLS) else f"Col{col_idx}"
        stat_name = SENSOR_STAT_NAMES[stat_idx] if stat_idx < len(SENSOR_STAT_NAMES) else f"stat{stat_idx}"
        top_features.append({
            "name": f"{col_name} ({stat_name})",
            "shap_value": float(sensor_shap_values[idx]),
            "feature_value": float(features[idx]),
            "direction": "defect" if sensor_shap_values[idx] > 0 else "good"
        })
    
    return {
        "modality_contributions": {
            "sensor": {"percentage": round(sensor_pct * 100, 1), "direction": "defect" if sensor_direction > 0 else "good"},
            "audio": {"percentage": round(audio_pct * 100, 1), "direction": "defect" if audio_direction > 0 else "good"},
            "image": {"percentage": round(image_pct * 100, 1), "direction": "defect" if image_direction > 0 else "good"},
        },
        "top_sensor_features": top_features,
        "base_value": float(bin_explainer.expected_value[1] if isinstance(bin_explainer.expected_value, (list, np.ndarray)) else bin_explainer.expected_value),
    }
