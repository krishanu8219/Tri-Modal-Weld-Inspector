from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import os
import glob
from src.inference import DefectClassifierPipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None

@app.on_event("startup")
def load_pipeline():
    global pipeline
    try:
        pipeline = DefectClassifierPipeline()
        print("Loaded Tri-Modal Classification Pipeline.")
    except Exception as e:
        print(f"Failed to load pipelines: {e}")

# Mount static files to serve images and audio natively.
app.mount("/static/sampleData", StaticFiles(directory="sampleData"), name="static_sample")
if os.path.exists("hackathon data"):
    app.mount("/static/hackathon_data", StaticFiles(directory="hackathon data"), name="static_hackathon")

# Mount test_data for serving test sample images/audio
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_data")
if os.path.exists(TEST_DATA_DIR):
    app.mount("/static/test_data", StaticFiles(directory=TEST_DATA_DIR), name="static_test")

def file_to_static_url(filepath: str) -> str:
    """Convert a local file path to a servable static URL."""
    if filepath.startswith("sampleData/") or filepath.startswith("sampleData\\"):
        return "/static/sampleData/" + filepath[len("sampleData/"):].replace("\\", "/")
    elif filepath.startswith("hackathon data/") or filepath.startswith("hackathon data\\"):
        return "/static/hackathon_data/" + filepath[len("hackathon data/"):].replace("\\", "/")
    return "/static/sampleData/" + filepath.replace("\\", "/")

import numpy as np
import json

@app.get("/stats")
def get_dataset_stats():
    """Dataset overview: counts, durations, missing stats, label distribution."""
    if not os.path.exists("runs_summary.csv"):
        raise HTTPException(status_code=404, detail="runs_summary.csv not found")

    df = pd.read_csv("runs_summary.csv")
    # Exclude sampleData demo runs
    df = df[df["data_dir"] != "sampleData"]
    train_pool = df[df["data_dir"] != "test_data"]
    test_pool = df[df["data_dir"] == "test_data"]

    # Label distribution
    label_counts = {}
    if not train_pool.empty:
        lc = train_pool["label_code"].astype(str).str.zfill(2).value_counts().sort_index()
        label_counts = lc.to_dict()

    # Duration stats
    dur_audio = df["flac_duration"].dropna().tolist() if "flac_duration" in df.columns else []
    dur_video = df["avi_duration"].dropna().tolist() if "avi_duration" in df.columns else []

    # Quality
    csv_missing = int((~df["csv_valid"]).sum()) if "csv_valid" in df.columns else 0
    flac_missing = int((~df["flac_valid"]).sum()) if "flac_valid" in df.columns else 0
    avi_missing = int((~df["avi_valid"]).sum()) if "avi_valid" in df.columns else 0

    return {
        "total_runs": len(df),
        "training_pool": len(train_pool),
        "test_samples": len(test_pool),
        "complete_runs": int(df["has_all_modalities"].sum()),
        "label_counts": label_counts,
        "audio_durations": dur_audio[:100],  # cap for JSON size
        "video_durations": dur_video[:100],
        "missing_csv": csv_missing,
        "missing_flac": flac_missing,
        "missing_avi": avi_missing,
    }

@app.get("/metrics")
def get_metrics():
    """Return all saved model metrics."""
    def load_json(path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    return {
        "binary": load_json("artifacts/binary_av_metrics.json"),
        "multiclass": load_json("artifacts/multiclass_av_metrics.json"),
        "pipeline": load_json("artifacts/pipeline_av_metrics.json"),
    }

@app.get("/diagnostics")
def get_diagnostics():
    """Return train vs validation fit diagnostics."""
    path = "artifacts/fit_diagnostics.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

@app.get("/runs")
def get_runs():
    if not os.path.exists("runs_summary.csv"):
        raise HTTPException(status_code=404, detail="runs_summary.csv not found")
        
    df_runs = pd.read_csv("runs_summary.csv")
    # Only show hackathon data (exclude sampleData demo runs)
    df_runs = df_runs[df_runs["data_dir"] != "sampleData"]
    valid_runs = df_runs[df_runs["has_all_modalities"] == True]
    
    # Deduplicate by run_id
    valid_runs = valid_runs.drop_duplicates(subset="run_id", keep="first")
    
    # Replace NaN with None for JSON serialization
    valid_runs = valid_runs.replace({np.nan: None})
    
    # Simple list of dicts
    runs = valid_runs.to_dict(orient="records")
    return {"runs": runs}

@app.get("/audio-waveform/{run_id}")
def get_audio_waveform(run_id: str):
    """Return downsampled audio waveform + error/anomaly region for visualization."""
    if not os.path.exists("runs_summary.csv"):
        raise HTTPException(status_code=404, detail="runs_summary.csv not found")

    df_runs = pd.read_csv("runs_summary.csv")
    run_df = df_runs[df_runs["run_id"] == run_id]

    if len(run_df) == 0:
        raise HTTPException(status_code=404, detail="Run not found")

    run_data = run_df.iloc[0]
    flac_path = run_data.get("flac_path", "")

    if not flac_path or not os.path.exists(flac_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    try:
        import librosa
        y, sr = librosa.load(flac_path, sr=None, mono=True)

        # Compute RMS envelope for a smooth waveform
        hop_length = max(1, len(y) // 500)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        waveform = rms.tolist()

        # Find anomaly region: window with highest variance in the RMS envelope
        window_size = max(1, len(waveform) // 8)
        rolling_var = pd.Series(waveform).rolling(window=window_size).var().fillna(0).to_numpy()
        hotspot_end = int(np.argmax(rolling_var))
        hotspot_start = max(0, hotspot_end - window_size)

        return {
            "run_id": run_id,
            "waveform": waveform,
            "sample_rate": sr,
            "duration": float(len(y) / sr),
            "error_region": [hotspot_start, hotspot_end],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")


@app.get("/infer/{run_id}")
def infer_run(run_id: str):
    global pipeline
    if pipeline is None:
        try:
            pipeline = DefectClassifierPipeline()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline initialization failed: {str(e)}")
        
    if not os.path.exists("runs_summary.csv"):
        raise HTTPException(status_code=404, detail="runs_summary.csv not found")
        
    df_runs = pd.read_csv("runs_summary.csv")
    run_df = df_runs[df_runs["run_id"] == run_id]
    
    if len(run_df) == 0:
        raise HTTPException(status_code=404, detail="Run not found")
        
    run_data = run_df.iloc[0]
    
    # Run Inference
    try:
        res = pipeline.infer_run(run_data["csv_path"], run_data["flac_path"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    # Get image paths
    run_dir = os.path.dirname(run_data["csv_path"])
    images = []
    
    img_dir = os.path.join(run_dir, "images")
    if os.path.exists(img_dir):
        image_files = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))
        for img in sorted(image_files)[:3]:
            images.append(file_to_static_url(img))
            
    # Extract Telemetry trace for frontend visualization
    sensor_data = []
    if os.path.exists(run_data["csv_path"]):
        df = pd.read_csv(run_data["csv_path"])
        # look for columns to plot
        primary_col = None
        targets = ["Primary Weld Current", "Current", "Pressure"]
        for t in targets:
            if t in df.columns:
                primary_col = t
                break
                
        if primary_col is not None and not df.empty:
            trace = df[primary_col].fillna(0).tolist()
            # Downsample if too large to avoid huge JSONs (e.g. max 200 points)
            step = max(1, len(trace) // 200)
            downsampled = trace[::step]
            
            # Find hotspot
            rolling_var = pd.Series(trace).rolling(window=min(200, max(1, len(trace)//5))).var().fillna(0).to_numpy()
            hotspot_end = int(np.argmax(rolling_var))
            hotspot_start = max(0, hotspot_end - min(200, max(1, len(trace)//5)))
            
            # Map downsampled indices properly or just return the full sequence up to 500
            if len(trace) > 500:
                step = len(trace) // 500
                trace = trace[::step]
                hotspot_start = hotspot_start // step
                hotspot_end = hotspot_end // step
                
            sensor_data = {
                "metric_name": primary_col,
                "values": trace,
                "hotspot": [hotspot_start, hotspot_end]
            }

    # Include audio static path
    flac_static_path = file_to_static_url(run_data["flac_path"])

    return {
        "run_id": run_id,
        "inference": res,
        "images": images,
        "audio": flac_static_path,
        "sensor_telemetry": sensor_data
    }

@app.get("/explain/{run_id}")
def explain_run(run_id: str):
    """Return SHAP feature importance breakdown explaining WHY the model flagged a run."""
    if not os.path.exists("runs_summary.csv"):
        raise HTTPException(status_code=404, detail="runs_summary.csv not found")
    
    df = pd.read_csv("runs_summary.csv")
    match = df[df["run_id"] == run_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    run_data = match.iloc[0].to_dict()
    csv_path = run_data["csv_path"]
    flac_path = run_data["flac_path"]
    run_dir = os.path.dirname(csv_path)
    
    try:
        from src.train_binary import extract_sensor_features
        from src.audio_features import extract_audio_features
        from src.image_features import extract_image_features
        from src.shap_explainer import explain_prediction
        
        sensor_f = extract_sensor_features(csv_path)
        audio_f = extract_audio_features(flac_path)
        image_f = extract_image_features(run_dir)
        features = np.concatenate([sensor_f, audio_f, image_f])
        
        # Get prediction first
        res = pipeline.infer_run(csv_path, flac_path)
        
        # Compute SHAP explanation
        explanation = explain_prediction(features, res["pred_label_code"])
        
        return {
            "run_id": run_id,
            "prediction": res,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


# ═══════════════════════════════════════════════════════
#  TEST DATA ENDPOINTS — serve test samples for inspection
# ═══════════════════════════════════════════════════════

def _load_submission():
    """Load submission.csv into a dict keyed by sample_id."""
    if os.path.exists("submission.csv"):
        df = pd.read_csv("submission.csv", dtype=str)
        return {row["sample_id"]: row for _, row in df.iterrows()}
    return {}


def _find_test_sample_dir(sample_id: str):
    """Return absolute path to a test sample directory."""
    d = os.path.join(TEST_DATA_DIR, sample_id)
    if os.path.isdir(d):
        return d
    return None


@app.get("/test-runs")
def get_test_runs():
    """Return list of test samples with predictions from submission.csv."""
    if not os.path.exists(TEST_DATA_DIR):
        raise HTTPException(status_code=404, detail="test_data directory not found")

    submissions = _load_submission()
    samples = sorted(
        d for d in os.listdir(TEST_DATA_DIR)
        if os.path.isdir(os.path.join(TEST_DATA_DIR, d)) and d.startswith("sample_")
    )

    runs = []
    for sid in samples:
        sample_dir = os.path.join(TEST_DATA_DIR, sid)
        flac_files = glob.glob(os.path.join(sample_dir, "*.flac"))
        img_dir = os.path.join(sample_dir, "images")
        n_images = len(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))) if os.path.isdir(img_dir) else 0

        pred = submissions.get(sid, {})
        runs.append({
            "sample_id": sid,
            "pred_label_code": pred.get("pred_label_code", "??"),
            "p_defect": float(pred.get("p_defect", 0)),
            "has_audio": len(flac_files) > 0,
            "n_images": n_images,
        })

    return {"runs": runs}


@app.get("/infer-test/{sample_id}")
def infer_test(sample_id: str):
    """Return prediction + media for a test sample."""
    sample_dir = _find_test_sample_dir(sample_id)
    if not sample_dir:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")

    submissions = _load_submission()
    pred = submissions.get(sample_id, {})

    # Find audio
    flac_files = glob.glob(os.path.join(sample_dir, "*.flac"))
    audio_url = f"/static/test_data/{sample_id}/{os.path.basename(flac_files[0])}" if flac_files else None

    # Find images
    img_dir = os.path.join(sample_dir, "images")
    images = []
    if os.path.isdir(img_dir):
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        for img in img_files[:6]:
            images.append(f"/static/test_data/{sample_id}/images/{os.path.basename(img)}")

    pred_code = pred.get("pred_label_code", "00")
    p_defect = float(pred.get("p_defect", 0))

    return {
        "run_id": sample_id,
        "inference": {
            "pred_label_code": pred_code,
            "p_defect": p_defect,
            "type_confidence": p_defect if pred_code != "00" else 1.0 - p_defect,
        },
        "images": images,
        "audio": audio_url,
        "sensor_telemetry": [],
    }


@app.get("/audio-waveform-test/{sample_id}")
def get_audio_waveform_test(sample_id: str):
    """Return downsampled audio waveform for a test sample."""
    sample_dir = _find_test_sample_dir(sample_id)
    if not sample_dir:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")

    flac_files = glob.glob(os.path.join(sample_dir, "*.flac"))
    if not flac_files:
        raise HTTPException(status_code=404, detail="No audio file found")

    try:
        import librosa
        y, sr = librosa.load(flac_files[0], sr=None, mono=True)
        hop_length = max(1, len(y) // 500)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        waveform = rms.tolist()

        window_size = max(1, len(waveform) // 8)
        rolling_var = pd.Series(waveform).rolling(window=window_size).var().fillna(0).to_numpy()
        hotspot_end = int(np.argmax(rolling_var))
        hotspot_start = max(0, hotspot_end - window_size)

        return {
            "run_id": sample_id,
            "waveform": waveform,
            "sample_rate": sr,
            "duration": float(len(y) / sr),
            "error_region": [hotspot_start, hotspot_end],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
