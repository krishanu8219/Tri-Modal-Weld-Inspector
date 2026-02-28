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
        "binary": load_json("artifacts/binary_metrics.json"),
        "multiclass": load_json("artifacts/multiclass_metrics.json"),
        "pipeline": load_json("artifacts/pipeline_metrics.json"),
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
