from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
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
app.mount("/static", StaticFiles(directory="sampleData"), name="static")

import numpy as np

@app.get("/runs")
def get_runs():
    if not os.path.exists("runs_summary.csv"):
        raise HTTPException(status_code=404, detail="runs_summary.csv not found")
        
    df_runs = pd.read_csv("runs_summary.csv")
    valid_runs = df_runs[df_runs["has_all_modalities"] == True]
    
    # Replace NaN with None for JSON serialization
    valid_runs = valid_runs.replace({np.nan: None})
    
    # Simple list of dicts
    runs = valid_runs.to_dict(orient="records")
    return {"runs": runs}

@app.get("/infer/{run_id}")
def infer_run(run_id: str):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    if not os.path.exists("runs_summary.csv"):
        raise HTTPException(status_code=404, detail="runs_summary.csv not found")
        
    df_runs = pd.read_csv("runs_summary.csv")
    run_df = df_runs[df_runs["run_id"] == run_id]
    
    if len(run_df) == 0:
        raise HTTPException(status_code=404, detail="Run not found")
        
    run_data = run_df.iloc[0]
    
    # Run Inference
    res = pipeline.infer_run(run_data["csv_path"], run_data["flac_path"])
    
    # Get image paths
    run_dir = os.path.dirname(run_data["csv_path"])
    images = []
    
    img_dir = os.path.join(run_dir, "images")
    if os.path.exists(img_dir):
        image_files = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))
        # We take up to 3 for visual summary
        for img in image_files[:3]:
            # Normalize for web path serving.
            parts = img.split(os.sep)
            static_path = "/static/" + "/".join(parts[1:])
            images.append(static_path)
            
    # Include audio static path
    flac_parts = run_data["flac_path"].split(os.sep)
    flac_static_path = "/static/" + "/".join(flac_parts[1:])
            
    return {
        "run_id": run_id,
        "inference": res,
        "images": images,
        "audio": flac_static_path
    }
