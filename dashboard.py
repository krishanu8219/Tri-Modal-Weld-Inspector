import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from src.inference import DefectClassifierPipeline
from src.visualizers import get_representative_images, plot_sensor_with_hotspot, plot_audio_spectrogram

st.set_page_config(page_title="Tri-Modal Weld Inspector", layout="wide")

st.title("Tri-Modal Weld Inspector")
st.markdown("Automated Quality Assurance Demo leveraging Sensor, Acoustic, and Vision streams.")

if 'pipeline' not in st.session_state:
    try:
        st.session_state.pipeline = DefectClassifierPipeline()
        st.success("Loaded Tri-Modal Classification Pipeline.")
    except Exception as e:
        st.error(f"Failed to load pipelines: {e}")

# Load runs
if os.path.exists("runs_summary.csv"):
    df_runs = pd.read_csv("runs_summary.csv")
    valid_runs = df_runs[df_runs["has_all_modalities"] == True]
else:
    st.warning("No runs_summary.csv found.")
    valid_runs = pd.DataFrame()

if not valid_runs.empty and 'pipeline' in st.session_state:
    st.markdown("---")
    
    col_input, col_pred = st.columns([1, 2])
    with col_input:
        selected_run_id = st.selectbox("Select Joint Run:", valid_runs["run_id"].tolist())
        run_data = valid_runs[valid_runs["run_id"] == selected_run_id].iloc[0]
        
        # Derive run directory from the csv_path structure
        # Assumes structure: sampleData/run_id/run_id.csv
        run_dir = os.path.dirname(run_data["csv_path"]) 
        
    with col_pred:
        # Run inference instantly on selection
        res = st.session_state.pipeline.infer_run(run_data["csv_path"], run_data["flac_path"])
        
        if res["pred_label_code"] == "00":
            st.success(f"### Assessment: **PASSED (Code 00)**\n**P(Defect):** `{res['p_defect']:.3f}`")
        else:
            st.error(f"### Assessment: **FAILED (Defect {res['pred_label_code']})**\n**P(Defect):** `{res['p_defect']:.3f}` | **Type Conf:** `{res['type_confidence']:.3f}`")
            
    st.markdown("---")
    
    # --- Tri-Modal Output Breakdown ---
    st.header("Modality Analysis")
    
    tab_sensors, tab_acoustic, tab_visual = st.tabs(["📊 Sensor Telemetry", "🎧 Acoustic Signature", "📷 Visual Frames"])
    
    with tab_sensors:
        st.subheader("Numeric Sensor Subsystems")
        st.markdown("Analyzes high-frequency continuous structural variables like Voltage, Current, and Wire Feed.")
        fig_sensor = plot_sensor_with_hotspot(run_data["csv_path"])
        if fig_sensor:
            st.pyplot(fig_sensor)
        else:
            st.info("No primary sensor streams detected.")
            
    with tab_acoustic:
        st.subheader("Process Spectrogram")
        st.markdown("Evaluates ambient sonic energy emitted during the weld progression.")
        if os.path.exists(run_data["flac_path"]):
            st.audio(run_data["flac_path"])
        
        fig_audio = plot_audio_spectrogram(run_data["flac_path"])
        if fig_audio:
            st.pyplot(fig_audio)
        else:
            st.info("Acoustic extraction unavailable.")
            
    with tab_visual:
        st.subheader("Structural Keyframes")
        st.markdown("Direct optical views extracted from auxiliary top-down cameras.")
        images = get_representative_images(run_dir, num_frames=3)
        
        if len(images) > 0:
            cols = st.columns(len(images))
            for i, img_path in enumerate(images):
                with cols[i]:
                    st.image(img_path, caption=f"Frame {os.path.basename(img_path)}", width='stretch')
        else:
            st.info("No visual frames available in this run's configuration folder.")

