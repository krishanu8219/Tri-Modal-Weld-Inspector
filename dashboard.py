import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from src.inference import DefectClassifierPipeline
from src.visualizers import get_representative_images, plot_sensor_with_hotspot, plot_audio_spectrogram

# ------------------------------------------------------------------
# Page Config
# ------------------------------------------------------------------
st.set_page_config(page_title="Tri-Modal Weld Inspector", layout="wide", page_icon="🔬")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Tri-Modal Weld Inspector")
st.caption("Automated Quality Assurance Dashboard — Sensor · Acoustic · Vision")

# ------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------
@st.cache_data
def load_runs():
    if os.path.exists("runs_summary.csv"):
        return pd.read_csv("runs_summary.csv")
    return pd.DataFrame()

df_runs = load_runs()

# Pipeline (lazy)
if 'pipeline' not in st.session_state:
    try:
        st.session_state.pipeline = DefectClassifierPipeline()
    except Exception as e:
        st.session_state.pipeline = None
        st.sidebar.error(f"Pipeline load failed: {e}")

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Dataset Overview",
    "🔍 Run Inspector",
    "📈 Evaluation Report",
    "📄 Export & Data Card"
])

# Helper: Load metrics from artifacts
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

bin_metrics = load_json("artifacts/binary_metrics.json")
multi_metrics = load_json("artifacts/multiclass_metrics.json")
pipeline_metrics = load_json("artifacts/pipeline_metrics.json")

# ==================================================================
# PAGE 1: DATASET OVERVIEW
# ==================================================================
if page == "📊 Dataset Overview":
    st.header("📊 Dataset Overview")

    if df_runs.empty:
        st.warning("No runs_summary.csv found. Run `python src/data_loader.py` first.")
    else:
        # Exclude test_data for training stats
        train_pool = df_runs[df_runs["data_dir"] != "test_data"]
        test_pool = df_runs[df_runs["data_dir"] == "test_data"]

        # --- Top-level counts ---
        st.subheader("Dataset Composition")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Runs", len(df_runs))
        c2.metric("Training Pool", len(train_pool))
        c3.metric("Test Samples", len(test_pool))
        c4.metric("Complete (All Modalities)", int(df_runs["has_all_modalities"].sum()))

        st.divider()

        # --- Label Distribution ---
        st.subheader("Label Distribution")
        col_chart, col_table = st.columns([2, 1])

        label_map = {
            "00": "good_weld", "01": "excessive_penetration", "02": "burn_through",
            "06": "overlap", "07": "lack_of_fusion", "08": "excessive_convexity",
            "11": "crater_cracks"
        }

        with col_chart:
            if not train_pool.empty:
                label_counts = train_pool["label_code"].astype(str).str.zfill(2).value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#10b981' if lbl == '00' else '#ef4444' for lbl in label_counts.index]
                bars = ax.bar(label_counts.index, label_counts.values, color=colors, edgecolor='white', linewidth=0.5)
                ax.set_xlabel("Defect Code")
                ax.set_ylabel("Count")
                ax.set_title("Class Distribution (Training Pool)")
                for bar, val in zip(bars, label_counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val),
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No training data found.")

        with col_table:
            if not train_pool.empty:
                lc = train_pool["label_code"].astype(str).str.zfill(2).value_counts().sort_index()
                summary = pd.DataFrame({
                    "Code": lc.index,
                    "Label": [label_map.get(c, "unknown") for c in lc.index],
                    "Count": lc.values,
                    "Pct": (lc.values / lc.sum() * 100).round(1)
                })
                st.dataframe(summary, hide_index=True, use_container_width=True)

                # Imbalance ratio
                majority = lc.max()
                minority = lc.min()
                if minority > 0:
                    st.metric("Imbalance Ratio", f"{majority / minority:.1f}:1")

        st.divider()

        # --- Duration Statistics ---
        st.subheader("Duration & Quality Statistics")
        dc1, dc2 = st.columns(2)

        with dc1:
            if "flac_duration" in df_runs.columns:
                dur_data = df_runs["flac_duration"].dropna()
                if not dur_data.empty:
                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    ax2.hist(dur_data, bins=30, color='#3b82f6', edgecolor='white', alpha=0.8)
                    ax2.set_title("Audio Duration Distribution")
                    ax2.set_xlabel("Duration (seconds)")
                    ax2.set_ylabel("Count")
                    ax2.axvline(dur_data.mean(), color='red', linestyle='--', label=f'Mean: {dur_data.mean():.1f}s')
                    ax2.legend()
                    fig2.tight_layout()
                    st.pyplot(fig2)

        with dc2:
            if "avi_duration" in df_runs.columns:
                vid_data = df_runs["avi_duration"].dropna()
                if not vid_data.empty:
                    fig3, ax3 = plt.subplots(figsize=(6, 3))
                    ax3.hist(vid_data, bins=30, color='#f59e0b', edgecolor='white', alpha=0.8)
                    ax3.set_title("Video Duration Distribution")
                    ax3.set_xlabel("Duration (seconds)")
                    ax3.set_ylabel("Count")
                    ax3.axvline(vid_data.mean(), color='red', linestyle='--', label=f'Mean: {vid_data.mean():.1f}s')
                    ax3.legend()
                    fig3.tight_layout()
                    st.pyplot(fig3)

        # --- Missing Modalities ---
        st.subheader("Data Quality Indicators")
        q1, q2, q3 = st.columns(3)
        csv_missing = int((~df_runs["csv_valid"]).sum()) if "csv_valid" in df_runs.columns else 0
        flac_missing = int((~df_runs["flac_valid"]).sum()) if "flac_valid" in df_runs.columns else 0
        avi_missing = int((~df_runs["avi_valid"]).sum()) if "avi_valid" in df_runs.columns else 0

        q1.metric("Missing Sensor CSVs", csv_missing, delta_color="inverse")
        q2.metric("Missing Audio (FLAC)", flac_missing, delta_color="inverse")
        q3.metric("Missing Video (AVI)", avi_missing, delta_color="inverse")

        incomplete = int(len(df_runs) - df_runs["has_all_modalities"].sum())
        if incomplete > 0:
            st.warning(f"⚠️ {incomplete} runs are missing at least one modality and will produce degraded predictions.")
        else:
            st.success("✅ All runs have complete multimodal data (sensor + audio + video).")


# ==================================================================
# PAGE 2: RUN INSPECTOR (Inference + Modality Analysis)
# ==================================================================
elif page == "🔍 Run Inspector":
    st.header("🔍 Single-Run Inspector")

    valid_runs = df_runs[df_runs["has_all_modalities"] == True] if not df_runs.empty else pd.DataFrame()

    if valid_runs.empty or st.session_state.pipeline is None:
        st.warning("No valid runs or pipeline not loaded.")
    else:
        col_input, col_pred = st.columns([1, 2])
        with col_input:
            selected_run_id = st.selectbox("Select Joint Run:", valid_runs["run_id"].tolist())
            run_data = valid_runs[valid_runs["run_id"] == selected_run_id].iloc[0]
            run_dir = os.path.dirname(run_data["csv_path"])

        with col_pred:
            res = st.session_state.pipeline.infer_run(run_data["csv_path"], run_data["flac_path"])
            if res["pred_label_code"] == "00":
                st.success(f"### Assessment: **PASSED (Code 00)**\n**P(Defect):** `{res['p_defect']:.3f}`")
            else:
                st.error(f"### Assessment: **FAILED (Defect {res['pred_label_code']})**\n"
                         f"**P(Defect):** `{res['p_defect']:.3f}` | **Type Conf:** `{res['type_confidence']:.3f}`")

        st.divider()

        # --- Tri-Modal Tabs ---
        tab_sensors, tab_acoustic, tab_visual, tab_video = st.tabs([
            "📊 Sensor Telemetry", "🎧 Acoustic Signature", "📷 Visual Frames", "🎬 Video Preview"
        ])

        with tab_sensors:
            st.subheader("Numeric Sensor Subsystems")
            fig_sensor = plot_sensor_with_hotspot(run_data["csv_path"])
            if fig_sensor:
                st.pyplot(fig_sensor)
            else:
                st.info("No primary sensor streams detected.")

        with tab_acoustic:
            st.subheader("Process Spectrogram")
            if os.path.exists(run_data["flac_path"]):
                st.audio(run_data["flac_path"])
            fig_audio = plot_audio_spectrogram(run_data["flac_path"])
            if fig_audio:
                st.pyplot(fig_audio)
            else:
                st.info("Acoustic extraction unavailable.")

        with tab_visual:
            st.subheader("Structural Keyframes")
            images = get_representative_images(run_dir, num_frames=5)
            if len(images) > 0:
                cols = st.columns(len(images))
                for i, img_path in enumerate(images):
                    with cols[i]:
                        st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
            else:
                st.info("No visual frames available.")

        with tab_video:
            st.subheader("Video Preview")
            avi_path = run_data.get("avi_path", "")
            if avi_path and os.path.exists(avi_path):
                # Extract a representative frame from video
                cap = cv2.VideoCapture(avi_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    # Show 3 frames: start, middle, end
                    frame_positions = [0, total_frames // 2, max(0, total_frames - 10)]
                    frame_cols = st.columns(3)
                    labels = ["Start", "Middle", "End"]
                    for idx, (pos, label) in enumerate(zip(frame_positions, labels)):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                        ret, frame = cap.read()
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            with frame_cols[idx]:
                                st.image(frame_rgb, caption=f"{label} (Frame {pos})", use_container_width=True)
                    cap.release()
                    st.caption(f"📹 Video: {os.path.basename(avi_path)} | {total_frames} frames")
                else:
                    st.warning("Could not open AVI file.")
            else:
                st.info("No video file found for this run.")


# ==================================================================
# PAGE 3: EVALUATION REPORT
# ==================================================================
elif page == "📈 Evaluation Report":
    st.header("📈 Model Evaluation Report")

    # --- Final Score ---
    st.subheader("Combined Final Score")
    final_score = pipeline_metrics.get("final_score", None)
    bin_f1 = pipeline_metrics.get("binary_f1", bin_metrics.get("f1", None))
    type_mf1 = pipeline_metrics.get("type_macro_f1", multi_metrics.get("macro_f1", None))

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("🏆 Final Score", f"{final_score:.4f}" if final_score is not None else "N/A",
               help="0.6 × Binary_F1 + 0.4 × Type_MacroF1")
    sc2.metric("Binary F1", f"{bin_f1:.4f}" if bin_f1 is not None else "N/A")
    sc3.metric("Type Macro-F1", f"{type_mf1:.4f}" if type_mf1 is not None else "N/A")

    st.divider()

    # --- Binary Metrics ---
    st.subheader("Phase 2: Binary Classification Metrics")
    bm1, bm2, bm3, bm4, bm5 = st.columns(5)
    bm1.metric("Precision", f"{bin_metrics.get('precision', 0):.4f}")
    bm2.metric("Recall", f"{bin_metrics.get('recall', 0):.4f}")
    bm3.metric("F1 Score", f"{bin_metrics.get('f1', 0):.4f}")
    bm4.metric("ROC-AUC", f"{bin_metrics.get('roc_auc', 0):.4f}")
    bm5.metric("ECE", f"{bin_metrics.get('ece', 0):.4f}")

    threshold = pipeline_metrics.get("best_pipeline_threshold", bin_metrics.get("best_threshold", 0.5))
    st.info(f"🎯 Optimal Binary Threshold: **{threshold:.3f}** (tuned on combined FinalScore)")

    st.divider()

    # --- Multiclass Metrics ---
    st.subheader("Phase 3: Multi-Class Metrics")
    mm1, mm2, mm3 = st.columns(3)
    mm1.metric("Macro F1", f"{multi_metrics.get('macro_f1', 0):.4f}")
    mm2.metric("Weighted F1", f"{multi_metrics.get('weighted_f1', 0):.4f}")
    mm3.metric("Macro Precision", f"{multi_metrics.get('macro_precision', 0):.4f}")

    st.divider()

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix (Validation)")

    # Try to generate confusion matrix from validation data
    if os.path.exists("val_split.csv") and st.session_state.pipeline is not None:
        if st.button("🔄 Generate Confusion Matrix from Validation Set"):
            with st.spinner("Running inference on validation set..."):
                df_val = pd.read_csv("val_split.csv")
                y_true = []
                y_pred = []

                for _, row in df_val.iterrows():
                    true_label = str(row["label_code"]).zfill(2)
                    try:
                        pred = st.session_state.pipeline.infer_run(row["csv_path"], row["flac_path"])
                        y_true.append(true_label)
                        y_pred.append(pred["pred_label_code"])
                    except Exception:
                        pass

                if len(y_true) > 0:
                    from sklearn.metrics import confusion_matrix, classification_report, f1_score
                    labels = sorted(set(y_true + y_pred))

                    # Confusion matrix heatmap
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=[f"Pred {l}" for l in labels],
                                yticklabels=[f"True {l}" for l in labels],
                                ax=ax_cm, linewidths=0.5)
                    ax_cm.set_title("Confusion Matrix")
                    ax_cm.set_ylabel("True Label")
                    ax_cm.set_xlabel("Predicted Label")
                    fig_cm.tight_layout()
                    st.pyplot(fig_cm)

                    # Per-class F1 bar chart
                    st.subheader("Per-Class Precision / Recall / F1")
                    report = classification_report(y_true, y_pred, labels=labels,
                                                   output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).T
                    # Filter to only class rows
                    class_rows = report_df[report_df.index.isin(labels)]
                    if not class_rows.empty:
                        fig_pc, ax_pc = plt.subplots(figsize=(10, 4))
                        x = np.arange(len(class_rows))
                        width = 0.25
                        ax_pc.bar(x - width, class_rows["precision"], width, label="Precision", color="#3b82f6")
                        ax_pc.bar(x, class_rows["recall"], width, label="Recall", color="#10b981")
                        ax_pc.bar(x + width, class_rows["f1-score"], width, label="F1", color="#f59e0b")
                        ax_pc.set_xticks(x)
                        ax_pc.set_xticklabels([f"Code {l}" for l in class_rows.index])
                        ax_pc.set_ylabel("Score")
                        ax_pc.set_title("Per-Class Performance")
                        ax_pc.legend()
                        ax_pc.set_ylim(0, 1.1)
                        fig_pc.tight_layout()
                        st.pyplot(fig_pc)

                    st.dataframe(class_rows[["precision", "recall", "f1-score", "support"]].round(4),
                                 use_container_width=True)

                    # --- Correctly vs Incorrectly Predicted Examples ---
                    st.divider()
                    st.subheader("Example Predictions (Correct ✅ vs Incorrect ❌)")

                    correct_examples = []
                    incorrect_examples = []

                    for i in range(len(y_true)):
                        row_data = df_val.iloc[i]
                        entry = {
                            "run_id": row_data["run_id"],
                            "true": y_true[i],
                            "pred": y_pred[i],
                            "csv_path": row_data["csv_path"]
                        }
                        if y_true[i] == y_pred[i]:
                            correct_examples.append(entry)
                        else:
                            incorrect_examples.append(entry)

                    ex_col1, ex_col2 = st.columns(2)

                    with ex_col1:
                        st.markdown("#### ✅ Correctly Predicted")
                        for ex in correct_examples[:3]:
                            with st.container(border=True):
                                st.markdown(f"**Run:** `{ex['run_id']}`")
                                st.markdown(f"True: `{ex['true']}` → Pred: `{ex['pred']}`")

                    with ex_col2:
                        st.markdown("#### ❌ Incorrectly Predicted")
                        if len(incorrect_examples) > 0:
                            for ex in incorrect_examples[:3]:
                                with st.container(border=True):
                                    st.markdown(f"**Run:** `{ex['run_id']}`")
                                    st.markdown(f"True: `{ex['true']}` → Pred: `{ex['pred']}`")
                        else:
                            st.success("No misclassifications found in the validation set.")

    else:
        st.info("Validation split (`val_split.csv`) not found or pipeline not loaded. "
                "Train models first to see evaluation results.")


# ==================================================================
# PAGE 4: EXPORT & DATA CARD
# ==================================================================
elif page == "📄 Export & Data Card":
    st.header("📄 Export & Data Card")

    # --- Download Buttons ---
    st.subheader("📥 Exportable Reports")

    col_dl1, col_dl2, col_dl3 = st.columns(3)

    with col_dl1:
        if os.path.exists("runs_summary.csv"):
            st.download_button(
                "⬇️ Download runs_summary.csv",
                data=open("runs_summary.csv", "rb").read(),
                file_name="runs_summary.csv",
                mime="text/csv"
            )
    with col_dl2:
        if os.path.exists("submission.csv"):
            st.download_button(
                "⬇️ Download submission.csv",
                data=open("submission.csv", "rb").read(),
                file_name="submission.csv",
                mime="text/csv"
            )
    with col_dl3:
        # Combined metrics export
        combined_metrics = {
            "binary": bin_metrics,
            "multiclass": multi_metrics,
            "pipeline": pipeline_metrics
        }
        st.download_button(
            "⬇️ Download All Metrics (JSON)",
            data=json.dumps(combined_metrics, indent=2),
            file_name="all_metrics.json",
            mime="application/json"
        )

    st.divider()

    # --- Data Card ---
    st.subheader("📋 Model Data Card")
    if os.path.exists("DataCard.md"):
        with open("DataCard.md") as f:
            st.markdown(f.read())
    else:
        st.info("No DataCard.md found. Create one to display here.")

    st.divider()

    # --- EDA Plots ---
    st.subheader("📊 EDA Report Artifacts")
    eda_dir = "eda_reports"
    if os.path.exists(eda_dir):
        eda_files = glob.glob(os.path.join(eda_dir, "*.png"))
        if eda_files:
            cols = st.columns(min(len(eda_files), 3))
            for i, f in enumerate(eda_files):
                with cols[i % len(cols)]:
                    st.image(f, caption=os.path.basename(f), use_container_width=True)
        # CSV stats
        stats_csv = os.path.join(eda_dir, "sensor_summary_stats_sample.csv")
        if os.path.exists(stats_csv):
            st.markdown("**Sensor Summary Statistics (Sampled)**")
            st.dataframe(pd.read_csv(stats_csv), use_container_width=True)
    else:
        st.info("Run `python src/eda.py` to generate EDA reports.")
