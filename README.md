# Tri-Modal Weld Inspector — AI-Powered Weld Defect Detection

> **Team AI-Beats** | Therness Hackathon 2025

An end-to-end machine learning pipeline for automated weld defect detection and classification using audio and visual data from industrial welding processes.

---

## Results

| Metric | Score |
|---|---|
| **Final Score** | **0.869** |
| Binary F1 | 0.902 |
| Binary Precision | 0.867 |
| Binary Recall | 0.940 |
| ROC AUC | 0.934 |
| Type Macro F1 | 0.819 |

*Scoring formula: `FinalScore = 0.6 x Binary_F1 + 0.4 x Type_MacroF1`*

### Per-Class Defect Type Performance (OOF)

| Code | Defect Type | F1 |
|---|---|---|
| 00 | Good weld | 0.860 |
| 01 | Excessive penetration | 0.763 |
| 02 | Burn through | 0.743 |
| 06 | Overlap | 0.933 |
| 07 | Lack of fusion | 0.709 |
| 08 | Excessive convexity | 0.921 |
| 11 | Crater cracks | 0.805 |

---

## Architecture

```
Raw Welding Sample
  audio.wav + images/*.png + (sensor.csv)
         |                    |
   Audio Features       Vision Features
    (180 dims)            (128 dims)
         |                    |
         +-------- + --------+
                   |
          Concatenate (308 features)
                   |
          2-Pass Feature Selection (93 kept)
                   |
       Stage 1: Binary Gate
       XGBoost + Isotonic Calibration
       Threshold: 0.455 (Youden-J)
                   |
            Good <-+-> Defective
                   |
       Stage 2: 7-Class Multiclass
       XGBoost (can override to "00")
                   |
           Final prediction
```

**Key design:** The multiclass model includes the "00" (good) class, allowing it to override binary false positives -- a major contributor to our precision improvement.

---

## Feature Engineering

### Audio Features (180 dimensions)

Physics-based features that capture weld defect acoustic signatures:

- **13 MFCCs** -- means, standard deviations, delta, and delta-delta coefficients
- **Sub-band energy ratios** -- configuration-invariant frequency analysis
- **Spectral entropy** -- acoustic disorder as a defect marker
- **Spectral centroid, rolloff, bandwidth, contrast, flatness** -- multi-scale frequency descriptors
- **Zero-crossing rate & RMS energy** -- temporal energy patterns
- **Temporal pooling** -- mean, std, percentiles across time frames

### Vision Features (128 dimensions)

Weld bead geometry and surface quality from keyframe extraction:

- **Bead width & centre brightness** -- physical geometry measurement
- **Laplacian variance** -- surface roughness quantification
- **GLCM texture features** -- homogeneity vs. pitting analysis
- **Temporal frame differencing** -- motion energy during welding
- **Edge orientation entropy** -- regularity of bead edges

### Feature Selection

- **308 features** extracted per sample
- **93 features** selected via two-pass importance ranking (top 80% cumulative importance)
- Removes noise features that could bias toward majority classes

---

## Training Pipeline

- **Cross-validation:** GroupKFold(5) on `config_folder` -- prevents leakage from similar weld setups
- **Hyperparameter search:** 60-iteration `RandomizedSearchCV` scored on Macro F1
- **Calibration:** Isotonic regression on out-of-fold predictions
- **Threshold optimization:** Youden J statistic (0.455) from CV probabilities
- **Parallel training:** Binary and multiclass models train simultaneously via `ThreadPoolExecutor`
- **Reproducibility:** Fixed seed (42), saved splits, cached features as `.npz`

---

## Quick Start

### Prerequisites

```bash
python 3.10+
pip install -r requirements.txt
```

### 1. Cache Features

```bash
python cache_features.py
```

Extracts audio + vision features for all training samples. Saves to `artifacts/feature_cache.npz`.

### 2. Train Models

```bash
python train_audiovisual.py
```

Trains binary + multiclass XGBoost models with parallel execution. Saves models and metrics to `artifacts/`.

### 3. Run Inference

```bash
python run_inference.py --test-dir ../test_data --output submission.csv
```

Generates submission CSV with `sample_id`, `pred_label_code`, `p_defect`.

### 4. Launch Dashboard

```bash
# Start API backend
python api.py

# Start frontend (in a separate terminal)
cd next-dashboard && npm run dev
```

Dashboard runs at `http://localhost:3000` with FastAPI backend at `http://localhost:8000`.

---

## Dashboard

Built with **Next.js** + **FastAPI**, the dashboard provides:

- **Overview** -- Dataset statistics, label distributions, data quality indicators
- **Sample Explorer** -- Drill into any sample: audio waveform, video frames, sensor traces
- **Evaluation** -- Binary metrics, confusion matrices, per-class F1 breakdown, ROC curves
- **Data Models** -- Pipeline architecture visualization, feature importance, fit diagnostics
- **Data Card** -- Dataset documentation, preprocessing assumptions, limitations

---

## Project Structure

```
api.py                    # FastAPI backend serving data & metrics
cache_features.py         # Feature extraction & caching pipeline
train_audiovisual.py      # Model training with parallel execution
run_inference.py          # Test inference & submission generation
diagnose_fit.py           # Overfitting diagnostics
requirements.txt          # Python dependencies
submission.csv            # Latest submission output
PITCH_AND_QA.md           # 4-min pitch script & Q&A prep
DataCard.md               # Dataset documentation
artifacts/                # Saved models, metrics, feature cache
next-dashboard/           # Next.js frontend dashboard
src/                      # Core source modules
sampleData/               # Sample data for dashboard preview
```

---

## Technical Decisions

| Decision | Rationale |
|---|---|
| XGBoost over deep learning | 2,330 samples too small for CNNs/transformers; XGBoost excels on tabular features |
| Audio-visual only (no sensor) | Only 2/90 test samples have sensor CSV; AV pipeline works universally |
| 7-class multiclass (inc. "00") | Allows model to override binary false positives |
| GroupKFold by config_folder | Prevents leakage from similar weld configurations |
| Isotonic calibration | Provides well-calibrated probabilities for reliable confidence scores |
| Parallel training | Binary + multiclass train simultaneously, halving training time |

---

## Team

**AI-Beats** -- Therness Hackathon 2025

---

## License

This project was developed for the Therness Hackathon competition.
