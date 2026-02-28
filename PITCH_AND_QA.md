# Tri-Modal Weld Inspector — 4-Minute Pitch Script

---

## SLIDE 1: The Problem (0:00 – 0:30)

> "Weld defects kill. A single missed crack in a pipeline, a bridge joint, or a pressure vessel can lead to catastrophic failure. Today, weld inspection is dominated by manual visual checks — slow, subjective, and error-prone.
>
> Our challenge: **given raw sensor data, audio recordings, and video from a welding process, can we automatically detect defects AND classify their exact type — in real time, with confidence scores?**
>
> The dataset has **2,330 labeled welding runs** across 7 classes — one good and six distinct defect types — with real industrial multimodal data."

---

## SLIDE 2: Our Approach — Physics-Based Feature Engineering (0:30 – 1:30)

> "Instead of throwing a black-box deep learning model at this problem, we took a **physics-first approach**. We asked: *what does a weld defect actually sound like, and look like?*
>
> **Audio (136 features):** We don't use generic MFCCs. We extract features that capture *defect signatures*:
> - **Sub-band energy ratios** — porosity creates high-frequency bursts
> - **Spectral entropy** — defects create acoustic disorder
> - **Kurtosis** — spatter produces impulsive pops
> - **Temporal variance** — arc instability means the sound is non-stationary
> - **Harmonic-to-noise ratio** — a clean weld has a steady hum; defects break it
>
> **Image/Video (128 features):** We extract weld bead geometry and surface quality:
> - **Laplacian variance** — surface roughness of the bead
> - **Bead width consistency** — defects cause irregular geometry across frames
> - **Spatter detection** — bright scattered spots outside the weld zone
> - **GLCM texture** — surface homogeneity vs. pitting
> - **Temporal frame differencing** — motion energy during welding
> - **Edge orientation entropy** — regular edges = good weld
>
> Total: **264 physics-informed features** per sample. Every feature has a physical justification. No black boxes."

---

## SLIDE 3: Model Architecture — Chained XGBoost Pipeline (1:30 – 2:30)

> "Our model is a **two-stage chained classifier**:
>
> **Stage 1 — Binary Gate:** Is this weld defective or good?
> - XGBoost binary classifier with **isotonic calibration** for reliable probability outputs
> - Threshold optimized via **Youden's J statistic** from cross-validation — no test-label peeking
> - Result: **Binary F1 = 0.730**, AUC = 0.821, **Recall = 1.000** (zero missed defects)
>
> **Stage 2 — Multiclass Assessor:** If defective, which of the 6 defect types?
> - Second XGBoost model trained only on defect samples
> - Handles class imbalance (overlap has 155 samples vs. excessive penetration with 479)
> - Result: **Type Macro F1 = 0.654**
>
> **Key design decisions:**
> - **GroupKFold on configuration folders** — prevents data leakage from similar weld setups
> - **Two-pass feature selection** — keeps only the 79 most important features (out of 264), reducing noise
> - **Hyperparameter search** with 40-iteration RandomizedSearchCV
> - **Multiprocessing** throughout — cache building, feature extraction, and fold training all parallelized
>
> **Combined Final Score: 0.700** (formula: 0.6 × Binary_F1 + 0.4 × Type_MacroF1)"

---

## SLIDE 4: Dashboard, Inference & Results (2:30 – 3:30)

> "We built an end-to-end production pipeline:
>
> **One-command inference:** `python run_inference.py --test-dir test_data` — reads any sample folder, auto-detects available modalities, outputs submission CSV.
>
> **Graceful fallback:** If a test sample is missing sensor CSV (which happens — only 2 out of 90 test samples have it), the pipeline automatically falls back to audio-visual mode. No crashes, no manual intervention.
>
> **Live dashboard** — Next.js frontend + FastAPI backend:
> - Dataset overview: label distributions, sample counts, data quality
> - Per-sample drill-down: audio waveform, video frames, sensor traces
> - Model predictions with confidence bars
> - Error analysis: false positives, false negatives, confusion patterns
>
> **Per-class performance highlights (on 115 unseen test samples):**
>
> | Defect Type | F1 Score |
> |---|---|
> | Overlap (06) | 1.000 |
> | Burn through (02) | 0.900 |
> | Excessive convexity (08) | 0.800 |
> | Good weld (00) | 0.673 |
> | Lack of fusion (07) | 0.632 |
> | Crater cracks (11) | 0.400 |
> | Excessive penetration (01) | 0.191 |
>
> Overlap detection is perfect. The weakest class (excessive penetration, F1 = 0.191) is the most physically subtle — it produces a slightly wider bead that is easily confused with good welds. Crater cracks (F1 = 0.400) are rare and occur only at weld termination zones."

---

## SLIDE 5: Why This Approach Wins (3:30 – 4:00)

> "Three things set us apart:
>
> 1. **Physics-grounded features** — not a black box. Every feature maps to a physical defect mechanism. An engineer can inspect and trust the model.
>
> 2. **Leakage-proof evaluation** — GroupKFold by configuration folder ensures our scores reflect real-world generalization, not data memorization.
>
> 3. **Production-ready** — one-command training, one-command inference, live dashboard, graceful fallback for missing modalities. This isn't a notebook prototype — it's deployable.
>
> Thank you."

---
---

# Possible Q&A — Preparation Guide

## Category 1: Model & Architecture

### Q1: Why XGBoost instead of deep learning (CNNs, transformers)?
**A:** Three reasons:
1. **Dataset size** — 2,330 samples is too small for deep learning to generalize without massive overfitting. XGBoost excels with tabular features on small-medium datasets.
2. **Interpretability** — XGBoost gives feature importances. We can tell an engineer *which* acoustic feature triggered a defect flag. A CNN is a black box.
3. **Speed** — Training takes ~2 minutes. Inference is <1 second per sample. No GPU required.
4. If we had 100K+ samples, a transformer on raw audio + video would be worth exploring.

### Q2: Why a two-stage pipeline instead of a single 7-class model?
**A:** The binary vs. multiclass split optimizes each task independently:
- The binary gate has **perfect recall (1.000)** — we miss zero defects on the test set. Precision is 0.575, meaning we over-flag some good welds, but safety-critical applications prefer false alarms over missed defects.
- The multiclass model only runs on predicted defects, so it sees a much cleaner class distribution.
- The combined score formula (0.6 × Binary + 0.4 × Macro) also rewards this separation — binary accuracy is weighted more.

### Q3: How do you handle class imbalance?
**A:** Multiple strategies:
- **GroupKFold** ensures each fold has representative class distribution.
- **Feature selection** removes noise features that could bias toward majority classes.
- **Hyperparameter tuning** is scored on Macro F1, which treats all classes equally.
- We experimented with sample weights and class_weight="balanced" but found they degraded performance — the XGBoost regularization (reg_alpha, reg_lambda) handles imbalance well implicitly.

### Q4: What is the Youden-J threshold and why use it?
**A:** Youden's J = Sensitivity + Specificity − 1. It finds the threshold that maximizes the trade-off between true positive rate and false positive rate. It's derived from cross-validation probabilities — never from test data — so there's no label leakage. Our optimal threshold is 0.69, not the default 0.5.

### Q5: What does isotonic calibration do?
**A:** XGBoost raw probabilities are often overconfident. Isotonic regression maps raw probabilities to calibrated ones using a non-parametric monotone function fitted on OOF (out-of-fold) predictions. After calibration, if the model says "80% chance of defect," it really is defective ~80% of the time.

---

## Category 2: Features & Engineering

### Q6: Why not just use raw MFCCs?
**A:** Raw MFCC means encode the *background environment* (factory hum, distance from mic) more than the defect. Different weld configurations have different ambient sounds. Instead, we use:
- **MFCC standard deviations** (variability = instability)
- **Sub-band energy ratios** (relative, not absolute — config-invariant)
- **Spectral entropy** (disorder is a defect signature regardless of environment)

### Q7: How do your image features differ from standard approaches?
**A:** Standard approaches use color histograms or pre-trained CNN embeddings. Color histograms vary by camera and lighting (config-specific). CNN embeddings require fine-tuning with more data. Our features measure:
- **Physical bead geometry** (width, centre brightness) — invariant to camera setup
- **Surface texture** (GLCM, Laplacian) — captures pitting, roughness
- **Temporal consistency** (frame differencing) — defects cause irregular motion

### Q8: Why 264 features but only 79 selected?
**A:** We extract 264 to cast a wide net, then use a two-pass importance-based selection:
1. Train a preliminary XGBoost, rank features by importance.
2. Keep features whose cumulative importance reaches 80%.
This removes noise and reduces overfitting. The 79 surviving features are the ones that genuinely discriminate defect types.

### Q9: What about sensor features? Why audio-visual only?
**A:** Only 2 out of 90 test samples have sensor CSVs. Building a model that relies on sensor data would fail on 88 samples at test time. Our audio-visual pipeline works universally. We do have a tri-modal model for the rare cases where sensor data exists, but the AV model is our primary submission.

---

## Category 3: Evaluation & Leakage

### Q10: What is GroupKFold and why is it critical?
**A:** Welding runs from the same configuration folder share similar equipment settings, materials, and ambient conditions. If we split randomly, training and validation might both contain runs from the same folder — the model would learn configuration, not defect patterns. GroupKFold ensures all runs from one configuration folder stay in the same fold. This gives a realistic estimate of performance on unseen setups.

### Q11: How confident are you in the 0.700 final score?
**A:** It's our real test score on 115 unseen samples — not a CV estimate:
- GroupKFold prevents leakage from similar configs during training.
- The threshold (0.69) is derived from CV, not test data.
- Feature selection is done inside the CV loop conceptually (two-pass on training data only).
- The gap from CV (0.893) to test (0.700) reflects the challenge of generalizing to truly unseen weld configurations — a realistic industrial scenario.

### Q12: What's the gap between CV score and actual test score?
**A:** CV score was 0.893; real test score is 0.700 — a 19-point gap. This is larger than expected, driven by:
- Test data contains weld configurations not seen in training.
- Class distribution in test differs significantly (e.g., more good welds causing higher false-positive rate).
- Binary precision dropped (0.575) while recall stayed perfect (1.000), indicating the model is conservative — it flags borderline cases as defective.
- Per-class: some classes generalized well (Overlap 1.000, Burn Through 0.900) while others struggled (Excessive Penetration 0.191).

---

## Category 4: Engineering & Production

### Q13: How fast is inference?
**A:** ~1 second per sample on a MacBook (no GPU). For 90 test samples, total inference is under 2 minutes. Feature extraction (audio + image) takes most of that time; the XGBoost prediction itself is <1ms.

### Q14: How does the fallback mechanism work?
**A:** The pipeline checks if `sensor.csv` exists. If yes → tri-modal (sensor + audio + image, 319 features). If no → audio-visual mode (264 features, different model). Both models are pre-trained and saved. The fallback is seamless — no code changes needed at test time.

### Q15: Can this run in production?
**A:** Yes:
- **No GPU required** — XGBoost + librosa + OpenCV are CPU-only.
- **Dependencies** are standard: scikit-learn, xgboost, librosa, opencv, numpy.
- **Model artifacts** are joblib files (~1MB total).
- **API** is already built (FastAPI) with CORS support for frontend integration.
- For real deployment, you'd add: input validation, logging, monitoring, and a message queue for batch processing.

### Q16: What's in the dashboard?
**A:** Built with Next.js + FastAPI:
- **Overview tab**: Dataset stats, label distributions, data quality indicators
- **Sample explorer**: Drill into any sample — see audio waveform, video frames, sensor traces
- **Prediction viewer**: Model outputs with confidence scores, color-coded by correctness
- **Error analysis**: Confusion matrix, worst predictions, failure patterns

---

## Category 5: Weaknesses & Future Work

### Q17: What are the main failure modes?
**A:** 
1. **Excessive penetration misclassification (F1 = 0.191)** — this defect produces a slightly wider bead that looks and sounds very similar to a good weld. The model confuses it with other classes.
2. **False alarms on good welds** — binary precision is 0.575 (34 false positives out of 69 non-defective). The model is safety-conservative: it flags borderline cases as defective rather than missing real defects.
3. **Crater cracks (F1 = 0.400)** — low support (rare defect type) and subtle termination-zone signatures make this hard to learn.
4. **Novel configurations** — test weld setups not seen in training cause feature distribution shift, explaining the CV-to-test gap.

### Q18: What would you do with more time?
**A:** 
1. **Temporal segmentation** — instead of global pooling, split audio/video into segments (start/middle/end of weld). Crater cracks happen at the end; overlap at the edges.
2. **MFCC delta features** — rate of change of spectral features captures transient defect events.
3. **Stacking ensemble** — combine XGBoost with LightGBM and ExtraTrees via a meta-learner.
4. **More image features** — LBP texture, Gabor filters, Hu moments for shape analysis.
5. **With 10× more data** — fine-tune a pre-trained audio model (e.g., AST) on weld sounds.

### Q19: Why not use the video (.avi) directly?
**A:** We do use it — the `images/` folder contains keyframes extracted from the video. Processing raw AVI would require video decoding and temporal modeling (3D CNNs or video transformers), which need more data and compute than justified. Our keyframe-based features capture the essential spatial information.

### Q20: How do you ensure reproducibility?
**A:** 
- Fixed random seeds (42) everywhere.
- Split files saved as CSVs (`train_split.csv`, `val_split.csv`).
- Feature cache saved as `.npz` for deterministic training.
- All hyperparameters logged in JSON metric files.
- Git-tracked codebase — one command to reproduce: `python cache_features.py && python train_audiovisual.py`.

---

## Quick Stats Cheat Sheet (for rapid answers)

| Metric | Value |
|---|---|
| Total training samples | 1,551 (1,240 train + 311 val) |
| Test samples | 115 |
| Total raw features | 264 (136 audio + 128 image) |
| Selected features | 79 (top 80% cumulative importance) |
| Binary F1 (test) | 0.730 |
| Binary AUC (test) | 0.821 |
| Binary Precision (test) | 0.575 |
| Binary Recall (test) | 1.000 |
| Binary Accuracy (test) | 0.704 |
| Type Macro F1 (test) | 0.654 |
| Final Score (test) | 0.700 |
| Best class F1 | Overlap (06) — 1.000 |
| Worst class F1 | Excessive penetration (01) — 0.191 |
| Binary threshold | 0.69 (Youden-J) |
| Training time | ~2 min (8-core CPU) |
| Inference time | ~1 sec/sample |
| CV strategy | GroupKFold(5) on config_folder |
| Model | Chained XGBoost (binary → multiclass) |
