# Tri-Modal Weld Inspector: Model Data Card

## 1. Sample Definition
A "sample" in this pipeline represents a single continuous welding run. It is fundamentally multimodal and is constructed from three distinct but aligned data streams:
*   **Sensor Telemetry (`sensor.csv`)**: Time-series structural dynamics (e.g., Primary Weld Current, Voltage).
*   **Acoustic Signature (`weld.flac`)**: High-frequency audio recorded during the welding process.
*   **Visual Keyframes (`images/*.jpg`)**: Representative images taken during the weld run.

A valid sample is expected to be contained within a single directory prefix (e.g., `sample_1234/`) with normalized filenames. The target variable is a 2-digit defect code where `"00"` signifies a nominal (passing) weld.

## 2. Preprocessing Assumptions
To guarantee generalization and prevent dataset leakage, the following strict preprocessing boundaries are enforced:
*   **Leakage Prevention**: Filenames, `run_id` tokens, and categorical metadata such as "Part No" are explicitly excluded from the feature matrix.
*   **Sensor Extraction**: Time-series are aggregated globally per-run using robust statistical descriptors (mean, std, percentiles, IQR, range) and localized end-of-run statistics to capture trailing craters.
*   **Acoustic Extraction**: `librosa` computes 13 MFCCs, their deltas, spectral centroids, rolloff, crossing rates, and RMS energy, outputting aggregated pooling vectors.
*   **Visual Extraction**: Representative keyframes are uniformly sampled, passed through OpenCV descriptors (Color Histograms & Canny Edge density), and mean-pooled across time.

## 3. Model Architecture & Outputs
The inference engine is a **Chained XGBoost Pipeline**:
1.  **Binary Gate**: First predicts `P(Defect)`. If `P(Defect) < Threshold`, the weld is classified as `"00"`.
2.  **Multiclass Assessor**: If a defect is detected, a secondary XGBoost model ranks the specific defect categories.

### Confidence Definition
*   **Defect Probability (`p_defect`)**: The calibrated probability output from the binary XGBoost classifier.
*   **Classification Confidence (`type_confidence`)**: The relative probability output from the multiclass XGBoost classifier indicating certainty of the specific defect *conditioned* on the weld being defective.

*Note: The binary threshold is optimized end-to-end against a combined metric: `0.6 * Binary F1 + 0.4 * Type Macro-F1`.*

## 4. Strengths, Weaknesses & Failure Cases

### Strengths
*   **Robust to Missing Modalities**: Late-fusion array concatenation replaces missing modalities with zero-vectors gracefully.
*   **Computationally Lightweight**: By aggregating temporal streams into descriptive global features and using OpenCV for visuals, inference operates in milliseconds without deep learning heavy-lifts.

### Weaknesses
*   **Loss of Exact Temporal Alignment**: Global pooling blurs the exact micro-second temporal correlation between a voltage spike and an acoustic pop.
*   **Class Imbalance Fragility**: Certain rare defects may remain hard to classify despite balanced class weightings during training.

### Common Failure Cases
*   **False Alarms on Minor Variations**: Nominal welds containing safe, expected "pops" (e.g., ignition start) might trigger the binary model if the audio energy exceeds the nominal threshold.
*   **Crater vs. Burn-through Confusion**: Both defects often manifest at the very end of a run with similar thermal/structural drops. The multiclass model may struggle to distinguish them without deeper localized context.
