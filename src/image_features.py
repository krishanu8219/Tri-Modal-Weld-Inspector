"""
Physics-based image/video feature extraction for weld defect detection.

Key insight: Instead of color histograms (which vary by camera/lighting = config-specific),
we extract features that capture WELD BEAD GEOMETRY and SURFACE IRREGULARITY,
which are physical properties of defect formation:

  - Porosity / crater cracks: irregular surface texture, pitting, bright spots
  - Burn through: dark holes, high local contrast in weld zone
  - Lack of fusion: irregular bead edges, low edge sharpness on one side
  - Spatter: bright scattered spots outside central weld bead
  - Overlap / excessive penetration: bead width inconsistency across frames

We use:
  1. Laplacian variance   — surface roughness / sharpness of bead edges
  2. Bead width variance  — consistency of weld geometry across frames
  3. Temporal frame differencing — changes during welding process
  4. Spatter detection    — bright spots outside weld zone
  5. GLCM texture features— surface homogeneity/contrast
  6. Edge orientation entropy — regularity of edge directions
  7. Bead centre brightness profile statistics
  8. Frame-to-frame motion energy

Output: 128-dimensional vector
"""

import os
import glob
import logging
import warnings
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

IMAGE_FEAT_DIM = 128


def _safe(arr):
    return np.nan_to_num(np.array(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _glcm_features(gray, distances=(1, 3), angles=(0, np.pi/4, np.pi/2)):
    """
    Compute GLCM (Grey-Level Co-occurrence Matrix) features:
    contrast, dissimilarity, homogeneity, energy, correlation.
    These capture surface TEXTURE independent of absolute brightness.
    """
    # Quantize to 32 levels for speed
    g = (gray // 8).astype(np.uint8)
    n = 32
    feats = []
    for d in distances:
        for angle in angles:
            dx = int(round(d * np.cos(angle)))
            dy = int(round(d * np.sin(angle)))
            # Build co-occurrence matrix
            g1 = g[max(-dy,0):g.shape[0]-max(dy,0), max(-dx,0):g.shape[1]-max(dx,0)]
            g2 = g[max(dy,0):g.shape[0]-max(-dy,0), max(dx,0):g.shape[1]-max(-dx,0)]
            glcm = np.zeros((n, n), dtype=np.float32)
            np.add.at(glcm, (g1.ravel(), g2.ravel()), 1)
            glcm /= (glcm.sum() + 1e-9)

            i, j = np.mgrid[0:n, 0:n]
            contrast     = float(np.sum(glcm * (i - j) ** 2))
            homogeneity  = float(np.sum(glcm / (1 + (i - j) ** 2)))
            energy       = float(np.sum(glcm ** 2))
            mu_i = np.sum(i * glcm)
            mu_j = np.sum(j * glcm)
            std_i = np.sqrt(np.sum(glcm * (i - mu_i) ** 2)) + 1e-9
            std_j = np.sqrt(np.sum(glcm * (j - mu_j) ** 2)) + 1e-9
            correlation = float(np.sum(glcm * (i - mu_i) * (j - mu_j)) / (std_i * std_j))

            feats.extend([contrast, homogeneity, energy, correlation])
    return feats   # 2 distances × 3 angles × 4 = 24 values


def _bead_profile_features(gray):
    """
    Find the brightest horizontal stripe (weld bead) and extract:
    - Bead width standard deviation across rows (geometry consistency)
    - Brightness profile statistics along the bead centre
    - Asymmetry of brightness left vs right of bead (fusion quality)
    """
    h, w = gray.shape
    # Vertical profile: find brightest row band (weld pool / bead)
    row_brightness = gray.mean(axis=1)
    bead_row = int(np.argmax(row_brightness))

    # Bead region: ±15% of height around brightest row
    margin = max(int(h * 0.15), 5)
    top    = max(0, bead_row - margin)
    bot    = min(h, bead_row + margin)
    bead   = gray[top:bot, :]

    # Bead width per row (pixels above 60th percentile brightness)
    thresh = np.percentile(bead, 60)
    widths = [(row > thresh).sum() for row in bead]
    widths = np.array(widths, dtype=float)

    width_mean = float(np.mean(widths))
    width_std  = float(np.std(widths))
    width_cv   = width_std / (width_mean + 1e-9)  # coefficient of variation (scale-free)

    # Brightness profile along bead centre
    centre_row = bead[len(bead)//2, :]
    profile_std  = float(np.std(centre_row))
    profile_kurt = float(np.mean(((centre_row - centre_row.mean()) /
                                  (centre_row.std() + 1e-9)) ** 4))

    # Left/right asymmetry (lack of fusion shows asymmetric bead)
    midcol = w // 2
    left_mean  = float(centre_row[:midcol].mean())
    right_mean = float(centre_row[midcol:].mean())
    asymmetry  = abs(left_mean - right_mean) / (max(left_mean, right_mean) + 1e-9)

    return [width_mean, width_std, width_cv, profile_std, profile_kurt, asymmetry]


def _spatter_features(gray, bead_mask_fraction=0.3):
    """
    Detect bright spots (spatter) OUTSIDE the central weld bead zone.
    High spatter = arc instability = likely defect.
    """
    h, w = gray.shape
    top    = int(h * bead_mask_fraction)
    bot    = int(h * (1 - bead_mask_fraction))
    peripheral = np.vstack([gray[:top, :], gray[bot:, :]])

    # Adaptive threshold for bright spots
    thresh = np.percentile(peripheral, 95)
    spatter_mask = peripheral > thresh
    spatter_count  = float(spatter_mask.sum())
    spatter_density = spatter_count / (peripheral.size + 1e-9)

    # Connectivity of spatter (small isolated = true spatter vs large = reflection)
    # Simple proxy: ratio of max blob size to total spatter pixels
    contours, _ = cv2.findContours(spatter_mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        areas = [cv2.contourArea(c) for c in contours]
        max_area = float(max(areas))
        n_blobs  = float(len(areas))
        avg_area = float(np.mean(areas))
    else:
        max_area = 0.0
        n_blobs  = 0.0
        avg_area = 0.0

    return [spatter_density, n_blobs, avg_area, max_area]


def _edge_orientation_entropy(gray):
    """
    Compute edge orientation distribution entropy.
    Good welds: edges aligned along weld direction (low entropy).
    Defects: irregular edges in all directions (high entropy).
    """
    sobelx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle     = np.arctan2(sobely, sobelx)   # -π to π

    # Weighted histogram of angles (weighted by magnitude)
    n_bins = 18  # 10-degree bins
    hist, _ = np.histogram(angle.ravel(), bins=n_bins, range=(-np.pi, np.pi),
                           weights=magnitude.ravel())
    hist = hist / (hist.sum() + 1e-9)
    entropy = float(-np.sum(hist * np.log(hist + 1e-9)))

    # Dominant direction strength (low = chaotic edge directions)
    dominant_strength = float(hist.max())

    return [entropy, dominant_strength]


def extract_image_features(run_dir, num_frames=8):
    """
    Extract physics-based image features from weld frames.
    Returns IMAGE_FEAT_DIM=128 floats.

    Feature groups:
    ──────────────────────────────────────────────────────
    A. Laplacian variance (sharpness)     per frame  → 4 stats
    B. Bead profile geometry              per frame  → 6 stats × agg
    C. Spatter detection                  per frame  → 4 stats × agg
    D. Edge orientation entropy           per frame  → 2 stats × agg
    E. GLCM texture (extracted from mean frame) → 24 stats
    F. Frame-to-frame temporal differences → 8 stats
    G. Overall brightness stats (normalised by frame mean) → 6 stats
    H. Local contrast map stats            → 6 stats
    ──────────────────────────────────────────────────────
    """
    img_dir = os.path.join(run_dir, "images")
    if not os.path.exists(img_dir):
        return np.zeros(IMAGE_FEAT_DIM)

    frames = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not frames:
        return np.zeros(IMAGE_FEAT_DIM)

    # Evenly sample frames
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]

    grays = []
    for f in frames:
        try:
            if os.path.getsize(f) > 10 * 1024 * 1024:
                continue
            img = cv2.imread(f)
            if img is None:
                continue
            img = cv2.resize(img, (320, 240))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grays.append(gray.astype(np.float32))
        except Exception:
            continue

    if not grays:
        return np.zeros(IMAGE_FEAT_DIM)

    feats = []

    # ─── A. Laplacian variance (focus/surface roughness) ────────────────────
    lap_vars = []
    for g in grays:
        lap = cv2.Laplacian(g.astype(np.uint8), cv2.CV_32F)
        lap_vars.append(lap.var())
    lap_vars = np.array(lap_vars)
    feats.extend([float(np.mean(lap_vars)),  float(np.std(lap_vars)),
                  float(np.min(lap_vars)),   float(np.max(lap_vars))])   # 4 dims

    # ─── B. Bead profile geometry features ──────────────────────────────────
    bead_stats_all = [_bead_profile_features(g.astype(np.uint8)) for g in grays]
    bead_arr = np.array(bead_stats_all)   # (N, 6)
    feats.extend(bead_arr.mean(axis=0).tolist())   # 6: mean over frames
    feats.extend(bead_arr.std(axis=0).tolist())    # 6: std over frames (= temporal consistency)
    # 12 dims — width_cv std is key: high = inconsistent bead = defect

    # ─── C. Spatter detection ────────────────────────────────────────────────
    spatter_all = [_spatter_features(g.astype(np.uint8)) for g in grays]
    spatter_arr = np.array(spatter_all)   # (N, 4)
    feats.extend(spatter_arr.mean(axis=0).tolist())   # 4
    feats.extend(spatter_arr.std(axis=0).tolist())    # 4
    # 8 dims

    # ─── D. Edge orientation entropy ─────────────────────────────────────────
    eoe_all = [_edge_orientation_entropy(g.astype(np.uint8)) for g in grays]
    eoe_arr = np.array(eoe_all)   # (N, 2)
    feats.extend(eoe_arr.mean(axis=0).tolist())   # 2
    feats.extend(eoe_arr.std(axis=0).tolist())    # 2
    # 4 dims

    # ─── E. GLCM texture on averaged frame ───────────────────────────────────
    mean_frame = np.mean(grays, axis=0).astype(np.uint8)
    feats.extend(_glcm_features(mean_frame))   # 24 dims

    # ─── F. Frame-to-frame temporal difference ───────────────────────────────
    if len(grays) > 1:
        diffs = [np.abs(grays[i+1] - grays[i]).mean() for i in range(len(grays)-1)]
        diffs = np.array(diffs)
        feats.extend([float(np.mean(diffs)),   float(np.std(diffs)),
                      float(np.max(diffs)),    float(np.percentile(diffs, 90)),
                      float(np.min(diffs)),    float(np.percentile(diffs, 10)),
                      float(diffs.max() - diffs.min()),
                      float(np.mean(diffs > diffs.mean()))])  # 8 dims
    else:
        feats.extend([0.0] * 8)

    # ─── G. Normalised brightness (mean-subtracted per frame) ────────────────
    norm_stds = [g.std() / (g.mean() + 1e-9) for g in grays]  # CV of brightness
    feats.extend([float(np.mean(norm_stds)),    float(np.std(norm_stds)),
                  float(np.max(norm_stds)),     float(np.min(norm_stds)),
                  float(np.percentile(norm_stds, 75)),
                  float(np.percentile(norm_stds, 25))])   # 6 dims

    # ─── H. Local contrast map ───────────────────────────────────────────────
    # Compute local std in 16×16 patches → stats of spatial contrast
    local_contrasts = []
    for g in grays:
        g8 = g.astype(np.uint8)
        h, w = g8.shape
        stds = []
        for r in range(0, h - 16, 16):
            for c in range(0, w - 16, 16):
                patch = g8[r:r+16, c:c+16]
                stds.append(float(patch.std()))
        local_contrasts.extend(stds)
    lc = np.array(local_contrasts)
    feats.extend([float(np.mean(lc)),     float(np.std(lc)),
                  float(np.max(lc)),      float(np.percentile(lc, 95)),
                  float(np.percentile(lc, 5)), float(lc.max() - lc.min())])   # 6 dims

    out = _safe(feats)

    # Pad / truncate to fixed dim
    if len(out) < IMAGE_FEAT_DIM:
        out = np.concatenate([out, np.zeros(IMAGE_FEAT_DIM - len(out))])
    else:
        out = out[:IMAGE_FEAT_DIM]

    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sampleData/08-17-22-0011-00"
    feats = extract_image_features(path)
    print(f"Extracted {len(feats)} image features. Non-zero: {(feats != 0).sum()}")
    print(f"Range: [{feats.min():.4f}, {feats.max():.4f}]")
