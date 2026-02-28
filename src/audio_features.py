"""
Physics-based audio feature extraction for weld defect detection.

Key insight: We extract features that capture DEFECT SIGNATURES, not
environment/config signatures. Welding defects produce physically distinct
acoustic events:
  - Porosity / crater cracks: high-frequency burst transients
  - Burn through: abrupt energy collapse followed by silencing
  - Spatter: short-duration impulsive pops
  - Arc instability: non-stationary frequency drift, high ZCR variability
  - Lack of fusion: low-energy, under-powered arc sound

We avoid MFCC means (which encode background hum = config-specific).
Instead we use: sub-band energy RATIOS, temporal VARIANCE, spectral
ENTROPY (disorder = defect), kurtosis (impulsive events), and
differential features (rate of change = arc instability).

Output: 136-dimensional vector (fully documented inline)
"""

import os
import logging
import warnings
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

AUDIO_FEAT_DIM = 136   # documented below


def _safe(arr):
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_audio_features(flac_path):
    """
    Extract physics-based audio features.  Returns AUDIO_FEAT_DIM=136 floats.

    Feature groups
    ──────────────
    A. Sub-band energy statistics  (12 bands × 4 stats  = 48 dims)
    B. Spectral entropy per frame  (4 stats              =  4 dims)
    C. Temporal energy envelope    (8 stats              =  8 dims)
    D. Delta-energy (arc stability)(4 stats              =  4 dims)
    E. Zero-crossing rate          (4 stats              =  4 dims)
    F. Kurtosis per sub-band       (12 bands             = 12 dims)
    G. High/Low energy ratio       (12 frame stats       = 12 dims)
    H. Spectral flux (rate of chg) (4 stats              =  4 dims)
    I. Onset strength envelope     (4 stats              =  4 dims)
    J. Band energy RATIOS          (6 pair ratios × 4    = 24 dims)
    K. Global percentile contrasts (4 dims               =  4 dims)
    L. Harmonic-to-noise ratio     (4 stats              =  4 dims)
    ──────────────────────────────────────────
    Total: 48+4+8+4+4+12+12+4+4+24+4+4 = 132 … padded to 136
    """
    if not os.path.exists(flac_path):
        return np.zeros(AUDIO_FEAT_DIM)

    try:
        import librosa

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y, sr = librosa.load(flac_path, sr=None, mono=True)

        if len(y) < 512:
            return np.zeros(AUDIO_FEAT_DIM)

        # ─── Shared STFT ────────────────────────────────────────────────────
        n_fft   = 1024
        hop     = 256
        S       = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))  # (F, T)
        freqs   = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        def stat4(x):
            """Return [mean, std, skewness, kurtosis] — invariant to amplitude scaling."""
            mu  = np.mean(x)
            std = np.std(x) + 1e-9
            sk  = float(np.mean(((x - mu) / std) ** 3))
            ku  = float(np.mean(((x - mu) / std) ** 4))
            return np.array([float(mu), float(std), sk, ku])

        feats = []

        # ─── A. Sub-band energy statistics (12 bands × 4 = 48) ─────────────
        # Bands on mel-scale to capture acoustic physics better
        band_edges_hz = [0, 200, 500, 1000, 1500, 2000, 3000, 4000,
                         5000, 6000, 8000, 10000, sr // 2]
        band_edges_hz = [b for b in band_edges_hz if b <= sr // 2]
        # Ensure 12 bands even if sr is low
        while len(band_edges_hz) < 13:
            band_edges_hz.append(sr // 2)

        band_energies = []   # (T,) per band
        for lo, hi in zip(band_edges_hz[:-1], band_edges_hz[1:]):
            mask = (freqs >= lo) & (freqs < hi)
            if mask.sum() == 0:
                band_energies.append(np.zeros(S.shape[1]))
            else:
                band_energies.append(S[mask, :].mean(axis=0))

        for be in band_energies:
            feats.extend(stat4(be))          # 12 × 4 = 48 dims  ← DEFECT-SENSITIVE
            # (variance/kurtosis of band energy over time captures burst events)

        # ─── B. Spectral entropy (disorder in frequency domain) (4) ─────────
        S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-9)
        spectral_entropy = -np.sum(S_norm * np.log(S_norm + 1e-9), axis=0)
        feats.extend(stat4(spectral_entropy))   # high entropy = erratic arc

        # ─── C. Temporal RMS energy envelope (8) ────────────────────────────
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        feats.extend(stat4(rms))
        # Energy change variability (coefficient of variation — scale-free!)
        cv = np.std(rms) / (np.mean(rms) + 1e-9)
        feats.append(float(cv))
        # Peak-to-median ratio (burst events = spatter)
        feats.append(float(np.max(rms) / (np.median(rms) + 1e-9)))
        # Fraction of frames with very low energy (dropouts = burn through)
        quiet_thresh = np.percentile(rms, 10)
        feats.append(float(np.mean(rms < quiet_thresh * 1.5)))
        # 90th-10th percentile range (dynamic range of arc)
        feats.append(float(np.percentile(rms, 90) - np.percentile(rms, 10)))

        # ─── D. Delta energy — arc stability signal (4) ─────────────────────
        delta_rms = np.diff(rms)
        feats.extend(stat4(delta_rms))   # large std = unstable arc

        # ─── E. Zero-crossing rate (4) ──────────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
        feats.extend(stat4(zcr))         # high ZCR std = erratic arc noise

        # ─── F. Per-band kurtosis (12) ──────────────────────────────────────
        for be in band_energies:
            mu  = np.mean(be)
            std = np.std(be) + 1e-9
            ku  = float(np.mean(((be - mu) / std) ** 4))
            feats.append(ku)              # kurtosis > 3 = impulsive = defect event

        # ─── G. High-freq to total energy ratio per frame (12) ──────────────
        high_mask = freqs >= 4000
        low_mask  = freqs < 4000
        if high_mask.sum() > 0 and low_mask.sum() > 0:
            high_e = S[high_mask, :].mean(axis=0)
            low_e  = S[low_mask,  :].mean(axis=0)
            hl_ratio = high_e / (low_e + 1e-9)
        else:
            hl_ratio = np.zeros(S.shape[1])
        feats.extend(stat4(hl_ratio))     # 4
        # Percentile stats of ratio
        feats.extend([float(np.percentile(hl_ratio, p)) for p in [10, 25, 75, 90]])  # 4
        # Fraction of frames where high-freq dominates
        feats.append(float(np.mean(hl_ratio > 1.0)))   # 1
        # Max burst of high-freq ratio
        feats.append(float(np.max(hl_ratio)))           # 1
        # Std of hl_ratio (variability)
        feats.append(float(np.std(hl_ratio)))           # 1

        # ─── H. Spectral flux (frame-to-frame spectral change) (4) ──────────
        S_diff = np.diff(S, axis=1)
        flux = np.sqrt(np.sum(S_diff ** 2, axis=0))
        feats.extend(stat4(flux))         # high flux = arc instability

        # ─── I. Onset strength envelope (4) ─────────────────────────────────
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        feats.extend(stat4(onset_env))    # many onsets = many arc events / spatter

        # ─── J. Band energy pair RATIOS (6 pairs × 4 = 24) ──────────────────
        # Pairs encode the spectral "shape" change over time (config-invariant)
        n_bands = len(band_energies)
        pairs = [(0, 5), (0, 11), (2, 7), (3, 8), (4, 9), (5, 10)]
        for lo_i, hi_i in pairs:
            if lo_i < n_bands and hi_i < n_bands:
                ratio = band_energies[hi_i] / (band_energies[lo_i] + 1e-9)
            else:
                ratio = np.zeros(S.shape[1])
            feats.extend(stat4(ratio))    # spectral slope indicator

        # ─── K. Global percentile contrasts (4) ─────────────────────────────
        sc = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
        feats.extend([float(np.mean(sc)),
                      float(np.std(sc)),
                      float(np.max(sc)),
                      float(np.percentile(sc.flatten(), 95))])

        # ─── L. Harmonic-to-noise ratio proxy (4) ───────────────────────────
        # Harmonic component vs percussive (spatter = high percussive)
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_rms = np.sqrt(np.mean(y_harm ** 2)) + 1e-9
        perc_rms = np.sqrt(np.mean(y_perc ** 2)) + 1e-9
        hnr = harm_rms / perc_rms
        feats.extend([float(hnr),
                      float(np.log1p(hnr)),
                      float(np.sqrt(np.mean(y_perc ** 2))),
                      float(np.sqrt(np.mean(y_harm ** 2)))])

        out = _safe(np.array(feats, dtype=np.float32))

        # Pad or truncate to fixed dim
        if len(out) < AUDIO_FEAT_DIM:
            out = np.concatenate([out, np.zeros(AUDIO_FEAT_DIM - len(out))])
        else:
            out = out[:AUDIO_FEAT_DIM]

        return out

    except Exception as e:
        logging.error(f"Audio feature error ({flac_path}): {e}")
        return np.zeros(AUDIO_FEAT_DIM)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sampleData/08-17-22-0011-00/08-17-22-0011-00.flac"
    feats = extract_audio_features(path)
    print(f"Extracted {len(feats)} audio features. Non-zero: {(feats != 0).sum()}")
    print(f"Range: [{feats.min():.4f}, {feats.max():.4f}]")
