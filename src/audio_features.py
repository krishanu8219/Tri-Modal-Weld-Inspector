import os
import logging
import warnings
import numpy as np
import librosa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def extract_audio_features(flac_path, n_mfcc=13):
    """
    Load a flac file and extract summary statistics from its MFCCs 
    plus spectral centroid, rolloff, zcr, and rms energy.
    """
    # 4 stats for 13 mfccs + 4 stats for 13 delta mfccs
    # + 4 stats each for centroid, rolloff, zcr, rms = 4 * 4 = 16
    # Total dim = 52 + 52 + 16 = 120
    feature_dim = (n_mfcc * 4 * 2) + 16
    
    if not os.path.exists(flac_path):
        return np.zeros(feature_dim)
        
    try:
        # We suppress numba/librosa warnings for cleaner logs
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y, sr = librosa.load(flac_path, sr=None, mono=True)
            
        if len(y) == 0:
            return np.zeros(feature_dim)
            
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        
        def summarize(feat_matrix):
            return np.concatenate([
                np.mean(feat_matrix, axis=1),
                np.std(feat_matrix, axis=1),
                np.min(feat_matrix, axis=1),
                np.max(feat_matrix, axis=1)
            ])
            
        mfcc_summary = summarize(mfccs)
        delta_summary = summarize(delta_mfccs)
        centroid_summary = summarize(centroid)
        rolloff_summary = summarize(rolloff)
        zcr_summary = summarize(zcr)
        rms_summary = summarize(rms)
        
        combined_features = np.concatenate([
            mfcc_summary, delta_summary, centroid_summary, 
            rolloff_summary, zcr_summary, rms_summary
        ])
        
        # Replace NaN/inf if any
        combined_features = np.nan_to_num(combined_features, nan=0.0)
        
        return combined_features

    except Exception as e:
        logging.error(f"Error processing audio {flac_path}: {e}")
        return np.zeros(feature_dim)

if __name__ == "__main__":
    # Smoke test on a known file from sampleData
    test_flac = "sampleData/08-17-22-0011-00/08-17-22-0011-00.flac"
    feats = extract_audio_features(test_flac)
    print(f"Extracted {len(feats)} audio features. Sample data preview: {feats[:5]}")
