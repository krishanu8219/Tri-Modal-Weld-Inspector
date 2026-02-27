import os
import logging
import warnings
import numpy as np
import librosa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def extract_audio_features(flac_path, n_mfcc=13):
    """
    Load a flac file and extract summary statistics from its MFCCs.
    Returns a unified 1D numpy array.
    Shape will be (n_mfcc * 4) representing Mean, Std, Min, Max.
    For n_mfcc=13, output dimensionality = 52.
    """
    feature_dim = n_mfcc * 4
    
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
        
        # mfccs shape is (n_mfcc, time_steps)
        # Summarize across time
        mean_mfcc = np.mean(mfccs, axis=1)
        std_mfcc = np.std(mfccs, axis=1)
        min_mfcc = np.min(mfccs, axis=1)
        max_mfcc = np.max(mfccs, axis=1)
        
        combined_features = np.concatenate([mean_mfcc, std_mfcc, min_mfcc, max_mfcc])
        
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
