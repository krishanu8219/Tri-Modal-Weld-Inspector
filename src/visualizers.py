import os
import glob
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

def get_representative_images(run_dir, num_frames=3):
    """
    Given an absolute or relative path to a run configuration directory
    (e.g. sampleData/08-17-22-0012-00/), extracts `num_frames` equally
    spaced .jpg files from the embedded images/ directory.
    """
    img_dir = os.path.join(run_dir, "images")
    if not os.path.exists(img_dir):
        return []
        
    frames = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    
    if len(frames) == 0:
        return []
    if len(frames) <= num_frames:
        return frames
        
    # Select evenly spaced indices
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    selected = [frames[i] for i in indices]
    
    return selected

def plot_sensor_with_hotspot(csv_path):
    """
    Heuristically finds the most volatile region in the structural sensor logic.
    Returns a matplotlib figure object capturing the Primary Weld Current and
    annotating the max variance hotspot window.
    """
    if not os.path.exists(csv_path):
        return None
        
    df = pd.read_csv(csv_path)
    
    # Try to find the primary operational trace
    primary_col = None
    targets = ["Primary Weld Current", "Current", "Pressure"]
    for t in targets:
        if t in df.columns:
            primary_col = t
            break
            
    if primary_col is None or df.empty:
        return None
        
    trace = df[primary_col].to_numpy()
    
    # Define an arbitrary hotspot heuristic: 200 frame rolling variance block
    window_size = min(200, max(1, len(trace) // 5))
    
    rolling_var = pd.Series(trace).rolling(window=window_size).var().fillna(0).to_numpy()
    hotspot_end = np.argmax(rolling_var)
    hotspot_start = max(0, hotspot_end - window_size)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(trace, label=primary_col, color="#1f77b4")
    ax.axvspan(hotspot_start, hotspot_end, color="red", alpha=0.3, label="Max Variance Hotspot")
    ax.set_title(f"{primary_col} Trace")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (samples)")
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    
    return fig

def plot_audio_spectrogram(flac_path):
    """
    Computes and plots a Mel-Spectrogram using librosa.
    Highlights the maximum energy column directly on the plot.
    """
    if not os.path.exists(flac_path):
        return None
        
    try:
        y, sr = librosa.load(flac_path, sr=None, mono=True)
        if len(y) == 0:
            return None
            
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Heuristic: Find temporal frame with max total energy
        frame_energies = np.sum(S, axis=0)
        hotspot_frame = np.argmax(frame_energies)
        
        # Convert frame index to seconds for the plot highlight
        hop_length = 512
        hotspot_time = librosa.frames_to_time(hotspot_frame, sr=sr, hop_length=hop_length)
        
        fig, ax = plt.subplots(figsize=(6, 3))
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='magma')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        ax.axvline(x=hotspot_time, color='cyan', linestyle='--', linewidth=2, label="Peak Acoustic Energy")
        ax.set_title('Audio Mel-Spectrogram')
        ax.legend(loc="upper right", fontsize="small")
        fig.tight_layout()
        
        return fig
    
    except Exception as e:
        import logging
        logging.error(f"Spectrogram failed on {flac_path}: {e}")
        return None
