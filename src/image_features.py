import os
import glob
import logging
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 3 channels * 32 bins + 1 edge density feature = 97 dims
IMAGE_FEAT_DIM = 97

def extract_image_features(run_dir, num_frames=3):
    """
    Reads representative images from a run, computes global color histograms 
    and edge density, and returns a mean-pooled 1D numpy array.
    """
    img_dir = os.path.join(run_dir, "images")
    if not os.path.exists(img_dir):
        return np.zeros(IMAGE_FEAT_DIM)
        
    frames = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    
    if len(frames) == 0:
        return np.zeros(IMAGE_FEAT_DIM)
        
    # Select evenly spaced frames
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        selected_frames = [frames[i] for i in indices]
    else:
        selected_frames = frames
        
    embeddings = []
    
    for f in selected_frames:
        try:
            # Skip files > 10MB
            if os.path.getsize(f) > 10 * 1024 * 1024:
                continue
                
            img = cv2.imread(f)
            if img is None:
                continue
                
            # Resize for speed
            img = cv2.resize(img, (224, 224))
            
            # 1. Color Histogram (3 channels, 32 bins each)
            hist_features = []
            for i in range(3):
                hist = cv2.calcHist([img], [i], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hist_features.extend(hist)
                
            # 2. Edge Density (Canny)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (224 * 224)
            
            # Combine
            feat = np.array(hist_features + [edge_density])
            embeddings.append(feat)
            
        except Exception as e:
            logging.error(f"Failed to extract cv2 features from {f}: {e}")
            
    if len(embeddings) == 0:
        return np.zeros(IMAGE_FEAT_DIM)
        
    # Mean pool all collected frame embeddings
    return np.mean(embeddings, axis=0)
