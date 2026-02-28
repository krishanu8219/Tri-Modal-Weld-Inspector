#!/usr/bin/env python3
"""
Fast feature caching with multiprocessing.
Only extracts audio(120) + image(97) = 217 features (skips sensor — not needed for AV model).
Uses all CPU cores for parallel extraction.

Output: artifacts/feature_cache.npz
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def extract_one(args):
    """Extract features for a single run (worker function)."""
    idx, flac_path, run_dir, label_code = args
    try:
        from src.audio_features import extract_audio_features
        from src.image_features import extract_image_features
        
        audio_f = extract_audio_features(flac_path)
        image_f = extract_image_features(run_dir)
        
        return idx, audio_f, image_f, None
    except Exception as e:
        return idx, np.zeros(120), np.zeros(97), str(e)


def cache_split(split_csv, split_name, n_workers=None):
    """Extract features for all runs in a split using multiprocessing."""
    df = pd.read_csv(split_csv)
    n = len(df)
    
    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    
    logging.info(f"[{split_name}] Processing {n} runs with {n_workers} workers...")
    
    # Prepare args
    tasks = []
    for i, (_, row) in enumerate(df.iterrows()):
        run_dir = os.path.dirname(row["csv_path"])
        tasks.append((i, row["flac_path"], run_dir, str(row["label_code"]).zfill(2)))
    
    # Run in parallel
    t0 = time.time()
    audio_arr = np.zeros((n, 120))
    image_arr = np.zeros((n, 97))
    errors = 0
    
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(extract_one, tasks, chunksize=10):
            idx, audio_f, image_f, err = result
            audio_arr[idx] = audio_f
            image_arr[idx] = image_f
            if err:
                errors += 1
            
            done = idx + 1
            if done % 100 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (n - done) / max(rate, 0.01)
                logging.info(f"  [{split_name}] {done}/{n} done ({rate:.1f} runs/s, ETA {eta:.0f}s)")
    
    elapsed = time.time() - t0
    logging.info(f"  [{split_name}] Finished {n} runs in {elapsed:.1f}s ({n/elapsed:.1f} runs/s, {errors} errors)")
    
    # Labels
    binary_labels = np.array([0 if str(r["label_code"]).zfill(2) == "00" else 1 for _, r in df.iterrows()])
    label_codes = np.array([str(r["label_code"]).zfill(2) for _, r in df.iterrows()])
    
    # Also extract sensor features (fast, serial — just CSV reads)
    from src.train_binary import extract_sensor_features
    sensor_arr = np.zeros((n, 102))
    for i, (_, row) in enumerate(df.iterrows()):
        sensor_arr[i] = extract_sensor_features(row["csv_path"])
    logging.info(f"  [{split_name}] Sensor features extracted (serial)")
    
    return sensor_arr, audio_arr, image_arr, binary_labels, label_codes


def main():
    cache_path = os.path.join("artifacts", "feature_cache.npz")
    
    logging.info("=== Fast Feature Cache Builder (Multiprocessing) ===")
    
    train_s, train_a, train_i, train_y, train_c = cache_split("train_split.csv", "TRAIN")
    val_s, val_a, val_i, val_y, val_c = cache_split("val_split.csv", "VAL")
    
    np.savez_compressed(
        cache_path,
        train_sensor=train_s, train_audio=train_a, train_image=train_i,
        train_labels=train_y, train_label_codes=train_c,
        val_sensor=val_s, val_audio=val_a, val_image=val_i,
        val_labels=val_y, val_label_codes=val_c,
    )
    
    fsize = os.path.getsize(cache_path) / (1024 * 1024)
    logging.info(f"Saved: {cache_path} ({fsize:.1f} MB)")
    logging.info(f"Train: {train_a.shape[0]} | Val: {val_a.shape[0]}")
    logging.info(f"Dims: sensor={train_s.shape[1]}, audio={train_a.shape[1]}, image={train_i.shape[1]}")


if __name__ == "__main__":
    main()
