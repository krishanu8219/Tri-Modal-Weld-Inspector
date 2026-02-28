#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║         Tri-Modal Weld Inspector — One-Command Inference    ║
║                                                              ║
║  Usage:                                                      ║
║    python run_inference.py                                   ║
║    python run_inference.py --test-dir path/to/test_data      ║
║    python run_inference.py --output my_submission.csv        ║
║                                                              ║
║  Reads test_data/sample_XXXX/{sensor.csv, weld.flac} and     ║
║  produces submission.csv with: sample_id, pred_label_code,   ║
║  p_defect                                                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
import time
import pandas as pd
from tqdm import tqdm

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference import DefectClassifierPipeline


ALLOWED_LABELS = {"00", "01", "02", "06", "07", "08", "11"}


def discover_samples(test_data_dir):
    """Discover all sample_XXXX folders and auto-detect csv/flac files inside."""
    import glob
    samples = []
    if not os.path.exists(test_data_dir):
        print(f"❌ ERROR: Test data directory '{test_data_dir}' not found.")
        print(f"   Place your test samples at: {os.path.abspath(test_data_dir)}")
        sys.exit(1)
    
    for item in sorted(os.listdir(test_data_dir)):
        sample_path = os.path.join(test_data_dir, item)
        if os.path.isdir(sample_path) and item.startswith("sample_"):
            # Auto-detect CSV: try sensor.csv first, else find any .csv
            csv_path = os.path.join(sample_path, "sensor.csv")
            if not os.path.exists(csv_path):
                csv_files = glob.glob(os.path.join(sample_path, "*.csv"))
                csv_path = csv_files[0] if csv_files else os.path.join(sample_path, "sensor.csv")
            
            # Auto-detect FLAC: try weld.flac first, else find any .flac
            flac_path = os.path.join(sample_path, "weld.flac")
            if not os.path.exists(flac_path):
                flac_files = glob.glob(os.path.join(sample_path, "*.flac"))
                flac_path = flac_files[0] if flac_files else os.path.join(sample_path, "weld.flac")
            
            has_csv = os.path.exists(csv_path)
            has_flac = os.path.exists(flac_path)
            
            if not has_csv:
                print(f"  ⚠️  {item}: No .csv found (will use zero sensor features)")
            if not has_flac:
                print(f"  ⚠️  {item}: No .flac found (will use zero audio features)")
            
            samples.append({
                "sample_id": item,
                "csv_path": csv_path,
                "flac_path": flac_path,
            })
    
    return samples


def validate_submission(df):
    """Validate the submission CSV against hackathon requirements."""
    print("\n━━━ Submission Validation ━━━")
    errors = []
    
    # Check 1: Row count
    if len(df) != 90:
        errors.append(f"Expected 90 rows, got {len(df)}")
    else:
        print(f"  ✅ Row count: {len(df)}/90")
    
    # Check 2: No duplicate sample_ids
    if df["sample_id"].nunique() != len(df):
        errors.append("Duplicate sample_ids found")
    else:
        print(f"  ✅ Unique sample_ids: {df['sample_id'].nunique()}")
    
    # Check 3: Valid label codes
    predicted_labels = set(df["pred_label_code"].astype(str).unique())
    invalid = predicted_labels - ALLOWED_LABELS
    if invalid:
        errors.append(f"Invalid label codes: {invalid}")
    else:
        print(f"  ✅ Label codes valid: {predicted_labels}")
    
    # Check 4: p_defect in [0, 1]
    if (df["p_defect"] < 0).any() or (df["p_defect"] > 1).any():
        errors.append("p_defect values outside [0, 1]")
    else:
        print(f"  ✅ p_defect range: [{df['p_defect'].min():.4f}, {df['p_defect'].max():.4f}]")
    
    # Check 5: Required columns
    required = {"sample_id", "pred_label_code", "p_defect"}
    if not required.issubset(set(df.columns)):
        errors.append(f"Missing columns: {required - set(df.columns)}")
    else:
        print(f"  ✅ Schema: {list(df.columns)}")
    
    if errors:
        print(f"\n  ❌ VALIDATION FAILED:")
        for e in errors:
            print(f"     • {e}")
        return False
    
    print(f"\n  ✅ ALL CHECKS PASSED — Ready to submit!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Tri-Modal Weld Inspector — Generate submission CSV from test data"
    )
    parser.add_argument(
        "--test-dir", default="test_data",
        help="Path to test_data directory (default: test_data/)"
    )
    parser.add_argument(
        "--output", default="submission.csv",
        help="Output submission CSV path (default: submission.csv)"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip submission validation (use if fewer than 90 samples)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Tri-Modal Weld Inspector — Inference Pipeline")
    print("=" * 60)
    
    # 1. Discover samples
    print(f"\n📂 Scanning: {os.path.abspath(args.test_dir)}")
    samples = discover_samples(args.test_dir)
    print(f"   Found {len(samples)} test samples\n")
    
    if len(samples) == 0:
        print("❌ No samples found. Exiting.")
        sys.exit(1)
    
    # 2. Load pipeline
    print("🔧 Loading models...")
    t0 = time.time()
    pipeline = DefectClassifierPipeline()
    print(f"   Models loaded in {time.time() - t0:.1f}s")
    print(f"   Binary threshold: {pipeline.binary_threshold:.3f}\n")
    
    # 3. Run inference
    print("🔬 Running inference...")
    results = []
    t_start = time.time()
    
    for sample in tqdm(samples, desc="   Processing"):
        try:
            res = pipeline.infer_run(sample["csv_path"], sample["flac_path"])
            results.append({
                "sample_id": sample["sample_id"],
                "pred_label_code": res["pred_label_code"],
                "p_defect": round(res["p_defect"], 4)
            })
        except Exception as e:
            print(f"\n   ⚠️  {sample['sample_id']} failed: {e}")
            results.append({
                "sample_id": sample["sample_id"],
                "pred_label_code": "00",
                "p_defect": 0.0
            })
    
    elapsed = time.time() - t_start
    print(f"\n   ✅ Processed {len(results)} samples in {elapsed:.1f}s ({elapsed/len(results):.2f}s/sample)")
    
    # 4. Build submission DataFrame
    df = pd.DataFrame(results).sort_values("sample_id").reset_index(drop=True)
    
    # 5. Summary stats
    n_defect = (df["pred_label_code"] != "00").sum()
    n_good = (df["pred_label_code"] == "00").sum()
    print(f"\n📊 Prediction Summary:")
    print(f"   Good welds:    {n_good}")
    print(f"   Defective:     {n_defect}")
    if n_defect > 0:
        print(f"   Defect types:  {df[df['pred_label_code'] != '00']['pred_label_code'].value_counts().to_dict()}")
    print(f"   Avg P(defect): {df['p_defect'].mean():.4f}")
    
    # 6. Validate
    if not args.skip_validation:
        validate_submission(df)
    
    # 7. Save
    df.to_csv(args.output, index=False)
    print(f"\n💾 Submission saved: {os.path.abspath(args.output)}")
    print(f"\n{'=' * 60}")
    print(f"  Preview:")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
