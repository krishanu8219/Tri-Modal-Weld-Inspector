import os
import pandas as pd
from tqdm import tqdm
from src.inference import DefectClassifierPipeline

def generate_submission(test_data_dir="test_data", output_csv="submission.csv"):
    pipeline = DefectClassifierPipeline()
    results = []
    
    # Check if test_data exists. For sample-level smoke tests, we might fallback to sampleData
    # but the submission strictly demands evaluating "sample_XXXX" structures.
    if not os.path.exists(test_data_dir):
        print(f"Directory {test_data_dir} not found. Ensure it is placed in the project root.")
        return
        
    for item in tqdm(os.listdir(test_data_dir), desc="Processing Test Samples"):
        sample_path = os.path.join(test_data_dir, item)
        if os.path.isdir(sample_path) and item.startswith("sample_"):
            csv_path = os.path.join(sample_path, "sensor.csv")
            flac_path = os.path.join(sample_path, "weld.flac")
            
            # Use dummy/empty feature outputs handled gracefully by our extractors if missing
            res = pipeline.infer_run(csv_path, flac_path)
            
            results.append({
                "sample_id": item,
                "pred_label_code": res["pred_label_code"],
                "p_defect": res["p_defect"]
            })
            
    if len(results) == 0:
        print(f"No samples matching 'sample_XXXX' found in {test_data_dir}.")
        return

    df_submission = pd.DataFrame(results)
    # Sort strictly by sample_id to be neat
    df_submission = df_submission.sort_values(by="sample_id").reset_index(drop=True)
    
    # ---------------------------------------------------------
    # STRICT SUBMISSION VALIDATOR
    # ---------------------------------------------------------
    print("\n--- Validating Submission ---")
    ALLOWED_LABELS = {"00", "01", "02", "06", "07", "08", "11"}
    
    # Check 1: Exactly 90 rows
    if len(df_submission) != 90:
        print(f"❌ VALIDATION FAILED: Expected exactly 90 rows, got {len(df_submission)}")
        # We don't necessarily raise Exception if they are testing on fewer, but warn loudly
        # The prompt instructed to fail loudly if rows != 90
        raise ValueError(f"Submission requires exactly 90 rows. Found {len(df_submission)}.")
        
    # Check 2: Unique sample IDs
    if df_submission["sample_id"].nunique() != len(df_submission):
        raise ValueError("❌ VALIDATION FAILED: Duplicate sample_ids found.")
        
    # Check 3: Allowed Labels
    invalid_labels = set(df_submission["pred_label_code"].astype(str).unique()) - ALLOWED_LABELS
    if len(invalid_labels) > 0:
        raise ValueError(f"❌ VALIDATION FAILED: Invalid labels predicted: {invalid_labels}. Allowed: {ALLOWED_LABELS}")
        
    # Check 4: Probabilities bounds
    if (df_submission["p_defect"] < 0).any() or (df_submission["p_defect"] > 1).any():
        raise ValueError("❌ VALIDATION FAILED: 'p_defect' values found outside [0.0, 1.0] range.")
        
    print("✅ All validation checks passed!")
    # ---------------------------------------------------------
    
    df_submission.to_csv(output_csv, index=False)
    print(f"\nSubmission generated successfully: {output_csv}")
    print(df_submission.head())

if __name__ == "__main__":
    generate_submission()
