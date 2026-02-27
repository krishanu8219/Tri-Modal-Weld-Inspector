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
    
    df_submission.to_csv(output_csv, index=False)
    print(f"\nSubmission generated successfully: {output_csv}")
    print(df_submission.head())

if __name__ == "__main__":
    generate_submission()
