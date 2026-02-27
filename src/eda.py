import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_eda(runs_csv="runs_summary.csv", output_dir="eda_reports"):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(runs_csv):
        print(f"{runs_csv} not found.")
        return
        
    df = pd.read_csv(runs_csv)
    
    # Exclude test_data for class distribution
    train_pool = df[df["data_dir"] != "test_data"]
    
    # 1. Class counts
    plt.figure(figsize=(10, 6))
    sns.countplot(data=train_pool, x="label_code", order=train_pool["label_code"].value_counts().index)
    plt.title("Class Counts in Training Pool")
    plt.xlabel("Defect Code / Label Code")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "class_counts.png"))
    plt.close()
    
    # 2. Duration distributions
    # We plot the distribution of durations for audio and video
    plt.figure(figsize=(10, 6))
    sns.histplot(df["flac_duration"].dropna(), kde=True, color='blue', label='Audio Duration')
    sns.histplot(df["avi_duration"].dropna(), kde=True, color='orange', label='Video Duration')
    plt.title("Run Durations (Audio vs Video)")
    plt.xlabel("Duration (seconds)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "duration_distributions.png"))
    plt.close()
    
    # 3. Missing Files Summary
    print("--- Missing Modalities Summary ---")
    print(f"Total runs: {len(df)}")
    missing_csv = len(df[~df["csv_valid"]])
    missing_audio = len(df[~df["flac_valid"]])
    missing_video = len(df[~df["avi_valid"]])
    incomplete = len(df[~df["has_all_modalities"]])
    
    print(f"Missing valid CSV: {missing_csv}")
    print(f"Missing valid Audio: {missing_audio}")
    print(f"Missing valid Video: {missing_video}")
    print(f"Runs missing at least one modality: {incomplete}")
    
    # 4. Sensor summary stats (sampled)
    # We will pick a few runs, load their CSVs, and compute aggregate stats
    print("\n--- Sensor Summary Stats Formulation ---")
    valid_csv_runs = df[df["csv_valid"]]["csv_path"].tolist()
    
    # Let's take up to 20 runs to compute quick aggregate stats to save time
    sample_csvs = valid_csv_runs[:20] 
    all_sensor_stats = []
    
    for c_path in sample_csvs:
        try:
            sdf = pd.read_csv(c_path)
            # Pick numeric columns
            numeric_cols = sdf.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                all_sensor_stats.append({
                    "column": col,
                    "mean": sdf[col].mean(),
                    "std": sdf[col].std(),
                    "min": sdf[col].min(),
                    "max": sdf[col].max()
                })
        except Exception as e:
            pass
            
    if all_sensor_stats:
        sensor_df = pd.DataFrame(all_sensor_stats)
        agg_stats = sensor_df.groupby("column").mean()
        print(agg_stats)
        agg_stats.to_csv(os.path.join(output_dir, "sensor_summary_stats_sample.csv"))
    
    print(f"\nEDA basic report saved to {output_dir}/ directory.")

if __name__ == "__main__":
    generate_eda()
