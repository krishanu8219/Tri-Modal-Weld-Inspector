import os
import glob
import pandas as pd
import librosa
import cv2
from tqdm import tqdm

class WeldDatasetLoader:
    def __init__(self, data_dirs):
        """
        data_dirs: list of directory paths (e.g. ['good_weld', 'defect_data_weld', 'sampleData'])
        """
        self.data_dirs = data_dirs

    def find_runs(self):
        """
        Returns a list of dicts with run metadata and paths.
        """
        runs = []
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                continue
            
            for root, dirs, files in os.walk(data_dir):
                dir_name = os.path.basename(root)
                csv_path = os.path.join(root, f"{dir_name}.csv")
                test_csv_path = os.path.join(root, "sensor.csv")
                
                is_train_run = os.path.exists(csv_path)
                is_test_run = os.path.exists(test_csv_path)
                
                if is_train_run or is_test_run:
                    label_code = "UNKNOWN"
                    if is_train_run:
                        # run_id pattern: 04-03-23-0010-11
                        parts = dir_name.rsplit("-", 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            label_code = parts[1]
                    
                    config_folder = os.path.basename(os.path.dirname(root))
                    split_group = f"{data_dir}_{config_folder}"
                    
                    csv_target = csv_path if is_train_run else test_csv_path
                    flac_target = os.path.join(root, f"{dir_name}.flac") if is_train_run else os.path.join(root, "weld.flac")
                    avi_target = os.path.join(root, f"{dir_name}.avi") if is_train_run else os.path.join(root, "weld.avi")
                    
                    runs.append({
                        "run_id": dir_name,
                        "data_dir": data_dir,
                        "config_folder": config_folder,
                        "split_group": split_group,
                        "label_code": label_code,
                        "csv_path": csv_target,
                        "flac_path": flac_target,
                        "avi_path": avi_target,
                        "dir_path": root
                    })
        return runs

    def validate_run(self, run_info, fast_mode=True):
        """
        Validates the run and extracts durations.
        If fast_mode is True, simply checks if files exist and are not empty,
        skipping time-consuming librosa and cv2 checks.
        """
        report = {
            "csv_exists": os.path.exists(run_info["csv_path"]) and os.path.getsize(run_info["csv_path"]) > 0,
            "flac_exists": os.path.exists(run_info["flac_path"]) and os.path.getsize(run_info["flac_path"]) > 0,
            "avi_exists": os.path.exists(run_info["avi_path"]) and os.path.getsize(run_info["avi_path"]) > 0,
            "csv_valid": False,
            "flac_valid": False,
            "avi_valid": False,
            "csv_duration": None,
            "flac_duration": None,
            "avi_duration": None,
            "csv_rows": 0,
            "flac_sample_rate": None,
            "avi_fps": None,
            "avi_frames": 0,
            "error_msg": []
        }
        
        if fast_mode:
            report["csv_valid"] = report["csv_exists"]
            report["flac_valid"] = report["flac_exists"]
            report["avi_valid"] = report["avi_exists"]
            report["has_all_modalities"] = report["csv_valid"] and report["flac_valid"] and report["avi_valid"]
            return report
            
        if report["csv_exists"]:
            try:
                df = pd.read_csv(run_info["csv_path"])
                report["csv_rows"] = len(df)
                if len(df) > 0:
                    report["csv_valid"] = True
            except Exception as e:
                report["error_msg"].append(f"CSV error: {e}")
                
        if report["flac_exists"]:
            try:
                y, sr = librosa.load(run_info["flac_path"], sr=None)
                report["flac_valid"] = True
                report["flac_duration"] = librosa.get_duration(y=y, sr=sr)
                report["flac_sample_rate"] = sr
            except Exception as e:
                report["error_msg"].append(f"FLAC error: {e}")
                
        if report["avi_exists"]:
            try:
                cap = cv2.VideoCapture(run_info["avi_path"])
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    report["avi_valid"] = True
                    if fps > 0:
                        report["avi_duration"] = frame_count / fps
                    report["avi_fps"] = fps
                    report["avi_frames"] = frame_count
                else:
                    report["error_msg"].append("AVI error: could not open")
                cap.release()
            except Exception as e:
                report["error_msg"].append(f"AVI error: {e}")
                
        report["has_all_modalities"] = report["csv_valid"] and report["flac_valid"] and report["avi_valid"]
        
        return report

    def load_dataset(self, run_validation=True, fast_mode=True):
        runs = self.find_runs()
        df_runs = pd.DataFrame(runs)
        
        if run_validation and not df_runs.empty:
            validation_records = []
            for run in tqdm(runs, desc="Validating runs"):
                validation_records.append(self.validate_run(run, fast_mode=fast_mode))
            
            df_val = pd.DataFrame(validation_records)
            df_runs = pd.concat([df_runs, df_val], axis=1)
            
        return df_runs

if __name__ == "__main__":
    new_data_dirs = [
        "sampleData", 
        "../therness/hackathon data/good_weld", 
        "../therness/hackathon data/defect-weld"
    ]
    loader = WeldDatasetLoader(new_data_dirs)
    # Use fast mode to parse thousands of runs efficiently
    df = loader.load_dataset(run_validation=True, fast_mode=True)
    if not df.empty:
        print(df.head())
        print(df["label_code"].value_counts())
        df.to_csv("runs_summary.csv", index=False)
    else:
        print("No runs found.")
