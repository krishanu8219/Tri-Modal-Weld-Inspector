import pandas as pd
from sklearn.model_selection import train_test_split

def create_run_level_split(runs_csv="runs_summary.csv", test_size=0.2, random_state=42):
    """
    Splits the dataset at the run-level, stratifying by defect label if possible.
    Since some configurations might have correlated runs, splitting by split_group
    (which contains the configuration_folder) might be safer to avoid leakage.
    However, the README mentions: "Split by run/group (not random rows)".
    We will group by `split_group` to ensure all runs from a configuration go into
    the same split OR we split runs directly but stratify securely.
    Let's split by configuration_folder (which maps to split_group).
    """
    df = pd.read_csv(runs_csv)
    
    # Exclude test_data from train/val splitting
    train_pool = df[df["data_dir"] != "test_data"].copy()
    
    if train_pool.empty:
        print("No training data available to split.")
        return None, None
        
    # We want to group by 'split_group' to avoid leakage.
    # We will compute the majority label for each split_group, and stratify based on that.
    group_labels = train_pool.groupby("split_group")["label_code"].agg(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else "00").reset_index()
    
    # Try stratified split on groups
    try:
        if len(group_labels) < 2:
            print("Warning: Only 1 group found. Splitting runs randomly for demonstration.")
            train_runs, val_runs = train_test_split(train_pool, test_size=test_size, random_state=random_state)
        else:
            train_groups, val_groups = train_test_split(
                group_labels["split_group"], 
                test_size=test_size, 
                random_state=random_state, 
                stratify=group_labels["label_code"]
            )
            train_runs = train_pool[train_pool["split_group"].isin(train_groups)]
            val_runs = train_pool[train_pool["split_group"].isin(val_groups)]
    except ValueError:
        # Fallback if there aren't enough samples for some classes to stratify
        print("Warning: Could not stratify by group label. Performing random split on groups.")
        train_groups, val_groups = train_test_split(
            group_labels["split_group"], 
            test_size=test_size, 
            random_state=random_state
        )
        train_runs = train_pool[train_pool["split_group"].isin(train_groups)]
        val_runs = train_pool[train_pool["split_group"].isin(val_groups)]
    
    print(f"Train runs: {len(train_runs)}, Val runs: {len(val_runs)}")
    
    # Let's see the class distribution
    print("\nTrain class distribution:")
    print(train_runs["label_code"].value_counts(normalize=True))
    
    print("\nVal class distribution:")
    print(val_runs["label_code"].value_counts(normalize=True))
    
    train_runs.to_csv("train_split.csv", index=False)
    val_runs.to_csv("val_split.csv", index=False)
    
    return train_runs, val_runs

if __name__ == "__main__":
    create_run_level_split()
