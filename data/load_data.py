"""
NYC Taxi Trip Duration - Data Ingestion & Splitting Module

This script manages the initial data tier of the pipeline. It handles locating 
raw data, loading specific subsets into memory, and isolating full dataset splits 
from development samples to optimize execution workflows.

Pipeline Workflow & Implementations:
----------------------------------
1. Directory Structure Navigation:
     - Configures robust multi-level path resolution to smoothly access data directories 
         from nested modules (e.g., handling cross-folder communication between 'data/' 
         and ingestion scripts).

2. Data Ingestion & Dynamic Selection:
   - Provides functional switches to load either the full training/validation splits 
     or the lighter development samples based on the runtime environment.

3. Split Synchronization:
   - Manages the separation between:
         * Full Tiers ('data/split/'): train.csv, val.csv, test.csv for final modeling.
         * Sample Tiers ('data/split_sample/'): Lightened deterministic versions 
       for fast iteration and iterative testing inside Jupyter Notebooks.
"""



import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from Config.load import load_config as config
from Enum.path_enums import DataTier, DataFile


def get_data_path(tier: DataTier = DataTier.FULL, filename: DataFile = DataFile.TRAIN) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, tier.value, filename.value)
    return path


def load_csv(tier: DataTier = DataTier.FULL, filename: DataFile = DataFile.TRAIN) -> pd.DataFrame:
    path = get_data_path(tier, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: The file {filename.value} does not exist in {tier.value} folder!")
        
    print(f"Loading {filename.value} from {tier.value}...")
    return pd.read_csv(path)


def load_train_val_test(tier: DataTier = DataTier.FULL):
    train_df = load_csv(tier, DataFile.TRAIN)
    val_df = load_csv(tier, DataFile.VAL)
    test_df = load_csv(tier, DataFile.TEST)
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    train, val, test = load_train_val_test(tier=DataTier.FULL)
    print(f"Train shape: {train.shape} | Val shape: {val.shape} | Test shape: {test.shape}")