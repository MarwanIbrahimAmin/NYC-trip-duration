import os
import sys
import shutil
import json
from datetime import datetime
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

CSV_PATH = os.path.join(PROJECT_ROOT, "Log/results/experiments_summary.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "Modeling/Saved_Models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
META_PATH = os.path.join(MODELS_DIR, "best_model_meta.json")

# Maps experiment_summary model_name to the actual filename on disk
MODEL_FILENAME_MAP = {
    "LinearRegression": "linearRegression_model.joblib",
    "RidgeRegression": "ridge_model.joblib",
    "LassoRegression": "lasso_model.joblib",
    "XGBoost": "xgboost_model.joblib",
    "Neural Network": "neuralNetwork_model.joblib",
    "LightGBM": "lightgbm_model.joblib"
}


def select_and_publish_best_model(penalty_weight: float = 0.5):
    print("Starting Dynamic Model Selection...")
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: Experiments summary file not found at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("Error: Experiments summary file is empty.")
        return
        
    # Clean and parse metrics
    df["train_rmse"] = pd.to_numeric(df["train_rmse"], errors="coerce")
    df["val_rmse"] = pd.to_numeric(df["val_rmse"], errors="coerce")
    df = df.dropna(subset=["train_rmse", "val_rmse"])
    
    # Calculate overfitting gap and selection score
    # Score = Val RMSE + Penalty * |Val RMSE - Train RMSE|
    df["overfitting_gap"] = (df["val_rmse"] - df["train_rmse"]).abs()
    df["selection_score"] = df["val_rmse"] + penalty_weight * df["overfitting_gap"]
    
    # Get the best run for each unique model to show comparison
    best_per_model = df.sort_values("selection_score").groupby("model_name").first().reset_index()
    
    print("\nModel Comparison (Best run per model category):")
    print("=" * 110)
    print(f"{'Model Name':<20} | {'Train RMSE':<12} | {'Val RMSE':<12} | {'Gap (Overfit)':<15} | {'Balance Score':<15} | {'Run ID':<10}")
    print("=" * 110)
    for _, row in best_per_model.sort_values("selection_score").iterrows():
        print(f"{row['model_name']:<20} | {row['train_rmse']:<12.6f} | {row['val_rmse']:<12.6f} | {row['overfitting_gap']:<15.6f} | {row['selection_score']:<15.6f} | {row['run_id']:<10}")
    print("=" * 110)
    
    # Select the overall absolute best run
    best_run = df.sort_values("selection_score").iloc[0]
    model_name = best_run["model_name"]
    run_id = best_run["run_id"]
    
    # Resolve file paths
    source_filename = MODEL_FILENAME_MAP.get(model_name)
    if not source_filename:
        print(f"Error: Model '{model_name}' does not have a mapped filename in MODEL_FILENAME_MAP.")
        return
        
    source_path = os.path.join(MODELS_DIR, source_filename)
    
    if not os.path.exists(source_path):
        print(f"Warning: Best model file '{source_filename}' not found at {source_path}.")
        # Try to search for any run-specific file or look for fallback
        print("Checking if we have another run file on disk...")
        return
        
    # Copy the champion model to the dynamic 'best_model.joblib' path
    print(f"\nWinner Model: {model_name} (Run ID: {run_id})")
    print(f"  - Validation RMSE: {best_run['val_rmse']:.6f}")
    print(f"  - Overfitting Gap: {best_run['overfitting_gap']:.6f}")
    print(f"  - Selection Score: {best_run['selection_score']:.6f}")
    print(f"Copying '{source_filename}' to '{BEST_MODEL_PATH}'...")
    
    shutil.copy2(source_path, BEST_MODEL_PATH)
    print("Model successfully copied.")
    
    # Write selection metadata JSON
    metadata = {
        "run_id": str(run_id),
        "model_name": str(model_name),
        "val_rmse": float(best_run["val_rmse"]),
        "train_rmse": float(best_run["train_rmse"]),
        "overfitting_gap": float(best_run["overfitting_gap"]),
        "selection_score": float(best_run["selection_score"]),
        "timestamp": str(best_run["timestamp"]),
        "selected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Selection metadata saved to '{META_PATH}'.")
    print("Done! FastAPI and Streamlit will now dynamically use the champion model.")


if __name__ == "__main__":
    select_and_publish_best_model()
