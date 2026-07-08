import os
import json
import uuid
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def log_experiment_results(
    model_name: str,
    config: Dict[str, Any],
    model_params: Dict[str, Any],
    train_rmse: float,
    val_rmse: float,
    log_file: str = "Log/results/experiments.log",
    extra_metrics: Optional[Dict[str, float]] = None,
    notes: Optional[str] = None,
    dataset_meta: Optional[Dict[str, Any]] = None,
    training_duration: Optional[float] = None,
    feature_importance: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """Append a rich, formatted experiment record to `log_file`, save a structured JSON run,
    and update a central CSV summary.

    Args:
        model_name: Friendly name for the model (e.g. 'LinearRegression').
        config: Full pipeline/config dictionary (will be logged in full).
        model_params: Model hyperparameters (e.g. {'fit_intercept': True}).
        train_rmse: Precomputed training RMSE (float).
        val_rmse: Precomputed validation RMSE (float).
        log_file: Path to append experiment logs to.
        extra_metrics: Optional additional scalar metrics to include.
        notes: Optional freeform notes to include in the entry.
        dataset_meta: Dict containing shapes and sizes of train/val/test datasets.
        training_duration: Training execution time in seconds.
        feature_importance: Dictionary of feature names mapped to their weights or importances.

    Returns:
        A dict containing `run_id` and resolved `log_path`.
    """

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")

    # 1. Prepare Performance Metrics
    metrics = {"train_rmse": train_rmse, "val_rmse": val_rmse}
    if extra_metrics:
        metrics.update(extra_metrics)

    # 2. Quick view of important pipeline flags
    highlighted_flags: Dict[str, Any] = {}
    for key in ("use_polynomial", "use_scaling", "scaling", "polynomial"):
        if key in config:
            highlighted_flags[key] = config[key]

    # 3. Formulate the human-readable log entry
    entry = []
    entry.append("=" * 80)
    entry.append(f"EXPERIMENT RUN: {timestamp}    RUN_ID: {run_id}")
    entry.append("=" * 80)
    entry.append(f"Model Name:        {model_name}")
    if training_duration is not None:
        entry.append(f"Training Time:     {training_duration:.4f} seconds")
    
    if dataset_meta:
        entry.append("")
        entry.append("Dataset Metadata:")
        for k, v in dataset_meta.items():
            entry.append(f"  {k}: {v}")

    entry.append("")
    entry.append("Model Hyperparameters:")
    entry.append(json.dumps(model_params, indent=2, sort_keys=True))
    
    entry.append("")
    entry.append("Pipeline Config Highlights:")
    entry.append(json.dumps(highlighted_flags, indent=2, sort_keys=True))
    
    entry.append("")
    entry.append("Performance Metrics:")
    for k, v in sorted(metrics.items()):
        entry.append(f"  {k:<15}: {v:.6f}" if isinstance(v, float) else f"  {k:<15}: {v}")
    
    if feature_importance:
        entry.append("")
        entry.append("Top Feature Weights / Coefficients:")
        # Sort features by absolute coefficient magnitude descending
        sorted_features = sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)
        # Display top 20 coefficients
        for feat, val in sorted_features[:20]:
            entry.append(f"  {feat:<45}: {val:+.6f}")
        if len(sorted_features) > 20:
            entry.append(f"  ... ({len(sorted_features) - 20} more features omitted from summary view)")

    if notes:
        entry.append("")
        entry.append("Notes:")
        entry.append(str(notes))
        
    entry.append("")
    entry.append("Full Config:")
    entry.append(json.dumps(config, indent=2, sort_keys=True))
    entry.append("=" * 80)
    entry.append("\n")

    # Append entry to the human-readable log file
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(entry))

    # 4. Save detailed run JSON
    runs_dir = path.parent / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    json_path = runs_dir / f"{run_id}.json"
    
    run_details = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_name": model_name,
        "training_duration_seconds": training_duration,
        "dataset_metadata": dataset_meta,
        "model_parameters": model_params,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "full_config": config
    }
    
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(run_details, jf, indent=2, sort_keys=True)

    # 5. Append to central CSV summary
    csv_path = path.parent / "experiments_summary.csv"
    csv_exists = csv_path.exists()
    
    csv_fields = [
        "run_id",
        "timestamp",
        "model_name",
        "train_rmse",
        "val_rmse",
        "train_r2",
        "val_r2",
        "train_mae",
        "val_mae",
        "train_mse",
        "val_mse",
        "training_duration_seconds",
        "train_shape",
        "val_shape"
    ]
    
    csv_row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_name": model_name,
        "train_rmse": metrics.get("train_rmse"),
        "val_rmse": metrics.get("val_rmse"),
        "train_r2": metrics.get("train_r2"),
        "val_r2": metrics.get("val_r2"),
        "train_mae": metrics.get("train_mae"),
        "val_mae": metrics.get("val_mae"),
        "train_mse": metrics.get("train_mse"),
        "val_mse": metrics.get("val_mse"),
        "training_duration_seconds": training_duration,
        "train_shape": str(dataset_meta.get("train_shape")) if dataset_meta else None,
        "val_shape": str(dataset_meta.get("val_shape")) if dataset_meta else None,
    }
    
    with csv_path.open("a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields, extrasaction="ignore")
        if not csv_exists:
            writer.writeheader()
        writer.writerow(csv_row)

    return {"run_id": run_id, "log_path": str(path.resolve())}


def log_results():
    """Backward-compatible placeholder.
    Use `log_experiment_results` instead.
    """
    raise RuntimeError("Use log_experiment_results(...) with proper arguments")