import os
import sys
import time

# Add project root and Modeling to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Modeling"))

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from Enum.path_enums import DataTier
from Config import load_config

config = load_config()

from Helper.prepare import Preparing
from sklearn.linear_model import Lasso
from Log.apply_log import log_experiment_results


class Train():
    def __init__(self):
        self.Lasso = Lasso(alpha=config["Model"]['Lasso']['alpha'],
                           fit_intercept=config['Model']['Lasso']['fit_intercept'],
                           positive=config["Model"]['Lasso']['positive'],
                           random_state=config['RANDOM_STATE'],
                           max_iter=config['Model']['Lasso']['max_iter']
                           )
        

    def train(self, X_train, y_train):
        self.Lasso.fit(X_train, y_train)


if __name__ == '__main__':
    print("Preparing data...")
    preparer = Preparing(tier=DataTier.SAMPLE)
    X_train, y_train, X_val, y_val, X_test, y_test = preparer.prepare_data()

    print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape} | y_val shape: {y_val.shape}")

    print("Training Lasso Regression model...")
    trainer = Train()
    
    start_time = time.time()
    trainer.train(X_train, y_train)
    training_duration = time.time() - start_time
    
    print(f"Training completed in {training_duration:.4f} seconds.")

    print("Evaluating Lasso model...")
    # Generate predictions
    train_pred = trainer.Lasso.predict(X_train)
    val_pred = trainer.Lasso.predict(X_val)

    # Compute detailed metrics
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    
    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)
    
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)

    print(f"Train Evaluation:\n MSE: {train_mse:.6f} | RMSE: {train_rmse:.6f} | R2: {train_r2:.6f} | MAE: {train_mae:.6f}")
    print(f"Val Evaluation:\n MSE: {val_mse:.6f} | RMSE: {val_rmse:.6f} | R2: {val_r2:.6f} | MAE: {val_mae:.6f}")

    print("Saving results to log...")
    model_params = config['Model']['Lasso'].copy()
    model_params['random_state'] = config['RANDOM_STATE']

    
    dataset_meta = {
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape) if X_test is not None else None,
        "num_features": X_train.shape[1]
    }
    
    # Extract coefficients / feature weights
    feature_names = X_train.columns.tolist()
    coefs = trainer.Lasso.coef_
    feature_importance = {name: float(coef) for name, coef in zip(feature_names, coefs)}
    feature_importance["_intercept"] = float(trainer.Lasso.intercept_)
    
    log_info = log_experiment_results(
        model_name="LassoRegression",
        config=config,
        model_params=model_params,
        train_rmse=train_rmse,
        val_rmse=val_rmse,
        log_file="Log/results/experiments.log",
        extra_metrics={
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "train_mae": train_mae,
            "val_mae": val_mae
        },
        dataset_meta=dataset_meta,
        training_duration=training_duration,
        feature_importance=feature_importance
    )
    print(f"Experiment saved with RUN_ID: {log_info['run_id']} to {log_info['log_path']}")

    print("Saving model and preprocessor...")
    from Helper.save import save_object
    preprocessor_path = config['save_paths']['preprocessor_path']
    save_object(preparer.preprocessor, preprocessor_path)
    
    model_save_path = os.path.join(config['save_paths']['models_dir'], "lasso_model.joblib")
    save_object(trainer.Lasso, model_save_path)

