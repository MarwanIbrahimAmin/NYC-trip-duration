import os
import sys
import pytest
from sklearn.linear_model import LinearRegression
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Modeling.Helper.save import save_object, load_object
from Modeling.Helper.evaluate import eval_model

def test_save_and_load_object(tmp_path):
    """Verifies that save and load functions correctly persist and retrieve objects with identical properties."""
    # tmp_path is a temporary directory fixture provided by pytest that is automatically cleaned up
    test_filepath = os.path.join(tmp_path, "test_dummy.joblib")
    data_to_save = {"alpha": 0.1, "solver": "saga", "max_iter": 100}
    
    # Test saving
    save_object(data_to_save, test_filepath)
    assert os.path.exists(test_filepath), "File was not written to disk."
    
    # Test loading
    loaded_data = load_object(test_filepath)
    assert loaded_data == data_to_save, "Retrieved object does not match the saved object."

def test_eval_model_output():
    """Verifies that the evaluation function returns mathematically sound RMSE and R2 values."""
    # Create and train a simple deterministic linear model (y = 2x)
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])  # Perfect linear relationship with zero error
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Evaluate on training data (expected RMSE is 0.0 and R2 is 1.0)
    rmse, r2 = eval_model(model, X, y, name="DummyTest")
    
    assert rmse == pytest.approx(0.0, abs=1e-7)
    assert r2 == pytest.approx(1.0)