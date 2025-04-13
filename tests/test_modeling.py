"""
Unit tests for the modeling module (src/modeling).
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression # Use a simple model for testing utils
from pathlib import Path
import sys
import os
import joblib

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Imports from src ---
from src import config
from src.modeling import evaluate, utils, predict

# --- Test Data ---
y_true_dummy = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
y_pred_dummy = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 0])
# Probabilities for class 1
y_prob_dummy = np.array([0.1, 0.8, 0.6, 0.9, 0.4, 0.2, 0.7, 0.75, 0.3, 0.45])
# Probabilities for class 0 and 1 (as predict_proba might return)
y_prob_2d_dummy = np.vstack([1 - y_prob_dummy, y_prob_dummy]).T

# --- Test evaluate.py ---
def test_calculate_metrics():
    """Tests calculation of evaluation metrics."""
    metrics = evaluate.calculate_metrics(y_true_dummy, y_pred_dummy, y_prob_dummy)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'roc_auc' in metrics
    # Check some expected values (can calculate manually or use sklearn directly)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    assert metrics['accuracy'] == pytest.approx(accuracy_score(y_true_dummy, y_pred_dummy))
    assert metrics['precision'] == pytest.approx(precision_score(y_true_dummy, y_pred_dummy))
    assert metrics['recall'] == pytest.approx(recall_score(y_true_dummy, y_pred_dummy))
    assert metrics['f1_score'] == pytest.approx(f1_score(y_true_dummy, y_pred_dummy))
    assert metrics['roc_auc'] == pytest.approx(roc_auc_score(y_true_dummy, y_prob_dummy))

def test_calculate_metrics_no_proba():
    """Tests metrics calculation when probability is not provided."""
    metrics = evaluate.calculate_metrics(y_true_dummy, y_pred_dummy)
    assert 'roc_auc' in metrics
    assert np.isnan(metrics['roc_auc']) # Should be NaN or similar indication

def test_calculate_metrics_2d_proba():
     """Tests metrics calculation with 2D probability array."""
     metrics = evaluate.calculate_metrics(y_true_dummy, y_pred_dummy, y_prob_2d_dummy)
     assert 'roc_auc' in metrics
     from sklearn.metrics import roc_auc_score
     assert metrics['roc_auc'] == pytest.approx(roc_auc_score(y_true_dummy, y_prob_2d_dummy[:, 1]))

# Test utils.py
@pytest.fixture
def temp_model_path(tmp_path) -> Path:
    """Provides a temporary path for saving/loading models."""
    return tmp_path / "test_model.joblib"

def test_save_load_object(temp_model_path):
    """Tests saving and loading a Python object."""
    dummy_object = {'a': 1, 'b': [1, 2, 3], 'c': 'test'}
    utils.save_object(dummy_object, temp_model_path)
    assert temp_model_path.exists()

    loaded_object = utils.load_object(temp_model_path)
    assert loaded_object == dummy_object

def test_load_object_not_found(tmp_path):
    """Tests loading a non-existent object."""
    non_existent_path = tmp_path / "not_real.joblib"
    with pytest.raises(FileNotFoundError):
        utils.load_object(non_existent_path)


# Test predict.py (requires a dummy trained model)
@pytest.fixture(scope="module")
def dummy_trained_model_and_path(tmp_path_factory):
    """Creates and saves a dummy trained model for prediction tests."""
    model = LogisticRegression()
    # Create dummy features/target for fitting
    X_dummy = np.random.rand(10, 3) # 10 samples, 3 features
    y_dummy = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    model.fit(X_dummy, y_dummy)

    # Save the model
    model_dir = tmp_path_factory.mktemp("models")
    model_path = model_dir / "dummy_predictor.joblib"
    joblib.dump(model, model_path)
    return model, model_path

@pytest.fixture
def dummy_preprocessed_input() -> np.ndarray:
    """Creates dummy preprocessed input data for prediction."""
    return np.random.rand(5, 3) # 5 samples, 3 features matching the dummy model

def test_make_prediction(dummy_trained_model_and_path, dummy_preprocessed_input):
    """Tests making predictions."""
    model, model_path = dummy_trained_model_and_path
    predictions = predict.make_prediction(dummy_preprocessed_input, model_path=model_path)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(dummy_preprocessed_input)
    assert all(p in [0, 1] for p in predictions) # Check if predictions are binary

def test_make_prediction_proba(dummy_trained_model_and_path, dummy_preprocessed_input):
    """Tests making probability predictions."""
    model, model_path = dummy_trained_model_and_path
    probabilities = predict.make_prediction_proba(dummy_preprocessed_input, model_path=model_path)
    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == len(dummy_preprocessed_input)
    assert probabilities.shape[1] == 2 # Should have probabilities for both classes
    assert np.all((probabilities >= 0) & (probabilities <= 1)) # Check range
    assert np.allclose(np.sum(probabilities, axis=1), 1.0) # Check probabilities sum to 1