"""
Integration tests for the training and prediction pipelines.
These tests are heavier and might take longer to run.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Imports from src ---
from src import config
# Need to import the pipeline runners
from src.pipelines import training_pipeline, prediction_pipeline

# --- Test Setup ---
# The pipelines read their file locations from src.config. Point every path
# at a temporary directory so running the tests never overwrites the real
# dataset in data/raw/ or the committed model artifacts in models/.
@pytest.fixture(scope="module", autouse=True)
def isolated_pipeline_paths(tmp_path_factory):
    """Creates a dummy raw CSV in a temp dir and redirects all pipeline paths to it."""
    tmp = tmp_path_factory.mktemp("pipeline")
    raw_file = tmp / "raw" / "predictive_maintainance.csv"
    model_dir = tmp / "models"
    raw_file.parent.mkdir(parents=True)
    model_dir.mkdir()

    rng = np.random.default_rng(42)
    n = 50
    data = {
        'Equipment_id': [f'EQ{i%5}' for i in range(n)], # 50 samples, 5 equipment IDs
        'Timestamp': pd.date_range(start='2024-01-01', periods=n, freq='h'),
        'Engine_temp': rng.uniform(400, 900, n),
        'oil_pressure': rng.uniform(2.0, 5.0, n),
        'Hydraulic_pressure': rng.uniform(290, 320, n),
        'Operating_hours': rng.uniform(50, 2000, n),
        'Vibrations': rng.uniform(20, 60, n),
        'Tool_wear': rng.integers(0, 50, n),
        'Fuel_rate': rng.uniform(5, 20, n),
        'Ambient_temp': rng.uniform(290, 310, n),
        'Terrain_type': rng.choice(['sandy', 'rocky', 'plain', 'mixed'], n),
        'Failure_event': np.tile([0, 1], n // 2), # balanced target so stratified split works
    }
    pd.DataFrame(data).to_csv(raw_file, index=False)

    mp = pytest.MonkeyPatch()
    mp.setattr(config, "RAW_DATA_FILE", raw_file)
    mp.setattr(config, "MODEL_SAVE_DIR", model_dir)
    mp.setattr(config, "FINAL_MODEL_FILE", model_dir / config.FINAL_MODEL_FILE.name)
    mp.setattr(prediction_pipeline, "PREPROCESSOR_PATH", model_dir / "preprocessor.joblib")
    mp.setattr(prediction_pipeline, "MODEL_PATH", model_dir / config.FINAL_MODEL_FILE.name)
    mp.setattr(prediction_pipeline, "preprocessor", None)
    yield
    mp.undo()


# --- Test Functions ---

# pytest runs tests in file order, so training runs before the prediction tests.
def test_training_pipeline_runs():
    """Tests if the training pipeline runs end-to-end without critical errors."""
    try:
        training_pipeline.run_training_pipeline()
        # Check if artifacts were created
        assert (config.MODEL_SAVE_DIR / "preprocessor.joblib").exists(), "Preprocessor artifact not created."
        assert (config.MODEL_SAVE_DIR / config.FINAL_MODEL_FILE.name).exists(), "Model artifact not created."
    except Exception as e:
        pytest.fail(f"Training pipeline failed with an exception: {e}")

def test_prediction_pipeline_runs_dataframe():
    """Tests if the prediction pipeline runs with DataFrame input."""
    # Need some raw input data (without target)
    input_df = pd.read_csv(config.RAW_DATA_FILE).head(5)
    if config.TARGET_COLUMN in input_df.columns:
        input_df = input_df.drop(columns=[config.TARGET_COLUMN])

    try:
        results_df = prediction_pipeline.run_prediction_pipeline(input_df)
        assert isinstance(results_df, pd.DataFrame)
        assert not results_df.empty
        assert 'prediction' in results_df.columns
        assert 'prediction_probability' in results_df.columns
        assert len(results_df) == len(input_df)
    except FileNotFoundError:
         pytest.fail("Prediction pipeline failed: Model or preprocessor artifacts not found. Ensure training pipeline ran first.")
    except Exception as e:
        pytest.fail(f"Prediction pipeline failed with DataFrame input: {e}")

def test_prediction_pipeline_runs_dict():
    """Tests if the prediction pipeline runs with dictionary input."""
    # Create a sample dictionary
    input_dict = pd.read_csv(config.RAW_DATA_FILE).head(1).to_dict(orient='records')[0]
    if config.TARGET_COLUMN in input_dict:
        del input_dict[config.TARGET_COLUMN] # Remove target

    try:
        results_df = prediction_pipeline.run_prediction_pipeline(input_dict)
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 1
        assert 'prediction' in results_df.columns
        assert 'prediction_probability' in results_df.columns
    except FileNotFoundError:
         pytest.fail("Prediction pipeline failed: Model or preprocessor artifacts not found.")
    except Exception as e:
        pytest.fail(f"Prediction pipeline failed with dictionary input: {e}")

def test_prediction_pipeline_keeps_input_row_order():
    """clean_data() sorts by equipment/time; results must still line up with the input rows."""
    input_df = pd.read_csv(config.RAW_DATA_FILE).head(10).drop(columns=[config.TARGET_COLUMN])
    shuffled = input_df.sample(frac=1, random_state=0).reset_index(drop=True)
    results_df = prediction_pipeline.run_prediction_pipeline(shuffled)
    # Input columns are returned unchanged and in the caller's order.
    pd.testing.assert_series_equal(results_df['Equipment_id'], shuffled['Equipment_id'])
    pd.testing.assert_series_equal(results_df['Engine_temp'], shuffled['Engine_temp'])
    # Same rows, different input order -> same prediction per row.
    ordered = prediction_pipeline.run_prediction_pipeline(input_df)
    key = ['Equipment_id', 'Timestamp']
    a = results_df.sort_values(key).reset_index(drop=True)['prediction_probability']
    b = ordered.sort_values(key).reset_index(drop=True)['prediction_probability']
    np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float))
