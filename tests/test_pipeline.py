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
# Ensure necessary directories exist or are created by fixtures/pipelines
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Create a dummy raw data file for testing pipelines
@pytest.fixture(scope="module", autouse=True) # Autouse ensures it runs for all tests in module
def create_dummy_raw_data_for_pipeline():
    """Creates a dummy raw CSV file in the expected location for pipeline tests."""
    data = {
        'Equipment_id': [f'EQ{i%5}' for i in range(50)], # 50 samples, 5 equipment IDs
        'Timestamp': pd.to_datetime(pd.date_range(start='2024-01-01', periods=50, freq='H')),
        'Engine_temp': np.random.uniform(400, 900, 50),
        'oil_pressure': np.random.uniform(2.0, 5.0, 50),
        'Hydraulic_pressure': np.random.uniform(290, 320, 50),
        'Operating_hours': np.random.uniform(50, 2000, 50),
        'Vibrations': np.random.uniform(20, 60, 50),
        'Tool_wear': np.random.randint(0, 50, 50),
        'Fuel_rate': np.random.uniform(5, 20, 50),
        'Ambient_temp': np.random.uniform(290, 310, 50),
        'Terrain_type': np.random.choice(['sandy', 'rocky', 'plain', 'mixed'], 50),
        'Failure_event': np.random.randint(0, 2, 50) # Random target
    }
    df = pd.DataFrame(data)
    df.to_csv(config.RAW_DATA_FILE, index=False)
    print(f"Dummy raw data created at {config.RAW_DATA_FILE}")

    yield # Allow tests to run

    # Teardown: Clean up created files (optional, pytest handles tmp dirs well)
    # print("Cleaning up pipeline test artifacts...")
    # if config.RAW_DATA_FILE.exists(): os.remove(config.RAW_DATA_FILE)
    # if config.PROCESSED_DATA_FILE.exists(): os.remove(config.PROCESSED_DATA_FILE)
    # preprocessor_file = config.MODEL_SAVE_DIR / "preprocessor.joblib"
    # model_file = config.MODEL_SAVE_DIR / config.FINAL_MODEL_FILE.name
    # if preprocessor_file.exists(): os.remove(preprocessor_file)
    # if model_file.exists(): os.remove(model_file)


# --- Test Functions ---

# Use 'order' marker if sequential execution is critical (training before prediction)
# Requires pytest-order: pip install pytest-order
@pytest.mark.order(1)
def test_training_pipeline_runs():
    """Tests if the training pipeline runs end-to-end without critical errors."""
    try:
        training_pipeline.run_training_pipeline()
        # Check if artifacts were created
        assert (config.MODEL_SAVE_DIR / "preprocessor.joblib").exists(), "Preprocessor artifact not created."
        assert (config.MODEL_SAVE_DIR / config.FINAL_MODEL_FILE.name).exists(), "Model artifact not created."
    except Exception as e:
        pytest.fail(f"Training pipeline failed with an exception: {e}")

@pytest.mark.order(2)
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

@pytest.mark.order(3)
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