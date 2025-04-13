"""
Pipeline script to make predictions on new, raw data:
1. Load Raw Input Data (e.g., from a CSV, dictionary, or DataFrame)
2. Clean the input data using the same cleaning functions.
3. Engineer features using the same feature engineering functions.
4. Load the saved Preprocessor.
5. Transform the new data using the loaded Preprocessor.
6. Load the saved Model.
7. Make Predictions using the loaded Model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List

from src import config
from src.data_processing.cleaning import clean_data # Re-use cleaning steps
from src.data_processing.feature_engineering import engineer_features # Re-use feature engineering
from src.modeling.utils import load_object # To load preprocessor and model
from src.modeling.predict import make_prediction, make_prediction_proba # To use the loaded model

# --- Configuration ---
PREPROCESSOR_PATH = config.MODEL_SAVE_DIR / "preprocessor.joblib"
MODEL_PATH = config.MODEL_SAVE_DIR / config.FINAL_MODEL_FILE.name

# --- Load Artifacts ---
# Load preprocessor and model once when the module is potentially imported or pipeline runs
try:
    print("Loading prediction artifacts...")
    preprocessor = load_object(PREPROCESSOR_PATH)
    # Model loading is handled lazily inside make_prediction/make_prediction_proba
    # model = load_object(MODEL_PATH) # Can optionally load it here too
    print("Prediction artifacts loaded (preprocessor). Model will be loaded on first prediction.")
except FileNotFoundError as e:
    print(f"Error: Required artifact not found: {e}")
    print("Please ensure the training pipeline has been run successfully.")
    preprocessor = None
    # model = None
except Exception as e:
    print(f"Error loading prediction artifacts: {e}")
    preprocessor = None
    # model = None


def run_prediction_pipeline(input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
    """
    Runs the full prediction pipeline on new raw input data.

    Args:
        input_data (Union[pd.DataFrame, Dict, List[Dict]]):
            The raw input data. Can be:
            - A pandas DataFrame with columns matching the original raw data.
            - A single dictionary representing one row of data.
            - A list of dictionaries, each representing a row.

    Returns:
        pd.DataFrame: A DataFrame containing the original input data (if provided as DF)
                      or derived input data, along with the 'prediction' (0 or 1)
                      and 'prediction_probability' (probability of failure, class 1).
                      Returns an empty DataFrame or raises error if artifacts are missing.

    Raises:
        ValueError: If essential artifacts (preprocessor, model) couldn't be loaded.
        TypeError: If input_data format is incorrect.
    """
    print("\n====== Starting Prediction Pipeline ======")

    if preprocessor is None:
         raise ValueError("Preprocessor artifact could not be loaded. Cannot run prediction pipeline.")
    # Check for model availability will happen within make_prediction call

    # 1. Standardize Input Data Format to DataFrame
    print("\n[Prediction Step 1/6] Standardizing Input Data...")
    if isinstance(input_data, pd.DataFrame):
        raw_df = input_data.copy()
        print(f"Input is a DataFrame with shape: {raw_df.shape}")
    elif isinstance(input_data, dict):
        raw_df = pd.DataFrame([input_data])
        print("Input is a single dictionary, converted to DataFrame.")
    elif isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
        raw_df = pd.DataFrame(input_data)
        print(f"Input is a list of dictionaries, converted to DataFrame with shape: {raw_df.shape}")
    else:
        raise TypeError("Input data must be a pandas DataFrame, a dictionary, or a list of dictionaries.")

    # Keep original columns for merging results later
    original_input_df = raw_df.copy()

    # 2. Clean Data
    # Apply the same cleaning steps used during training
    print("\n[Prediction Step 2/6] Cleaning Input Data...")
    try:
        # Ensure columns are stripped, handle timestamp etc.
        raw_df.columns = raw_df.columns.str.strip()
        cleaned_df = clean_data(raw_df) # clean_data handles timestamp conversion and sorting
    except Exception as e:
        print(f"Prediction pipeline failed during data cleaning: {e}")
        # Decide handling: return empty, raise error?
        return pd.DataFrame() # Return empty DataFrame on error

    # 3. Engineer Features
    # Apply the same feature engineering steps used during training
    print("\n[Prediction Step 3/6] Engineering Features...")
    try:
        engineered_df = engineer_features(cleaned_df)
    except Exception as e:
        print(f"Prediction pipeline failed during feature engineering: {e}")
        return pd.DataFrame()

    # 4. Prepare Features for Preprocessing
    # Select the same features that the preprocessor was trained on.
    # The preprocessor itself knows which columns to expect (numerical vs categorical)
    # but we need to ensure we provide a DataFrame with at least those columns present.
    # Drop columns NOT used as features *before* transforming.
    print("\n[Prediction Step 4/6] Preparing Features for Preprocessing...")
    try:
        # Get expected feature names from the preprocessor if possible
        # Note: This relies on the structure saved by ColumnTransformer
        expected_features = []
        for name, trans, columns in preprocessor.transformers_:
            if trans != 'drop' and columns: # 'passthrough' might have empty list? Check logic.
                 expected_features.extend(columns)
        # Handle 'remainder' if it was set to 'passthrough' and captured features
        if preprocessor.remainder == 'passthrough' and hasattr(preprocessor, 'feature_names_in_'):
            remainder_cols = [col for col in preprocessor.feature_names_in_ if col not in expected_features]
            expected_features.extend(remainder_cols)

        # Ensure only expected features are present and in the right order (DataFrame handles order usually)
        missing_features = [col for col in expected_features if col not in engineered_df.columns]
        if missing_features:
            # Handle missing features: error, impute defaults? For now, error.
            raise ValueError(f"Input data is missing expected features after engineering: {missing_features}")

        # Select only the features the preprocessor expects
        features_to_process = engineered_df[expected_features]
        print(f"Features prepared for preprocessing: {features_to_process.columns.tolist()}")

    except AttributeError:
        # Fallback if getting features from preprocessor fails
        print("Warning: Could not definitively determine expected features from preprocessor. Assuming engineered_df has necessary columns.")
        # We need to be careful here. Assume the feature engineering step produces the correct columns.
        # Drop ID/Timestamp if they are still present and not part of 'expected_features'
        cols_to_drop_pre_transform = [config.TIMESTAMP_COLUMN, config.EQUIPMENT_ID_COLUMN]
        features_to_process = engineered_df.drop(columns=[col for col in cols_to_drop_pre_transform if col in engineered_df.columns], errors='ignore')
        print(f"Features prepared for preprocessing (fallback): {features_to_process.columns.tolist()}")

    except Exception as e:
        print(f"Prediction pipeline failed during feature preparation: {e}")
        return pd.DataFrame()


    # 5. Transform Data using Loaded Preprocessor
    print("\n[Prediction Step 5/6] Transforming Data...")
    try:
        processed_data = preprocessor.transform(features_to_process)
        # Convert back to DataFrame if needed (predict functions can handle numpy)
        # processed_df = pd.DataFrame(processed_data, columns=preprocessor.get_feature_names_out(), index=features_to_process.index)
        print(f"Data transformed successfully. Shape: {processed_data.shape}")
    except Exception as e:
        print(f"Prediction pipeline failed during data transformation: {e}")
        return pd.DataFrame()


    # 6. Make Predictions using Loaded Model
    print("\n[Prediction Step 6/6] Making Predictions...")
    try:
        predictions = make_prediction(processed_data, model_path=MODEL_PATH)
        try:
            # Get probability for the positive class (failure)
            probabilities = make_prediction_proba(processed_data, model_path=MODEL_PATH)
            # Assuming binary classification, probability of failure is the second column
            failure_probability = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
        except AttributeError:
            print("Model does not support predict_proba. Probabilities set to NaN.")
            failure_probability = np.nan

        print("Predictions generated.")

    except Exception as e:
        print(f"Prediction pipeline failed during prediction generation: {e}")
        return pd.DataFrame()

    # 7. Format and Return Results
    print("\nFormatting results...")
    results_df = original_input_df.copy() # Start with original input
    results_df['prediction'] = predictions
    results_df['prediction_probability'] = failure_probability

    print("====== Prediction Pipeline Finished Successfully ======")
    return results_df


# Example Usage (can be commented out or run within a main block)
if __name__ == '__main__':
    # Example 1: Using a dictionary for a single prediction
    sample_input_dict = {
        'Equipment_id': 'FLT_TEST',
        'Timestamp': '2025-01-15 10:00:00', # Needs realistic timestamp
        'Engine_temp': 750,
        'oil_pressure': 3.1,
        'Hydraulic_pressure': 309.5,
        'Operating_hours': 1850,
        'Vibrations': 45.0,
        'Tool_wear': 25,
        'Fuel_rate': 12.5,
        'Ambient_temp': 299.0,
        'Terrain_type': 'rocky'
        # Note: Failure_event is NOT included in new data
    }
    try:
        prediction_result_single = run_prediction_pipeline(sample_input_dict)
        print("\n--- Prediction Result (Single Input) ---")
        print(prediction_result_single)
    except Exception as e:
        print(f"\nError running prediction pipeline for single dict: {e}")


    # Example 2: Using a DataFrame for batch predictions
    # Create a small DataFrame (e.g., using first few rows of raw data, *without* Failure_event)
    try:
        print("\nLoading raw data for batch prediction example...")
        raw_example_df = pd.read_csv(config.RAW_DATA_FILE).head(3)
        raw_example_df.columns = raw_example_df.columns.str.strip() # Ensure columns are clean
        if config.TARGET_COLUMN in raw_example_df.columns:
             raw_example_df = raw_example_df.drop(columns=[config.TARGET_COLUMN]) # Drop target if present

        prediction_result_batch = run_prediction_pipeline(raw_example_df)
        print("\n--- Prediction Result (Batch Input) ---")
        print(prediction_result_batch)
    except FileNotFoundError:
         print("\nRaw data file not found, skipping batch prediction example.")
    except Exception as e:
         print(f"\nError running prediction pipeline for batch DataFrame: {e}")