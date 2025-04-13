"""
Functions for making predictions using a trained model.
NOTE: This assumes the input data is ALREADY PREPROCESSED
in the same way as the training data. A full prediction pipeline
would include the preprocessing step.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from src import config
from src.modeling.utils import load_object
from typing import Union

# Global variable to hold the loaded model (lazy loading)
_model = None

def load_prediction_model(model_path: str = config.FINAL_MODEL_FILE):
    """Loads the trained model globally for prediction."""
    global _model
    if _model is None:
        print("Loading prediction model...")
        _model = load_object(Path(model_path))
    return _model

def make_prediction(input_data: Union[pd.DataFrame, np.ndarray],
                    model_path: str = config.FINAL_MODEL_FILE) -> np.ndarray:
    """
    Makes predictions on new (preprocessed) input data using the loaded model.

    Args:
        input_data (Union[pd.DataFrame, np.ndarray]): The preprocessed input data
                                                      for which predictions are needed.
                                                      Should have the same features
                                                      as the training data.
        model_path (str): Path to the saved trained model file.

    Returns:
        np.ndarray: The predicted labels (0 or 1).

    Raises:
        ValueError: If the model is not loaded or input data is invalid.
    """
    model = load_prediction_model(model_path)

    if model is None:
        raise ValueError("Model has not been loaded. Run load_prediction_model first or check path.")

    if isinstance(input_data, pd.DataFrame):
        # Potentially ensure column order matches training data if model is sensitive
        # Required features can be obtained from model.feature_names_in_ if available (sklearn >= 0.24)
        # Or saved separately during training. For now, assume columns are correct.
        print(f"Making predictions on DataFrame with shape: {input_data.shape}")
        input_array = input_data.values
    elif isinstance(input_data, np.ndarray):
        print(f"Making predictions on NumPy array with shape: {input_data.shape}")
        input_array = input_data
    else:
        raise ValueError("Input data must be a pandas DataFrame or NumPy array.")

    try:
        predictions = model.predict(input_array)
        print("Predictions generated successfully.")
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


def make_prediction_proba(input_data: Union[pd.DataFrame, np.ndarray],
                          model_path: str = config.FINAL_MODEL_FILE) -> np.ndarray:
    """
    Makes probability predictions on new (preprocessed) input data using the loaded model.

    Args:
        input_data (Union[pd.DataFrame, np.ndarray]): The preprocessed input data.
        model_path (str): Path to the saved trained model file.

    Returns:
        np.ndarray: The predicted probabilities for each class (e.g., [prob_class_0, prob_class_1]).

    Raises:
        ValueError: If the model is not loaded or input data is invalid.
        AttributeError: If the loaded model does not have a 'predict_proba' method.
    """
    model = load_prediction_model(model_path)

    if model is None:
        raise ValueError("Model has not been loaded. Run load_prediction_model first or check path.")

    if not hasattr(model, "predict_proba"):
        raise AttributeError(f"The loaded model ({model.__class__.__name__}) does not support 'predict_proba'.")

    if isinstance(input_data, pd.DataFrame):
        print(f"Making probability predictions on DataFrame with shape: {input_data.shape}")
        input_array = input_data.values
    elif isinstance(input_data, np.ndarray):
         print(f"Making probability predictions on NumPy array with shape: {input_data.shape}")
         input_array = input_data
    else:
        raise ValueError("Input data must be a pandas DataFrame or NumPy array.")

    try:
        probabilities = model.predict_proba(input_array)
        print("Probability predictions generated successfully.")
        return probabilities
    except Exception as e:
        print(f"Error during probability prediction: {e}")
        raise


# Example usage (can be commented out or run within a main block):
# if __name__ == '__main__':
#     from src.data_processing.load_data import load_processed_data
#     from sklearn.model_selection import train_test_split # To get some test data
#
#     try:
#         # 1. Load some processed data (just to get test data for demonstration)
#         df = load_processed_data()
#         X = df.drop(columns=[config.TARGET_COLUMN])
#         y = df[config.TARGET_COLUMN]
#         _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE)
#
#         # Ensure model is trained and saved first by running train.py
#
#         # 2. Make predictions on the test data (first 5 rows)
#         sample_data = X_test.head(5)
#         predictions = make_prediction(sample_data)
#         print("\nSample Predictions (Labels):")
#         print(predictions)
#
#         # 3. Make probability predictions
#         try:
#             probabilities = make_prediction_proba(sample_data)
#             print("\nSample Predictions (Probabilities):")
#             print(probabilities)
#         except AttributeError as e:
#             print(f"\nCould not get probabilities: {e}")
#
#     except FileNotFoundError:
#         print("\nRun training script (src/modeling/train.py) first to create the model file.")
#     except Exception as e:
#         print(f"\nAn error occurred during prediction example: {e}")