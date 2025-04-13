# -*- coding: utf-8 -*-
"""
Script to train the predictive maintenance model.
Loads processed data, trains a model, evaluates it, and saves the model.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Example: Using RandomForest
# from sklearn.linear_model import LogisticRegression # Alternative model
# from xgboost import XGBClassifier # Another alternative

from src import config
from src.data_processing.load_data import load_processed_data
from src.modeling.evaluate import print_evaluation_report, calculate_metrics
from src.modeling.utils import save_object
import joblib # Using joblib directly here for simplicity, could use utils

def train_model(data_path: str = config.PROCESSED_DATA_FILE,
                model_save_path: str = config.FINAL_MODEL_FILE):
    """
    Loads processed data, trains a classification model, evaluates it,
    and saves the trained model.

    Args:
        data_path (str): Path to the processed data CSV file.
        model_save_path (str): Path where the trained model should be saved.
    """
    print("\n--- Starting Model Training ---")

    # 1. Load Processed Data
    try:
        df = load_processed_data(data_path)
    except FileNotFoundError:
        print(f"Error: Processed data not found at {data_path}. Run preprocessing first.")
        return
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return

    # 2. Separate Features (X) and Target (y)
    if config.TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{config.TARGET_COLUMN}' not found in processed data.")
        return

    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Check if features exist
    if X.empty:
        print("Error: No features found in the processed data.")
        return

    # 3. Split Data into Training and Testing Sets
    print(f"Splitting data into train/test sets (Test size: {config.TEST_SPLIT_RATIO}, Random state: {config.RANDOM_STATE})")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SPLIT_RATIO,
            random_state=config.RANDOM_STATE,
            stratify=y # Important for potentially imbalanced classification
        )
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
    except ValueError as e:
         print(f"Error during train/test split (potentially too few samples for stratification): {e}")
         # Fallback without stratification if needed, though less ideal
         try:
             print("Retrying split without stratification...")
             X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config.TEST_SPLIT_RATIO,
                random_state=config.RANDOM_STATE
             )
             print(f"Training set size: {X_train.shape[0]}")
             print(f"Test set size: {X_test.shape[0]}")
         except Exception as split_e:
             print(f"Failed to split data even without stratification: {split_e}")
             return

    # 4. Initialize and Train the Model
    # --- Choose your model here ---
    # model = LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000)
    model = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1)
    # model = XGBClassifier(random_state=config.RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
    # -----------------------------

    print(f"\nTraining model: {model.__class__.__name__}...")
    try:
        model.fit(X_train, y_train)
        print("Model training completed.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # 5. Evaluate the Model on the Test Set
    print("\nEvaluating model performance on the test set...")
    try:
        y_pred = model.predict(X_test)
        # Get probabilities for ROC AUC calculation (requires predict_proba method)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test) # Returns probabilities for all classes
            # y_prob_positive = y_prob[:, 1] # Probability of positive class (usually index 1)
        else:
            y_prob = None
            print("Model does not support predict_proba, ROC AUC will not be calculated.")

        print_evaluation_report(y_test, y_pred, y_prob)

    except Exception as e:
        print(f"Error during model evaluation: {e}")
        # Optionally, still try to save the model even if evaluation failed partially
        # save_object(model, model_save_path)
        # return # Or continue to save

    # 6. Save the Trained Model
    print(f"\nSaving the trained model to {model_save_path}...")
    save_object(model, model_save_path)

    print("--- Model Training Script Finished ---")

# Entry point for running the script directly
if __name__ == '__main__':
    train_model()