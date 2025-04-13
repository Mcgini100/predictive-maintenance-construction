"""
Main pipeline script to run the full training process:
1. Load Raw Data
2. Clean Data
3. Engineer Features
4. Preprocess Features (Scaling, Encoding) -> Save processed data
5. Train Model -> Save trained model
"""

from src import config
from src.data_processing.load_data import load_raw_data
from src.data_processing.cleaning import clean_data
from src.data_processing.feature_engineering import engineer_features
from src.modeling.train import train_model # train_model loads the *processed* data itself

# Imports for preprocessing (if applying here instead of saving/reloading)
# We need to decide if preprocessing happens here or if train_model expects preprocessed data.
# Current setup: engineer_features outputs data -> train_model loads it.
# Let's refine this. Preprocessing (scaling/encoding) should ideally be part of the *training*
# script or saved as a separate object to apply consistently during prediction.

# ---> REVISED APPROACH: Preprocessing is integrated within the training pipeline before saving final data <---
# Let's modify the flow slightly:
# Raw -> Clean -> Engineer -> Preprocess (Fit on Train, Transform Train/Test) -> Train Model -> Save Model & Preprocessor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.modeling.utils import save_object # Use utils to save preprocessor

# --- Preprocessing Function (similar to notebook script but modular) ---
def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Creates a ColumnTransformer for preprocessing."""
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    print(f"Preprocessor: Identifying numerical features: {numerical_features}")
    print(f"Preprocessor: Identifying categorical features: {categorical_features}")

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any - adjust if necessary
    )
    return preprocessor, numerical_features, categorical_features


def run_training_pipeline():
    """Executes the full training pipeline."""
    print("====== Starting Training Pipeline ======")

    # 1. Load Raw Data
    print("\n[Pipeline Step 1/6] Loading Raw Data...")
    try:
        raw_df = load_raw_data(config.RAW_DATA_FILE)
    except Exception as e:
        print(f"Pipeline failed at Load Raw Data stage: {e}")
        return

    # 2. Clean Data
    print("\n[Pipeline Step 2/6] Cleaning Data...")
    try:
        cleaned_df = clean_data(raw_df)
    except Exception as e:
        print(f"Pipeline failed at Clean Data stage: {e}")
        return

    # 3. Engineer Features
    print("\n[Pipeline Step 3/6] Engineering Features...")
    try:
        engineered_df = engineer_features(cleaned_df)
    except Exception as e:
        print(f"Pipeline failed at Engineer Features stage: {e}")
        return

    # 4. Prepare Data for Preprocessing & Split
    print("\n[Pipeline Step 4/6] Preparing Data & Splitting...")
    try:
        if config.TARGET_COLUMN not in engineered_df.columns:
            raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found after feature engineering.")

        y = engineered_df[config.TARGET_COLUMN]
        # Drop target and potentially other identifiers NOT used as features
        # We keep Equipment ID if rolling features used it, but drop it before modeling
        features_to_drop_final = [config.TARGET_COLUMN, config.TIMESTAMP_COLUMN]
        if config.EQUIPMENT_ID_COLUMN in engineered_df.columns:
             features_to_drop_final.append(config.EQUIPMENT_ID_COLUMN)

        X = engineered_df.drop(columns=[col for col in features_to_drop_final if col in engineered_df.columns])

        print(f"Final features for modeling: {X.columns.tolist()}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SPLIT_RATIO,
            random_state=config.RANDOM_STATE,
            stratify=y
        )
        print(f"Data split complete. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    except Exception as e:
        print(f"Pipeline failed at Data Preparation/Split stage: {e}")
        return

    # 5. Preprocess Features (Fit on Train, Transform Train/Test) & Save Preprocessor
    print("\n[Pipeline Step 5/6] Preprocessing Features...")
    try:
        # Create the preprocessor based on the training data's structure
        preprocessor, num_feats, cat_feats = create_preprocessor(X_train)

        # Fit the preprocessor on the training data ONLY
        print("Fitting preprocessor on training data...")
        preprocessor.fit(X_train)

        # Save the fitted preprocessor
        preprocessor_save_path = config.MODEL_SAVE_DIR / "preprocessor.joblib"
        save_object(preprocessor, preprocessor_save_path)

        # Transform both training and test data
        print("Transforming training data...")
        X_train_processed = preprocessor.transform(X_train)
        print("Transforming test data...")
        X_test_processed = preprocessor.transform(X_test)

        # Get feature names after transformation (important for model interpretability/debugging)
        try:
            ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_feats)
            processed_feature_names = num_feats + list(ohe_feature_names)
            print(f"Number of features after preprocessing: {len(processed_feature_names)}")
        except Exception:
            processed_feature_names = None # Fallback if names can't be retrieved
            print("Warning: Could not retrieve feature names after preprocessing.")

        # Convert processed arrays back to DataFrames (optional but good practice)
        X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_feature_names, index=X_train.index)
        X_test_processed_df = pd.DataFrame(X_test_processed, columns=processed_feature_names, index=X_test.index)

        # --- Optional: Save processed train/test splits ---
        # This allows train.py to directly load these if preferred over passing them in memory
        # save_object(X_train_processed_df, config.TRAIN_FEATURES_FILE)
        # save_object(X_test_processed_df, config.TEST_FEATURES_FILE)
        # save_object(y_train, config.TRAIN_TARGET_FILE) # Use save_object for Series too
        # save_object(y_test, config.TEST_TARGET_FILE)
        # -----------------------------------------------

    except Exception as e:
        print(f"Pipeline failed at Preprocessing stage: {e}")
        return

    # 6. Train Model (Using the preprocessed data)
    print("\n[Pipeline Step 6/6] Training Model...")
    try:
        # We need to slightly modify train_model to accept preprocessed data directly
        # Or adjust the pipeline to save processed data and have train_model load it.
        # Let's adapt train_model concept slightly for pipeline integration:
        # We'll reuse the core logic but pass data instead of loading files.

        # Initialize Model (example: RandomForest)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1)

        # Train
        print(f"Training {model.__class__.__name__}...")
        model.fit(X_train_processed_df, y_train) # Use DataFrame if names are important, else use NumPy array
        print("Model training completed.")

        # Evaluate
        print("Evaluating model on test set...")
        from src.modeling.evaluate import print_evaluation_report
        y_pred = model.predict(X_test_processed_df)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_processed_df)
        else:
            y_prob = None
        print_evaluation_report(y_test, y_pred, y_prob)

        # Save Model
        model_save_path = config.MODEL_SAVE_DIR / config.FINAL_MODEL_FILE.name # Ensure using Path object correctly
        save_object(model, model_save_path)

    except Exception as e:
        print(f"Pipeline failed at Model Training stage: {e}")
        return

    print("\n====== Training Pipeline Finished Successfully ======")


if __name__ == "__main__":
    run_training_pipeline()