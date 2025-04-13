# -*- coding: utf-8 -*-
"""
Configuration file for the Predictive Maintenance project.
Stores file paths, constants, and parameters.
"""

import os
from pathlib import Path

# --- Project Root ---
# Assumes this config.py file is in the 'src' directory
PROJECT_ROOT = Path(__file__).parent.parent

# --- Data Paths ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure processed data directory exists
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_FILE = RAW_DATA_DIR / "predictive_maintainance.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "preprocessed_data.csv"
# Optional: Paths for saving train/test splits
TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "X_train.csv"
TEST_FEATURES_FILE = PROCESSED_DATA_DIR / "X_test.csv"
TRAIN_TARGET_FILE = PROCESSED_DATA_DIR / "y_train.csv"
TEST_TARGET_FILE = PROCESSED_DATA_DIR / "y_test.csv"


# --- Feature Engineering Parameters ---
# Example: Rolling window size
ROLLING_WINDOW_SIZE = 3


# --- Data Columns ---
# Define column names to avoid typos and manage changes easily
TARGET_COLUMN = 'Failure_event'
TIMESTAMP_COLUMN = 'Timestamp'
EQUIPMENT_ID_COLUMN = 'Equipment_id'

# Features to potentially drop before modeling (add others if needed)
FEATURES_TO_DROP = [TARGET_COLUMN, TIMESTAMP_COLUMN, EQUIPMENT_ID_COLUMN]


# --- Modeling Parameters (Placeholder for Phase 4) ---
MODEL_SAVE_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_FILE = MODEL_SAVE_DIR / "final_failure_classifier.joblib"

TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42 # for reproducibility


# --- Add other configurations as needed ---