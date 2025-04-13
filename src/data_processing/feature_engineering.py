# -*- coding: utf-8 -*-
"""
Functions for creating new features from the data.
"""
import pandas as pd
import numpy as np
from src import config

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates example interaction features.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with added interaction features.
    """
    print("Creating interaction features...")
    df_eng = df.copy()
    if 'Engine_temp' in df_eng.columns and 'Vibrations' in df_eng.columns:
        df_eng['temp_x_vibration'] = df_eng['Engine_temp'] * df_eng['Vibrations']
        print("  - Created 'temp_x_vibration'.")
    else:
        print("  - Skipping 'temp_x_vibration' (required columns missing).")

    # Add more interaction features as needed
    # e.g., if 'oil_pressure' and 'Hydraulic_pressure' exist:
    # if 'oil_pressure' in df_eng.columns and 'Hydraulic_pressure' in df_eng.columns:
    #     df_eng['pressure_ratio'] = df_eng['oil_pressure'] / (df_eng['Hydraulic_pressure'] + 1e-6) # Avoid division by zero
    #     print("  - Created 'pressure_ratio'.")

    print("Interaction feature creation finished.")
    return df_eng

def create_rolling_features(df: pd.DataFrame,
                             group_col: str = config.EQUIPMENT_ID_COLUMN,
                             target_cols: list = ['Engine_temp', 'Vibrations'], # Example columns
                             window: int = config.ROLLING_WINDOW_SIZE) -> pd.DataFrame:
    """
    Creates rolling window features (e.g., mean, std) for specified columns, grouped by equipment ID.
    Uses shift(1) to prevent data leakage from the current observation.

    Args:
        df (pd.DataFrame): Input DataFrame (should be sorted by group_col and time).
        group_col (str): Column to group by (e.g., Equipment ID).
        target_cols (list): List of column names to calculate rolling features for.
        window (int): The size of the rolling window.

    Returns:
        pd.DataFrame: DataFrame with added rolling features.
    """
    print(f"Creating rolling features (window={window}) grouped by '{group_col}'...")
    df_eng = df.copy()

    if group_col not in df_eng.columns:
        print(f"  - Warning: Group column '{group_col}' not found. Skipping rolling features.")
        return df_eng

    valid_target_cols = [col for col in target_cols if col in df_eng.columns]
    if not valid_target_cols:
        print("  - Warning: None of the specified target columns found. Skipping rolling features.")
        return df_eng

    print(f"  - Calculating for columns: {valid_target_cols}")

    # Ensure the DataFrame is sorted correctly before applying rolling functions
    if config.TIMESTAMP_COLUMN in df_eng.columns:
         df_eng = df_eng.sort_values(by=[group_col, config.TIMESTAMP_COLUMN])
    else:
         print(f"  - Warning: Timestamp column '{config.TIMESTAMP_COLUMN}' not found. Assuming data is pre-sorted.")


    for col in valid_target_cols:
        # Calculate rolling mean, shifted to use past data only
        roll_mean_col = f'{col}_roll_mean_{window}'
        df_eng[roll_mean_col] = df_eng.groupby(group_col)[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        print(f"    - Created '{roll_mean_col}'")

        # Calculate rolling std deviation, shifted
        roll_std_col = f'{col}_roll_std_{window}'
        df_eng[roll_std_col] = df_eng.groupby(group_col)[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().shift(1)
        )
        print(f"    - Created '{roll_std_col}'")

        # Fill NaNs created by shift/rolling (especially at the beginning of each group)
        # Option 1: Backfill first, then fill remaining with global mean/median or original value
        df_eng[roll_mean_col].fillna(method='bfill', inplace=True) # Backfill within groups
        df_eng[roll_std_col].fillna(method='bfill', inplace=True)
        # Fill any remaining NaNs (e.g., if a group has < window points) - use original value as approximation
        df_eng[roll_mean_col].fillna(df_eng[col], inplace=True)
        df_eng[roll_std_col].fillna(0, inplace=True) # Std dev can be 0 if constant

    print("Rolling feature creation finished.")
    return df_eng


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a sequence of feature engineering steps to the DataFrame.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with engineered features.
    """
    print("\n--- Starting Feature Engineering ---")
    df_engineered = df.copy()
    df_engineered = create_interaction_features(df_engineered)
    df_engineered = create_rolling_features(df_engineered) # Make sure data is sorted before this step!
    # Add other feature engineering functions here
    print("--- Feature Engineering Finished ---")
    return df_engineered

# Example usage (can be commented out or run within a main block):
# if __name__ == '__main__':
#     from load_data import load_raw_data
#     from cleaning import clean_data
#     try:
#         raw_df = load_raw_data()
#         cleaned_df = clean_data(raw_df) # Cleaning includes sorting needed for rolling features
#         engineered_df = engineer_features(cleaned_df)
#         print("\nEngineered Data Head:")
#         print(engineered_df.head())
#         print("\nEngineered Data Info:")
#         engineered_df.info()
#     except Exception as e:
#         print(f"Error during feature engineering process: {e}")