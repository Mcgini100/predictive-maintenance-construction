# -*- coding: utf-8 -*-
"""
Functions for cleaning the raw data.
Handles timestamp conversion, sorting, and potentially outlier removal (if defined).
"""
import pandas as pd
from src import config

def convert_timestamp(df: pd.DataFrame, column: str = config.TIMESTAMP_COLUMN) -> pd.DataFrame:
    """
    Converts the specified timestamp column to datetime objects.
    Drops rows where conversion fails.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Name of the timestamp column. Defaults to config.TIMESTAMP_COLUMN.

    Returns:
        pd.DataFrame: DataFrame with the timestamp column converted.
    """
    if column in df.columns:
        print(f"Converting '{column}' to datetime...")
        initial_rows = len(df)
        # Use errors='coerce' to turn unparseable dates into NaT (Not a Time)
        df[column] = pd.to_datetime(df[column], errors='coerce')
        # Drop rows with NaT timestamps
        df.dropna(subset=[column], inplace=True)
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} rows due to invalid timestamp format.")
        print(f"'{column}' converted successfully.")
    else:
        print(f"Warning: Timestamp column '{column}' not found in DataFrame.")
    return df

def sort_data(df: pd.DataFrame,
              sort_by_cols: list = [config.EQUIPMENT_ID_COLUMN, config.TIMESTAMP_COLUMN]) -> pd.DataFrame:
    """
    Sorts the DataFrame by the specified columns (typically Equipment ID and Timestamp).

    Args:
        df (pd.DataFrame): Input DataFrame.
        sort_by_cols (list): List of column names to sort by.
                             Defaults to [config.EQUIPMENT_ID_COLUMN, config.TIMESTAMP_COLUMN].

    Returns:
        pd.DataFrame: Sorted DataFrame.
    """
    existing_sort_cols = [col for col in sort_by_cols if col in df.columns]
    if not existing_sort_cols:
        print("Warning: None of the specified sort columns exist. Skipping sort.")
        return df

    if len(existing_sort_cols) < len(sort_by_cols):
        missing = set(sort_by_cols) - set(existing_sort_cols)
        print(f"Warning: Sort columns not found: {missing}. Sorting by available columns: {existing_sort_cols}")

    print(f"Sorting data by {existing_sort_cols}...")
    df_sorted = df.sort_values(by=existing_sort_cols).reset_index(drop=True)
    print("Data sorted successfully.")
    return df_sorted

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a sequence of cleaning steps to the raw DataFrame.

    Args:
        df (pd.DataFrame): The raw DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    print("\n--- Starting Data Cleaning ---")
    df_cleaned = df.copy()
    df_cleaned = convert_timestamp(df_cleaned)
    df_cleaned = sort_data(df_cleaned)
    # Add other cleaning steps here if needed (e.g., outlier handling)
    # Example: handle_outliers(df_cleaned, column='Engine_temp', method='IQR')
    print("--- Data Cleaning Finished ---")
    return df_cleaned

# Example usage (can be commented out or run within a main block):
# if __name__ == '__main__':
#     from load_data import load_raw_data
#     try:
#         raw_df = load_raw_data()
#         cleaned_df = clean_data(raw_df)
#         print("\nCleaned Data Head:")
#         print(cleaned_df.head())
#         print("\nCleaned Data Info:")
#         cleaned_df.info()
#     except Exception as e:
#         print(f"Error during cleaning process: {e}")