# -*- coding: utf-8 -*-
"""
Functions for loading data.
"""
import pandas as pd
from src import config  # Import the configuration file

def load_raw_data(file_path: str = config.RAW_DATA_FILE) -> pd.DataFrame:
    """
    Loads the raw data from the specified CSV file.

    Args:
        file_path (str): The path to the raw CSV data file.
                         Defaults to the path defined in config.py.

    Returns:
        pd.DataFrame: The loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        Exception: For other potential loading errors.
    """
    print(f"Attempting to load raw data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Raw data loaded successfully. Shape: {df.shape}")
        # Basic cleanup: Remove potential leading/trailing spaces in column names
        df.columns = df.columns.str.strip()
        print("Column names stripped of leading/trailing whitespace.")
        return df
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading raw data from {file_path}: {e}")
        raise

def load_processed_data(file_path: str = config.PROCESSED_DATA_FILE) -> pd.DataFrame:
    """
    Loads the preprocessed data from the specified CSV file.

    Args:
        file_path (str): The path to the processed CSV data file.
                         Defaults to the path defined in config.py.

    Returns:
        pd.DataFrame: The loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        Exception: For other potential loading errors.
    """
    print(f"Attempting to load processed data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Processed data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {file_path}. Run preprocessing first.")
        raise
    except Exception as e:
        print(f"Error loading processed data from {file_path}: {e}")
        raise

# Example usage (can be commented out or run within a main block):
# if __name__ == '__main__':
#     try:
#         raw_df = load_raw_data()
#         print("\nRaw Data Head:")
#         print(raw_df.head())
#     except Exception as e:
#         print(f"Failed to load raw data: {e}")

#     # Example for loading processed data (will fail if preprocessing hasn't run)
#     # try:
#     #     processed_df = load_processed_data()
#     #     print("\nProcessed Data Head:")
#     #     print(processed_df.head())
#     # except Exception as e:
#     #     print(f"Failed to load processed data: {e}")