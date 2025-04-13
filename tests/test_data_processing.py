"""
Unit tests for the data processing module (src/data_processing).
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
# Import modules to be tested after adjusting path
from src import config
from src.data_processing import load_data, cleaning, feature_engineering

# --- Test Fixtures (Optional but good practice) ---
@pytest.fixture(scope="module") # Run once per module
def sample_raw_dataframe() -> pd.DataFrame:
    """Creates a small sample DataFrame mimicking raw data."""
    data = {
        'Equipment_id': ['EQ01', 'EQ01', 'EQ02', 'EQ01'],
        'Timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 10:30:00', '2024-01-01 09:00:00'],
        'Engine_temp': [500, 550, 600, 480],
        'oil_pressure': [2.5, 2.6, 3.0, 2.4],
        'Hydraulic_pressure': [300, 301, 305, 299],
        'Operating_hours': [100, 101, 50, 99],
        'Vibrations': [30, 32, 35, 29],
        'Tool_wear': [5, 6, 2, 4],
        'Fuel_rate': [8.0, 8.5, 9.0, 7.8],
        'Ambient_temp': [295, 295, 296, 294],
        'Terrain_type': ['sandy', 'sandy', 'rocky', 'sandy'],
        'Failure_event': [0, 0, 1, 0] # Example target
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="module")
def temp_raw_csv(tmp_path_factory, sample_raw_dataframe) -> Path:
    """Creates a temporary CSV file with sample raw data."""
    # Use tmp_path_factory for module-scoped temporary directory
    temp_dir = tmp_path_factory.mktemp("raw_data")
    file_path = temp_dir / "temp_raw_data.csv"
    sample_raw_dataframe.to_csv(file_path, index=False)
    return file_path

# --- Test Functions ---

# Test load_data.py
def test_load_raw_data_success(temp_raw_csv):
    """Tests successful loading of raw data."""
    df = load_data.load_raw_data(temp_raw_csv)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'Engine_temp' in df.columns # Check if a known column exists

def test_load_raw_data_file_not_found():
    """Tests FileNotFoundError when loading raw data."""
    with pytest.raises(FileNotFoundError):
        load_data.load_raw_data(Path("non_existent_file.csv"))

# Test cleaning.py
def test_convert_timestamp(sample_raw_dataframe):
    """Tests timestamp conversion."""
    df = sample_raw_dataframe.copy()
    df_cleaned = cleaning.convert_timestamp(df, column=config.TIMESTAMP_COLUMN)
    assert pd.api.types.is_datetime64_any_dtype(df_cleaned[config.TIMESTAMP_COLUMN])
    # Test handling of bad data (optional: add a row with bad timestamp)
    df['Timestamp'] = ['2024-01-01 10:00:00', 'bad-date', '2024-01-01 10:30:00', '2024-01-01 09:00:00']
    df_cleaned_bad = cleaning.convert_timestamp(df.copy(), column=config.TIMESTAMP_COLUMN)
    assert len(df_cleaned_bad) < len(df) # Row with 'bad-date' should be dropped

def test_sort_data(sample_raw_dataframe):
    """Tests data sorting by Equipment ID and Timestamp."""
    df = sample_raw_dataframe.copy()
    # Convert timestamp first for correct sorting
    df = cleaning.convert_timestamp(df, column=config.TIMESTAMP_COLUMN)
    df_sorted = cleaning.sort_data(df)

    # Check if sorted correctly within EQ01 group
    eq01_sorted = df_sorted[df_sorted['Equipment_id'] == 'EQ01']['Timestamp'].is_monotonic_increasing
    assert eq01_sorted

    # Check overall sorting order (EQ01 before EQ02 conceptually, then time)
    # Example check: first row should be EQ01 at 09:00:00
    assert df_sorted.iloc[0]['Equipment_id'] == 'EQ01'
    assert df_sorted.iloc[0]['Timestamp'] == pd.Timestamp('2024-01-01 09:00:00')
    assert df_sorted.iloc[-1]['Equipment_id'] == 'EQ02' # Based on sample data

def test_clean_data_pipeline(sample_raw_dataframe):
    """Tests the overall clean_data function."""
    df = sample_raw_dataframe.copy()
    cleaned_df = cleaning.clean_data(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df[config.TIMESTAMP_COLUMN])
    # Check if still sorted (or if clean_data calls sort_data correctly)
    eq01_sorted = cleaned_df[cleaned_df['Equipment_id'] == 'EQ01']['Timestamp'].is_monotonic_increasing
    assert eq01_sorted


# Test feature_engineering.py
def test_create_interaction_features(sample_raw_dataframe):
    """Tests creation of interaction features."""
    df = sample_raw_dataframe.copy()
    engineered_df = feature_engineering.create_interaction_features(df)
    assert 'temp_x_vibration' in engineered_df.columns
    expected_value = df.loc[0, 'Engine_temp'] * df.loc[0, 'Vibrations']
    assert engineered_df.loc[0, 'temp_x_vibration'] == expected_value

def test_create_rolling_features(sample_raw_dataframe):
    """Tests creation of rolling features."""
    df = sample_raw_dataframe.copy()
    # Requires cleaning (sorting) first
    cleaned_df = cleaning.clean_data(df)
    engineered_df = feature_engineering.create_rolling_features(cleaned_df, target_cols=['Engine_temp'], window=2)

    assert 'Engine_temp_roll_mean_2' in engineered_df.columns
    assert 'Engine_temp_roll_std_2' in engineered_df.columns

    # Check shifted calculation for EQ01
    eq01_df = engineered_df[engineered_df['Equipment_id'] == 'EQ01']
    # First entry for EQ01 (index 0 in original, likely index 0 after sorting) should have NaN initially, then backfilled/filled
    assert not pd.isna(eq01_df.iloc[0]['Engine_temp_roll_mean_2']) # Check if filled
    # Second entry (index 1 in original, index 1 after sorting) should use value from first entry
    assert eq01_df.iloc[1]['Engine_temp_roll_mean_2'] == cleaned_df.loc[0, 'Engine_temp'] # 500
    # Third entry (index 3 in original, index 2 after sorting) should use mean of first two
    expected_mean = cleaned_df.loc[[0, 1], 'Engine_temp'].mean() # Mean of 500, 550 -> 525
    # Need to be careful with indices after sorting! Let's re-verify:
    # Sorted order for EQ01: Index 3 (9:00), Index 0 (10:00), Index 1 (11:00)
    # Rolling mean for Index 0 (10:00) should use Index 3 (9:00) value: 480
    # Rolling mean for Index 1 (11:00) should use mean(Index 3, Index 0): mean(480, 500) = 490
    assert engineered_df[engineered_df['Timestamp'] == pd.Timestamp('2024-01-01 10:00:00')]['Engine_temp_roll_mean_2'].iloc[0] == 480.0
    assert engineered_df[engineered_df['Timestamp'] == pd.Timestamp('2024-01-01 11:00:00')]['Engine_temp_roll_mean_2'].iloc[0] == 490.0

def test_engineer_features_pipeline(sample_raw_dataframe):
    """Tests the overall engineer_features function."""
    df = sample_raw_dataframe.copy()
    cleaned_df = cleaning.clean_data(df)
    engineered_df = feature_engineering.engineer_features(cleaned_df)
    assert 'temp_x_vibration' in engineered_df.columns
    assert 'Engine_temp_roll_mean_3' in engineered_df.columns # Default window is 3 in config