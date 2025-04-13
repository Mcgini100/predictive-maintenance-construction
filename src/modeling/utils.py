# -*- coding: utf-8 -*-
"""
Utility functions for modeling tasks, like saving and loading objects.
"""
import joblib
from pathlib import Path
import pandas as pd
from typing import Any
from src import config # Import configuration

def save_object(obj: Any, filepath: Path):
    """
    Saves a Python object (like a model or preprocessor) to a file using joblib.

    Args:
        obj (Any): The Python object to save.
        filepath (Path): The path where the object should be saved.
    """
    try:
        # Ensure the directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, filepath)
        print(f"Object saved successfully to: {filepath}")
    except Exception as e:
        print(f"Error saving object to {filepath}: {e}")
        raise

def load_object(filepath: Path) -> Any:
    """
    Loads a Python object (like a model or preprocessor) from a file using joblib.

    Args:
        filepath (Path): The path from where the object should be loaded.

    Returns:
        Any: The loaded Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For other loading errors.
    """
    try:
        if not filepath.exists():
             raise FileNotFoundError(f"Object file not found at: {filepath}")
        obj = joblib.load(filepath)
        print(f"Object loaded successfully from: {filepath}")
        return obj
    except FileNotFoundError as e:
        print(e)
        raise
    except Exception as e:
        print(f"Error loading object from {filepath}: {e}")
        raise

# Example usage (can be commented out)
# if __name__ == '__main__':
#     # Create a dummy object and path
#     dummy_model = {"param1": 10, "type": "dummy"}
#     save_path = config.MODEL_SAVE_DIR / "dummy_model.joblib"
#
#     # Save the object
#     save_object(dummy_model, save_path)
#
#     # Load the object
#     loaded_model = load_object(save_path)
#     print(f"Loaded object: {loaded_model}")