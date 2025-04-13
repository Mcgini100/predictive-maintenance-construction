"""
Streamlit Dashboard for Predictive Maintenance Predictions.

Allows users to input equipment data manually or upload a CSV
and get failure predictions using the trained model and pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import traceback # To display errors more clearly

# --- Add project root to sys.path ---
# This allows importing modules from 'src'
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Import the Prediction Pipeline ---
try:
    # Ensure artifacts are loaded when the pipeline module is imported
    from src.pipelines.prediction_pipeline import run_prediction_pipeline, PREPROCESSOR_PATH, MODEL_PATH
    PIPELINE_AVAILABLE = True
    if not PREPROCESSOR_PATH.exists() or not MODEL_PATH.exists():
         st.error(f"Error: Model artifacts not found! Expected Preprocessor at '{PREPROCESSOR_PATH}' and Model at '{MODEL_PATH}'. Please run the training pipeline first.")
         PIPELINE_AVAILABLE = False

except ImportError as e:
    st.error(f"Error importing prediction pipeline: {e}. Ensure src is in PYTHONPATH and required modules exist.")
    PIPELINE_AVAILABLE = False
    # Define a dummy function to prevent crashes later if import fails
    def run_prediction_pipeline(input_data):
        st.error("Prediction pipeline is not available due to import errors.")
        return pd.DataFrame({'prediction': [np.nan], 'prediction_probability': [np.nan]})

except Exception as e:
    st.error(f"An unexpected error occurred during pipeline import or artifact check: {e}")
    st.error(traceback.format_exc()) # Show detailed error in the app
    PIPELINE_AVAILABLE = False
    def run_prediction_pipeline(input_data):
        st.error("Prediction pipeline is not available due to initialization errors.")
        return pd.DataFrame({'prediction': [np.nan], 'prediction_probability': [np.nan]})


# --- Streamlit App Configuration ---
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("🚧 Predictive Maintenance for Construction Equipment")
st.markdown("Use this dashboard to predict potential equipment failures based on operational data.")

# --- Sidebar for Input Method Selection ---
st.sidebar.header("Input Data Method")
input_method = st.sidebar.radio("Choose input method:", ("Manual Input", "Upload CSV File"))

# --- Global variable for sample data structure (based on raw CSV) ---
# Fetch column names from a sample or define explicitly if pipeline not available
# This helps structure the manual input form.
# Ideally, get this from training data columns used by the preprocessor.
# For now, use expected raw columns minus the target.
EXPECTED_COLUMNS = [
    'Equipment_id', 'Timestamp', 'Engine_temp', 'oil_pressure',
    'Hydraulic_pressure', 'Operating_hours', 'Vibrations', 'Tool_wear',
    'Fuel_rate', 'Ambient_temp', 'Terrain_type'
]

input_data = None

# --- Input Form Logic ---
if PIPELINE_AVAILABLE:
    if input_method == "Manual Input":
        st.sidebar.subheader("Enter Equipment Data Manually")
        input_dict = {}
        with st.sidebar.form(key='manual_input_form'):
            # Create input fields based on expected columns
            input_dict['Equipment_id'] = st.text_input("Equipment ID", "EQ_TEST_01")
            input_dict['Timestamp'] = st.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)", pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
            input_dict['Engine_temp'] = st.number_input("Engine Temperature (°C)", min_value=0.0, value=650.0, step=10.0)
            input_dict['oil_pressure'] = st.number_input("Oil Pressure (bar)", min_value=0.0, value=3.5, step=0.1, format="%.4f")
            input_dict['Hydraulic_pressure'] = st.number_input("Hydraulic Pressure (bar)", min_value=0.0, value=305.0, step=1.0)
            input_dict['Operating_hours'] = st.number_input("Operating Hours", min_value=0, value=1500, step=50)
            input_dict['Vibrations'] = st.number_input("Vibrations (mm/s)", min_value=0.0, value=40.0, step=0.5)
            input_dict['Tool_wear'] = st.number_input("Tool Wear (units)", min_value=0, value=10, step=1)
            input_dict['Fuel_rate'] = st.number_input("Fuel Rate (L/hr)", min_value=0.0, value=10.0, step=0.5)
            input_dict['Ambient_temp'] = st.number_input("Ambient Temperature (K)", min_value=250.0, value=298.0, step=0.1)
            input_dict['Terrain_type'] = st.selectbox("Terrain Type", ['rocky', 'sandy', 'plain', 'mixed', 'marshy', 'sandy-wet'], index=0) # Add all expected types

            submit_button = st.form_submit_button(label='Predict Failure')

            if submit_button:
                input_data = input_dict

    elif input_method == "Upload CSV File":
        st.sidebar.subheader("Upload CSV File")
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                # Read the uploaded CSV
                df_upload = pd.read_csv(uploaded_file)
                st.sidebar.success("CSV file uploaded successfully!")

                # Basic validation: Check if expected columns are present (case-insensitive check example)
                df_upload.columns = df_upload.columns.str.strip() # Clean column names
                uploaded_cols_lower = [col.lower() for col in df_upload.columns]
                missing_cols = [col for col in map(str.lower, EXPECTED_COLUMNS) if col not in uploaded_cols_lower]

                if missing_cols:
                    st.sidebar.warning(f"Warning: Uploaded CSV is missing expected columns (case-insensitive): {', '.join(missing_cols)}. Predictions might fail or be inaccurate.")
                else:
                    st.sidebar.markdown("CSV columns look okay (basic check).")

                # Use the uploaded data for prediction
                input_data = df_upload

            except Exception as e:
                st.sidebar.error(f"Error reading or validating CSV: {e}")
                input_data = None


# --- Prediction Execution and Display ---
st.subheader("Prediction Results")

if input_data is not None and PIPELINE_AVAILABLE:
    st.markdown("---")
    st.write("Input Data Provided:")
    # Display input data (handle single dict vs DataFrame)
    display_input = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data
    st.dataframe(display_input.head()) # Show head if it's a larger uploaded file

    # Run the prediction pipeline
    with st.spinner("Running prediction pipeline..."):
        try:
            results_df = run_prediction_pipeline(input_data)

            if not results_df.empty:
                st.write("Prediction Output:")
                # Display results - customize columns shown if needed
                st.dataframe(results_df[['prediction', 'prediction_probability'] + [col for col in display_input.columns if col not in ['prediction', 'prediction_probability'] ]]) # Show prediction first

                # --- Alert System ---
                st.markdown("---")
                st.subheader("Maintenance Alerts")
                alert_threshold = 0.7 # Define a threshold for high failure probability

                high_prob_failures = results_df[results_df['prediction_probability'] >= alert_threshold]

                if not high_prob_failures.empty:
                     st.warning(f"🚨 **Alert!** Potential failure predicted for {len(high_prob_failures)} instance(s) with probability >= {alert_threshold}:")
                     # Display details of the high-probability predictions
                     alert_cols = ['prediction', 'prediction_probability']
                     # Try to include Equipment_id if available
                     if 'Equipment_id' in high_prob_failures.columns:
                         alert_cols.append('Equipment_id')
                     st.dataframe(high_prob_failures[alert_cols])
                else:
                     st.success("✅ No high-probability failures detected (Probability < 0.7).")

                # Optionally show probability distribution if multiple predictions
                if len(results_df) > 1:
                    st.write("Prediction Probability Distribution:")
                    fig, ax = plt.subplots() # Requires matplotlib
                    results_df['prediction_probability'].hist(ax=ax, bins=10)
                    ax.set_xlabel("Predicted Failure Probability")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig) # Requires adding 'matplotlib' to requirements_app.txt

            else:
                st.error("Prediction pipeline returned empty results. Check logs or input data.")

        except Exception as e:
            st.error("An error occurred during prediction:")
            st.error(traceback.format_exc()) # Show detailed error

elif not PIPELINE_AVAILABLE:
    st.warning("Prediction functionality is unavailable because the prediction pipeline or necessary model artifacts could not be loaded. Please check the application logs and ensure the training pipeline has run successfully.")
else:
    st.info("Please provide input data using the sidebar options to generate a prediction.")

st.markdown("---")
st.markdown("Developed as part of the University Project.")