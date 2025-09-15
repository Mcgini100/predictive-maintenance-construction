# Predictive Maintenance for Construction Equipment 

## 1. Introduction

This project implements a data-driven machine learning model to predict potential failures in construction equipment. By analyzing sensor data, operational logs, and maintenance records, the system aims to shift maintenance practices from reactive/scheduled to proactive and predictive. This enhances equipment reliability, minimizes costly unplanned downtime, optimizes maintenance schedules, and improves overall project efficiency.

This repository contains the code and documentation for the university project phase, focusing on building and evaluating the core predictive model using historical data provided in CSV format.

## 2. Problem Statement

Construction operations frequently suffer from:
*   **Unplanned Equipment Breakdowns:** Leading to significant project delays and increased costs.
*   **High Maintenance Costs:** Resulting from emergency repairs and potentially inefficient preventative schedules.
*   **Suboptimal Scheduling:** Difficulty in knowing the optimal time for maintenance interventions.

## 3. Project Goal & Objectives

**Goal:** To enhance construction machinery reliability and efficiency through early failure prediction.

**Objectives:**
*   Develop ML models for early failure detection based on historical data.
*   Provide predictive insights to potentially optimize maintenance scheduling.
*   Demonstrate cost reduction potential by minimizing simulated unplanned downtime.
*   Build a proof-of-concept system including data processing, model training, prediction, and a user interface.

## 4. Key Features

*   **Data Processing:** Includes cleaning, timestamp handling, and sorting of raw equipment data.
*   **Feature Engineering:** Creates new informative features (e.g., interaction terms, rolling averages/standard deviations) to improve model performance.
*   **Machine Learning Model:** Implements a classification model (e.g., RandomForestClassifier) to predict failure events (0 or 1).
*   **Model Training Pipeline (`src/pipelines/training_pipeline.py`):** Orchestrates the end-to-end process of loading raw data, cleaning, engineering features, preprocessing (scaling/encoding), training the model, evaluating performance, and saving the trained model and preprocessor artifacts.
*   **Prediction Pipeline (`src/pipelines/prediction_pipeline.py`):** Loads the saved artifacts and applies the full processing and prediction sequence to new, raw input data.
*   **Interactive Dashboard (`app/dashboard.py`):** A Streamlit application providing a user interface to:
    *   Input equipment data manually.
    *   Upload a CSV file with new equipment data.
    *   View failure predictions and probabilities.
    *   Receive basic alerts for high-probability failures.
*   **Testing:** Includes unit and integration tests (`tests/`) using `pytest` to ensure code reliability.
*   **Modularity:** Code is organized into logical modules (`data_processing`, `modeling`, `pipelines`).

## 5. Project Structure

Its there its the project itself, open it dzemit!!


## 6. Technology Stack

*   **Language:** Python 3.8+
*   **Core Libraries:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn
*   **Dashboard:** Streamlit
*   **Model/Object Serialization:** Joblib
*   **Testing:** Pytest
*   **Environment Management:** Conda (Recommended) or venv

## 7. Setup Instructions

**Prerequisites:**
*   Python (3.8 or higher)
*   Conda or `venv` installed

**Steps:**

1.  **Clone the Repository (or Download Code):**
    ```bash
    git clone <your-repository-url>
    cd predictive-maintenance-construction
    ```
    (If downloaded, extract and navigate to the `predictive-maintenance-construction` directory)

2.  **Create and Activate Virtual Environment:**
    *   **Using Conda:**
        ```bash
        conda create --name predmaint_env python=3.9 -y
        conda activate predmaint_env
        ```
    *   **Using `venv`:**
        ```bash
        python -m venv venv
        # Windows: .\venv\Scripts\activate
        # macOS/Linux: source venv/bin/activate
        ```
    *(Ensure the environment name `(predmaint_env)` appears in your terminal prompt)*

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Also install app requirements if running the dashboard
    pip install -r app/requirements_app.txt
    ```

4.  **Place Raw Data:**
    *   Ensure your raw data file is named `predictive_maintainance.csv`.
    *   Place it inside the `data/raw/` directory. The final path should be `predictive-maintenance-construction/data/raw/predictive_maintainance.csv`.

## 8. Usage Instructions

**Important:** Run all Python commands from the **project root directory** (`predictive-maintenance-construction/`).

1.  **Run the Training Pipeline:**
    *   This processes the raw data, trains the model, and saves the `preprocessor.joblib` and model file (e.g., `final_failure_classifier.joblib`) to the `models/` directory.
    ```bash
    python -m src.pipelines.training_pipeline
    ```

2.  **Run Predictions (Choose one method):**

    *   **Method A: Prediction Pipeline Script (Batch/Example):**
        *   Runs prediction on sample raw data defined within the script (currently uses `data/raw/predictive_maintainance.csv` as input source for the example).
        ```bash
        python -m src.pipelines.prediction_pipeline
        ```
        *   Outputs predictions to the console.

    *   **Method B: Interactive Dashboard:**
        *   Starts the Streamlit web application.
        ```bash
        streamlit run app/dashboard.py
        ```
        *   Open the local URL provided (e.g., `http://localhost:8501`) in your browser.
        *   Use the sidebar to input data manually or upload a CSV (must match raw format, without the `Failure_event` column).
        *   View results and alerts directly in the dashboard.

## 9. Running Tests

*   To verify the functionality of the code components:
    ```bash
    pytest
    ```
    *(Ensure you have installed `pytest` via `requirements.txt`)*

## 10. Future Work & Potential Improvements

*   **Real-time Data:** Integrate with live IoT sensors using MQTT or Kafka.
*   **Cloud Deployment:** Deploy pipelines and the dashboard to a cloud platform (AWS, Azure, GCP) for scalability.
*   **Advanced Models:** Explore Deep Learning (LSTMs, Transformers) for sequence data or Survival Analysis for Remaining Useful Life (RUL) estimation.
*   **Explainability (XAI):** Integrate techniques like SHAP or LIME to understand model predictions.
*   **Monitoring:** Implement model performance monitoring and drift detection.
*   **CMMS Integration:** Connect predictive alerts to existing Computerized Maintenance Management Systems.
*   **Hyperparameter Optimization:** Use more advanced techniques (e.g., Optuna, Hyperopt) for tuning.

## 11. License

Free, will add apache 2.0 

