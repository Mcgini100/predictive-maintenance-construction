# Technology Stack

This document outlines the primary technologies and libraries used in the Predictive Maintenance for Construction Equipment project (University Project Phase).

## Core Language & Environment

*   **Programming Language:** Python (Version 3.8+)
*   **Environment Management:** Anaconda / Conda (Recommended), or `venv`

## Data Handling & Processing

*   **Data Storage (Input):** CSV Files (`.csv`)
*   **Core Data Manipulation:** Pandas
*   **Numerical Operations:** NumPy

## Machine Learning & Modeling

*   **Core ML Library:** Scikit-learn
    *   Preprocessing: `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`
    *   Models: `RandomForestClassifier` (Example, easily swappable with `LogisticRegression`, etc.)
    *   Evaluation: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `classification_report`
    *   Utilities: `train_test_split`
*   **Model Serialization:** Joblib

## User Interface (Proof-of-Concept)

*   **Dashboarding Library:** Streamlit

## Version Control

*   **System:** Git
*   **Platform:** GitHub / GitLab (or similar)

## Development Environment

*   **IDE:** VS Code / PyCharm / Jupyter Lab (for notebooks)

## Future / Scaled Considerations (Out of Scope for Initial Phase)

*   **Real-time Data Ingestion:** Kafka, MQTT, Cloud Pub/Sub
*   **Scalable Data Storage:** SQL Databases (PostgreSQL, MySQL), NoSQL Databases (MongoDB), Data Lakes (S3, ADLS, GCS)
*   **Distributed Processing:** Spark, Dask
*   **Cloud Platform:** AWS (SageMaker, S3, Lambda), Azure (Azure ML, Blob Storage, Functions), GCP (Vertex AI, GCS, Cloud Functions)
*   **MLOps Platforms:** MLflow, Kubeflow, Vertex AI Pipelines
*   **Monitoring:** Prometheus, Grafana, CloudWatch/Azure Monitor