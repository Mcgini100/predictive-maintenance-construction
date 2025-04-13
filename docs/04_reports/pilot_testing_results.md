# Pilot Testing Results (Simulated)

## 1. Objective

To simulate the deployment of the predictive maintenance system on a representative set of equipment data and evaluate its end-to-end functionality, prediction accuracy on unseen data, and the usability of the dashboard interface.

## 2. Methodology

*   **Test Data:** A hold-out set comprising [Number]% of the original `predictive_maintainance.csv` dataset was used. This data was not used during the model training or hyperparameter tuning phases. The 'Failure_event' column was removed to simulate new, incoming data.
    *   Alternatively: The test set generated during the `training_pipeline.py` split (Size: [Number] samples) was used.
*   **Execution:**
    *   [Method 1 Used: Batch Prediction] The hold-out data (as a CSV) was processed using the `src/pipelines/prediction_pipeline.py` script.
    *   [Method 2 Used: Dashboard UI] Sample data points from the hold-out set were manually entered into the Streamlit dashboard (`app/dashboard.py`), and a subset was uploaded via the CSV upload feature.
*   **Evaluation:**
    *   The generated predictions (`prediction` and `prediction_probability`) were compared against the true `Failure_event` values from the hold-out set.
    *   Standard classification metrics (Accuracy, Precision, Recall, F1-score) were calculated for the hold-out set performance.
    *   The dashboard alerts (triggered for `prediction_probability >= 0.7`) were reviewed for correctness (True Positives, False Positives).
    *   Qualitative feedback on the dashboard usability was gathered (simulated via discussion with [Supervisor/Peers]).

## 3. Quantitative Results

*   **Hold-out Set Performance:**
    *   Accuracy: [Value]%
    *   Precision: [Value]%
    *   Recall: [Value]%
    *   F1-score: [Value]%
    *   Confusion Matrix:
        ```
        [[TN, FP],
         [FN, TP]]
        ```
        (Replace TN, FP, FN, TP with actual numbers from the hold-out set)

*   **Alert Analysis:**
    *   Number of True Alerts (High probability & Actual Failure): [Number]
    *   Number of False Alerts (High probability & No Actual Failure): [Number]
    *   Number of Missed Failures (Low probability & Actual Failure): [Number]

## 4. Qualitative Findings & Feedback

*   **Dashboard Usability:**
    *   (Feedback Summary) e.g., "The dashboard was generally clear...", "Inputting data manually was straightforward...", "CSV upload worked as expected...", "Visualizations were helpful/could be improved..."
*   **Alert System:**
    *   (Feedback Summary) e.g., "Alert threshold of 0.7 seemed reasonable/too high/too low...", "More context needed with alerts (e.g., specific sensor values contributing?)...", "Email/SMS notification would be essential in practice..."
*   **Overall System Functionality:**
    *   (Feedback Summary) e.g., "The end-to-end flow from raw data to prediction was demonstrated successfully...", "Potential for reducing downtime seems plausible based on recall..."

## 5. Issues Encountered & Limitations

*   [Issue 1, e.g., Specific data format required for CSV upload]
*   [Issue 2, e.g., Timestamp parsing strictness]
*   Limitation: Simulation based on historical data; real-world sensor noise, data delays, and operational changes are not fully captured.
*   Limitation: Small scale of testing compared to a full fleet deployment.

## 6. Conclusion & Recommendations from Pilot

*   The simulated pilot test indicates that the system [is promising / has potential / requires further refinement].
*   The predictive performance on unseen data achieved [Level] accuracy/recall.
*   Recommendations:
    *   [Recommendation 1, e.g., Refine feature engineering based on error analysis]
    *   [Recommendation 2, e.g., Enhance dashboard with feature importance or historical trends]
    *   [Recommendation 3, e.g., Investigate adjusting the alert threshold]