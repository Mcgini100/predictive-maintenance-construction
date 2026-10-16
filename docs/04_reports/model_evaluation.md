# Model evaluation

All numbers come from `python -m src.pipelines.training_pipeline` (scikit-learn 1.9, pandas 2.3) and are
reproduced, with plots, in [`notebooks/eda_and_model_evaluation.ipynb`](../../notebooks/eda_and_model_evaluation.ipynb).

## Setup

| Item | Value |
|---|---|
| Data | `data/raw/predictive_maintainance.csv`, 10,000 rows (1 dropped for a missing timestamp) |
| Split | stratified random 80/20, `random_state=42` → 7,999 train / 2,000 test |
| Preprocessing | median imputation + standard scaling (numeric), most-frequent imputation + one-hot (`Terrain_type`) |
| Model | `RandomForestClassifier(n_estimators=100, random_state=42)` |
| Positive class | `Failure_event = 1` (63.1% of rows) |

## Test-set results (2,000 rows)

| Metric | Value |
|---|---|
| Accuracy | 0.6435 |
| Precision (failure) | 0.6849 |
| Recall (failure) | 0.8049 |
| F1 (failure) | 0.7401 |
| ROC-AUC | 0.6676 |
| Majority-class baseline accuracy | 0.6305 |

Confusion matrix:

| | Predicted no failure | Predicted failure |
|---|---|---|
| **Actual no failure** (739) | 272 | 467 |
| **Actual failure** (1,261) | 246 | 1,015 |

### Dashboard alert rule (probability ≥ 0.7)

| | Count |
|---|---|
| Alerts raised | 727 |
| True alerts (actual failure) | 585 |
| False alerts (no failure) | 142 |
| Failures not alerted | 676 |

The 0.7 threshold trades recall for precision: 80% of alerts are real failures, but fewer than half of the
failures (585 of 1,261) trigger an alert.

## Interpretation

* Accuracy is only 1.3 points above always predicting "failure", so accuracy is not a useful headline here.
  ROC-AUC of 0.67 shows the model ranks failures above non-failures better than chance, but weakly.
* `Engine_temp` (correlation −0.27 with failure) and `Fuel_rate` (+0.12) carry most of the signal; the other
  sensor columns have near-zero correlation with the target.

## Limitations found during evaluation

* **No time dimension.** Every `Equipment_id` appears exactly once and all rows share one timestamp, so the
  rolling-window features have no history: `*_roll_mean_3` equals the raw value and `*_roll_std_3` is 0.
  They add no information on this dataset.
* **Dirty category labels.** `Terrain_type` values include literal quote characters, and `sandy_wet` /
  `sandy-wet` are one category spelled two ways. Failure rates are almost identical across terrains (62–64%).
* **Random split.** With repeated readings per machine, a time-based or machine-grouped split would be required
  to avoid optimistic estimates. That is not possible with the current snapshot data.

## Next steps

1. Normalise `Terrain_type` labels and drop the constant rolling features.
2. Tune the decision threshold (or use class weights) for the recall/precision trade-off the maintenance team wants.
3. Try gradient boosting and calibrated probabilities, compared against the baseline above.
4. Collect repeated readings per machine so time-based features and a time-based validation split become possible.
