# Data quality report

Computed from `data/raw/predictive_maintainance.csv`; see `notebooks/eda_and_model_evaluation.ipynb`.

| Check | Result | Handling in the pipeline |
|---|---|---|
| Rows / columns | 10,000 / 12 | |
| Duplicate rows | 0 | |
| Missing `Timestamp` | 1 row | Row dropped in `cleaning.convert_timestamp` |
| Missing `Terrain_type` | 98 rows | Imputed with the most frequent value |
| Other missing values | none | |
| Distinct `Equipment_id` | 10,000 (one row per machine) | Rolling features carry no history (see model evaluation) |
| Distinct `Timestamp` | 1 (`2024-12-01`) | No time-based split possible |
| `Terrain_type` formatting | values wrapped in literal `"` characters; `sandy_wet` and `sandy-wet` both present | Not cleaned yet (listed as a next step) |
| Class balance | 63.1% failure / 36.9% no failure | Stratified train/test split |
