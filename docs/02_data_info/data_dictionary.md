# Data dictionary

File: `data/raw/predictive_maintainance.csv` (10,000 rows, 12 columns). Ranges are computed from the file.
Units are not recorded in the source file, so none are claimed here.

| Column | Type | Range / values | Notes |
|---|---|---|---|
| `Equipment_id` | string | 10,000 distinct values, e.g. `FLT48`, `FCSB…` | One row per machine; prefix appears to encode equipment type |
| `Timestamp` | date | `2024-12-01` for every row (1 missing) | Single snapshot date |
| `Engine_temp` | integer | 77 – 2,239 (mean 791) | Strongest single predictor (r = −0.27 with target) |
| `oil_pressure` | float | 0.004 – 7.27 (mean 3.30) | |
| `Hydraulic_pressure` | float | 305.7 – 313.8 (mean 310.0) | |
| `Operating_hours` | integer | 1,168 – 2,886 (mean 1,539) | |
| `Vibrations` | float | 3.8 – 76.6 (mean 40.0) | |
| `Tool_wear` | integer | 0 – 253 (mean 108) | |
| `Fuel_rate` | float | 0.003 – 21.1 (mean 6.62) | r = +0.12 with target |
| `Ambient_temp` | float | 295.3 – 304.5 (mean 300.0) | |
| `Terrain_type` | string | 9 labels: plain, alluvial, paved, mixed, marshy, sandy_wet, rocky, sandy, sandy-wet | Stored with literal quotes; 98 missing |
| `Failure_event` | 0 / 1 | 1 = failure (63.1%), 0 = no failure (36.9%) | Target |

## Engineered features (`src/data_processing/feature_engineering.py`)

| Feature | Definition |
|---|---|
| `temp_x_vibration` | `Engine_temp × Vibrations` |
| `Engine_temp_roll_mean_3`, `Vibrations_roll_mean_3` | Mean of the previous 3 readings for the same machine |
| `Engine_temp_roll_std_3`, `Vibrations_roll_std_3` | Std of the previous 3 readings for the same machine |

On this file every machine has one reading, so the rolling features fall back to the raw value (mean) and 0 (std).
