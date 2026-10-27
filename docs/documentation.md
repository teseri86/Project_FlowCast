# FlowCast code documentation

What each module does, its major functions, and how they work. Every script is run from the repo root
with `python -m src.flowcast.<folder>.<script>` (or `python bin/quickstart.py` for the demo).

**Contents**
1. [How the pieces fit](#1-how-the-pieces-fit)
2. [storage.py: where data lives](#2-storagepy-where-data-lives)
3. [ingestion/: downloading data](#3-ingestion-downloading-data)
4. [model/build_training_data.py: scores, inputs, labels](#4-modelbuild_training_datapy-scores-inputs-labels)
5. [model/compare_models.py: the 9-year test](#5-modelcompare_modelspy-the-9-year-test)
6. [model/train_final_models.py: the deployed models](#6-modeltrain_final_modelspy-the-deployed-models)
7. [api/predict.py: one prediction](#7-apipredictpy-one-prediction)
8. [bin/: examples](#8-bin-examples)

---

## 1. How the pieces fit

```
ingestion/*_backfill.py, daily_update.py ──► raw data (USGS river readings, Open-Meteo weather) ──► S3
                                                                │
model/build_training_data.py ◄──────────────────────────────────┘   scores + inputs + labels, one row per gauge per day
        │
        ├──► model/compare_models.py      tests models year by year against "same as today"
        └──► model/train_final_models.py  trains the two final models on all years ──► S3 models/final/
                                                                │
api/predict.py ◄────────────────────────────────────────────────┘   address ─► gauge ─► live data ─► two answers
```

Key words used below:

| Word | Meaning |
|---|---|
| **gauge** | one USGS monitoring location (e.g. `USGS-0212419274`, Coddle Creek) |
| **measure** | what a gauge records: `streamflow` (cubic feet per second) or `gauge_height` (feet) |
| **level** | a gauge's daily average of its chosen measure |
| **ruler** | a gauge's past levels, used to judge whether today is unusual |
| **hw_score** | 0–100: share of the gauge's past days (other years) with a lower daily level |
| **dry_score** | 0–100: same, for the 7-day average level, comparing only days within ±15 calendar days |
| **high water** | `hw_score` ≥ 90 |
| **drought** | `dry_score` < 10 (and not high water) |

---

## 2. storage.py: where data lives

Every dataset has one name, one local path and one S3 folder, listed in the `DATASETS` dictionary.
Scripts never hard-code paths; they call these functions. S3 is the default; set
`FLOWCAST_DATA_SOURCE=local` (or pass `--local`) to use the local `data/` folder.

### `read_dataset(name, source=None, **read_csv_kwargs)`
Loads a dataset as a pandas DataFrame: from S3 (the newest dated copy) or from the local file.
- `name`: a key of `DATASETS`, e.g. `"usgs_daily"`, `"weather_daily"`, `"training_data"`.
- `source`: `"s3"` or `"local"`; defaults to `FLOWCAST_DATA_SOURCE`.
- extra arguments go to `pd.read_csv` (e.g. `dtype={"monitoring_location_id": str}`).
```python
from src.flowcast.storage import read_dataset
usgs = read_dataset("usgs_daily", dtype={"monitoring_location_id": str})
```

### `write_dataset(name, df, source=None)`
Saves a DataFrame to its local path (written to a temporary file first, then renamed, so a crash
never leaves a half-written file) and, unless local, uploads it to `<folder>/<today>/<file>` in S3.

### `upload_file(local_path, folder)` / `latest_s3_key(name)`
Upload any file to `<folder>/<today>/`; find the newest dated copy of a dataset in S3.

---

## 3. ingestion/: downloading data

All download scripts save each request's result to its own small file before combining, so an
interrupted run can simply be started again: finished parts are skipped. Raw data is downloaded in
full (all gauges, all measures); choosing what to use happens later, in `build_training_data.py`.

| Script | Downloads | Run |
|---|---|---|
| `usgs_backfill.py` | daily streamflow + gauge height for every active gauge, 2020 to now | once |
| `weather_backfill.py` | daily weather for every gauge's grid square, 2020 to now | once |
| `usgs_history_backfill.py` | the same USGS series, 2007–2019 | once |
| `weather_history_backfill.py` | the same weather, 2007–2019 | once (about 2 days: free-tier limit) |
| `daily_update.py` | the last 30 days of both sources | regularly |
| `upload_to_s3.py` | uploads the raw files to S3 | after downloads |

Major functions:

- **`usgs_backfill.find_daily_series(refresh=False)`**: asks USGS which daily series exist for the
  384 candidate sites and keeps those with data since 2020 (186 series at 102 sites). Saved to
  `usgs_daily_series.csv`.
- **`usgs_backfill.pull_batches(series)`**: downloads the series 20 at a time (one API page each). If
  USGS stops it (e.g. rate limit), it returns `False`; rerunning continues from the next batch.
- **`usgs_backfill.combine(series)`**: stacks all batch files and daily update files, keeps the
  newest value for any repeated day, writes `usgs_daily_2020_present.csv` and `usgs_active_sites.csv`.
- **`weather_backfill.cell_for(lat, lon)`**: snaps a location to the centre of its 0.1° (~11 km) grid
  square, so nearby gauges share one weather download. Returns `(cell_id, lat, lon)`.
- **`weather_backfill.pull_all(mapping)`**: downloads each grid square's weather, pausing between
  requests to stay under Open-Meteo's hourly limit, and stops cleanly at the daily limit.
- **`weather_backfill.get_recent_and_forecast(lat, lon, past_days, forecast_days)`**: recent days plus
  the forecast for one location (for the live app).
- **`usgs_history_backfill.series_to_pull()`** / **`weather_history_backfill.pull_all(cells)`**: the
  same download logic for 2007–2019; the results go to S3 `reference/` once and never change.

---

## 4. model/build_training_data.py: scores, inputs, labels

Turns raw readings into one row per gauge per day. Output: `data/training_data.csv`.

### `choose_measures(usgs)`
Decision rule: use streamflow if it has readings on ≥ 90% of 2020–2024 days, else gauge height if
≥ 90%, else leave the gauge out. Returns `gauge id → "streamflow" / "gauge_height"`.

### `load_level()`
Combines 2007–2019 and 2020+ readings, keeps each gauge's chosen measure, and puts every gauge on a
complete calendar (missing days become empty rows, so "7 days ago" always means 7 calendar days).
Columns: `monitoring_location_id, date, level_value, measure`.

### `score_against(values, history)`
The core scoring step. Returns, for each value, the percentage of `history` below it (ties count half).
Example: history `[2, 3, 4, 5, 6, 8, 10, 15, 40, 120]`, value `50` → 9 of 10 lower → **90**.

### `add_scores(df, split_date)`
Adds `hw_score`, `dry_score`, `is_high` and `is_drought` to every day. Each day is compared only
against **other years** before `split_date`: a 2022 day against 2007–2021 and 2023+, never 2022
itself. This keeps a dry year from being judged against its own dry days. Gauges with fewer than
10 years of history are left out.

### `weather_features(weather)`
Per grid square: rain over 1/3/7/30 days, evapotranspiration over 7/30 days, the 30-day water balance
(rain minus evapotranspiration), the 7-day average high temperature, and the rain over the next
1 and 7 days (`rain_next_1d`, `rain_next_7d`).

### `add_features_and_targets(df, cell_map, weather, sites)`
Adds the score changes (`hw_score_change_1d/3d`, `hw_score_mean_30d`, `dry_score_change_7d`), the weather
inputs, `month`, `log_drainage_area`, and the answers: `target_high_1d` (high water tomorrow) and
`target_drought_7d` (drought in 7 days).

### `build_table(level, split_date, cell_map, weather, sites)`
Runs the three steps above for one split date and labels rows `train`/`test`. Used by every script.

### `train_test_masks(df, target, split_date)`
Returns `(train, test)` row masks for one target. Training rows whose answer day falls on or after the
split date are dropped (the **gap**), so no training answer comes from the test period.

### `features_for(target, with_future_rain=False)`
The input columns for a model: the 16 standard inputs, plus the rain over that model's horizon when
`with_future_rain=True`.

---

## 5. model/compare_models.py: the 9-year test

Answers "which model, if any, is better than 'same as today'?" for four setups: high water and drought,
each with and without future rain. One round per test year, 2017 to 2025+, always training on earlier
years only.

- **`make_tree()`** / **`make_boosted()`**: the two models (scikit-learn `DecisionTreeClassifier`,
  `HistGradientBoostingClassifier`), both with `class_weight="balanced"` so rare yes-days count more.
- **`pick_cutoff(make, table, cols, target, fit_part, holdout, fill)`**: a model outputs a number from
  0 to 1; it says "yes" above the cut-off. This holds back the last training year, trains on the
  earlier years, and returns the cut-off from 0.05–0.95 with the best F1 on the held-back year.
  The test year never influences it.
- **`grade(y_true, y_pred)`**: F1, recall ("caught") and precision ("right when warning") for "yes".
- **`run_round(table, split_date, test_end)`**: trains and grades every model for one test year.
- **`summarize(results)`**: averages over the years and applies the winner rule: a model must beat the
  baseline's F1 in **every** year; the tree is kept unless boosted trees beat it by ≥ 0.05 average F1.

Outputs: `data/model/compare_results.csv` (every model/year) and `compare_summary.csv` (with the verdict).

---

## 6. model/train_final_models.py: the deployed models

Trains the two winners (boosted trees for high water and for drought, both using the rain over their
horizon) on **all** years, 2007 to the latest data, and saves:

| File | Contents |
|---|---|
| `final_models.joblib` | both models, their input columns, fill values for blanks, yes/no cut-off, horizon, tested accuracy |
| `final_models_info.json` | the same, human-readable |
| `ruler.csv` | each gauge's level history, used to score new readings exactly as in training |

```python
import joblib
bundle = joblib.load("data/model/final_models.joblib")
bundle["models"]["drought"]["cutoff"]     # e.g. 0.76
```

---

## 7. api/predict.py: one prediction

`python -m src.flowcast.api.predict --address "..."` (or `--gauge USGS-...`).

| Function | What it does |
|---|---|
| `geocode(address)` | US Census Geocoder → `(lat, lon, matched address)` |
| `nearest_gauges(lat, lon, bundle)` | gauges with a model within 10 miles, nearest first |
| `fetch_recent_levels(gauge_id, measure)` | the gauge's last 45 days of daily values from USGS |
| `fetch_weather(cell_lat, cell_lon)` | last 45 days + 8-day forecast from Open-Meteo |
| `build_inputs(levels, weather, ...)` | scores the latest complete day against `ruler.csv` and builds the same inputs as training, with forecast rain |
| `predict_gauge(gauge, bundle, ruler)` | runs both models; falls back to "same as today" if there is no forecast |
| `describe(result, bundle, miles)` | the plain-language answer |

If the nearest gauge has no recent readings, `main()` tries the next one. Answers are labeled with
the date they are for; dates already past are marked "(already past)".

---

## 8. bin/: examples

| File | What it shows |
|---|---|
| [`bin/quickstart.py`](../bin/quickstart.py) | trains and tests both models on a 3-gauge sample, then predicts for an address (`--address "..."`, or `--offline`); about a minute, no keys or AWS |
| [`bin/run_pipeline.ps1`](../bin/run_pipeline.ps1) / [`bin/run_pipeline.sh`](../bin/run_pipeline.sh) | the full pipeline in order (Windows / macOS-Linux) |
| `bin/sample_data/` | the quickstart data: 3 gauges' streamflow, their weather and USGS series IDs, 2007 to Oct 2026 |
