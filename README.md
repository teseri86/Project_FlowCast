# FlowCast

High-water and drought prediction for rivers and streams in Mecklenburg County, NC and 15 surrounding
NC/SC counties, built on USGS river readings and Open-Meteo weather, with a serverless AWS pipeline.

**Status:** prototype. The data pipeline, training data, model evaluation, final models and a local
prediction script work. The AWS prediction API, daily automation, monitoring and the Streamlit app are
not built yet. FlowCast is **not** an official warning service; see [Limitations](#limitations).

## What it predicts

For the USGS gauge nearest an address, FlowCast gives two yes/no answers:

| Answer | Question | Meaning of "yes" |
|---|---|---|
| **High water** | Will the river be in high water 1 day after the latest complete day of readings? | The daily level is in the top 10% of that gauge's history (any time of year) |
| **Drought** | Will the river be in drought 7 days after the latest complete day? | The 7-day average level is in the bottom 10% of that gauge's history *for that time of year* (within ±15 calendar days) |

If a day would be both, it counts as high water. "High water" means *unusually high for this river*,
not official flood stage.

**Timing:** USGS publishes a day's average after the day ends, so the latest complete day is usually
yesterday. Each answer is labeled with the date it is for (answers for a date that has already passed
are marked "already past").

## How it works

```
Address ─► US Census Geocoder ─► nearest gauge with a model (≤ 10 miles; skips gauges with no recent readings)
        ─► latest USGS readings + Open-Meteo recent weather and rain forecast
        ─► score today against the gauge's history ("ruler")  ─► two models ─► yes / no answers
```

1. **Ingestion** downloads USGS daily river readings and Open-Meteo daily weather (all gauges, all
   measures; nothing is filtered at download time) and stores them in S3.
2. **Training data** turns each gauge's readings into two 0–100 scores compared against the gauge's own
   history, adds weather inputs, and labels each day.
3. **Model comparison** tests the models year by year (2017–2025+) against a "same as today" baseline.
4. **Final models** are trained on all years and saved, with everything needed to use them, to S3.
5. **Prediction** (`predict.py`) runs both models for one address or gauge.

## Project structure

```
src/flowcast/
  storage.py                     where every dataset lives, locally and in S3 (read_dataset / write_dataset)
  ingestion/
    pull_usgs.py                 builds the candidate USGS site list (rarely rerun)
    usgs_backfill.py             one-time: USGS daily values, 2020 to now
    weather_backfill.py          one-time: Open-Meteo weather, 2020 to now
    usgs_history_backfill.py     one-time: USGS daily values, 2007-2019
    weather_history_backfill.py  one-time: Open-Meteo weather, 2007-2019 (about 2 days: free-tier limit)
    daily_update.py              run regularly: last 30 days of both sources
    upload_to_s3.py              uploads the raw data to S3
  model/
    build_training_data.py       scores, inputs and labels: one row per gauge per day
    compare_models.py            9-year test of baseline vs decision tree vs boosted trees
    train_final_models.py        trains and saves the two final models + the ruler
    validate_labels.py           LEGACY - checks the OLD labels against NWS flood stages / Drought Monitor
  api/
    predict.py                   address or gauge -> both answers (core of the future API)
  geocoding/                     (empty - geocoding currently lives in api/predict.py)
notebooks/
  eda_raw_data.ipynb             exploratory analysis; section 9 lists the design decisions
resources/                       USGS candidate site list
data/                            local data files (not in git)
app/  docs/  tests/              (not built yet)
```

## Setup

1. Clone the repository.
2. Create the environment (Python 3.12):
   ```
   conda create -n flowcast_env python=3.12 -y
   conda activate flowcast_env
   python -m pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your USGS token, AWS keys, and
   `S3_BUCKET_NAME=flowcast-pred-app-data`. Never commit `.env`.
4. For Jupyter, register the kernel:
   ```
   python -m pip install ipykernel
   python -m ipykernel install --user --name flowcast_env --display-name "Python (flowcast_env)"
   ```

## Running the pipeline

Run everything from the repo root in the `flowcast_env` terminal, with `python -m ...`.
Scripts read from S3 by default; add `--local` to use the local `data/` folder instead.

**1. One-time downloads** (already done for this project; all are safe to rerun and resume where they stopped):
```
python -m src.flowcast.ingestion.usgs_backfill
python -m src.flowcast.ingestion.weather_backfill
python -m src.flowcast.ingestion.usgs_history_backfill
python -m src.flowcast.ingestion.weather_history_backfill    # stops at the daily limit; rerun the next day
```
Add `--status` to any backfill to check progress without calling the API.

**2. Keeping the data current** (after a gap of more than 30 days, raise `LOOKBACK_DAYS` in `daily_update.py` for one run):
```
python -m src.flowcast.ingestion.daily_update
python -m src.flowcast.ingestion.upload_to_s3
```

**3. Models** (after new data, or when changing the design):
```
python -m src.flowcast.model.build_training_data    # ~1 min; optional (the next two rebuild it themselves)
python -m src.flowcast.model.compare_models         # ~25 min; the 9-year test
python -m src.flowcast.model.train_final_models     # ~1-2 min; saves the final models to S3
```

**4. A prediction:**
```
python -m src.flowcast.api.predict --address "600 E 4th St, Charlotte, NC"
python -m src.flowcast.api.predict --gauge USGS-0212419274
```

## Main files

| File (in `data/`) | What it holds |
|---|---|
| `usgs_daily_2020_present.csv`, `usgs_daily_2007_2019.csv` | Daily streamflow and gauge height, every series, one row per series per day |
| `weather_daily_2020_present.csv`, `weather_daily_2007_2019.csv` | Daily weather per ~11 km grid square |
| `usgs_active_sites.csv` | Active gauges: name, county, coordinates, drainage area |
| `site_weather_cell_map.csv` | Which weather grid square each gauge is in |
| `training_data.csv` | One row per gauge per day: scores, inputs, labels |
| `model/compare_results.csv`, `model/compare_summary.csv` | The 9-year test, every model and year / averaged |
| `model/final_models.joblib` | Both final models, their inputs, fill values and yes/no cut-offs |
| `model/final_models_info.json` | The same, readable, with tested accuracy |
| `model/ruler.csv` | Each gauge's river history used to score new readings |

**S3** (`flowcast-pred-app-data`), each file in a dated folder (`<folder>/<YYYY-MM-DD>/<file>`); scripts read the newest:

| Folder | Contents |
|---|---|
| `raw/usgs/...`, `raw/openmeteo/...` | Downloaded data, 2020 to now, and each daily update |
| `reference/usgs/...`, `reference/openmeteo/...` | 2007–2019 history (uploaded once; never changes) |
| `processed/training_data/`, `processed/model/`, `processed/validation/` | Training data, test results, validation files |
| `models/final/` | The deployed models, their info file and the ruler |

## Method

- **Gauges and measure:** streamflow if it has readings on ≥ 90% of 2020–2024 days, else gauge height,
  else the gauge is left out. A gauge also needs 10 years of history before it is used.
- **Scores ("the ruler"):** each day is ranked against the gauge's readings from every *other* year
  since 2007 (never its own year), so dry and wet years show up as they really were.
  `hw_score` ranks the daily level against all days; `dry_score` ranks the 7-day average against days
  within ±15 calendar days. Comparing high water to all days but drought to the same time of year follows
  the recommendation of a Cabarrus County Soil and Water Conservation District specialist.
- **Inputs (16 + future rain):** today's scores and their recent changes, rain over 1/3/7/30 days,
  evapotranspiration, a 30-day water balance, temperature, month and river size. Each model also gets
  the rain over its own horizon (1 day / 7 days): the rain that actually fell in training, the
  Open-Meteo forecast in the app.
- **Models compared:** "same as today" baseline, one decision tree, boosted trees (scikit-learn
  `HistGradientBoostingClassifier`), each with class weighting and a yes/no cut-off chosen on the last
  training year only.
- **Test:** one round per year, 2017 to 2025+, always training on earlier years only.
  Main measure: F1 (balances catching events and being right when warning). A model must beat the
  baseline in **every** year; the simpler tree is kept unless boosted trees beat it by ≥ 0.05 F1.

## Results (9 test years)

| Model | Without rain information | With the rain that actually fell |
|---|---|---|
| High water, 1 day (boosted trees) | does not beat "same as today" | catches ~72% of high-water days; ~71% of warnings right |
| Drought, 7 days (boosted trees) | does not beat "same as today" | catches ~66% of drought days; ~56% of warnings right |

Most of the predictive power comes from knowing the rain ahead. Real accuracy with real forecasts will be
**lower** than the right-hand column; it has not been measured yet.

## Limitations

- Accuracy with real rain forecasts is unmeasured (the tests used rain that actually fell).
- Daily averages hide short flood peaks on fast creeks; "high water" is relative to each river, not
  official flood stage.
- History starts in 2007; gauges with less than 10 years are not covered.
- Weather is a ~11 km grid estimate; small thunderstorms can be missed.
- Gauges concentrate in urban Mecklenburg; rural areas have fewer or none.
- A gauge's chosen measure can stop reporting (e.g. Big Dutchman Creek streamflow stopped in June 2026);
  the predictor then skips to the next-nearest gauge.

## Use of AI

This project used an AI assistant (Claude, by Anthropic) to help draft and debug code and draft documentation and write-ups.

- **Decisions were made by the team.** The AI proposed options and explained trade-offs; team members
  made the final decision.
- **Every change was reviewed.** Team members read the code and documentation the AI drafted, verified code, made corrections, and approved changes before they were kept.
- **Results were verified by running them.** All scripts were run on the team's own setup, and their
  outputs were checked against expected results before being used or reported.

The team is responsible for the final code, results and conclusions.

## Data sources

- USGS Water Data API (streamflow, gauge height). Token: https://api.waterdata.usgs.gov/signup/
- Open-Meteo Historical Weather and Forecast APIs (free for non-commercial use): https://open-meteo.com/
- U.S. Census Bureau Geocoder (address to coordinates)

Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).

## Acknowledgments

We thank the resource education specialist at the **Cabarrus County Soil and Water Conservation District**
for consulting on how to define high water and drought, and which metrics to utilise. Their recommendation, to judge high water against
all days of the year but drought against the same time of year, is the labeling rule FlowCast uses.

## License

MIT License — see LICENSE file.

## Contributors

- Andrew Farmer
- Samip Khanal
- Eric Testen
- Sanchit Thakare