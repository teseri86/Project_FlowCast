# FlowCast

Predictive flood and drought risk tool combining USGS streamflow data with Open-Meteo weather data, built for Mecklenburg County, NC and surrounding counties.

## Overview

FlowCast estimates flood and drought risk for rivers and streams in the Charlotte region by combining USGS river measurements (streamflow and gauge height) with daily weather data (precipitation, snow, temperature, evapotranspiration). FlowCast is designed as a live, continuously updated pipeline that serves predictions through an API and monitors its own model performance over time.

## Features

- Automated ingestion of USGS streamflow/gauge-height data and Open-Meteo weather data
- A combined flood/drought risk model covering multiple rivers and streams across the region
- Address-based lookup: enter an address, get current risk, historical trends, and forecasts
- Predictions served via an API
- Ongoing monitoring for model performance degradation with scheduled retraining

## Project Structure

```
/src
  /flowcast
    /ingestion    - Scripts to pull USGS and weather data, upload to S3
    /geocoding    - Address-to-coordinates and nearest gauge matching
    /model        - Data preprocessing, feature engineering, model training
    /api          - Lambda/API Gateway prediction endpoint code
/app              - Streamlit user interface
/resources        - Small static reference files (e.g. the USGS site list)
/data             - Local data files created by the ingestion scripts (not in git)
/notebooks        - Exploratory analysis
/docs             - Project documentation
/tests            - Unit tests
```

Code under `/src/flowcast` is structured as an importable Python package. `/app`, `/notebooks`, and `/resources` sit outside it since they aren't library code.

## Setup

1. Clone this repository
2. Create a Python environment (Python 3.12 recommended):
   ```
   conda create -n flowcast_env python=3.12 -y
   conda activate flowcast_env
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your own API keys/credentials (never commit `.env`)
4. Configure AWS credentials locally: `aws configure`
5. Get the USGS site list (CSV files are not stored in git). Either download
   `usgs_monitoring_locations.csv` from S3 (`raw/usgs/monitoring_locations/`) into `resources/`,
   or rebuild it:
   ```
   python -m src.flowcast.ingestion.pull_usgs
   ```

## Data Pipeline

Run every command from the repo root, in the `flowcast_env` terminal. Always use
`python -m ...` (not the Run button) so the imports work.

**One-time history (2020-01-01 to today).** Both scripts are safe to rerun: if one stops
partway (rate limit, lost connection), run it again and it continues where it left off.
```
python -m src.flowcast.ingestion.usgs_backfill      # run first: finds active sites, pulls river data
python -m src.flowcast.ingestion.weather_backfill   # weather for those sites (may span 2 days: free-tier limits)
```
Add `--status` to either one to check progress without calling the API.

**Keeping it current (run each day, or after a gap of up to 30 days):**
```
python -m src.flowcast.ingestion.daily_update
```

**Sending the data to S3:**
```
python -m src.flowcast.ingestion.upload_to_s3
```

**Main output files** (in `data/`):

| File | What it holds |
|---|---|
| `usgs_daily_2020_present.csv` | Daily streamflow and gauge height, one row per site per measurement per day |
| `usgs_active_sites.csv` | The active USGS sites: name, county, latitude/longitude, drainage area |
| `weather_daily_2020_present.csv` | Daily weather, one row per grid square per day |
| `site_weather_cell_map.csv` | Which weather grid square each USGS site falls in |

## Quick Usage Example

```python
from flowcast import RiskPredictor

predictor = RiskPredictor(region="Mecklenburg-NC")
risk = predictor.predict(date="2026-09-20")
print(risk)  # {'flood_risk': 'low', 'drought_risk': 'moderate'}
```

*(Note: API not yet implemented — package is under active development.)*

## Data Sources

- USGS Water Data API (streamflow, gauge height). Request a token here: https://api.waterdata.usgs.gov/signup/
- Open-Meteo Historical Weather and Forecast APIs (no key needed for non-commercial use): https://open-meteo.com/
- U.S. Census Bureau Geocoder (address-to-coordinates)

Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).

## License

MIT License — see LICENSE file.

## Contributors

- Andrew Farmer
- Samip Khanal
- Eric Testen
- Sanchit Thakare