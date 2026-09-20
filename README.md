# FlowCast

Predictive flood and drought risk tool combining USGS streamflow data with NOAA precipitation data, built for Mecklenburg County, NC and surrounding counties.

## Overview

FlowCast estimates flood and drought risk for U.S. watersheds by combining real-time USGS streamflow data with NOAA precipitation data. FlowCast is designed as a live, continuously updated pipeline that serves predictions through an API and monitors its own model performance over time.

## Features

- Automated ingestion of USGS streamflow/gauge-height data and NOAA precipitation data
- A combined flood/drought risk classification model covering multiple watersheds/regions, not just one location
- Address-based lookup: enter an address, get current risk, historical trends, and forecasts
- Predictions served via an API, rather than only existing as a notebook
- Ongoing monitoring for model performance degradation with scheduled retraining

## Project Structure

```
/ingestion      - Scripts to pull USGS and NOAA data, upload to Amazon S3
/geocoding      - Address-to-coordinates and nearest gauge/station matching
/model          - Data preprocessing, feature engineering, model training using Sagemaker
/api            - Lambda/API Gateway prediction endpoint code
/app            - Streamlit user interface
/notebooks      - Exploratory analysis
/docs           - Project documentation
```

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

## Quick Usage Example

```python
from flowcast import RiskPredictor

predictor = RiskPredictor(region="Mecklenburg-NC")
risk = predictor.predict(date="2026-09-20")
print(risk)  # {'flood_risk': 'low', 'drought_risk': 'moderate'}
```

*(Note: API not yet implemented — package is under active development.)*

## Data Sources

- USGS Water Data API (streamflow, gauge height)
- NOAA Climate Data Online API (GHCN-Daily precipitation)
- U.S. Census Bureau Geocoder (address-to-coordinates)

## License

MIT License — see LICENSE file.

## Contributors

*(add group member names here)*
