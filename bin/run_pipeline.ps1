# FlowCast - the full pipeline, in order (Windows PowerShell).
# Run from the repo root:  .\bin\run_pipeline.ps1  (or copy the commands one at a time).
# Needs: conda env flowcast_env, a .env file (USGS token, AWS keys, S3_BUCKET_NAME), AWS access.
# Every script reads/writes S3 by default; add --local to use the local data/ folder instead.
# To try FlowCast WITHOUT any of this, run:  python bin/quickstart.py

$ErrorActionPreference = "Stop"

# 1. One-time downloads (already done for this project; safe to rerun - they resume where they stopped)
python -m src.flowcast.ingestion.usgs_backfill              # USGS daily values, 2020 to now
python -m src.flowcast.ingestion.weather_backfill           # Open-Meteo weather, 2020 to now
python -m src.flowcast.ingestion.usgs_history_backfill      # USGS daily values, 2007-2019
python -m src.flowcast.ingestion.weather_history_backfill   # weather 2007-2019 (stops at the daily limit; rerun next day)

# 2. Keep the data current (re-pulls the last 30 days), then send it to S3
python -m src.flowcast.ingestion.daily_update
python -m src.flowcast.ingestion.upload_to_s3

# 3. Models
python -m src.flowcast.model.build_training_data    # ~1 min: scores, inputs, labels (optional - steps below rebuild it)
python -m src.flowcast.model.compare_models         # ~25 min: 9-year test, baseline vs tree vs boosted
python -m src.flowcast.model.train_final_models     # ~1-2 min: trains and saves the two final models to S3

# 4. Predictions for an address, or for one gauge
python -m src.flowcast.api.predict --address "600 E 4th St, Charlotte, NC"
python -m src.flowcast.api.predict --gauge USGS-0212419274
