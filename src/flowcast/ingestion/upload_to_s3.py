"""
Upload the project's data files to S3, in two zones:

    raw/<source>/<dataset>/<run date>/<file>         data as downloaded from USGS / Open-Meteo
    processed/<dataset>/<run date>/<file>            data our code built from raw/
                                                     (training table, validation results)

Anything in processed/ can be rebuilt from raw/ by rerunning the scripts.
Files that don't exist yet are skipped, so this is safe to run at any stage.

Reads S3_BUCKET_NAME (and AWS credentials) from .env.
Run from the repo root with:
    python -m src.flowcast.ingestion.upload_to_s3
"""

import os
from datetime import date

import boto3
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.environ["S3_BUCKET_NAME"]

# local file -> folder in S3 (the run date and file name are added after it)
UPLOADS = {
    # ---- raw
    os.path.join("resources", "usgs_monitoring_locations.csv"): "raw/usgs/monitoring_locations",
    os.path.join("data", "usgs_daily_series.csv"): "raw/usgs/daily_series",
    os.path.join("data", "usgs_daily_2020_present.csv"): "raw/usgs/daily_values",
    os.path.join("data", "usgs_active_sites.csv"): "raw/usgs/active_sites",
    os.path.join("data", "site_weather_cell_map.csv"): "raw/openmeteo/site_cell_map",
    os.path.join("data", "weather_daily_2020_present.csv"): "raw/openmeteo/daily_weather",
    # ---- processed
    os.path.join("data", "training_table.csv"): "processed/training_table",
    os.path.join("data", "validation", "nws_flood_stages.csv"): "processed/validation",
    os.path.join("data", "validation", "flood_check_by_gauge.csv"): "processed/validation",
    os.path.join("data", "validation", "official_flood_days.csv"): "processed/validation",
    os.path.join("data", "validation", "usdm_county_weekly.csv"): "processed/validation",
    os.path.join("data", "validation", "drought_check_weekly.csv"): "processed/validation",
}


def upload_all():
    s3 = boto3.client("s3")
    run_date = date.today().isoformat()

    # Today's daily-update pulls (from daily_update.py), if any, so S3 keeps a
    # dated copy of exactly what each update downloaded.
    uploads = dict(UPLOADS)
    uploads[os.path.join("data", "raw", "usgs_updates", f"{run_date}.csv")] = "raw/usgs/daily_updates"
    uploads[os.path.join("data", "raw", "weather_updates", f"{run_date}.csv")] = "raw/openmeteo/daily_updates"

    for local_path, folder in uploads.items():
        if not os.path.exists(local_path):
            print(f"SKIPPED (not found yet): {local_path}")
            continue
        key = f"{folder}/{run_date}/{os.path.basename(local_path)}"
        s3.upload_file(local_path, S3_BUCKET, key)  # streams from disk; handles large files
        size_mb = os.path.getsize(local_path) / 1_000_000
        print(f"Uploaded {local_path} ({size_mb:.1f} MB) -> s3://{S3_BUCKET}/{key}")


if __name__ == "__main__":
    upload_all()
    print("\nDone.")