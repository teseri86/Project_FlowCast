"""
Upload the backfill outputs to the project's S3 bucket, using the raw-zone layout:
    raw/<source>/<dataset>/<run date>/<file name>

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

# local file -> (source, dataset) folder in S3
UPLOADS = {
    os.path.join("resources", "usgs_monitoring_locations.csv"): ("usgs", "monitoring_locations"),
    os.path.join("data", "usgs_daily_series.csv"): ("usgs", "daily_series"),
    os.path.join("data", "usgs_daily_2020_present.csv"): ("usgs", "daily_values"),
    os.path.join("data", "usgs_active_sites.csv"): ("usgs", "active_sites"),
    os.path.join("data", "site_weather_cell_map.csv"): ("openmeteo", "site_cell_map"),
    os.path.join("data", "weather_daily_2020_present.csv"): ("openmeteo", "daily_weather"),
}


def upload_all():
    s3 = boto3.client("s3")
    run_date = date.today().isoformat()

    # Today's daily-update pulls (from daily_update.py), if any, so S3 keeps a
    # dated copy of exactly what each update downloaded.
    uploads = dict(UPLOADS)
    uploads[os.path.join("data", "raw", "usgs_updates", f"{run_date}.csv")] = ("usgs", "daily_updates")
    uploads[os.path.join("data", "raw", "weather_updates", f"{run_date}.csv")] = ("openmeteo", "daily_updates")

    for local_path, (source, dataset) in uploads.items():
        if not os.path.exists(local_path):
            print(f"SKIPPED (not found yet): {local_path}")
            continue
        key = f"raw/{source}/{dataset}/{run_date}/{os.path.basename(local_path)}"
        s3.upload_file(local_path, S3_BUCKET, key)  # streams from disk; handles large files
        size_mb = os.path.getsize(local_path) / 1_000_000
        print(f"Uploaded {local_path} ({size_mb:.1f} MB) -> s3://{S3_BUCKET}/{key}")


if __name__ == "__main__":
    upload_all()
    print("\nDone.")