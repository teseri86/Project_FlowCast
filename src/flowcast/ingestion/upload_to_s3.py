"""
Upload the ingestion outputs (the raw zone) to S3:

    raw/<source>/<dataset>/<run date>/<file>

Run this after the backfills or after daily_update. Downstream scripts then 
read the newest copies from S3, and upload their own outputs to processed/ themselves.

Where each file goes is defined once, in src/flowcast/storage.py.
Files that don't exist yet are skipped, so this is safe to run at any stage.

Run from the repo root with:
    python -m src.flowcast.ingestion.upload_to_s3
"""

import os
from datetime import date

from src.flowcast.storage import DATASETS, upload_file


def upload_all():
    run_date = date.today().isoformat()
    uploads = [(path, folder) for path, folder in DATASETS.values() if folder.startswith("raw/")]
    # Today's daily-update pulls (from daily_update.py), if any, so S3 keeps a
    # dated copy of exactly what each update downloaded.
    uploads += [
        (os.path.join("data", "raw", "usgs_updates", f"{run_date}.csv"), "raw/usgs/daily_updates"),
        (os.path.join("data", "raw", "weather_updates", f"{run_date}.csv"), "raw/openmeteo/daily_updates"),
    ]
    for local_path, folder in uploads:
        if not os.path.exists(local_path):
            print(f"  SKIPPED (not found yet): {local_path}")
            continue
        upload_file(local_path, folder)


if __name__ == "__main__":
    upload_all()
    print("\nDone.")