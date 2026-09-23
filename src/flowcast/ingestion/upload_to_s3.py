"""
Upload local CSVs (from pull_usgs.py and pull_noaa.py) to the project's S3 bucket.

Reads credentials from a .env file (see .env.example at the repo root).
AWS credentials should be configured separately via `aws configure`.
Run from the repo root with: python -m src.flowcast.ingestion.upload_to_s3
"""

import os
import io
from datetime import datetime

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.environ["S3_BUCKET_NAME"]
s3_client = boto3.client("s3")


def upload_dataframe_to_s3(df: pd.DataFrame, s3_key: str):
    """Upload a pandas DataFrame directly to S3 as a CSV, no local file needed."""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=csv_buffer.getvalue(),
    )
    print(f"Uploaded to s3://{S3_BUCKET}/{s3_key}")


def upload_daily_pulls(usgs_path="resources/usgs_monitoring_locations.csv",
                        noaa_path="resources/noaa_precipitation_by_site.csv"):
    """Upload the day's USGS and NOAA pulls to S3, partitioned by source and date."""
    run_date = datetime.now().strftime("%Y-%m-%d")

    usgs_sites = pd.read_csv(usgs_path)
    upload_dataframe_to_s3(
        usgs_sites,
        f"raw/usgs/monitoring_locations/{run_date}/usgs_monitoring_locations.csv",
    )

    noaa_precip = pd.read_csv(noaa_path)
    upload_dataframe_to_s3(
        noaa_precip,
        f"raw/noaa/precipitation/{run_date}/noaa_precipitation_by_site.csv",
    )


if __name__ == "__main__":
    upload_daily_pulls()
    print("\nAll files uploaded successfully.")