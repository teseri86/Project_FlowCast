"""
One place that knows where every dataset lives, locally and in S3.

S3 is the source of truth: ingestion uploads there, and everything downstream
(EDA, training table, validation, model training) reads from there. That way any
teammate - or an AWS training job - gets the same data without running the
backfills themselves.

USAGE
    from src.flowcast.storage import read_dataset, write_dataset

    usgs = read_dataset("usgs_daily")          # newest copy from S3
    write_dataset("training_table", df)        # saves locally AND uploads to S3

WHERE TO READ FROM
    Default is S3. To use the local data/ folder instead (offline, or testing):
      - set FLOWCAST_DATA_SOURCE=local in your .env, or
      - pass source="local", or --local on the scripts that accept it.

S3 LAYOUT: <folder>/<YYYY-MM-DD>/<file name>. "Newest copy" = the latest date folder.
"""

import io
import os
from datetime import date

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# name -> (local file, S3 folder). The file name in S3 is the same as locally.
DATASETS = {
    # ---- raw zone: written by the ingestion scripts (uploaded by upload_to_s3)
    "usgs_sites_all":      ("resources/usgs_monitoring_locations.csv",   "raw/usgs/monitoring_locations"),
    "usgs_daily_series":   ("data/usgs_daily_series.csv",                "raw/usgs/daily_series"),
    "usgs_daily":          ("data/usgs_daily_2020_present.csv",          "raw/usgs/daily_values"),
    "usgs_active_sites":   ("data/usgs_active_sites.csv",                "raw/usgs/active_sites"),
    "weather_cell_map":    ("data/site_weather_cell_map.csv",            "raw/openmeteo/site_cell_map"),
    "weather_daily":       ("data/weather_daily_2020_present.csv",       "raw/openmeteo/daily_weather"),
    # ---- reference zone: downloaded ONCE and never changes, so it is not under raw/ (upload_to_s3 skips it)
    "usgs_history":        ("data/usgs_daily_2007_2019.csv",             "reference/usgs/daily_values_2007_2019"),
    "weather_history":     ("data/weather_daily_2007_2019.csv",          "reference/openmeteo/daily_weather_2007_2019"),
    # ---- processed zone: written by our own scripts via write_dataset()
    "training_table":      ("data/training_table.csv",                   "processed/training_table"),
    "training_data":       ("data/training_data.csv",                    "processed/training_data"),   # new pipeline (build_training_data.py)
    "nws_flood_stages":    ("data/validation/nws_flood_stages.csv",      "processed/validation"),
    "flood_check_by_gauge": ("data/validation/flood_check_by_gauge.csv", "processed/validation"),
    "official_flood_days": ("data/validation/official_flood_days.csv",   "processed/validation"),
    "usdm_county_weekly":  ("data/validation/usdm_county_weekly.csv",    "processed/validation"),
    "drought_check_weekly": ("data/validation/drought_check_weekly.csv", "processed/validation"),
    # ---- model results: written by train_gradient_boosting.py
    "model_metrics":       ("data/model/gradient_boosting_metrics.csv",            "processed/model"),
    "feature_importance":  ("data/model/gradient_boosting_feature_importance.csv", "processed/model"),
    "level_report":        ("data/model/gradient_boosting_level_report.csv",       "processed/model"),
    "scenario_predictions": ("data/model/gradient_boosting_scenarios.csv",         "processed/model"),
    # ---- cross-validation results: written by cross_validate.py
    "cv_results":          ("data/model/cv_results.csv",                           "processed/model"),
    "cv_summary":          ("data/model/cv_summary.csv",                           "processed/model"),
    # ---- model comparison (new pipeline): written by compare_models.py
    "compare_results":     ("data/model/compare_results.csv",                      "processed/model"),
    "compare_summary":     ("data/model/compare_summary.csv",                      "processed/model"),
    # ---- final models (train_final_models.py); the .joblib/.json go to the same S3 folder
    "ruler":               ("data/model/ruler.csv",                                "models/final"),
}


def default_source():
    return os.environ.get("FLOWCAST_DATA_SOURCE", "s3").strip().lower()


def _s3():
    import boto3  # imported here so local-only use doesn't need AWS set up
    return boto3.client("s3"), os.environ["S3_BUCKET_NAME"]


def latest_s3_key(name):
    """Full S3 key of the newest dated copy of a dataset (or None if none exist)."""
    local_path, folder = DATASETS[name]
    file_name = os.path.basename(local_path)
    s3, bucket = _s3()
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=folder + "/"):
        keys += [obj["Key"] for obj in page.get("Contents", []) if obj["Key"].endswith("/" + file_name)]
    return max(keys) if keys else None   # date folders sort correctly as text (YYYY-MM-DD)


def read_dataset(name, source=None, **read_csv_kwargs):
    """Load a dataset as a DataFrame. Extra arguments go to pd.read_csv (e.g. dtype=...)."""
    source = source or default_source()
    local_path, folder = DATASETS[name]
    if source == "local":
        print(f"  reading {local_path}")
        return pd.read_csv(local_path, **read_csv_kwargs)

    key = latest_s3_key(name)
    if key is None:
        raise FileNotFoundError(
            f"No copy of '{name}' in S3 under {folder}/. Upload it first "
            f"(python -m src.flowcast.ingestion.upload_to_s3), or use source='local'.")
    s3, bucket = _s3()
    print(f"  reading s3://{bucket}/{key}")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(body), **read_csv_kwargs)


def write_dataset(name, df, source=None):
    """Save a dataset locally, and (unless source='local') upload it to S3 under today's date."""
    source = source or default_source()
    local_path, folder = DATASETS[name]
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    tmp = local_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, local_path)
    print(f"  saved {local_path}")
    if source != "local":
        upload_file(local_path, folder)


def upload_file(local_path, folder):
    """Upload one local file to <folder>/<today>/<file name>."""
    s3, bucket = _s3()
    key = f"{folder}/{date.today().isoformat()}/{os.path.basename(local_path)}"
    s3.upload_file(local_path, bucket, key)
    size_mb = os.path.getsize(local_path) / 1_000_000
    print(f"  uploaded {local_path} ({size_mb:.1f} MB) -> s3://{bucket}/{key}")