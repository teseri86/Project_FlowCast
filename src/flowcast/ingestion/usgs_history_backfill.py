"""
USGS river HISTORY, 2007-2019, for the "ruler" that defines normal / high water / drought.

WHY: build_training_data.py scores each day against the gauge's past (e.g. "lower than 95% of past
mid-Septembers -> drought"). With only 2020+ readings, that ruler rests on 4-5 years, so one odd year
can move the lines. These older readings give it 15-18 years. They are NOT training rows: the models
still learn from 2020 onward, so no older weather is needed.

WHAT IT DOWNLOADS (raw data: everything relevant; choosing gauges/measures happens later,
in build_training_data.py)
    EVERY series in data/usgs_daily_series.csv - all active gauges, both streamflow and gauge
    height, daily means - from 2007-01-01 to 2019-12-31.
    Series whose record starts after 2019 are skipped (USGS has nothing to send for them).

HOW TO RUN (from the repo root, in the flowcast_env terminal; run ONCE)
    python -m src.flowcast.ingestion.usgs_history_backfill            # pull (safe to rerun)
    python -m src.flowcast.ingestion.usgs_history_backfill --status   # progress only, no API calls

    About 19 requests, a few minutes. If USGS stops it (e.g. the hourly rate limit), run it again later:
    finished batches are kept and skipped.

OUTPUTS
    data/raw/usgs_history/batch_*.csv   one file per batch
    data/usgs_daily_2007_2019.csv       all batches combined
        -> also uploaded ONCE to S3: reference/usgs/daily_values_2007_2019/<today>/
           (not under raw/, so the daily upload_to_s3 run does not re-upload it)
"""

import argparse
import hashlib
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # must run BEFORE importing dataretrieval so it sees API_USGS_PAT

from dataretrieval import waterdata  # noqa: E402
from dataretrieval.exceptions import DataRetrievalError  # noqa: E402

from src.flowcast.storage import read_dataset, write_dataset  # noqa: E402

# ---------------------------------------------------------------- settings
HISTORY_START = "2007-01-01"
HISTORY_END = "2019-12-31"            # 2020 onward is already in usgs_daily_2020_present.csv
SERIES_PER_BATCH = 10                 # 10 series x ~4,750 days = ~47,500 rows = one API page
PARAMETER_TO_VARIABLE = {"00060": "streamflow_cfs_mean", "00065": "gauge_height_ft_mean"}
BATCH_DIR = os.path.join("data", "raw", "usgs_history")
TEXT = {"monitoring_location_id": str, "parameter_code": str, "statistic_id": str, "time_series_id": str}


def series_to_pull(verbose=True):
    """Every daily-mean series in the series list that has data before 2020 (all gauges, both measures)."""
    series = read_dataset("usgs_daily_series", dtype=TEXT)
    series = series[(series["statistic_id"] == "00003") & series["parameter_code"].isin(PARAMETER_TO_VARIABLE)]
    begins = pd.to_datetime(series["begin"].astype(str).str[:10], errors="coerce")
    has_old = begins <= pd.Timestamp(HISTORY_END)
    if verbose:
        old = series[has_old]
        print(f"  {len(old)} of {len(series)} series (at {old['monitoring_location_id'].nunique()} of "
              f"{series['monitoring_location_id'].nunique()} gauges) have records before 2020; "
              f"{(~has_old).sum()} start in 2020 or later - nothing to download for them")
    return series[has_old]


def make_batches(series):
    """Fixed batches; each file name comes from its contents, so reruns find the same files."""
    ids = sorted(series["time_series_id"].unique())
    batches = []
    for i in range(0, len(ids), SERIES_PER_BATCH):
        chunk = ids[i:i + SERIES_PER_BATCH]
        fingerprint = hashlib.md5(",".join(chunk).encode()).hexdigest()[:8]
        batches.append((chunk, os.path.join(BATCH_DIR, f"batch_{i // SERIES_PER_BATCH:03d}_{fingerprint}.csv")))
    return batches


def pull_batches(series):
    os.makedirs(BATCH_DIR, exist_ok=True)
    batches = make_batches(series)
    todo = [(ids, path) for ids, path in batches if not os.path.exists(path)]
    print(f"Pulling: {len(batches)} batches total, {len(batches) - len(todo)} already saved, {len(todo)} to pull.")

    for n, (ids, path) in enumerate(todo, start=1):
        print(f"  [{n}/{len(todo)}] {len(ids)} series -> {os.path.basename(path)}")
        try:
            df, _ = waterdata.get_daily(time_series_id=ids, time=f"{HISTORY_START}/{HISTORY_END}",
                                        skip_geometry=True)
        except DataRetrievalError as e:
            print(f"\n  Stopped: {type(e).__name__}: {e}")
            print("  Saved batches are safe. Wait ~1 hour and run this again to continue.")
            return False

        df = df[["monitoring_location_id", "parameter_code", "time", "value", "time_series_id"]].copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.drop_duplicates(subset=["time_series_id", "time"])

        tmp = path + ".tmp"                      # temp file, then rename: a half-written file never counts
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        print(f"      saved {len(df):,} rows")
    return True


def combine(series):
    paths = [p for _, p in make_batches(series) if os.path.exists(p)]
    df = pd.concat([pd.read_csv(p, dtype=TEXT) for p in paths], ignore_index=True)
    df["variable"] = df["parameter_code"].map(PARAMETER_TO_VARIABLE)
    df = (df[["monitoring_location_id", "variable", "time", "value"]]
          .dropna(subset=["value"]).sort_values(["monitoring_location_id", "time"]))

    print(f"\nCombined {len(df):,} rows, {df['monitoring_location_id'].nunique()} gauges, "
          f"{df['time'].min()} to {df['time'].max()}")
    years = df.assign(year=df["time"].str[:4]).groupby(["monitoring_location_id", "variable"])["year"].nunique()
    print(f"  Years of history per series (2007-2019, max 13): median {years.median():.0f}, "
          f"{(years >= 10).sum()} of {len(years)} series with 10+")
    print("\nSaving and uploading (one time)...")
    write_dataset("usgs_history", df)


def status():
    series = series_to_pull(verbose=False)
    batches = make_batches(series)
    done = sum(os.path.exists(p) for _, p in batches)
    print(f"Batches saved: {done}/{len(batches)}  |  combined file exists: "
          f"{os.path.exists(os.path.join('data', 'usgs_daily_2007_2019.csv'))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="show progress without calling USGS")
    parser.add_argument("--local", action="store_true", help="read/write the local data/ folder only (no S3)")
    args = parser.parse_args()
    if args.local:
        os.environ["FLOWCAST_DATA_SOURCE"] = "local"

    if args.status:
        status()
    else:
        print("Finding the series to download...")
        series = series_to_pull()
        if pull_batches(series):
            combine(series)
            print("\nRiver history download complete. Next: python -m src.flowcast.model.build_training_data")
