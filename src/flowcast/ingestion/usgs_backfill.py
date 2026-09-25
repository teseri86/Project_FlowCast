"""
USGS historical backfill: daily streamflow (00060) and gauge height (00065),
2020-01-01 through today, for every monitoring site in
resources/usgs_monitoring_locations.csv.

*** RUN THIS FIRST BEFORE weather_backfill.py ***

  Step 1 - "Which sites actually have daily data since 2020?"
      One metadata lookup for all 384 sites. Many "Stream" sites are old or
      discontinued, so we throw those away before pulling anything.

  Step 2 - Pull the daily values in BATCHES of time series (20 at a time).
      One batch is ~50,000 rows, which USGS returns in a single page, so the
      whole backfill is only a handful of requests.

  Each batch is saved to its own small file in data/raw/usgs_daily/.
  If a run stops partway, just run it again: batches that already have a file are skipped. 
  Nothing is ever appended to an existing file, so duplicate rows can't pile up.

HOW TO RUN (from the repo root, in the flowcast_env terminal)
    python -m src.flowcast.ingestion.usgs_backfill              # pull (safe to rerun)
    python -m src.flowcast.ingestion.usgs_backfill --status     # progress only, no API calls
    python -m src.flowcast.ingestion.usgs_backfill --refresh    # redo Step 1 (re-check which sites are active)

OUTPUTS
    data/usgs_daily_series.csv        which site/parameter time series exist (Step 1)
    data/raw/usgs_daily/batch_*.csv   one file per batch (Step 2)
    data/usgs_daily_2020_present.csv  all batches combined, one row per site/parameter/day
    data/usgs_active_sites.csv        one row per active site: name, county, lat/lon, drainage area
                                      (join to the daily file on monitoring_location_id)
"""

import argparse
import hashlib
import os
from datetime import date

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # must run BEFORE importing dataretrieval so it sees API_USGS_PAT

from dataretrieval import waterdata  # noqa: E402
from dataretrieval.exceptions import DataRetrievalError  # noqa: E402

# ---------------------------------------------------------------- settings
START_DATE = "2020-01-01"
PARAMETERS = {"00060": "streamflow_cfs", "00065": "gauge_height_ft"}

# 00003 = daily mean, 00001 = daily max. Daily MAX gauge height matters for
# floods (a river can spike and drop within one day, which the mean hides).
STATISTICS = {"00003": "mean", "00001": "max"}
SERIES_PER_BATCH = 20  # 20 series x ~2,450 days = ~49,000 rows = one API page

SITES_FILE = os.path.join("resources", "usgs_monitoring_locations.csv")
DATA_DIR = "data"
SERIES_FILE = os.path.join(DATA_DIR, "usgs_daily_series.csv")
BATCH_DIR = os.path.join(DATA_DIR, "raw", "usgs_daily")
COMBINED_FILE = os.path.join(DATA_DIR, "usgs_daily_2020_present.csv")
ACTIVE_SITES_FILE = os.path.join(DATA_DIR, "usgs_active_sites.csv")
UPDATE_DIR = os.path.join(DATA_DIR, "raw", "usgs_updates")  # written by daily_update.py

# Codes like "00060" must stay TEXT. If pandas reads them as numbers they
# become 60, which is exactly the bug that broke the old script.
TEXT_COLUMNS = {
    "monitoring_location_id": str,
    "parameter_code": str,
    "statistic_id": str,
    "time_series_id": str,
}

KEEP_COLUMNS = [
    "monitoring_location_id",
    "parameter_code",
    "statistic_id",
    "time",
    "value",
    "unit_of_measure",
    "approval_status",
    "qualifier",
    "time_series_id",
]


# ------------------------------------------------------- Step 1: discovery
def find_daily_series(refresh=False):
    """Return a table of daily time series (one row per site/parameter/statistic)
    that have data on or after START_DATE. Cached in SERIES_FILE."""
    if os.path.exists(SERIES_FILE) and not refresh:
        series = pd.read_csv(SERIES_FILE, dtype=TEXT_COLUMNS)
        print(f"Step 1: using saved series list ({len(series)} series). "
              f"Use --refresh to re-check USGS.")
        return series

    sites = pd.read_csv(SITES_FILE, dtype=TEXT_COLUMNS)
    site_ids = sites["monitoring_location_id"].dropna().unique().tolist()
    print(f"Step 1: asking USGS which of {len(site_ids)} sites have daily data...")

    meta, _ = waterdata.get_time_series_metadata(
        monitoring_location_id=site_ids,
        parameter_code=list(PARAMETERS),
        statistic_id=list(STATISTICS),
        skip_geometry=True,
    )
    print(f"  USGS returned {len(meta)} matching time series (all dates, all types).")

    # Keep only DAILY series (the same codes also exist for 15-minute data).
    if "computation_period_identifier" in meta.columns:
        is_daily = meta["computation_period_identifier"].astype(str).str.lower().eq("daily")
        meta = meta[is_daily]

    # Keep only series whose last value is on/after START_DATE (still active in our window).
    meta_end = pd.to_datetime(meta["end"], errors="coerce", utc=True)
    meta = meta[meta_end >= pd.Timestamp(START_DATE, tz="UTC")]

    cols = [c for c in ["time_series_id", "monitoring_location_id", "parameter_code",
                        "statistic_id", "begin", "end", "unit_of_measure", "primary"]
            if c in meta.columns]
    series = meta[cols].sort_values(["monitoring_location_id", "parameter_code", "statistic_id"])
    series = series.astype({c: str for c in TEXT_COLUMNS if c in series.columns})

    os.makedirs(DATA_DIR, exist_ok=True)
    series.to_csv(SERIES_FILE, index=False)

    print(f"  -> {len(series)} daily series active since {START_DATE}, "
          f"across {series['monitoring_location_id'].nunique()} sites.")
    print(series.groupby(["parameter_code", "statistic_id"]).size()
          .rename("series_count").to_string())
    print(f"  Saved to {SERIES_FILE}\n")
    return series


# ---------------------------------------------------------- Step 2: pulling
def make_batches(series):

    """Split series IDs into fixed batches. Each batch gets a file name built
    from its contents, so the same batch always maps to the same file."""

    ids = sorted(series["time_series_id"].unique())
    batches = []
    for i in range(0, len(ids), SERIES_PER_BATCH):
        chunk = ids[i:i + SERIES_PER_BATCH]
        fingerprint = hashlib.md5(",".join(chunk).encode()).hexdigest()[:8]
        path = os.path.join(BATCH_DIR, f"batch_{i // SERIES_PER_BATCH:03d}_{fingerprint}.csv")
        batches.append((chunk, path))
    return batches


def clean(df):

    """Consistent columns and formats before anything touches disk."""

    df = df[[c for c in KEEP_COLUMNS if c in df.columns]].copy()
    # One date format everywhere ("2020-01-01"), the fix for the old
    # duplicate-rows bug.
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.drop_duplicates(subset=["time_series_id", "time"])


def pull_batches(series):
    os.makedirs(BATCH_DIR, exist_ok=True)
    batches = make_batches(series)
    todo = [(ids, path) for ids, path in batches if not os.path.exists(path)]
    print(f"Step 2: {len(batches)} batches total, {len(batches) - len(todo)} already saved, "
          f"{len(todo)} to pull.")

    for n, (ids, path) in enumerate(todo, start=1):
        print(f"  [{n}/{len(todo)}] pulling {len(ids)} series -> {os.path.basename(path)}")
        try:
            df, _ = waterdata.get_daily(
                time_series_id=ids,
                time=f"{START_DATE}/..",
                skip_geometry=True,
            )
        except DataRetrievalError as e:

            # Most likely the hourly rate limit (HTTP 429). Everything saved so
            # far is safe; rerunning later picks up from this batch.

            print(f"\n  Stopped: {type(e).__name__}: {e}")
            print("  Saved batches are safe. Wait ~1 hour and run this again to continue.")
            return False

        df = clean(df)

        # Write to a temp file, then rename. A half-written file can never be
        # mistaken for a finished batch.

        tmp = path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        print(f"      saved {len(df):,} rows")

    return True


# ------------------------------------------------------ Step 3: combining
def update_files():

    """Daily update files (from daily_update.py), oldest first. The file names
    are dates, so sorting by name = sorting by date."""

    if not os.path.isdir(UPDATE_DIR):
        return []
    return sorted(os.path.join(UPDATE_DIR, f) for f in os.listdir(UPDATE_DIR) if f.endswith(".csv"))


def combine(series):

    """Backfill batches first, then daily updates stacked underneath. For any
    series+day that appears more than once, keep="last" keeps the newest value."""
    
    batches = make_batches(series)
    paths = [p for _, p in batches if os.path.exists(p)] + update_files()
    frames = [pd.read_csv(p, dtype=TEXT_COLUMNS) for p in paths]
    if not frames:
        print("Nothing to combine yet.")
        return
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time_series_id", "time"], keep="last")
    df["variable"] = df["parameter_code"].map(PARAMETERS) + "_" + df["statistic_id"].map(STATISTICS)
    df = df.sort_values(["monitoring_location_id", "variable", "time"])
    tmp = COMBINED_FILE + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, COMBINED_FILE)

    n_updates = len(update_files())
    extra = f" + {n_updates} daily update file(s)" if n_updates else ""
    print(f"\nStep 3: combined {len(df):,} rows (backfill{extra}) into {COMBINED_FILE}")
    print(f"  Sites with data: {df['monitoring_location_id'].nunique()}")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    summary = df.groupby("variable").agg(
        sites=("monitoring_location_id", "nunique"),
        rows=("value", "size"),
    )
    print(summary.to_string())
    write_active_sites(df)

    # Sanity check: no series should have more rows than there are days.
    max_days = (date.today() - date.fromisoformat(START_DATE)).days + 1
    worst = df.groupby("time_series_id").size().max()
    flag = "OK" if worst <= max_days else "PROBLEM - duplicates!"
    print(f"  Most rows in any one series: {worst:,} (max possible {max_days:,}) -> {flag}")


def write_active_sites(df):
    """One row per ACTIVE site: name, county, coordinates, drainage area, and
    which measurements it has. Coordinates live here (once per site) instead of
    being repeated on every daily row. The app's nearest-gauge lookup uses this."""
    sites = pd.read_csv(SITES_FILE, dtype={"monitoring_location_id": str})
    coverage = df.groupby("monitoring_location_id").agg(
        first_date=("time", "min"),
        last_date=("time", "max"),
        has_streamflow=("parameter_code", lambda p: (p == "00060").any()),
        has_gauge_height=("parameter_code", lambda p: (p == "00065").any()),
    ).reset_index()
    active = sites.merge(coverage, on="monitoring_location_id", how="inner")
    active.to_csv(ACTIVE_SITES_FILE, index=False)
    print(f"  Saved {len(active)} active sites (with coordinates) to {ACTIVE_SITES_FILE}")


# ----------------------------------------------------------- status only
def status():
    if not os.path.exists(SERIES_FILE):
        print("Step 1 hasn't run yet (no series list). Run without --status first.")
        return
    series = pd.read_csv(SERIES_FILE, dtype=TEXT_COLUMNS)
    batches = make_batches(series)
    done = sum(os.path.exists(p) for _, p in batches)
    print(f"Series found: {len(series)} across {series['monitoring_location_id'].nunique()} sites")
    print(f"Batches saved: {done}/{len(batches)} ({done / len(batches):.0%})")
    print(f"Combined file exists: {os.path.exists(COMBINED_FILE)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="show progress without calling USGS")
    parser.add_argument("--refresh", action="store_true", help="redo Step 1 (series discovery)")
    args = parser.parse_args()

    if args.status:
        status()
    else:
        series = find_daily_series(refresh=args.refresh)
        finished = pull_batches(series)
        if finished:
            combine(series)
            print("\nUSGS backfill complete.")