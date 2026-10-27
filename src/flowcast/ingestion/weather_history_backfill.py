"""
Open-Meteo weather HISTORY, 2007-2019, so the models can also TRAIN on 2007-2019 days
(the years with the big droughts: 2007-2008 and 2011-2012).

Same grid squares, variables and units as weather_backfill.py (it reuses that script's code),
so old and new weather numbers are directly comparable.

COST AND TIME (free tier: 1 call per 2 weeks of data, 10,000 calls/day, 5,000/hour)
    13 years = ~339 calls per grid square x 57 squares = ~19,300 calls
    -> 2 days (~29 squares per day). The script paces itself for the hourly limit, and when Open-Meteo says
       the DAILY limit is used up it stops cleanly. Run it again the next day: finished squares
       are kept and skipped.

HOW TO RUN (from the repo root, in the flowcast_env terminal)
    python -m src.flowcast.ingestion.weather_history_backfill            # pull (safe to rerun)
    python -m src.flowcast.ingestion.weather_history_backfill --status   # progress, no API calls

    It needs data/site_weather_cell_map.csv (made by weather_backfill.py; already on your computer).
    Leave the terminal open while it runs (~2-2.5 hours per day, mostly waiting between squares).

OUTPUTS
    data/raw/weather_history/<cell>.csv   one file per grid square
    data/weather_daily_2007_2019.csv      all squares combined
        -> also uploaded ONCE to S3: reference/openmeteo/daily_weather_2007_2019/<today>/
           (not under raw/, so upload_to_s3 does not re-upload it)
"""

import argparse
import math
import os
import time

import pandas as pd
import requests

from src.flowcast.ingestion.weather_backfill import (ARCHIVE_URL, DAILY_VARIABLES, HOURLY_CALL_BUDGET,
                                                     MAP_FILE, UNITS, DailyLimitReached, _daily_to_frame,
                                                     _get, estimated_calls)
from src.flowcast.storage import write_dataset

HISTORY_START = "2007-01-01"
HISTORY_END = "2019-12-31"     # 2020 onward is already in weather_daily_2020_present.csv
CELL_DIR = os.path.join("data", "raw", "weather_history")
COMBINED_FILE = os.path.join("data", "weather_daily_2007_2019.csv")
DAILY_CALL_LIMIT = 10_000      # Open-Meteo free tier


def cell_path(cell_id):
    return os.path.join(CELL_DIR, f"{cell_id}.csv")


def load_cells():
    if not os.path.exists(MAP_FILE):
        raise SystemExit(f"{MAP_FILE} not found. Run python -m src.flowcast.ingestion.weather_backfill first.")
    mapping = pd.read_csv(MAP_FILE, dtype={"monitoring_location_id": str, "cell_id": str})
    return mapping.drop_duplicates("cell_id").set_index("cell_id")[["cell_latitude", "cell_longitude"]]


def cost_report(cells):
    todo = [c for c in cells.index if not os.path.exists(cell_path(c))]
    per_cell = estimated_calls(HISTORY_START, HISTORY_END)
    total = per_cell * len(todo)
    print(f"  {len(cells) - len(todo)} of {len(cells)} grid squares done; {len(todo)} left.")
    squares_per_day = max(1, int(DAILY_CALL_LIMIT // per_cell))
    print(f"  Cost: ~{per_cell:.0f} calls per square -> ~{total:,.0f} calls left "
          f"(~{squares_per_day} squares fit in one day's limit -> {math.ceil(len(todo) / squares_per_day)} day(s)).")
    return todo, per_cell


def pull_all(cells):
    os.makedirs(CELL_DIR, exist_ok=True)
    todo, per_cell = cost_report(cells)
    wait_seconds = 3600 * per_cell / HOURLY_CALL_BUDGET      # stay under the hourly limit

    for n, cell_id in enumerate(todo, start=1):
        row = cells.loc[cell_id]
        try:
            body = _get(ARCHIVE_URL, {
                "latitude": row["cell_latitude"], "longitude": row["cell_longitude"],
                "start_date": HISTORY_START, "end_date": HISTORY_END,
                "daily": ",".join(DAILY_VARIABLES), **UNITS,
            })
        except DailyLimitReached as e:
            print(f"\n  Open-Meteo says: {e}")
            print("  Everything saved so far is safe. Run this again tomorrow (or in an hour if it")
            print("  mentions an hourly limit) and it will continue from here.")
            return False
        except requests.RequestException as e:
            print(f"  [{n}/{len(todo)}] {cell_id}: FAILED ({e}). Will retry on next run.")
            continue

        df = _daily_to_frame(body)
        df.insert(0, "cell_id", cell_id)
        tmp = cell_path(cell_id) + ".tmp"                    # temp file, then rename
        df.to_csv(tmp, index=False)
        os.replace(tmp, cell_path(cell_id))
        print(f"  [{n}/{len(todo)}] {cell_id}: {len(df):,} days saved", end="")
        if n < len(todo):
            print(f"  (waiting {wait_seconds:.0f}s for the hourly limit)")
            time.sleep(wait_seconds)
        else:
            print()
    return all(os.path.exists(cell_path(c)) for c in cells.index)


def combine(cells):
    df = pd.concat([pd.read_csv(cell_path(c), dtype={"cell_id": str}) for c in sorted(cells.index)],
                   ignore_index=True)
    df = df.drop_duplicates(subset=["cell_id", "date"]).sort_values(["cell_id", "date"])
    per_cell = df.groupby("cell_id").size()
    print(f"\nCombined {len(df):,} rows from {len(per_cell)} grid squares, {df['date'].min()} to {df['date'].max()}")
    print(f"  Days per square: min {per_cell.min():,}, max {per_cell.max():,} (expected 4,748)")
    print(f"  Days missing precipitation: {df['precipitation_sum'].isna().sum():,}")
    print("\nSaving and uploading (one time)...")
    write_dataset("weather_history", df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="show progress without calling Open-Meteo")
    parser.add_argument("--local", action="store_true", help="save locally only (no S3 upload)")
    args = parser.parse_args()
    if args.local:
        os.environ["FLOWCAST_DATA_SOURCE"] = "local"

    cells = load_cells()
    if args.status:
        cost_report(cells)
        print(f"  Combined file exists: {os.path.exists(COMBINED_FILE)}")
    elif pull_all(cells):
        combine(cells)
        print("\nWeather history complete. Next: python -m src.flowcast.model.build_training_data")
