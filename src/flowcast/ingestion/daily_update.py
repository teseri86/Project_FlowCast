"""
Daily update: add new days to the USGS and weather data (run AFTER both backfills).

WHAT IT DOES
  1. Pulls the last LOOKBACK_DAYS (30) of USGS river data and Open-Meteo weather.
     Re-pulling 30 days, not just yesterday, picks up corrections:
       - USGS "Provisional" values get checked and sometimes fixed in the weeks after.
       - Open-Meteo replaces its most recent days with better data about 5 days later.
  2. Saves each pull as its own file, named by today's date:
       data/raw/usgs_updates/2026-10-01.csv
       data/raw/weather_updates/2026-10-01.csv
  3. Rebuilds the combined files: backfill files first, update files stacked
     underneath, and for any day that appears twice the NEWEST value is kept.

  The backfill files are never changed. Running this twice in one day just
  replaces that day's update file. Cost: ~10 USGS requests and ~115
  Open-Meteo calls, a few minutes total.

HOW TO RUN (from the repo root, in the flowcast_env terminal)
    python -m src.flowcast.ingestion.daily_update
    python -m src.flowcast.ingestion.upload_to_s3        # then send it to S3
"""

import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

from src.flowcast.ingestion import usgs_backfill as usgs
from src.flowcast.ingestion import weather_backfill as weather

LOOKBACK_DAYS = 30 # number of days to look back when pulling updates. increase this if it's past 30 days since last update.
TODAY = date.today().isoformat()
START = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()


def save(df, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{TODAY}.csv")
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- USGS
def update_usgs():
    print(f"USGS: pulling {START} to today for all active series...")
    series = pd.read_csv(usgs.SERIES_FILE, dtype=usgs.TEXT_COLUMNS)
    ids = sorted(series["time_series_id"].unique())

    frames = []
    for i in range(0, len(ids), usgs.SERIES_PER_BATCH):
        chunk = ids[i:i + usgs.SERIES_PER_BATCH]
        try:
            df, _ = usgs.waterdata.get_daily(time_series_id=chunk, time=f"{START}/..", skip_geometry=True)
        except usgs.DataRetrievalError as e:
            print(f"  Stopped: {type(e).__name__}: {e}")
            print("  USGS update NOT saved. Try again later.")
            return False
        frames.append(usgs.clean(df))

    new = pd.concat(frames, ignore_index=True)
    path = save(new, usgs.UPDATE_DIR)
    print(f"  saved {len(new):,} rows to {path} "
          f"(dates {new['time'].min()} to {new['time'].max()})")
    usgs.combine(series)
    return True


# ------------------------------------------------------------- weather
def update_weather():
    print(f"\nWeather: pulling {START} to {YESTERDAY} for every grid square...")
    mapping = pd.read_csv(weather.MAP_FILE, dtype={"monitoring_location_id": str, "cell_id": str})
    cells = mapping.drop_duplicates("cell_id")

    frames = []
    for _, row in cells.iterrows():
        try:
            body = weather._get(weather.ARCHIVE_URL, {
                "latitude": row["cell_latitude"], "longitude": row["cell_longitude"],
                "start_date": START, "end_date": YESTERDAY,
                "daily": ",".join(weather.DAILY_VARIABLES),
                **weather.UNITS,
            })
        except (weather.DailyLimitReached, requests.RequestException) as e:
            print(f"  Stopped at {row['cell_id']}: {e}")
            print("  Weather update NOT saved. Try again later.")
            return False
        df = weather._daily_to_frame(body)
        df.insert(0, "cell_id", row["cell_id"])
        frames.append(df)
        time.sleep(0.5)  # well under Open-Meteo's 600 calls/minute

    new = pd.concat(frames, ignore_index=True)
    path = save(new, weather.UPDATE_DIR)
    print(f"  saved {len(new):,} rows ({len(frames)} squares) to {path}")
    weather.combine(mapping)
    return True


if __name__ == "__main__":
    missing = [f for f in (usgs.COMBINED_FILE, weather.COMBINED_FILE) if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"Finish the backfills first. Missing: {', '.join(missing)}")

    ok_usgs = update_usgs()
    ok_weather = update_weather()

    print("\n" + ("Daily update complete." if ok_usgs and ok_weather
                  else "Daily update partly failed (see above). Safe to run again."))
    print("Next: python -m src.flowcast.ingestion.upload_to_s3")