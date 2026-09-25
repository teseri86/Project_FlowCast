"""
Weather backfill from Open-Meteo: daily precipitation, snow, temperature and
evaporation, 2020-01-01 through yesterday, for every USGS site that has river data.

WHY OPEN-METEO
  - Gives a value for ANY latitude/longitude, for any day, including today and
    the forecast. So training and live predictions get the same kind of number.
  - No station matching, no stations that stop reporting, no token needed.
  - The trade-off: it's a weather-model estimate on a grid (~11 km squares),
    not a physical rain gauge.

HOW IT SAVES API CALLS
  Open-Meteo's free tier counts every 2 weeks of data as 1 "call", so
  2020-today is ~175 calls PER LOCATION. Limits: 5,000 calls/hour and
  10,000 calls/day. To stay efficient:
    1. Only sites with USGS daily data are used (from usgs_backfill Step 1).
    2. Sites are grouped into grid squares (GRID_DEGREES). Sites in the same
       square get identical weather anyway, so each square is pulled once.
    3. The script waits between requests so it never trips the hourly limit.
  If it hits the DAILY limit it stops; run it again tomorrow and it continues
  where it left off (each grid square is saved to its own file).

HOW TO RUN (from the repo root, in the flowcast_env terminal)
    python -m src.flowcast.ingestion.usgs_backfill      # run this first (finds active sites)
    python -m src.flowcast.ingestion.weather_backfill             # pull (safe to rerun)
    python -m src.flowcast.ingestion.weather_backfill --status    # progress + cost estimate
    python -m src.flowcast.ingestion.weather_backfill --refresh   # rebuild the site -> grid map

OUTPUTS
    data/site_weather_cell_map.csv       each USGS site -> its grid square (cell_id)
    data/raw/weather_daily/<cell>.csv    one file per grid square
    data/weather_daily_2020_present.csv  all squares combined, one row per cell/day

UNITS: inches (precipitation, snow, evapotranspiration), Fahrenheit (temperature).
Dates are local Eastern time days.
"""

import argparse
import math
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

# ---------------------------------------------------------------- settings
START_DATE = "2020-01-01"
END_DATE = (date.today() - timedelta(days=1)).isoformat()  # today isn't finished yet

# 0.1 degrees is about 11 km, which matches the resolution of Open-Meteo's
# underlying data. Setting 0.25 (~25 km) means fewer squares and fewer calls,
# but coarser rain.

GRID_DEGREES = 0.1

# Keep this list IDENTICAL to what the live prediction code requests, so the
# model sees the same kind of numbers in training and in production.

DAILY_VARIABLES = [
    "precipitation_sum",           # total rain + snow (as water), inches
    "rain_sum",                    # rain only
    "snowfall_sum",                # snow depth that fell
    "precipitation_hours",         # how many hours it was wet (storm length)
    "et0_fao_evapotranspiration",  # water pulled back out by heat/sun (drought signal)
    "temperature_2m_max",
    "temperature_2m_min",
]
UNITS = {"precipitation_unit": "inch", "temperature_unit": "fahrenheit", "timezone": "America/New_York"}

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_CALL_BUDGET = 4500  # stay safely under the 5,000/hour limit

SITES_FILE = os.path.join("resources", "usgs_monitoring_locations.csv")
ACTIVE_SERIES_FILE = os.path.join("data", "usgs_daily_series.csv")  # from usgs_backfill Step 1
DATA_DIR = "data"
MAP_FILE = os.path.join(DATA_DIR, "site_weather_cell_map.csv")
CELL_DIR = os.path.join(DATA_DIR, "raw", "weather_daily")
COMBINED_FILE = os.path.join(DATA_DIR, "weather_daily_2020_present.csv")
UPDATE_DIR = os.path.join(DATA_DIR, "raw", "weather_updates")  # written by daily_update.py


# ------------------------------------------------- shared helpers
def cell_for(lat, lon, grid=GRID_DEGREES):

    """Snap a point to the center of its grid square. Returns (cell_id, lat, lon).
    Use this in the live app too, so an address lands in the same square the
    model was trained on."""

    cell_lat = round(math.floor(lat / grid) * grid + grid / 2, 4)
    cell_lon = round(math.floor(lon / grid) * grid + grid / 2, 4)
    return f"{cell_lat:.3f}_{cell_lon:.3f}", cell_lat, cell_lon


def estimated_calls(start, end, n_variables=len(DAILY_VARIABLES)):

    """Open-Meteo's counting rule: 1 call per 2 weeks of data, and more if
    over 10 variables."""

    days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    return max(1.0, days / 14) * max(1.0, n_variables / 10)


class DailyLimitReached(Exception):
    pass


def _get(url, params):
    resp = requests.get(url, params=params, timeout=120)
    if resp.status_code == 429 or (resp.status_code == 400 and "limit" in resp.text.lower()):
        reason = resp.json().get("reason", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        raise DailyLimitReached(reason)
    resp.raise_for_status()
    return resp.json()


def _daily_to_frame(body):
    df = pd.DataFrame(body["daily"]).rename(columns={"time": "date"})
    df["grid_latitude"] = body.get("latitude")    # the model grid point Open-Meteo actually used
    df["grid_longitude"] = body.get("longitude")
    return df


def get_recent_and_forecast(lat, lon, past_days=14, forecast_days=7):

    """FOR THE LIVE APP (not the backfill): last `past_days` of weather plus the
    next `forecast_days` for one location, same variables and units as training,
    avoiding training/prediction mismatch for the app. Costs ~1-2 calls."""

    _, cell_lat, cell_lon = cell_for(lat, lon)
    body = _get(FORECAST_URL, {
        "latitude": cell_lat, "longitude": cell_lon,
        "daily": ",".join(DAILY_VARIABLES),
        "past_days": past_days, "forecast_days": forecast_days,
        **UNITS,
    })
    df = _daily_to_frame(body)
    df["is_forecast"] = pd.to_datetime(df["date"]).dt.date >= date.today()
    return df


# ------------------------------------------------- Step 1: sites -> grid squares
def build_cell_map(refresh=False):
    if os.path.exists(MAP_FILE) and not refresh:
        mapping = pd.read_csv(MAP_FILE, dtype={"monitoring_location_id": str, "cell_id": str})
        print(f"Step 1: using saved site -> grid map ({mapping['cell_id'].nunique()} squares). "
              f"Use --refresh to rebuild.")
        return mapping

    sites = pd.read_csv(SITES_FILE, dtype={"monitoring_location_id": str})
    sites = sites.dropna(subset=["latitude", "longitude"])

    if os.path.exists(ACTIVE_SERIES_FILE):
        active = pd.read_csv(ACTIVE_SERIES_FILE, dtype=str)["monitoring_location_id"].unique()
        sites = sites[sites["monitoring_location_id"].isin(active)]
        print(f"Step 1: using the {len(sites)} sites that have USGS daily data.")
    else:
        print("Step 1: NOTE - usgs_backfill hasn't run yet, so using ALL sites.\n"
              "        Run usgs_backfill first to cut the number of API calls a lot.")

    cells = sites.apply(lambda r: cell_for(r["latitude"], r["longitude"]), axis=1)
    sites["cell_id"] = [c[0] for c in cells]
    sites["cell_latitude"] = [c[1] for c in cells]
    sites["cell_longitude"] = [c[2] for c in cells]
    mapping = sites[["monitoring_location_id", "cell_id", "cell_latitude", "cell_longitude"]]

    os.makedirs(DATA_DIR, exist_ok=True)
    mapping.to_csv(MAP_FILE, index=False)
    print(f"  -> {len(mapping)} sites fall in {mapping['cell_id'].nunique()} grid squares. Saved to {MAP_FILE}\n")
    return mapping


# ------------------------------------------------- Step 2: pull each square
def cell_path(cell_id):
    return os.path.join(CELL_DIR, f"{cell_id}.csv")


def cost_report(mapping):
    cells = mapping["cell_id"].unique()
    todo = [c for c in cells if not os.path.exists(cell_path(c))]
    per_cell = estimated_calls(START_DATE, END_DATE)
    total = per_cell * len(todo)
    print(f"  Cost: ~{per_cell:.0f} calls per square x {len(todo)} squares left = ~{total:,.0f} calls")
    print(f"  Free limits: 5,000/hour, 10,000/day -> about {total / HOURLY_CALL_BUDGET:.1f} hours of pulling"
          f"{', spread over ' + str(math.ceil(total / 9500)) + ' days' if total > 9500 else ''}.")
    return todo, per_cell


def pull_all(mapping):
    os.makedirs(CELL_DIR, exist_ok=True)
    cells = mapping.drop_duplicates("cell_id").set_index("cell_id")
    print(f"Step 2: {len(cells)} grid squares total.")
    todo, per_cell = cost_report(mapping)
    # Wait between requests so we use at most HOURLY_CALL_BUDGET calls per hour.
    wait_seconds = 3600 * per_cell / HOURLY_CALL_BUDGET

    for n, cell_id in enumerate(todo, start=1):
        row = cells.loc[cell_id]
        try:
            body = _get(ARCHIVE_URL, {
                "latitude": row["cell_latitude"], "longitude": row["cell_longitude"],
                "start_date": START_DATE, "end_date": END_DATE,
                "daily": ",".join(DAILY_VARIABLES),
                **UNITS,
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
        tmp = cell_path(cell_id) + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, cell_path(cell_id))
        print(f"  [{n}/{len(todo)}] {cell_id}: {len(df):,} days saved", end="")

        if n < len(todo):
            print(f"  (waiting {wait_seconds:.0f}s to respect the hourly limit)")
            time.sleep(wait_seconds)
        else:
            print()

    return all(os.path.exists(cell_path(c)) for c in cells.index)


# ------------------------------------------------- Step 3: combine
def update_files():
    """Daily update files (from daily_update.py), oldest first (names are dates)."""
    if not os.path.isdir(UPDATE_DIR):
        return []
    return sorted(os.path.join(UPDATE_DIR, f) for f in os.listdir(UPDATE_DIR) if f.endswith(".csv"))


def combine(mapping):

    """Backfill squares first, then daily updates stacked underneath. For any
    square+day that appears more than once, keep="last" keeps the newest value."""

    paths = [cell_path(c) for c in sorted(mapping["cell_id"].unique())]
    paths = [p for p in paths if os.path.exists(p)] + update_files()
    frames = [pd.read_csv(p, dtype={"cell_id": str}) for p in paths]
    if not frames:
        print("Nothing to combine yet.")
        return
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["cell_id", "date"], keep="last")
    df = df.sort_values(["cell_id", "date"])
    tmp = COMBINED_FILE + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, COMBINED_FILE)

    expected = (date.fromisoformat(END_DATE) - date.fromisoformat(START_DATE)).days + 1
    per_cell = df.groupby("cell_id").size()
    n_updates = len(update_files())
    extra = f" + {n_updates} daily update file(s)" if n_updates else ""
    print(f"\nStep 3: combined {len(df):,} rows from {len(per_cell)} grid squares{extra} into {COMBINED_FILE}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Days per square: min {per_cell.min():,}, max {per_cell.max():,} (expected {expected:,})")
    print(f"  Days missing precipitation: {df['precipitation_sum'].isna().sum():,}")


def status():
    if not os.path.exists(MAP_FILE):
        print("Step 1 hasn't run yet. Run without --status first.")
        return
    mapping = pd.read_csv(MAP_FILE, dtype={"cell_id": str})
    cells = mapping["cell_id"].unique()
    done = sum(os.path.exists(cell_path(c)) for c in cells)
    print(f"Sites mapped: {len(mapping)}  |  Grid squares saved: {done}/{len(cells)}")
    cost_report(mapping)
    print(f"Combined file exists: {os.path.exists(COMBINED_FILE)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="show progress without calling Open-Meteo")
    parser.add_argument("--refresh", action="store_true", help="rebuild the site -> grid square map")
    args = parser.parse_args()

    if args.status:
        status()
    else:
        mapping = build_cell_map(refresh=args.refresh)
        if pull_all(mapping):
            combine(mapping)
            print("\nWeather backfill complete.")