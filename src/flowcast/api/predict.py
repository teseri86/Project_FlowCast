"""
Make a FlowCast prediction for one place: the core of the future prediction API.

    address -> coordinates (US Census Geocoder) -> nearest gauge that has a model
            -> that gauge's latest complete days of readings (USGS)
            -> recent weather + forecast for its grid square (Open-Meteo)
            -> both models -> plain answers, labeled by the date they are for

HOW TO RUN (from the repo root, after train_final_models.py)
    python -m src.flowcast.api.predict --address "600 E 4th St, Charlotte, NC"
    python -m src.flowcast.api.predict --gauge USGS-0212419274
    Add --local to read the saved models/ruler from the local data/ folder instead of S3.

HOW EACH ANSWER IS MADE (exactly like the training data)
    - The latest COMPLETE day of river readings is "day 0" (usually yesterday: USGS publishes a
      daily average after the day ends).
    - Day 0 and the days before it are scored against ruler.csv, each against the OTHER years
      (hw_score: daily level vs all days; dry_score: 7-day average vs days within +/-15 calendar days).
    - The rain after day 0 comes from the Open-Meteo FORECAST (in training it was the rain that
      actually fell, so real accuracy is at most the measured "perfect forecast" accuracy).
    - High water answers "day 0 + 1"; drought answers "day 0 + 7".
    - The answer is yes/no only. The models' raw outputs are NOT chances (class weighting pushes
      them up), so they are not shown as percentages.
"""

import argparse
import math
import os
import tempfile
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import requests

from src.flowcast.model.build_training_data import day_of_year_365, score_against
from src.flowcast.storage import DATASETS, latest_s3_key, read_dataset

MAX_DISTANCE_MILES = 10
HISTORY_DAYS = 45        # days of readings/weather needed before day 0 (30-day features + changes)
GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_VARIABLES = ["precipitation_sum", "et0_fao_evapotranspiration", "temperature_2m_max"]
WEATHER_UNITS = {"precipitation_unit": "inch", "temperature_unit": "fahrenheit", "timezone": "America/New_York"}
PARAMETER = {"streamflow": "00060", "gauge_height": "00065"}


# ---------------------------------------------------------------- loading the saved models
def load_final(local=False):
    """The saved models bundle and the ruler (from S3 unless local)."""
    path = os.path.join("data", "model", "final_models.joblib")
    if not local:
        from src.flowcast.storage import _s3
        s3, bucket = _s3()
        keys = [o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix="models/final/").get("Contents", [])
                if o["Key"].endswith("final_models.joblib")]
        if not keys:
            raise FileNotFoundError("No final models in S3. Run python -m src.flowcast.model.train_final_models")
        path = os.path.join(tempfile.gettempdir(), "final_models.joblib")
        s3.download_file(bucket, max(keys), path)
    bundle = joblib.load(path)
    ruler = read_dataset("ruler", source="local" if local else None,
                         dtype={"monitoring_location_id": str}, parse_dates=["date"])
    return bundle, ruler


# ---------------------------------------------------------------- the three outside services
def geocode(address):
    """Address -> (lat, lon, matched address), via the US Census Geocoder (free, no key)."""
    r = requests.get(GEOCODER_URL, params={"address": address, "benchmark": "Public_AR_Current",
                                           "format": "json"}, timeout=30)
    r.raise_for_status()
    matches = r.json()["result"]["addressMatches"]
    if not matches:
        raise ValueError(f"The Census Geocoder could not find: {address}")
    m = matches[0]
    return m["coordinates"]["y"], m["coordinates"]["x"], m["matchedAddress"]


def fetch_recent_levels(gauge_id, measure, days=HISTORY_DAYS):
    """The gauge's daily averages for the last `days` days, from USGS. Columns: date, level_value.
    Uses the same request as daily_update.py (by time series ID), which is known to work."""
    from src.flowcast.ingestion import usgs_backfill as usgs        # loads .env (API_USGS_PAT) first
    series = read_dataset("usgs_daily_series", dtype=usgs.TEXT_COLUMNS)
    ids = series[(series["monitoring_location_id"] == gauge_id) & (series["parameter_code"] == PARAMETER[measure])
                 & (series["statistic_id"] == "00003")]["time_series_id"].unique().tolist()
    if not ids:
        raise ValueError(f"No daily {measure} series for {gauge_id} in usgs_daily_series.csv")
    start = (date.today() - timedelta(days=days)).isoformat()
    df, _ = usgs.waterdata.get_daily(time_series_id=ids, time=f"{start}/..", skip_geometry=True)
    df = usgs.clean(df)
    df = df.assign(date=pd.to_datetime(df["time"], errors="coerce"),
                   level_value=pd.to_numeric(df["value"], errors="coerce")).dropna(subset=["date"])
    if df["level_value"].notna().sum() == 0:
        raise ValueError(f"USGS returned no {measure} readings for {gauge_id} since {start}.")
    return df.groupby("date", as_index=False)["level_value"].mean()


def fetch_weather(cell_lat, cell_lon, past_days=HISTORY_DAYS, forecast_days=8):
    """Recent days + forecast for one grid square, from Open-Meteo. Columns: date + WEATHER_VARIABLES."""
    r = requests.get(FORECAST_URL, params={"latitude": cell_lat, "longitude": cell_lon,
                                           "daily": ",".join(WEATHER_VARIABLES), "past_days": past_days,
                                           "forecast_days": forecast_days, **WEATHER_UNITS}, timeout=60)
    r.raise_for_status()
    w = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
    w["date"] = pd.to_datetime(w["date"])
    return w


# ---------------------------------------------------------------- the prediction
_CACHE = {}


def _cell_map():
    if "cells" not in _CACHE:
        _CACHE["cells"] = read_dataset("weather_cell_map", dtype={"monitoring_location_id": str, "cell_id": str})
    return _CACHE["cells"]


def nearest_gauges(lat, lon, bundle):
    """Gauges that have a model, nearest first, with distance in miles. Only those within the limit."""
    sites = read_dataset("usgs_active_sites", dtype={"monitoring_location_id": str})
    sites = sites[sites["monitoring_location_id"].isin(bundle["trained_on"]["gauges"])].copy()
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2, lon2 = np.radians(sites["latitude"]), np.radians(sites["longitude"])
    a = np.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    sites["miles"] = 3958.8 * 2 * np.arcsin(np.sqrt(a))
    sites = sites.sort_values("miles")
    return sites[sites["miles"] <= MAX_DISTANCE_MILES], float(sites["miles"].iloc[0])


def score_days(recent, ruler_gauge, settings):
    """hw_score and dry_score for each recent day, each against the ruler's OTHER years."""
    recent = recent.copy()
    recent["hw_score"] = np.nan
    recent["dry_score"] = np.nan
    r_year = ruler_gauge["date"].dt.year.to_numpy()
    r_doy = day_of_year_365(ruler_gauge["date"])
    r_level = ruler_gauge["level_value"].to_numpy(dtype=float)
    r_level7 = ruler_gauge["level_7d"].to_numpy(dtype=float)
    for i, row in recent.iterrows():
        other = r_year != row["date"].year
        if pd.notna(row["level_value"]):
            recent.at[i, "hw_score"] = score_against(np.array([row["level_value"]]), r_level[other])[0]
        if pd.notna(row["level_7d"]):
            d = day_of_year_365(pd.Series([row["date"]]))[0]
            dist = np.minimum(np.abs(r_doy - d), 365 - np.abs(r_doy - d))
            near = other & (dist <= settings["window_days"])
            recent.at[i, "dry_score"] = score_against(np.array([row["level_7d"]]), r_level7[near])[0]
    return recent


def build_inputs(levels, weather, drainage_area, settings, ruler_gauge, day0=None):
    """The model inputs for day 0, computed the same way as build_training_data.py."""
    full = pd.DataFrame({"date": pd.date_range(levels["date"].min(), levels["date"].max())})
    lv = full.merge(levels, on="date", how="left")
    lv["level_7d"] = lv["level_value"].rolling(settings["smooth_days"], min_periods=5).mean().round(6)  # same rounding as training
    if day0 is None:
        day0 = lv.loc[lv["level_value"].notna(), "date"].max()          # latest complete day
    lv = lv[lv["date"] <= day0]
    lv = score_days(lv, ruler_gauge, settings)

    s = lv.set_index("date")
    today = s.loc[day0]
    x = {"hw_score": today["hw_score"],
         "hw_score_change_1d": today["hw_score"] - s["hw_score"].get(day0 - pd.Timedelta(days=1), np.nan),
         "hw_score_change_3d": today["hw_score"] - s["hw_score"].get(day0 - pd.Timedelta(days=3), np.nan),
         "hw_score_mean_30d": (s["hw_score"].loc[day0 - pd.Timedelta(days=29):day0].mean()
                               if s["hw_score"].loc[day0 - pd.Timedelta(days=29):day0].notna().sum() >= 20 else np.nan),
         "dry_score": today["dry_score"],
         "dry_score_change_7d": today["dry_score"] - s["dry_score"].get(day0 - pd.Timedelta(days=7), np.nan)}

    w = weather.set_index("date")
    def past(col, days, how="sum"):
        win = w[col].loc[day0 - pd.Timedelta(days=days - 1):day0]
        return getattr(win, how)() if win.notna().sum() == days else np.nan
    def ahead(days):
        win = w["precipitation_sum"].loc[day0 + pd.Timedelta(days=1):day0 + pd.Timedelta(days=days)]
        return win.sum() if win.notna().sum() == days else np.nan
    x.update({"precip_1d": w["precipitation_sum"].get(day0, np.nan), "precip_3d": past("precipitation_sum", 3),
              "precip_7d": past("precipitation_sum", 7), "precip_30d": past("precipitation_sum", 30),
              "et0_7d": past("et0_fao_evapotranspiration", 7), "et0_30d": past("et0_fao_evapotranspiration", 30),
              "temp_max_mean_7d": past("temperature_2m_max", 7, "mean"),
              "month": day0.month, "log_drainage_area": np.log1p(drainage_area),
              "rain_next_1d": ahead(1), "rain_next_7d": ahead(7)})
    x["water_balance_30d"] = x["precip_30d"] - x["et0_30d"]
    return x, day0, today


def predict_gauge(gauge, bundle, ruler, levels=None, weather=None, day0=None):
    """Both answers for one gauge (a row of usgs_active_sites). levels/weather can be passed in
    (e.g. for testing); otherwise they are downloaded."""
    gid = gauge["monitoring_location_id"]
    ruler_gauge = ruler[ruler["monitoring_location_id"] == gid]
    measure = ruler_gauge["measure"].iloc[0]
    cells = _cell_map()
    cell = cells[cells["monitoring_location_id"] == gid].iloc[0]
    if levels is None:
        levels = fetch_recent_levels(gid, measure)
    if weather is None:
        weather = fetch_weather(cell["cell_latitude"], cell["cell_longitude"])

    x, day0, today = build_inputs(levels, weather, float(gauge["drainage_area"]), bundle["scoring"], ruler_gauge, day0)
    if (pd.Timestamp(date.today()) - day0).days > 3:
        print(f"  NOTE: the latest complete reading for {gid} is {day0.date()} - more than 3 days old.")
    answers = {}
    for name, m in bundle["models"].items():
        row = pd.DataFrame([{c: x.get(c, np.nan) for c in m["features"]}])
        if row[m["features"][-1]].isna().all():                     # no forecast -> fall back to "same as today"
            label = "is_high" if name == "high_water" else "is_drought"
            same = (today["hw_score"] >= bundle["scoring"]["high_pct"]) if name == "high_water" else \
                   (today["dry_score"] < bundle["scoring"]["low_pct"] and today["hw_score"] < bundle["scoring"]["high_pct"])
            answers[name] = {"answer": bool(same), "method": f"same as today (no forecast; {label})"}
        else:
            prob = float(m["model"].predict_proba(row.fillna(m["fill_values"]))[:, 1][0])
            answers[name] = {"answer": prob >= m["cutoff"], "method": f"{m['model_type']} model"}
        answers[name]["for_date"] = (day0 + pd.Timedelta(days=m["horizon_days"])).date()
    return {"gauge": gid, "gauge_name": gauge["monitoring_location_name"], "measure": measure,
            "day0": day0.date(), "latest_reading": float(today["level_value"]),
            "hw_score": float(today["hw_score"]), "dry_score": float(today["dry_score"]), "answers": answers}


def _compared(score):
    """'higher than 86%' or 'lower than 92%' - whichever reads naturally for this score."""
    return f"higher than {score:.0f}%" if score >= 50 else f"lower than {100 - score:.0f}%"


def describe(result, bundle, miles=None):
    units = "cfs" if result["measure"] == "streamflow" else "ft"
    lines = [f"Gauge: {result['gauge_name']} ({result['gauge']})" + (f", {miles:.1f} miles away" if miles is not None else ""),
             f"Latest complete day: {result['day0']}  reading {result['latest_reading']:g} {units}  "
             f"({_compared(result['hw_score'])} of past days; "
             f"7-day average {_compared(result['dry_score'])} of past days at this time of year)"]
    names = {"high_water": "High water", "drought": "Drought"}
    for name, a in result["answers"].items():
        past = "  (already past)" if a["for_date"] < date.today() else ""
        lines.append(f"  {names[name]} on {a['for_date']:%a %b %d}: {'YES' if a['answer'] else 'no'}{past}   [{a['method']}]")
        acc = bundle["models"][name].get("measured_accuracy")
        if a["method"].startswith("same as today"):
            lines.append("      no rain forecast available, so this just repeats today's condition")
        elif acc:
            lines.append(f"      in past testing: caught about {acc['caught_avg']:.0%} of {names[name].lower()} days; "
                         f"about {acc['right_when_flagged_avg']:.0%} of its warnings were right")
    lines.append("  Official sources: NWS alerts (weather.gov), U.S. Drought Monitor (droughtmonitor.unl.edu).")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--address")
    where.add_argument("--gauge")
    parser.add_argument("--local", action="store_true", help="read the saved models and ruler from the local data/ folder")
    args = parser.parse_args()
    if args.local:
        os.environ["FLOWCAST_DATA_SOURCE"] = "local"

    bundle, ruler = load_final(local=args.local)
    sites = read_dataset("usgs_active_sites", dtype={"monitoring_location_id": str})
    miles = None
    if args.address:
        lat, lon, matched = geocode(args.address)
        print(f"Address found: {matched} ({lat:.4f}, {lon:.4f})")
        candidates, closest = nearest_gauges(lat, lon, bundle)
        if candidates.empty:
            print(f"No gauge with a model within {MAX_DISTANCE_MILES} miles (nearest is {closest:.1f} miles away).")
            return
        # Nearest first; if a gauge has no recent readings (e.g. USGS stopped reporting it), try the next one.
        for _, gauge in candidates.iterrows():
            try:
                result = predict_gauge(gauge, bundle, ruler)
            except ValueError as e:
                print(f"  Skipping {gauge['monitoring_location_name']} ({gauge['miles']:.1f} mi): {e}")
                continue
            print(describe(result, bundle, gauge["miles"]))
            return
        print(f"None of the {len(candidates)} gauges within {MAX_DISTANCE_MILES} miles has recent readings.")
        return
    else:
        if args.gauge not in bundle["trained_on"]["gauges"]:
            raise SystemExit(f"{args.gauge} has no model (not one of the {len(bundle['trained_on']['gauges'])} gauges).")
        gauge = sites[sites["monitoring_location_id"] == args.gauge].iloc[0]
        print(describe(predict_gauge(gauge, bundle, ruler), bundle, miles))


if __name__ == "__main__":
    main()
