"""
FlowCast quickstart: train both models on a small sample, then get a prediction for an address,
just like the app (about a minute).

No AWS account or API keys needed. The training sample is included in bin/sample_data/:
    river.csv     daily streamflow, 2007 to Oct 2026, for 3 USGS gauges:
                    USGS-0214642825  Briar Creek near Charlotte, NC        (small creek,  5 sq mi)
                    USGS-0212419274  Coddle Creek near Davidson, NC         (stream,      23 sq mi)
                    USGS-02146000    Catawba River near Rock Hill, SC       (large river, 3,050 sq mi)
    weather.csv   daily rain, evapotranspiration and high temperature for their weather grid squares
    sites.csv     gauge names, counties, drainage areas, coordinates
    cell_map.csv  which weather grid square each gauge is in
    series.csv    each gauge's USGS series ID (to download its latest readings)

HOW TO RUN (from the repo root, after installing requirements.txt)
    python bin/quickstart.py                                   # asks for an address
    python bin/quickstart.py --address "600 E 4th St, Charlotte, NC"
    python bin/quickstart.py --offline                         # no internet: replays a recent day from the sample

WHAT IT DOES (the same steps as the full pipeline, on 3 gauges instead of 82)
    1. TRAIN + TEST: scores every day against the gauge's other years, trains a decision tree and boosted
       trees for each question (high water tomorrow; drought in 7 days), and tests them on 2024 to
       Oct 2026 against "same as today".
    2. TRAIN THE DEMO MODELS on all sample years (boosted trees, like the deployed models).
    3. PREDICT FOR AN ADDRESS: US Census Geocoder -> nearest sample gauge -> its latest readings (USGS)
       and rain forecast (Open-Meteo) -> both answers, labeled with the date they are for.
       Only the 3 sample gauges are available here, so use an address in or near Charlotte, Davidson or
       Rock Hill. The full app (src/flowcast/api/predict.py) covers 82 gauges.

Weather data by Open-Meteo.com (CC BY 4.0). River data: U.S. Geological Survey.
"""

import argparse
import math
import os
import sys
import time
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.flowcast.api import predict as P                                    # noqa: E402
from src.flowcast.model import compare_models as cm                          # noqa: E402
from src.flowcast.model.build_training_data import (HIGH_PCT, LOW_PCT, SMOOTH_DAYS, WINDOW_DAYS,  # noqa: E402
                                                    build_table, features_for, train_test_masks)

SAMPLE = os.path.join(REPO, "bin", "sample_data")
OUTPUT = os.path.join(REPO, "bin", "quickstart_output")
SPLIT_DATE = "2024-01-01"                       # step 1: train before this date, test on/after it
DEFAULT_ADDRESS = "600 E 4th St, Charlotte, NC"
MAX_MILES = 25                                  # only 3 gauges in the sample, so a wider limit than the app's 10
SETTINGS = {"high_pct": HIGH_PCT, "low_pct": LOW_PCT, "window_days": WINDOW_DAYS, "smooth_days": SMOOTH_DAYS}

QUESTIONS = [  # (key, name, target column, today's label used by the baseline)
    ("high_water", "High water, 1 day ahead", "target_high_1d", "is_high"),
    ("drought", "Drought, 7 days ahead", "target_drought_7d", "is_drought"),
]


# ---------------------------------------------------------------- data
def load_sample():
    """The sample files, shaped like the full pipeline's inputs."""
    ids = {"monitoring_location_id": str, "cell_id": str, "time_series_id": str}
    read = lambda name, **kw: pd.read_csv(os.path.join(SAMPLE, name), dtype=ids, **kw)
    river = read("river.csv", parse_dates=["date"])
    weather = read("weather.csv", parse_dates=["date"])
    sites, cell_map, series = read("sites.csv"), read("cell_map.csv"), read("series.csv")

    # One row per gauge per calendar day (missing days become empty rows), like load_level().
    days = pd.date_range(river["date"].min(), river["date"].max())
    grid = pd.MultiIndex.from_product([river["monitoring_location_id"].unique(), days],
                                      names=["monitoring_location_id", "date"])
    level = river.set_index(["monitoring_location_id", "date"])[["level_value"]].reindex(grid).reset_index()
    level["measure"] = level["monitoring_location_id"].map(river.groupby("monitoring_location_id")["measure"].first())
    return level, weather, sites, cell_map, series


def make_ruler(level):
    """Each gauge's history with its 7-day average (rounded exactly like training)."""
    r = level.sort_values(["monitoring_location_id", "date"]).copy()
    r["level_7d"] = (r.groupby("monitoring_location_id")["level_value"]
                     .transform(lambda s: s.rolling(SMOOTH_DAYS, min_periods=5).mean())).round(6)
    return r.dropna(subset=["level_value", "level_7d"], how="all")


# ---------------------------------------------------------------- steps 1 and 2
def train_and_test(table):
    """Step 1: tree vs boosted vs 'same as today' on 2024-Oct 2026. Returns the results table."""
    dates = pd.to_datetime(table["date"])
    rows = []
    for key, name, target, today_label in QUESTIONS:
        horizon = int(target.rsplit("_", 1)[1].rstrip("d"))
        cols = features_for(target, with_future_rain=True)
        train, test = train_test_masks(table, target, SPLIT_DATE)
        fill = table.loc[train, cols].median()
        holdout_start = pd.Timestamp(SPLIT_DATE) - pd.DateOffset(years=1)
        fit_part = train & (dates + pd.Timedelta(days=horizon) < holdout_start)
        holdout = train & (dates >= holdout_start)
        y_test = table.loc[test, target].astype(int)
        rows.append({"key": key, "question": name, "model": "same as today",
                     **cm.grade(y_test, table.loc[test, today_label].fillna(0).astype(int))})
        for model_name, make in [("decision tree", cm.make_tree), ("boosted trees", cm.make_boosted)]:
            cutoff = cm.pick_cutoff(make, table, cols, target, fit_part, holdout, fill)
            model = make().fit(table.loc[train, cols].fillna(fill), table.loc[train, target].astype(int))
            pred = (model.predict_proba(table.loc[test, cols].fillna(fill))[:, 1] >= cutoff).astype(int)
            rows.append({"key": key, "question": name, "model": model_name, **cm.grade(y_test, pred)})
    return pd.DataFrame(rows)


def train_demo_models(table, results):
    """Step 2: boosted trees on ALL sample days, packaged like the deployed models."""
    dates = pd.to_datetime(table["date"])
    bundle = {"models": {}}
    for key, name, target, _ in QUESTIONS:
        horizon = int(target.rsplit("_", 1)[1].rstrip("d"))
        cols = features_for(target, with_future_rain=True)
        has_answer = table[target].notna()
        fill = table.loc[has_answer, cols].median()
        holdout_start = dates.max() - pd.DateOffset(years=1)
        cutoff = cm.pick_cutoff(cm.make_boosted, table, cols, target,
                                has_answer & (dates + pd.Timedelta(days=horizon) < holdout_start),
                                has_answer & (dates >= holdout_start), fill)
        model = cm.make_boosted().fit(table.loc[has_answer, cols].fillna(fill), table.loc[has_answer, target].astype(int))
        tested = results[(results["key"] == key) & (results["model"] == "boosted trees")].iloc[0]
        bundle["models"][key] = {"model": model, "model_type": "boosted", "features": cols, "cutoff": cutoff,
                                 "fill_values": fill.to_dict(), "horizon_days": horizon,
                                 "measured_accuracy": {"caught_avg": tested["caught (recall)"],
                                                       "right_when_flagged_avg": tested["right when flagged (precision)"]}}
    os.makedirs(OUTPUT, exist_ok=True)
    joblib.dump(bundle, os.path.join(OUTPUT, "demo_models.joblib"))
    return bundle


# ---------------------------------------------------------------- step 3
def nearest(lat, lon, sites):
    lat2, lon2 = np.radians(sites["latitude"]), np.radians(sites["longitude"])
    a = (np.sin((lat2 - math.radians(lat)) / 2) ** 2
         + math.cos(math.radians(lat)) * np.cos(lat2) * np.sin((lon2 - math.radians(lon)) / 2) ** 2)
    s = sites.assign(miles=3958.8 * 2 * np.arcsin(np.sqrt(a))).sort_values("miles")
    return s.iloc[0]


def fetch_live(gauge, series, cell):
    """The gauge's latest 45 days of USGS readings and its grid square's weather + forecast."""
    from src.flowcast.ingestion import usgs_backfill as usgs                  # loads .env if there is one
    ids = series.loc[series["monitoring_location_id"] == gauge["monitoring_location_id"], "time_series_id"].tolist()
    start = (date.today() - timedelta(days=P.HISTORY_DAYS)).isoformat()
    df, _ = usgs.waterdata.get_daily(time_series_id=ids, time=f"{start}/..", skip_geometry=True)
    df = usgs.clean(df).assign(date=lambda d: pd.to_datetime(d["time"]),
                               level_value=lambda d: pd.to_numeric(d["value"], errors="coerce"))
    if df["level_value"].notna().sum() == 0:
        raise ValueError("USGS returned no recent readings for this gauge")
    levels = df.groupby("date", as_index=False)["level_value"].mean()
    weather = P.fetch_weather(cell["cell_latitude"], cell["cell_longitude"])
    return levels, weather, None


def replay_sample(gauge, level, weather, cell):
    """Offline: replay the sample's day 8 days before its end, with the rain that actually fell after it."""
    day0 = level["date"].max() - pd.Timedelta(days=8)
    lv = level[(level["monitoring_location_id"] == gauge["monitoring_location_id"])
               & (level["date"] > day0 - pd.Timedelta(days=60)) & (level["date"] <= day0)][["date", "level_value"]]
    w = weather[weather["cell_id"] == cell["cell_id"]]
    return lv, w, day0


def predict(gauge, levels, weather, ruler, bundle, day0=None):
    """Both answers for one gauge, using the same input builder as the app (api/predict.py)."""
    x, day0, today = P.build_inputs(levels, weather, float(gauge["drainage_area"]), SETTINGS,
                                    ruler[ruler["monitoring_location_id"] == gauge["monitoring_location_id"]], day0)
    answers = {}
    for key, m in bundle["models"].items():
        row = pd.DataFrame([{c: x.get(c, np.nan) for c in m["features"]}])
        if row[m["features"][-1]].isna().all():                     # no rain forecast -> "same as today"
            same = (today["hw_score"] >= HIGH_PCT) if key == "high_water" else \
                   (today["dry_score"] < LOW_PCT and today["hw_score"] < HIGH_PCT)
            answers[key] = {"answer": bool(same), "method": "same as today (no forecast)"}
        else:
            prob = float(m["model"].predict_proba(row.fillna(m["fill_values"]))[:, 1][0])
            answers[key] = {"answer": prob >= m["cutoff"], "method": "boosted model"}
        answers[key]["for_date"] = (day0 + pd.Timedelta(days=m["horizon_days"])).date()
    return {"gauge": gauge["monitoring_location_id"], "gauge_name": gauge["monitoring_location_name"],
            "measure": "streamflow", "day0": day0.date(), "latest_reading": float(today["level_value"]),
            "hw_score": float(today["hw_score"]), "dry_score": float(today["dry_score"]), "answers": answers}


# ---------------------------------------------------------------- run
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--address", help=f'an address near Charlotte, Davidson or Rock Hill (default: "{DEFAULT_ADDRESS}")')
    parser.add_argument("--offline", action="store_true", help="don't use the internet: replay a recent day from the sample")
    args = parser.parse_args()
    start = time.time()

    print("1. Training and testing on the sample (3 gauges, 2007 to Oct 2026)...")
    level, weather, sites, cell_map, series = load_sample()
    table = build_table(level, SPLIT_DATE, cell_map, weather, sites, verbose=False)
    print(f"   {len(table):,} gauge-days; train 2007-2023, test 2024 to Oct 2026\n")
    results = train_and_test(table)
    for name in results["question"].unique():
        r = results[results["question"] == name].set_index("model").drop(columns=["key", "question"])
        r.columns = ["F1 (higher is better)", "caught", "right when warning"]
        print(f"   {name}\n   " + r.round(2).to_string().replace("\n", "\n   ") + "\n")

    print("2. Training the demo models on all sample years...")
    all_days = build_table(level, (level["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                           cell_map, weather, sites, verbose=False)
    bundle = train_demo_models(all_days, results)
    ruler = make_ruler(level)
    print(f"   saved to {os.path.relpath(OUTPUT, REPO)}/demo_models.joblib\n")

    print("3. Prediction for an address")
    address = args.address
    if address is None:
        address = DEFAULT_ADDRESS
        if sys.stdin.isatty():
            typed = input(f'   Address near Charlotte, Davidson or Rock Hill [Enter = "{DEFAULT_ADDRESS}"]: ').strip()
            address = typed or DEFAULT_ADDRESS

    offline = args.offline
    if not offline:
        try:
            lat, lon, matched = P.geocode(address)
            print(f"   Address found: {matched} ({lat:.4f}, {lon:.4f})")
        except Exception as e:
            print(f"   Could not reach the Census Geocoder ({type(e).__name__}); switching to offline mode.")
            offline = True
    if offline:
        lat, lon = 35.2215, -80.8378            # 600 E 4th St, Charlotte (no geocoding offline)
        print(f"   OFFLINE: using {DEFAULT_ADDRESS} ({lat}, {lon})")

    gauge = nearest(lat, lon, sites)
    if gauge["miles"] > MAX_MILES:
        print(f"   The nearest sample gauge is {gauge['miles']:.0f} miles away - this demo only covers the "
              f"Charlotte / Davidson / Rock Hill area. Try an address there.")
        return
    cell = cell_map[cell_map["monitoring_location_id"] == gauge["monitoring_location_id"]].iloc[0]

    day0 = None
    if not offline:
        try:
            levels, live_weather, day0 = fetch_live(gauge, series, cell)
        except Exception as e:
            print(f"   Could not get live data ({type(e).__name__}: {e}); switching to offline mode.")
            offline = True
    if offline:
        levels, live_weather, day0 = replay_sample(gauge, level, weather, cell)
        print(f"   OFFLINE: replaying {day0.date()} from the sample, with the rain that actually fell after it.")

    result = predict(gauge, levels, live_weather, ruler, bundle, day0)
    print("\n" + P.describe(result, bundle, gauge["miles"]).replace("in past testing", "in the sample test"))
    print(f"\nDone in {time.time() - start:.0f} s. (Demo models: 3 gauges. The full app uses 82 gauges and "
          f"9 test years - see README.md.)")


if __name__ == "__main__":
    main()
