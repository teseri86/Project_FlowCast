"""
Train and save the two FINAL models the app uses (chosen by compare_models.py, 9-year test):

    high water, 1 day ahead  -> boosted trees,  with the rain forecast for the next day
    drought,    7 days ahead -> boosted trees,  with the rain forecast for the next 7 days

Both are trained on EVERY available day (2007 to the latest data). The 9-year test in
compare_models.py already measured how well this setup does on unseen years; for the deployed
models we use all the data, including the 2025-2026 drought.

HOW TO RUN (from the repo root, after build_training_data's inputs are in S3)
    python -m src.flowcast.model.train_final_models            # read from S3, save + upload
    python -m src.flowcast.model.train_final_models --local    # local data/ folder only

    Takes a few minutes.

OUTPUTS (also uploaded to S3: models/final/<today>/)
    data/model/final_models.joblib    both models + everything needed to use them:
                                      input columns, fill values, yes/no cut-off, horizon
    data/model/final_models_info.json the same facts in readable form, plus the measured accuracy
                                      from compare_models (to show users)
    data/model/ruler.csv              each gauge's river history that defines "normal". The app
                                      scores today's reading against it exactly like training did.

USING THE MODELS IN THE APP (for the prediction API, later)
    1. Get the gauge's latest COMPLETE days of readings and the grid square's weather + forecast.
    2. Score the latest day against ruler.csv (hw_score: daily level vs all ruler days;
       dry_score: 7-day average vs ruler days within +/-15 calendar days).
    3. Build the same input columns, fill blanks with the saved fill values,
       probability = model.predict_proba(X)[:, 1], "yes" if probability >= cutoff.
    TIMING: USGS daily values are published after the day ends, so the latest complete day is
    usually yesterday. A model run each morning on yesterday's data predicts horizon_days after
    YESTERDAY (high water: today; drought: 6 days from today). Label answers by the date predicted.
"""

import argparse
import json
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd

from src.flowcast.model import compare_models as cm
from src.flowcast.model.build_training_data import (FEATURE_COLUMNS, LOW_PCT, HIGH_PCT, SMOOTH_DAYS,
                                                    WINDOW_DAYS, build_table, features_for, load_inputs,
                                                    load_level)
from src.flowcast.storage import read_dataset, upload_file, write_dataset

MODEL_DIR = os.path.join("data", "model")
MODEL_FILE = os.path.join(MODEL_DIR, "final_models.joblib")
INFO_FILE = os.path.join(MODEL_DIR, "final_models_info.json")
S3_FOLDER = "models/final"

# The two final models (decided by the 9-year test and the winner rule).
FINAL = {
    "high_water": {"target": "target_high_1d", "model_type": "boosted", "make": cm.make_boosted,
                   "experiment": "high water 1d, perfect forecast",
                   "question": "Will this river be in high water 1 day after the latest complete day?"},
    "drought": {"target": "target_drought_7d", "model_type": "boosted", "make": cm.make_boosted,
                "experiment": "drought 7d, perfect forecast",
                "question": "Will this river be in drought 7 days after the latest complete day?"},
}


def measured_accuracy(experiment, model_type):
    """The 9-year test results from compare_models.py, if they've been saved."""
    try:
        s = read_dataset("compare_summary")
    except FileNotFoundError:
        return None
    row = s[(s["experiment"] == experiment) & (s["model"] == model_type)]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"f1_avg": float(r["f1 (avg)"]), "caught_avg": float(r["caught (avg)"]),
            "right_when_flagged_avg": float(r["right when flagged (avg)"]),
            "note": "Average over 9 test years (2017-2025+) using ACTUAL future rain (a perfect forecast). "
                    "With a real forecast, accuracy will be lower."}


def main():
    print("Loading data...")
    level = load_level()
    cell_map, weather, sites = load_inputs()

    # A split date after the last day makes every row a training row, and puts every year
    # (2007 to now) on the ruler. Each day is still scored against the OTHER years.
    final_split = (level["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Building training data on all days (ruler: every year before {final_split})...")
    table = build_table(level, final_split, cell_map, weather, sites)
    dates = pd.to_datetime(table["date"])
    last_day = dates.max()

    bundle = {"trained_on": {"first_day": dates.min().strftime("%Y-%m-%d"), "last_day": last_day.strftime("%Y-%m-%d"),
                             "gauges": sorted(table["monitoring_location_id"].unique())},
              "scoring": {"high_pct": HIGH_PCT, "low_pct": LOW_PCT, "window_days": WINDOW_DAYS,
                          "smooth_days": SMOOTH_DAYS, "ruler_file": "ruler.csv"},
              "models": {}}
    info = {k: v for k, v in bundle.items() if k != "models"} | {"models": {}}
    info["trained_on"] = {**info["trained_on"], "gauges": len(info["trained_on"]["gauges"])}

    for name, spec in FINAL.items():
        target = spec["target"]
        horizon = int(target.rsplit("_", 1)[1].rstrip("d"))
        cols = features_for(target, with_future_rain=True)
        has_answer = table[target].notna()
        fill = table.loc[has_answer, cols].median()

        # Cut-off: hold back the last 12 months, train on the earlier days, pick the best-F1 cut-off.
        holdout_start = last_day - pd.DateOffset(years=1)
        fit_part = has_answer & (dates + pd.Timedelta(days=horizon) < holdout_start)
        holdout = has_answer & (dates >= holdout_start)
        cutoff = cm.pick_cutoff(spec["make"], table, cols, target, fit_part, holdout, fill)

        print(f"\nTraining {name} ({spec['model_type']}) on {int(has_answer.sum()):,} days; cut-off {cutoff:.2f}")
        model = spec["make"]().fit(table.loc[has_answer, cols].fillna(fill), table.loc[has_answer, target].astype(int))

        bundle["models"][name] = {"model": model, "model_type": spec["model_type"], "target": target,
                                  "horizon_days": horizon, "features": cols, "fill_values": fill.to_dict(),
                                  "cutoff": cutoff, "question": spec["question"]}
        accuracy = measured_accuracy(spec["experiment"], spec["model_type"])
        bundle["models"][name]["measured_accuracy"] = accuracy
        info["models"][name] = {"model_type": spec["model_type"], "target": target, "horizon_days": horizon,
                                "features": cols, "cutoff": cutoff, "question": spec["question"],
                                "training_days": int(has_answer.sum()),
                                "yes_share_in_training": round(float(table.loc[has_answer, target].mean()), 4),
                                "measured_accuracy": accuracy}

    # The ruler: each kept gauge's river history up to the latest day.
    kept = set(bundle["trained_on"]["gauges"])
    ruler = level[level["monitoring_location_id"].isin(kept)].sort_values(["monitoring_location_id", "date"]).copy()
    ruler["level_7d"] = (ruler.groupby("monitoring_location_id")["level_value"]
                         .transform(lambda s: s.rolling(SMOOTH_DAYS, min_periods=5).mean())).round(6)
    ruler = ruler.dropna(subset=["level_value", "level_7d"], how="all")   # keep days that count for either score
    ruler["date"] = ruler["date"].dt.strftime("%Y-%m-%d")

    print("\nSaving...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    with open(INFO_FILE, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  saved {MODEL_FILE}\n  saved {INFO_FILE}")
    write_dataset("ruler", ruler[["monitoring_location_id", "measure", "date", "level_value", "level_7d"]])
    if os.environ.get("FLOWCAST_DATA_SOURCE", "s3").lower() != "local":
        upload_file(MODEL_FILE, S3_FOLDER)
        upload_file(INFO_FILE, S3_FOLDER)

    print(f"\nDone. Trained on {info['trained_on']['first_day']} to {info['trained_on']['last_day']}, "
          f"{info['trained_on']['gauges']} gauges.")
    for name, m in info["models"].items():
        acc = m["measured_accuracy"]
        acc_text = (f"9-year test (perfect forecast): catches {acc['caught_avg']:.0%}, "
                    f"right when flagged {acc['right_when_flagged_avg']:.0%}") if acc else "run compare_models for accuracy"
        print(f"  {name:10s} {m['model_type']:8s} cut-off {m['cutoff']:.2f}  |  {acc_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="read/write the local data/ folder only (no S3)")
    if parser.parse_args().local:
        os.environ["FLOWCAST_DATA_SOURCE"] = "local"
    main()
