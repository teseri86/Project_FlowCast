"""
Compare every model fairly across SEVERAL test years (time-series cross-validation).

Instead of one train/test split, the data is split several times, always training
on the past and testing on the year that follows:

    Round 1: train 2020-2021  -> test 2022
    Round 2: train 2020-2022  -> test 2023
    Round 3: train 2020-2023  -> test 2024
    Round 4: train 2020-2024  -> test 2025-present   (the same split the training scripts use)

For EACH round the 0-100 scores are rebuilt from that round's training years only,
so no test year ever shapes the scale. The last HORIZON_DAYS training days are
dropped (the "gap"), since their answers fall in the test year.

Every model is graded with the same measures (see evaluation.py):
    baseline (same as today) | decision trees: 3-level, 7-level, 0-100 score | gradient boosting

HOW TO RUN (from the repo root; reads the raw data from S3 by default)
    python -m src.flowcast.model.cross_validate
    python -m src.flowcast.model.cross_validate --local

Takes a few minutes (4 rounds x 5 models).

OUTPUTS (also uploaded to S3: processed/model/<today>/)
    data/model/cv_results.csv   every model, every round
    data/model/cv_summary.csv   each model averaged over the rounds, with best and worst round
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.flowcast.model.build_training_table import (FEATURE_COLUMNS, build_table, load_river,
                                                     load_sites, load_weather)
from src.flowcast.model.evaluation import MEASURE_ORDER, grade
from src.flowcast.model import train_decision_trees as dt
from src.flowcast.model import train_gradient_boosting as gb
from src.flowcast.storage import write_dataset

# (test starts, test ends) - the test ends the day before the end date; None = end of data
ROUNDS = [
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", None),
]


def run_models(train, test):
    """Train each model on `train`, grade it on `test`. Returns a list of result dicts."""
    X_train, X_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
    y_true = test["target_pct"].to_numpy()
    results = []

    def add(name, kind, predicted, seconds):
        row = {"model": name, **grade(y_true, predicted, kind), "train_seconds": round(seconds, 1)}
        results.append(row)
        print(f"    {name:<32} drought caught {row['drought_caught_%']:5.1f}%   "
              f"high water caught {row['high_water_caught_%']:5.1f}%   "
              f"balanced accuracy {row['balanced_accuracy_%']:5.1f}%")

    # Baseline: the river in 3 days = the river today.
    add("baseline (same as today)", "score", test["level_pct"].to_numpy(), 0)

    # Decision trees need gaps filled (training-period medians only).
    medians = X_train.median()
    Xf_train, Xf_test = X_train.fillna(medians), X_test.fillna(medians)
    for name, make, target, kind in [
        ("decision tree, 3-level", dt.make_classifier_3level, "target_level3", "level3"),
        ("decision tree, 7-level", dt.make_classifier_7level, "target_level7", "level7"),
        ("decision tree, 0-100 score", dt.make_regressor, "target_pct", "score"),
    ]:
        start = time.time()
        model = make().fit(Xf_train, train[target])
        add(name, kind, model.predict(Xf_test), time.time() - start)

    # Gradient boosting handles gaps on its own.
    start = time.time()
    model = gb.make_model().fit(X_train, train["target_pct"])
    add("gradient boosting, 0-100 score", "score", model.predict(X_test), time.time() - start)
    return results


def main():
    print("Loading raw data (once for all rounds)...")
    river = load_river()
    cell_map, weather = load_weather()
    sites = load_sites()

    all_results = []
    for n, (test_start, test_end) in enumerate(ROUNDS, start=1):
        label = f"{test_start[:4]}" if test_end else f"{test_start[:4]}-present"
        print(f"\nRound {n}: train before {test_start}, test {label}")
        table = build_table(river, test_start, cell_map, weather, sites, verbose=False)
        train = table[table["split"] == "train"]
        test = table[table["split"] == "test"]
        if test_end:
            test = test[test["date"] < test_end]
        if train.empty or test.empty:
            print("  (skipped - not enough data)")
            continue

        # Context: how dry or wet was this test year? (By design ~10% each in training.)
        dry = (test["target_pct"] < 10).mean() * 100
        wet = (test["target_pct"] >= 90).mean() * 100
        print(f"  train {len(train):,} rows | test {len(test):,} rows | "
              f"test year: {dry:.0f}% drought days, {wet:.0f}% high-water days")

        for row in run_models(train, test):
            all_results.append({"round": n, "test_period": label, "train_rows": len(train),
                                "test_rows": len(test), "test_drought_days_%": round(dry, 1),
                                "test_high_water_days_%": round(wet, 1), **row})

    results = pd.DataFrame(all_results)
    model_order = list(dict.fromkeys(results["model"]))

    # ---- readable tables: one per key measure, models x test years
    for measure, title in [
        ("drought_caught_%", "Drought days caught (%)"),
        ("drought_right_%", "When it said drought, how often right (%)"),
        ("high_water_caught_%", "High-water days caught (%)"),
        ("high_water_right_%", "When it said high water, how often right (%)"),
        ("balanced_accuracy_%", "Balanced accuracy (%) - the fairest single number"),
        ("avg_miss_points", "Average miss, 0-100 points (score models only)"),
    ]:
        table = results.pivot(index="model", columns="test_period", values=measure).reindex(model_order)
        table["AVERAGE"] = table.mean(axis=1)
        print(f"\n{title}")
        print(table.round(1).to_string())

    # ---- summary: average over rounds, plus best and worst round
    summary = results.groupby("model")[MEASURE_ORDER].mean().reindex(model_order)
    for measure in ["drought_caught_%", "high_water_caught_%", "balanced_accuracy_%"]:
        stats = results.groupby("model")[measure].agg(["min", "max"]).reindex(model_order)
        summary[f"{measure}_worst_round"] = stats["min"]
        summary[f"{measure}_best_round"] = stats["max"]
    summary = summary.round(1).reset_index()

    best = summary.sort_values("balanced_accuracy_%", ascending=False).iloc[0]
    print(f"\nHighest average balanced accuracy: {best['model']} ({best['balanced_accuracy_%']:.1f}%)")

    print("\nSaving...")
    write_dataset("cv_results", results.round(2))
    write_dataset("cv_summary", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="read/write the local data/ folder only (no S3)")
    if parser.parse_args().local:
        os.environ["FLOWCAST_DATA_SOURCE"] = "local"
    main()