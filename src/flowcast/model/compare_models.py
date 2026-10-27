"""
Compare the three models fairly across nine test years (2017 to 2025-present), and pick a winner for each target
(the plan in notebooks/eda_raw_data.ipynb, section 9).

WHAT IS COMPARED
    Targets (yes/no):
        high water 1 day ahead, WITHOUT future rain   (worst case: no forecast)
        high water 1 day ahead, WITH future rain      (best case: perfect forecast)
        drought    7 days ahead, WITHOUT future rain  (worst case: no forecast)
        drought    7 days ahead, WITH next-7-day rain (best case: perfect forecast)
    Models:
        baseline      "same as today": predicts yes if today is already high water / drought
        tree          one decision tree (easy to explain)
        boosted       many small trees added together (scikit-learn HistGradientBoostingClassifier)

THE ROUNDS (always train on the past, test on the year after)
    Round 1: train 2007-2016 -> test 2017
    Round 2: train 2007-2017 -> test 2018
    ...
    Round 9: train 2007-2024 -> test 2025-present
    Needs the 2007-2019 river and weather history. The test years cover dry years (2017),
    very wet years (2018-2020) and the 2025-2026 drought.
    Gauges need 10 years on the ruler BEFORE each test year, so early rounds use fewer gauges
    (gauges with shorter records join in later rounds).
    In each round the scores are rebuilt using only years before that round's test year
    (minus each day's own year), and training rows whose answer falls in the test year are
    dropped (the gap).

THE YES/NO CUT-OFF (tree and boosted)
    A model gives a probability; it says "yes" when the probability is at least the cut-off.
    In each round the cut-off is chosen on the TRAINING years only: hold back the last training
    year, train on the earlier years, pick the cut-off with the best F1 on the held-back year,
    then retrain on all training years and use that cut-off on the test year. The chosen cut-off
    is saved in the `cutoff` column of compare_results.csv.

HOW THE WINNER IS PICKED (decided before seeing results)
    Main metric: F1 for the "yes" class (balances "caught" and "right when flagged").
    A model qualifies only if it beats the baseline's F1 in EVERY round.
    Between tree and boosted: keep the tree unless boosted's average F1 is higher by >= 0.05.

HOW TO RUN (from the repo root)
    python -m src.flowcast.model.compare_models            # read from S3 (default)
    python -m src.flowcast.model.compare_models --local    # read from the local data/ folder
    Takes about 25 minutes (9 rounds x 4 targets; each model is trained twice per round for the cut-off).

OUTPUTS (also uploaded to S3: processed/model/<today>/)
    data/model/compare_results.csv   every model, target and round
    data/model/compare_summary.csv   averaged over rounds, with the worst round and the verdict
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from src.flowcast.model.build_training_data import (build_table, features_for, load_inputs,
                                                    load_level, train_test_masks)
from src.flowcast.storage import write_dataset

# (test starts, test ends) - test runs up to the day before the end; None = end of data
# One round per test year, 2017 through 2025-present. 2017 is the first year whose ruler
# (2007 up to the year before) has the 10 years build_training_data.py requires.
ROUNDS = [(f"{y}-01-01", f"{y + 1}-01-01") for y in range(2017, 2025)] + [("2025-01-01", None)]

# (name shown in results, target column, use future rain?, today's label for the baseline)
EXPERIMENTS = [
    ("high water 1d, no forecast",      "target_high_1d",    False, "is_high"),
    ("high water 1d, perfect forecast", "target_high_1d",    True,  "is_high"),
    ("drought 7d, no forecast",         "target_drought_7d", False, "is_drought"),
    ("drought 7d, perfect forecast",    "target_drought_7d", True,  "is_drought"),
]

BOOSTING_MARGIN = 0.05   # boosted must beat the tree's average F1 by at least this much to win


def make_tree():
    # Same settings as the old 3-level tree (they did best in the old cross-validation).
    return DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42)


def make_boosted():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=50, l2_regularization=1.0,
                                          class_weight="balanced", random_state=42)


def grade(y_true, y_pred):
    """The three numbers we report for the 'yes' class."""
    return {"f1": f1_score(y_true, y_pred, zero_division=0),
            "caught (recall)": recall_score(y_true, y_pred, zero_division=0),
            "right when flagged (precision)": precision_score(y_true, y_pred, zero_division=0)}


def run_round(table, split_date, test_end):
    """Train and grade every model for every experiment in one round."""
    rows = []
    dates = pd.to_datetime(table["date"])
    for name, target, future_rain, today_label in EXPERIMENTS:
        train, test = train_test_masks(table, target, split_date)
        if test_end is not None:
            test &= dates < pd.Timestamp(test_end)
        cols = features_for(target, with_future_rain=future_rain)

        # Fill the few blanks with TRAINING medians (never test values), same for both models.
        fill = table.loc[train, cols].median()
        X_train = table.loc[train, cols].fillna(fill)
        X_test = table.loc[test, cols].fillna(fill)
        y_train = table.loc[train, target].astype(int)
        y_test = table.loc[test, target].astype(int)

        base = {"experiment": name, "test_year": split_date[:4] if test_end else f"{split_date[:4]}+",
                "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                "test_yes_rate": round(y_test.mean() * 100, 1)}

        baseline_pred = table.loc[test, today_label].fillna(0).astype(int)
        rows.append({**base, "model": "baseline", **grade(y_test, baseline_pred), "seconds": 0.0})

        # For the cut-off: hold back the LAST training year (e.g. 2024 when testing 2025+).
        horizon = int(target.rsplit("_", 1)[1].rstrip("d"))
        holdout_start = pd.Timestamp(split_date) - pd.DateOffset(years=1)
        fit_part = train & (dates + pd.Timedelta(days=horizon) < holdout_start)
        holdout = train & (dates >= holdout_start)

        for model_name, make in [("tree", make_tree), ("boosted", make_boosted)]:
            start = time.time()
            cutoff = pick_cutoff(make, table, cols, target, fit_part, holdout, fill)
            model = make().fit(X_train, y_train)
            pred = (model.predict_proba(X_test)[:, 1] >= cutoff).astype(int)
            rows.append({**base, "model": model_name, **grade(y_test, pred), "cutoff": cutoff,
                         "seconds": round(time.time() - start, 1)})
    return rows


def pick_cutoff(make, table, cols, target, fit_part, holdout, fill):
    """The yes/no cut-off ("say yes when the model is at least this sure") with the best F1 on the
    held-back last training year. Uses training years only - the test year never sees this choice."""
    model = make().fit(table.loc[fit_part, cols].fillna(fill), table.loc[fit_part, target].astype(int))
    prob = model.predict_proba(table.loc[holdout, cols].fillna(fill))[:, 1]
    truth = table.loc[holdout, target].astype(int)
    candidates = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [f1_score(truth, (prob >= c).astype(int), zero_division=0) for c in candidates]
    return float(candidates[int(np.argmax(scores))])


def summarize(results):
    """Average over rounds, worst round, 'beats baseline every round', and the winner per experiment."""
    summary_rows = []
    for name in results["experiment"].unique():
        r = results[results["experiment"] == name]
        f1 = r.pivot(index="test_year", columns="model", values="f1")
        stats = {}
        for model in ["baseline", "tree", "boosted"]:
            mr = r[r["model"] == model]
            stats[model] = {
                "experiment": name, "model": model,
                "f1 (avg)": mr["f1"].mean(), "f1 (worst round)": mr["f1"].min(),
                "caught (avg)": mr["caught (recall)"].mean(),
                "right when flagged (avg)": mr["right when flagged (precision)"].mean(),
                "beats baseline every round": (bool((f1[model] > f1["baseline"]).all())
                                               if model != "baseline" else None),
            }

        tree_ok = stats["tree"]["beats baseline every round"]
        boost_ok = stats["boosted"]["beats baseline every round"]
        if tree_ok and boost_ok:
            winner = ("boosted" if stats["boosted"]["f1 (avg)"] - stats["tree"]["f1 (avg)"] >= BOOSTING_MARGIN
                      else "tree")
        elif tree_ok:
            winner = "tree"
        elif boost_ok:
            winner = "boosted"
        else:
            winner = "none (no model beats the baseline every round)"
        for model in stats:
            stats[model]["winner"] = winner if model == "baseline" else ""
            summary_rows.append(stats[model])
    return pd.DataFrame(summary_rows)


def main():
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) < (1, 2):
        raise SystemExit(f"scikit-learn {sklearn.__version__} is too old (need 1.2+ for class_weight in "
                         f"boosting). Run: python -m pip install -U scikit-learn")

    print("Loading data once...")
    level = load_level()
    cell_map, weather, sites = load_inputs()

    all_rows = []
    for i, (split_date, test_end) in enumerate(ROUNDS, 1):
        label = f"{split_date[:4]}" if test_end else f"{split_date[:4]}-present"
        table = build_table(level, split_date, cell_map, weather, sites, verbose=False)
        print(f"\nRound {i}/{len(ROUNDS)}: train {table['date'].min()[:4]}-{int(split_date[:4]) - 1} -> test {label}")
        start = time.time()
        all_rows += run_round(table, split_date, test_end)
        print(f"  done in {time.time() - start:.0f}s")

    results = pd.DataFrame(all_rows).round(3)
    summary = summarize(results).round(3)

    print("\nSaving...")
    write_dataset("compare_results", results)
    write_dataset("compare_summary", summary)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n================ F1 by round (higher is better; 'yes %' = share of test days that were yes)")
    for name in results["experiment"].unique():
        r = results[results["experiment"] == name]
        table = r.pivot(index="test_year", columns="model", values="f1")[["baseline", "tree", "boosted"]]
        table["yes %"] = r.groupby("test_year")["test_yes_rate"].first()
        print(f"\n{name}\n{table.to_string()}")

    print("\n================ Summary and verdict")
    for name in summary["experiment"].unique():
        s = summary[summary["experiment"] == name].set_index("model")
        print(f"\n{name}")
        print(s.drop(columns=["experiment", "winner"]).to_string())
        print(f"  -> WINNER: {s.loc['baseline', 'winner']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="read/write the local data/ folder only (no S3)")
    if parser.parse_args().local:
        os.environ["FLOWCAST_DATA_SOURCE"] = "local"
    main()
