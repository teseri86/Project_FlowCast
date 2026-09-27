"""
Train a GRADIENT-BOOSTED TREE model (scikit-learn HistGradientBoostingRegressor) to
predict each gauge's 0-100 river score HORIZON_DAYS (3) ahead.

Gradient boosting builds hundreds of small decision trees in a row; each new tree
focuses on the mistakes the trees before it made, and the prediction adds up all
their answers. (Compare: train_decision_trees.py uses a single tree per target.)

WHAT IT DOES
  1. Loads the training table (from S3 by default).
  2. BASELINE: "the river in 3 days = the river today" (no model at all).
     Our model is only worth using if it beats this.
  3. Trains a gradient-boosted tree model on 2020-2024.
  4. Grades both on 2025-present (data the model never saw):
       - average miss, in score points
       - after turning scores into the 7 levels: how often the level is right,
         and how many real drought / very-high-flow days were caught
  5. Shows which inputs the model relies on most.
  6. Predicts a few made-up "ideal data" scenarios (big storm, long dry spell,
     ordinary week) - for the demo.
  7. Saves the model and all results locally and to S3.

HOW TO RUN (from the repo root, after build_training_table)
    python -m src.flowcast.model.train_gradient_boosting            # read/write S3 (default)
    python -m src.flowcast.model.train_gradient_boosting --local    # local data/ folder only

OUTPUTS
    data/model/gradient_boosting.joblib                   the trained model (also S3: models/gradient_boosting/<today>/)
    data/model/gradient_boosting_metrics.csv              baseline vs model scores
    data/model/gradient_boosting_level_report.csv         per-level results (how often each level was caught)
    data/model/gradient_boosting_feature_importance.csv   which inputs matter most
    data/model/gradient_boosting_scenarios.csv            the "ideal data" demo predictions
  (all CSVs also uploaded to S3: processed/model/<today>/)
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from src.flowcast.model.build_training_table import FEATURE_COLUMNS, LEVELS_7, to_level
from src.flowcast.storage import read_dataset, upload_file, write_dataset, default_source

MODEL_FILE = os.path.join("data", "model", "gradient_boosting.joblib")
LEVEL_ORDER = [label for _, label in LEVELS_7]


def make_model():
    """Model settings (one place, also used by cross_validate.py)."""
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=False, random_state=42)


# ------------------------------------------------------------- grading
def grade(name, true_pct, pred_pct):
    """Score one set of predictions. All numbers are plain-language measures."""
    true_lvl = to_level(true_pct, LEVELS_7)
    pred_lvl = to_level(pred_pct, LEVELS_7)
    rank = {label: i for i, label in enumerate(LEVEL_ORDER)}
    level_gap = (pd.Series(pred_lvl).map(rank).values - pd.Series(true_lvl).map(rank).values)

    dry = true_pct < 10          # truly in a drought level
    very_high = true_pct >= 99   # truly very high flow
    return {
        "model": name,
        "avg_miss_points": np.mean(np.abs(pred_pct - true_pct)),              # MAE
        "level_exactly_right_%": np.mean(level_gap == 0) * 100,
        "level_within_one_%": np.mean(np.abs(level_gap) <= 1) * 100,
        "drought_days_caught_%": np.mean(pred_pct[dry] < 10) * 100 if dry.any() else np.nan,
        "very_high_days_caught_as_high_or_very_high_%":
            np.mean(pred_pct[very_high] >= 95) * 100 if very_high.any() else np.nan,
    }


def level_report(true_pct, pred_pct):
    """For each true level: how many days, and how often the model predicted that same level."""
    true_lvl = pd.Series(to_level(true_pct, LEVELS_7), name="true_level")
    pred_lvl = pd.Series(to_level(pred_pct, LEVELS_7), name="predicted_level")
    table = pd.DataFrame({"true_level": true_lvl, "correct": true_lvl == pred_lvl})
    report = table.groupby("true_level").agg(test_days=("correct", "size"), caught_pct=("correct", "mean"))
    report["caught_pct"] = (report["caught_pct"] * 100).round(1)
    most_common = pd.crosstab(true_lvl, pred_lvl).idxmax(axis=1).rename("most_often_predicted_as")
    return report.join(most_common).reindex(LEVEL_ORDER).reset_index()


# ------------------------------------------------------ demo scenarios
def ideal_scenarios(train):
    """Hand-made input rows describing clear situations, starting from a typical
    training day and changing only what the scenario is about."""
    typical = train[FEATURE_COLUMNS].median()

    def scenario(name, description, **changes):
        row = typical.copy()
        for k, v in changes.items():
            row[k] = v
        return name, description, row

    s = [
        scenario("Ordinary autumn week", "Normal river, light rain, mild weather.",
                 level_pct=50, level_pct_1d_ago=51, level_pct_3d_ago=52, level_pct_7d_ago=53,
                 level_pct_change_7d=-3, level_pct_mean_30d=50, precip_1d=0.0, precip_3d=0.2,
                 precip_7d=0.6, precip_14d=1.2, precip_30d=3.0, precip_next_3d=0.3, month=10),
        scenario("Big storm coming", "River already a bit high after a wet week; 4 inches of rain forecast.",
                 level_pct=70, level_pct_1d_ago=65, level_pct_3d_ago=60, level_pct_7d_ago=55,
                 level_pct_change_7d=15, level_pct_mean_30d=60, precip_1d=0.8, precip_3d=1.5,
                 precip_7d=2.5, precip_14d=3.5, precip_30d=6.0, precip_hours_3d=20,
                 precip_next_3d=4.0, month=9),
        scenario("Long hot dry spell", "Almost no rain for a month, hot, high evaporation; no rain forecast.",
                 level_pct=12, level_pct_1d_ago=13, level_pct_3d_ago=15, level_pct_7d_ago=20,
                 level_pct_change_7d=-8, level_pct_mean_30d=18, precip_1d=0.0, precip_3d=0.0,
                 precip_7d=0.0, precip_14d=0.1, precip_30d=0.4, precip_hours_3d=0,
                 et0_7d=1.4, et0_30d=5.8, water_balance_30d=-5.4, temp_max_mean_7d=94,
                 precip_next_3d=0.0, month=8),
        scenario("Drought, then a small shower", "Same dry spell, but 0.3 inches of rain forecast.",
                 level_pct=12, level_pct_1d_ago=13, level_pct_3d_ago=15, level_pct_7d_ago=20,
                 level_pct_change_7d=-8, level_pct_mean_30d=18, precip_1d=0.0, precip_3d=0.0,
                 precip_7d=0.0, precip_14d=0.1, precip_30d=0.4, et0_7d=1.4, et0_30d=5.8,
                 water_balance_30d=-5.4, temp_max_mean_7d=94, precip_next_3d=0.3, month=8),
    ]
    names = [x[0] for x in s]
    descriptions = [x[1] for x in s]
    rows = pd.DataFrame([x[2] for x in s])[FEATURE_COLUMNS]
    return names, descriptions, rows


# ----------------------------------------------------------------- main
def main():
    print("Loading the training table...")
    t = read_dataset("training_table", dtype={"monitoring_location_id": str})
    train, test = t[t["split"] == "train"], t[t["split"] == "test"]
    X_train, y_train = train[FEATURE_COLUMNS], train["target_pct"].to_numpy()
    X_test, y_test = test[FEATURE_COLUMNS], test["target_pct"].to_numpy()
    print(f"  Train: {len(train):,} rows ({train['date'].min()} to {train['date'].max()})")
    print(f"  Test:  {len(test):,} rows ({test['date'].min()} to {test['date'].max()})")

    # Baseline: tomorrow looks like today.
    baseline_pred = test["level_pct"].to_numpy()

    print("\nTraining the gradient-boosted tree model (400 trees)...")
    model = make_model()
    model.fit(X_train, y_train)          # handles missing values on its own
    model_pred = np.clip(model.predict(X_test), 0, 100)

    metrics = pd.DataFrame([grade("baseline: same as today", y_test, baseline_pred),
                            grade("gradient boosting", y_test, model_pred)]).set_index("model")
    print("\nRESULTS on the test period (data the model never saw):")
    print(metrics.round(1).T.to_string())
    better = metrics.loc["baseline: same as today", "avg_miss_points"] - metrics.loc["gradient boosting", "avg_miss_points"]
    print(f"\n  The model misses by {better:+.1f} points less than the baseline on average "
          f"({'better' if better > 0 else 'NOT better'} than just using today's score).")

    report = level_report(y_test, model_pred)
    print("\nBy level (test period): how often the model predicted the right level")
    print(report.to_string(index=False))

    print("\nWhich inputs matter most (how much worse the model gets when each is scrambled)...")
    sample = test.sample(min(len(test), 20000), random_state=0)
    imp = permutation_importance(model, sample[FEATURE_COLUMNS], sample["target_pct"],
                                 scoring="neg_mean_absolute_error", n_repeats=3, random_state=0)
    importance = (pd.DataFrame({"feature": FEATURE_COLUMNS, "extra_miss_points": imp.importances_mean})
                  .sort_values("extra_miss_points", ascending=False))
    print(importance.head(10).round(2).to_string(index=False))

    names, descriptions, rows = ideal_scenarios(train)
    scen_pred = np.clip(model.predict(rows), 0, 100)
    scenarios = pd.DataFrame({
        "scenario": names, "description": descriptions,
        "river_score_today": rows["level_pct"].values,
        "rain_next_3_days_in": rows["precip_next_3d"].values,
        "predicted_score_in_3_days": scen_pred.round(1),
        "predicted_level": to_level(pd.Series(scen_pred), LEVELS_7),
    })
    print("\n'Ideal data' scenarios (made-up inputs, for the demo):")
    print(scenarios.drop(columns="description").to_string(index=False))

    print("\nSaving...")
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "levels": LEVELS_7}, MODEL_FILE)
    print(f"  saved {MODEL_FILE}")
    if default_source() != "local":
        upload_file(MODEL_FILE, "models/gradient_boosting")
    write_dataset("model_metrics", metrics.reset_index())
    write_dataset("level_report", report)
    write_dataset("feature_importance", importance)
    write_dataset("scenario_predictions", scenarios)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="read/write the local data/ folder only (no S3)")
    if parser.parse_args().local:
        os.environ["FLOWCAST_DATA_SOURCE"] = "local"
    main()