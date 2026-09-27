"""
Train three single decision trees, one per target in the training table:
  - 3-level classifier  (drought / normal / high_water)
  - 7-level classifier  (severe drought ... very high flow)
  - regressor           (the 0-100 river score, 3 days ahead)

Single trees are simple and easy to explain (each one is a set of if-then
questions), which makes them a good first model to compare against.

HOW TO RUN (from the repo root, after build_training_table)
    python -m src.flowcast.model.train_decision_trees            # read/write S3 (default)
    python -m src.flowcast.model.train_decision_trees --local    # local data/ folder only

OUTPUT
    data/model/decision_trees.joblib   (also uploaded to S3: models/decision_trees/<today>/)
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.flowcast.model.build_training_table import FEATURE_COLUMNS
from src.flowcast.storage import default_source, read_dataset, upload_file

MODEL_FILE = os.path.join("data", "model", "decision_trees.joblib")


# ---- model settings (one place, also used by cross_validate.py)
def make_classifier_3level():
    # max_depth controls overfitting (essential for Decision Trees)
    # class_weight='balanced' ensures rare events (high water / drought) are not ignored
    return DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42)


def make_classifier_7level():
    return DecisionTreeClassifier(max_depth=8, min_samples_leaf=15, class_weight="balanced", random_state=42)


def make_regressor():
    return DecisionTreeRegressor(max_depth=7, min_samples_leaf=25, random_state=42)


def load_and_split_data():
    """Loads dataset and splits strictly along the time boundary set by build_training_table.py."""
    print("Loading training table...")
    df = read_dataset("training_table", dtype={"monitoring_location_id": str})

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Fill gaps with the TRAINING medians only, so nothing from the test period leaks in.
    train_medians = train_df[FEATURE_COLUMNS].median()
    train_df[FEATURE_COLUMNS] = train_df[FEATURE_COLUMNS].fillna(train_medians)
    test_df[FEATURE_COLUMNS] = test_df[FEATURE_COLUMNS].fillna(train_medians)

    print(f"Dataset split: {len(train_df):,} training rows | {len(test_df):,} testing rows")
    return train_df, test_df, train_medians


def train_3level_classifier(train_df, test_df):
    """Trains a Decision Tree for 3-class risk level (drought / normal / high_water)."""
    print("\n" + "=" * 50)
    print("Training Decision Tree Classifier: 3-Level Risk Target")
    print("=" * 50)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_level3"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target_level3"]

    clf = make_classifier_3level()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, zero_division=0))

    return clf


def train_7level_classifier(train_df, test_df):
    """Trains a Decision Tree for 7-class detailed risk level."""
    print("\n" + "=" * 50)
    print("Training Decision Tree Classifier: 7-Level Risk Target")
    print("=" * 50)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_level7"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target_level7"]

    clf = make_classifier_7level()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, zero_division=0))

    return clf


def train_percentile_regressor(train_df, test_df):
    """Trains a Decision Tree Regressor to predict exact 0-100 target_pct."""
    print("\n" + "=" * 50)
    print("Training Decision Tree Regressor: target_pct (0-100 Continuous Score)")
    print("=" * 50)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_pct"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target_pct"]

    reg = make_regressor()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))   # works on every scikit-learn version
    r2 = r2_score(y_test, y_pred)

    print(f"Test Root Mean Squared Error (RMSE): {rmse:.2f} percentage points")
    print(f"Test R² Score: {r2:.3f}")

    return reg


def inspect_feature_importance(name, model, feature_names):
    """Prints top features driving the Decision Tree decisions."""
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"\nTop 10 Feature Importances ({name}):")
    print(((importance.head(10) * 100).round(2).astype(str) + "%").to_string())


def main():
    train_df, test_df, train_medians = load_and_split_data()

    # 1. Train models
    clf_3 = train_3level_classifier(train_df, test_df)
    clf_7 = train_7level_classifier(train_df, test_df)
    reg_pct = train_percentile_regressor(train_df, test_df)

    # 2. Inspect key drivers
    inspect_feature_importance("3-level classifier", clf_3, FEATURE_COLUMNS)
    inspect_feature_importance("0-100 regressor", reg_pct, FEATURE_COLUMNS)

    # 3. Save all three (plus the gap-filling medians, needed to predict later)
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump({"classifier_3level": clf_3, "classifier_7level": clf_7, "regressor_pct": reg_pct,
                 "features": FEATURE_COLUMNS, "fill_values": train_medians}, MODEL_FILE)
    print(f"\nSaved models to {MODEL_FILE}")
    if default_source() != "local":
        upload_file(MODEL_FILE, "models/decision_trees")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="read/write the local data/ folder only (no S3)")
    if parser.parse_args().local:
        os.environ["FLOWCAST_DATA_SOURCE"] = "local"
    main()