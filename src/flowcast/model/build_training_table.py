"""
Build the model's training table: one row per USGS site per day, with
  - FEATURES: what we know on that day (river level, recent weather, season)
  - TARGETS:  how high the river is HORIZON_DAYS later (what the model learns to predict)

HOW TO RUN (from the repo root, after the ingestion data is in S3)
    python -m src.flowcast.model.build_training_table            # read from S3 (default)
    python -m src.flowcast.model.build_training_table --local    # read from the local data/ folder

INPUTS (newest copies in S3, via src/flowcast/storage.py)
    usgs_daily, usgs_active_sites, weather_cell_map, weather_daily

OUTPUT
    data/training_table.csv, also uploaded to S3: processed/training_table/<today>/

KEY IDEAS (plain language)
  1. River level as a PERCENTILE (0-100), per site.
     A big river and a small creek can't be compared in raw numbers (5,000 vs 20
     cubic feet per second). So each day's level is turned into "how high is this
     compared to this site's own history": 50 = a typical day, 97 = higher than
     97% of days, 3 = lower than 97% of days. Now all sites share one scale and
     one model can learn from all of them.
     The percentiles are measured against the TRAINING years only, so the test
     years never leak into how the scale is defined.

  2. Train/test split BY TIME, not at random.
     Neighboring days are almost identical, so a random split would let the model
     "peek" at the day before and after each test day. Instead: train on
     2020-2024, test on 2025-present, the way it will actually be used (learn
     from the past, predict the future).
     GAP: the last HORIZON_DAYS training days are dropped, because their answers
     (the river HORIZON_DAYS later) fall inside the test period.
     cross_validate.py reuses build_table() with several split dates.

  3. Three target columns, so the team can pick the model type later:
       target_pct      0-100 score, HORIZON_DAYS ahead          (regression)
       target_level7   7 levels, severe drought ... very high flow (classifier)
       target_level3   drought / normal / high water             (classifier)
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.flowcast.storage import read_dataset, write_dataset

# ---------------------------------------------------------------- settings
HORIZON_DAYS = 3             # predict the river this many days ahead
SPLIT_DATE = "2025-01-01"    # train on days before this, test on this day and after
MIN_TRAIN_DAYS = 365         # a site needs at least a year of training history

# 7 levels: (upper percentile limit, label). Same cut-offs the app will use.
# Drought names were checked against the U.S. Drought Monitor and hold up.
# The high side is named for FLOW, not "flood": checked against NWS flood stages,
# most top-1% days never reached official flood stage. "Flood" is decided
# separately, by comparing to each gauge's official NWS flood stage.
LEVELS_7 = [
    (2, "severe_drought"), (5, "moderate_drought"), (10, "mild_drought"),
    (90, "normal"),
    (95, "above_normal_flow"), (99, "high_flow"), (100.01, "very_high_flow"),
]
LEVELS_3 = [(10, "drought"), (90, "normal"), (100.01, "high_water")]


def to_level(pct, levels):
    """Turn a 0-100 score into a label using the cut-off table."""
    edges = [0] + [upper for upper, _ in levels]
    labels = [label for _, label in levels]
    return pd.cut(pct, bins=edges, labels=labels, right=False, include_lowest=True).astype("object")


# ------------------------------------------------ Step 1: river data, wide
def load_river():
    """One row per site per day, with streamflow and gauge height side by side,
    on a complete calendar (missing days become empty rows, so "7 days ago"
    always means 7 calendar days, not 7 rows)."""
    usgs = read_dataset("usgs_daily", dtype={"monitoring_location_id": str, "parameter_code": str})
    usgs["date"] = pd.to_datetime(usgs["time"])
    wide = usgs.pivot_table(index=["monitoring_location_id", "date"], columns="parameter_code",
                            values="value", aggfunc="mean")
    wide = wide.rename(columns={"00060": "streamflow_cfs", "00065": "gauge_height_ft"}).reset_index()
    for col in ["streamflow_cfs", "gauge_height_ft"]:
        if col not in wide.columns:
            wide[col] = np.nan

    full_dates = pd.date_range(wide["date"].min(), wide["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([wide["monitoring_location_id"].unique(), full_dates],
                                      names=["monitoring_location_id", "date"])
    wide = wide.set_index(["monitoring_location_id", "date"]).reindex(grid).reset_index()

    # Which measurement drives the score: streamflow when the site has it
    # (the more standard flood/drought measure), otherwise gauge height.
    has_flow = wide.groupby("monitoring_location_id")["streamflow_cfs"].transform(lambda s: s.notna().any())
    wide["level_source"] = np.where(has_flow, "streamflow", "gauge_height")
    wide["level_value"] = np.where(has_flow, wide["streamflow_cfs"], wide["gauge_height_ft"])
    return wide


# ------------------------------------------ Step 2: level -> percentile
def add_percentiles(df, split_date=SPLIT_DATE, verbose=True):
    """Score each day 0-100 against that site's TRAINING-period history
    (days before split_date only, so test years never shape the scale)."""
    split = pd.Timestamp(split_date)
    pct = pd.Series(np.nan, index=df.index)
    kept_sites = []
    for site, rows in df.groupby("monitoring_location_id"):
        history = np.sort(rows.loc[rows["date"] < split, "level_value"].dropna().to_numpy())
        if len(history) < MIN_TRAIN_DAYS:
            continue
        values = rows["level_value"].to_numpy()
        # Share of history below the value (ties count half), as 0-100.
        below = np.searchsorted(history, values, side="left")
        at_or_below = np.searchsorted(history, values, side="right")
        pct.loc[rows.index] = (below + at_or_below) / 2 / len(history) * 100
        kept_sites.append(site)

    dropped = df["monitoring_location_id"].nunique() - len(kept_sites)
    if dropped and verbose:
        print(f"  Dropped {dropped} site(s) with under {MIN_TRAIN_DAYS} days of training history.")
    df["level_pct"] = pct.where(df["level_value"].notna())
    return df[df["monitoring_location_id"].isin(kept_sites)].copy()


# -------------------------------------------------- Step 3: add weather
def load_weather():
    """The site -> grid square map and the daily weather, ready to join."""
    cell_map = read_dataset("weather_cell_map", dtype={"monitoring_location_id": str, "cell_id": str})
    weather = read_dataset("weather_daily", dtype={"cell_id": str})
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.drop(columns=["grid_latitude", "grid_longitude"], errors="ignore")
    return cell_map, weather


def load_sites():
    return read_dataset("usgs_active_sites", dtype={"monitoring_location_id": str})


def add_weather(df, cell_map=None, weather=None, verbose=True):
    if cell_map is None or weather is None:
        cell_map, weather = load_weather()

    df = df.merge(cell_map[["monitoring_location_id", "cell_id"]], on="monitoring_location_id", how="left")
    df = df.merge(weather, on=["cell_id", "date"], how="left")
    if verbose:
        print(f"  Days with no matching weather: {df['precipitation_sum'].isna().mean():.1%}")
    return df


# ------------------------------------------- Step 4: features and targets
def add_features_and_targets(df, sites=None):
    df = df.sort_values(["monitoring_location_id", "date"]).reset_index(drop=True)
    g = df.groupby("monitoring_location_id", group_keys=False)

    def rolling_sum(col, days):
        return g[col].transform(lambda s: s.rolling(days, min_periods=max(1, days // 2)).sum())

    def rolling_mean(col, days):
        return g[col].transform(lambda s: s.rolling(days, min_periods=max(1, days // 2)).mean())

    # River: where is it now, and which way is it heading?
    df["level_pct_1d_ago"] = g["level_pct"].shift(1)
    df["level_pct_3d_ago"] = g["level_pct"].shift(3)
    df["level_pct_7d_ago"] = g["level_pct"].shift(7)
    df["level_pct_change_7d"] = df["level_pct"] - df["level_pct_7d_ago"]
    df["level_pct_mean_30d"] = rolling_mean("level_pct", 30)     # long dry/wet stretches

    # Rain: recent totals over different windows (soil soaks up the first rain,
    # so several days of rain matter more than one day).
    df["precip_1d"] = df["precipitation_sum"]
    for days in (3, 7, 14, 30):
        df[f"precip_{days}d"] = rolling_sum("precipitation_sum", days)
    df["precip_hours_3d"] = rolling_sum("precipitation_hours", 3)
    df["snowfall_7d"] = rolling_sum("snowfall_sum", 7)

    # Drying: evaporation, heat, and a simple water balance (rain in minus
    # evaporation out over 30 days; very negative = drying out = drought signal).
    df["et0_7d"] = rolling_sum("et0_fao_evapotranspiration", 7)
    df["et0_30d"] = rolling_sum("et0_fao_evapotranspiration", 30)
    df["water_balance_30d"] = df["precip_30d"] - df["et0_30d"]
    df["temp_max_mean_7d"] = rolling_mean("temperature_2m_max", 7)

    # Rain in the NEXT HORIZON_DAYS days. In training this is the rain that
    # actually fell; in the live app it comes from the Open-Meteo forecast.
    df["precip_next_3d"] = g["precipitation_sum"].transform(
        lambda s: s.rolling(HORIZON_DAYS).sum().shift(-HORIZON_DAYS))

    # Season and site size.
    df["month"] = df["date"].dt.month
    if sites is None:
        sites = load_sites()
    df = df.merge(sites[["monitoring_location_id", "drainage_area", "county_name"]],
                  on="monitoring_location_id", how="left")
    df["log_drainage_area"] = np.log1p(pd.to_numeric(df["drainage_area"], errors="coerce"))

    # TARGETS: the score HORIZON_DAYS later, plus the same thing as levels.
    df["target_pct"] = df.groupby("monitoring_location_id")["level_pct"].shift(-HORIZON_DAYS)
    df["target_level7"] = to_level(df["target_pct"], LEVELS_7)
    df["target_level3"] = to_level(df["target_pct"], LEVELS_3)
    return df


FEATURE_COLUMNS = [
    "level_pct", "level_pct_1d_ago", "level_pct_3d_ago", "level_pct_7d_ago",
    "level_pct_change_7d", "level_pct_mean_30d",
    "precip_1d", "precip_3d", "precip_7d", "precip_14d", "precip_30d",
    "precip_hours_3d", "snowfall_7d",
    "et0_7d", "et0_30d", "water_balance_30d", "temp_max_mean_7d",
    "precip_next_3d", "month", "log_drainage_area",
]
ID_COLUMNS = ["monitoring_location_id", "date", "split", "county_name", "cell_id",
              "level_source", "level_value"]
TARGET_COLUMNS = ["target_pct", "target_level7", "target_level3"]


def assign_split(dates, split_date):
    """train = before the split (minus the gap), test = on/after the split,
    gap = the last HORIZON_DAYS before the split (their answers fall in the test period)."""
    split = pd.Timestamp(split_date)
    gap_start = split - pd.Timedelta(days=HORIZON_DAYS)
    return np.select([dates < gap_start, dates < split], ["train", "gap"], default="test")


def build_table(river, split_date=SPLIT_DATE, cell_map=None, weather=None, sites=None, verbose=True):
    """Everything after loading: scores (for this split date), weather, features,
    targets, and the train/gap/test label. Gap rows are removed.

    `river` is the output of load_river(). Pass preloaded cell_map/weather/sites to
    avoid re-downloading them (cross_validate.py does this for each round)."""
    df = add_percentiles(river.copy(), split_date, verbose=verbose)
    df = add_weather(df, cell_map, weather, verbose=verbose)
    df = add_features_and_targets(df, sites)

    # A row is usable only if we know today's level and the level HORIZON days later.
    before = len(df)
    df = df.dropna(subset=["level_pct", "target_pct"])
    df["split"] = assign_split(df["date"], split_date)
    gap_rows = (df["split"] == "gap").sum()
    df = df[df["split"] != "gap"]
    if verbose:
        print(f"  Kept {len(df):,} of {before:,} rows (dropped days missing today's or the future level, "
              f"and {gap_rows:,} gap rows).")
    df = df.copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[ID_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS]


def main():
    print("Step 1: loading river data...")
    river = load_river()
    print(f"  {river['monitoring_location_id'].nunique()} sites, "
          f"{river['date'].min().date()} to {river['date'].max().date()}")

    print(f"Steps 2-4: 0-100 scores, weather, features and {HORIZON_DAYS}-day-ahead targets "
          f"(split date {SPLIT_DATE})...")
    df = build_table(river, SPLIT_DATE)

    print("\nSaving...")
    write_dataset("training_table", df)

    print(f"\n{len(df):,} rows x {len(df.columns)} columns")
    print(f"  Sites: {df['monitoring_location_id'].nunique()}  |  "
          f"Train rows: {(df['split'] == 'train').sum():,}  |  Test rows: {(df['split'] == 'test').sum():,}")
    print("\n  How often each level happens (share of rows):")
    order = [label for _, label in LEVELS_7]
    counts = (df.groupby("split")["target_level7"].value_counts(normalize=True)
              .unstack(0).reindex(order).fillna(0))
    print((counts * 100).round(1).astype(str).add("%").to_string())
    print("\n  Missing values per feature (the model can handle some):")
    missing = df[FEATURE_COLUMNS].isna().mean()
    print((missing[missing > 0] * 100).round(1).astype(str).add("%").to_string() or "  none")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="read/write the local data/ folder only (no S3)")
    if parser.parse_args().local:
        os.environ["FLOWCAST_DATA_SOURCE"] = "local"
    main()