"""
Build the training data for the new FlowCast models (the plan in notebooks/eda_raw_data.ipynb, section 9).

One row per gauge per day, with
  - FEATURES: what we know on that day (river scores, recent weather, season, river size)
  - TARGETS:  yes/no answers some days later, one column per model:
        target_high_1d       will the river be in HIGH WATER tomorrow?
        target_drought_7d    will the river be in DROUGHT 7 days from now?
    Other horizons can be added later by editing HIGH_WATER_HORIZONS / DROUGHT_HORIZONS.

HOW TO RUN (from the repo root, after daily_update + upload_to_s3)
    python -m src.flowcast.model.build_training_data            # read from S3 (default)
    python -m src.flowcast.model.build_training_data --local    # read from the local data/ folder

INPUTS (newest copies in S3, via src/flowcast/storage.py)
    usgs_daily, usgs_active_sites, weather_cell_map, weather_daily
    usgs_history    (2007-2019 river readings, from usgs_history_backfill.py) - part of the "ruler"
                    that defines normal / high water / drought.
    weather_history (2007-2019 weather, from weather_history_backfill.py) - lets 2007-2019 days
                    become TRAINING ROWS too.
    Both are optional: training rows start when the weather starts (2007 with the weather history,
    2020 without it), and the ruler uses whatever river years exist. The script says which it used.

OUTPUT
    data/training_data.csv, also uploaded to S3: processed/training_data/<today>/

THE STEPS (each matches a decision in the EDA notebook)
  1. Pick one measure per gauge (Decision 2): streamflow if it has readings on >= 90% of the
     2020-2024 days, else gauge height if >= 90%, else the gauge is left out.
  2. Two scores per day (Decisions 4 and 5, rule B):
       hw_score   0-100: today's DAILY level vs ALL of the gauge's training days.
                  >= 90 -> high water.
       dry_score  0-100: today's 7-DAY AVERAGE level vs the gauge's training days within
                  +/-15 calendar days of today's date (any year). < 10 -> drought.
     Tie rule: a day that is both is labeled high water (not drought).
     WHAT EACH DAY IS COMPARED AGAINST (the "ruler") - every year from 2007 up to the split date,
     except the day's own year:
       - a TRAINING day in year Y  -> 2007 .. (split year - 1), except Y
       - a TEST day                -> 2007 .. (split year - 1)
     Example (split 2025): a 2022 day is scored against 2007-2021 and 2023-2024 (17 years).
     A gauge needs at least MIN_RULER_YEARS years on its ruler, or it is left out of that round.
     Why: if a day were compared against a history that includes its own year, a dry year could only
     ever have ~10% of its days labeled drought, so training droughts would look like they always end
     within a week, and the models learn the opposite of reality.
  3. Weather features for each gauge's weather grid square (Decisions 6 and 8).
  4. Targets: the labels HORIZON days later.

TRAIN / TEST
    Rows before SPLIT_DATE are "train", rows on/after it are "test" (column `split`).
    Each model must also drop the training rows whose answer falls in the test period
    (the GAP = that model's horizon). Use train_test_masks() below so every script does
    this the same way.

FUTURE RAIN (Decision B, extended to drought)
    rain_next_1d = rain that actually fell tomorrow            (for the 1-day high-water model)
    rain_next_7d = rain that actually fell over the next 7 days (for the 7-day drought model)
    In training this is the rain that really fell (a "perfect forecast"); in the app it comes from
    the Open-Meteo forecast. Every model is trained and reported both WITH it (best case) and
    WITHOUT it (worst case); real accuracy falls between the two.
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.flowcast.storage import read_dataset, write_dataset

# ---------------------------------------------------------------- settings
SPLIT_DATE = "2025-01-01"     # train = before this date, test = on/after
TRAIN_START = "2020-01-01"     # start of the "recent" data: used to choose each gauge's measure, and the
                               # first training row when the 2007-2019 weather history isn't downloaded
HISTORY_START = "2007-01-01"   # first year of the RULER (river readings only)
MIN_RULER_YEARS = 10           # a gauge needs at least this many years on its ruler
MIN_COMPLETE_PCT = 90         # Decision 2
LOW_PCT, HIGH_PCT = 10, 90    # Decision D: bottom 10% = drought, top 10% = high water
WINDOW_DAYS = 15              # Decision 5: drought compares to +/- this many calendar days
SMOOTH_DAYS = 7               # Decision 7: drought uses the 7-day average level
HIGH_WATER_HORIZONS = (1,)    # days ahead
DROUGHT_HORIZONS = (7,)       # days ahead

# ---------------------------------------------------------------- columns
# INPUT COLUMNS (item E, approved). Every one is known on the day of the prediction.
FEATURE_COLUMNS = [
    # River, high-water side (daily level vs all days)
    "hw_score",              # today's high-water score, 0-100
    "hw_score_change_1d",    # today minus yesterday: is the river rising or falling right now?
    "hw_score_change_3d",    # today minus 3 days ago
    "hw_score_mean_30d",     # how wet the river has been over the last month
    # River, drought side (7-day average vs same time of year)
    "dry_score",             # today's drought score, 0-100
    "dry_score_change_7d",   # today minus 7 days ago: drying out or recovering?
    # Rain (Decision 6)
    "precip_1d", "precip_3d", "precip_7d", "precip_30d",
    # Drying out (Decision 8: evapotranspiration and max temperature)
    "et0_7d", "et0_30d",
    "water_balance_30d",     # rain in minus evapotranspiration out, last 30 days (very negative = drying)
    "temp_max_mean_7d",
    # Season and river size
    "month", "log_drainage_area",
]
FUTURE_RAIN_COLUMNS = {h: f"rain_next_{h}d" for h in sorted(set(HIGH_WATER_HORIZONS) | set(DROUGHT_HORIZONS))}
TODAY_LABEL_COLUMNS = ["is_high", "is_drought"]                 # used by the "same as today" baseline
TARGET_COLUMNS = ([f"target_high_{h}d" for h in HIGH_WATER_HORIZONS]
                  + [f"target_drought_{h}d" for h in DROUGHT_HORIZONS])
ID_COLUMNS = ["monitoring_location_id", "date", "split", "measure", "level_value", "county_name", "cell_id"]


# ------------------------------------------------ Step 1: one measure per gauge
MEASURE_NAMES = {"streamflow_cfs_mean": "streamflow", "gauge_height_ft_mean": "gauge_height"}


def choose_measures(usgs, verbose=True):
    """Decision 2: streamflow if it has readings on >= 90% of the 2020-2024 days, else gauge height
    if >= 90%, else the gauge is left out. Returns gauge id -> "streamflow" / "gauge_height".
    Uses the main SPLIT_DATE on purpose, so the same gauges are used in every cross-validation round."""
    start, split = pd.Timestamp(TRAIN_START), pd.Timestamp(SPLIT_DATE)
    train_days = (split - start).days
    dates = pd.to_datetime(usgs["time"])
    in_train = usgs[(dates >= start) & (dates < split)].assign(measure=lambda d: d["variable"].map(MEASURE_NAMES))
    pct_days = (in_train.groupby(["monitoring_location_id", "measure"])["value"].count()
                .div(train_days).mul(100).unstack())
    flow_ok = pct_days.get("streamflow", pd.Series(dtype=float)).reindex(pct_days.index) >= MIN_COMPLETE_PCT
    gh_ok = pct_days.get("gauge_height", pd.Series(dtype=float)).reindex(pct_days.index) >= MIN_COMPLETE_PCT
    choice = pd.Series(np.select([flow_ok, gh_ok], ["streamflow", "gauge_height"], default="drop"),
                       index=pct_days.index, name="measure")
    keep = choice[choice != "drop"]
    if verbose:
        print(f"  Gauges kept: {len(keep)} ({(keep == 'streamflow').sum()} streamflow, "
              f"{(keep == 'gauge_height').sum()} gauge height); dropped: {(choice == 'drop').sum()}")
    return keep


def load_history(verbose=True):
    """2007-2019 river readings (usgs_history_backfill.py), or None if they haven't been downloaded."""
    try:
        return read_dataset("usgs_history", dtype={"monitoring_location_id": str})
    except FileNotFoundError:
        if verbose:
            print("  NOTE: no 2007-2019 river history found, so 'normal' is defined from 2020 onward only.")
            print("        Run: python -m src.flowcast.ingestion.usgs_history_backfill")
        return None


def load_level(split_date=SPLIT_DATE, verbose=True):
    """One row per kept gauge per calendar day (from 2007 if the history is available), with that
    gauge's chosen measure in `level_value`. Missing days become empty rows, so "7 days ago"
    always means 7 calendar days."""
    usgs = read_dataset("usgs_daily", dtype={"monitoring_location_id": str})
    keep = choose_measures(usgs, verbose)

    history = load_history(verbose)
    frames = [usgs[["monitoring_location_id", "variable", "time", "value"]]]
    if history is not None:
        frames.insert(0, history[["monitoring_location_id", "variable", "time", "value"]])
    readings = pd.concat(frames, ignore_index=True)
    readings["date"] = pd.to_datetime(readings["time"])
    readings["measure"] = readings["variable"].map(MEASURE_NAMES)

    level = readings.merge(keep.reset_index(), on=["monitoring_location_id", "measure"])
    level = level.groupby(["monitoring_location_id", "date"], as_index=False)["value"].mean()
    if verbose and history is not None:
        first = level.groupby("monitoring_location_id")["date"].min()
        print(f"  River history: {len(history):,} rows from {history['time'].min()}; "
              f"{(first < pd.Timestamp(TRAIN_START)).sum()} of {len(keep)} kept gauges have readings before 2020")

    start = min(level["date"].min(), pd.Timestamp(TRAIN_START))
    full_dates = pd.date_range(start, level["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([keep.index, full_dates], names=["monitoring_location_id", "date"])
    out = (level.set_index(["monitoring_location_id", "date"])
           .reindex(grid).reset_index().rename(columns={"value": "level_value"}))
    out["measure"] = out["monitoring_location_id"].map(keep)
    return out


# ------------------------------------------------ Step 2: the two scores
def day_of_year_365(dates):
    """Day of year 1-365 on a non-leap calendar (Feb 29 counts as Feb 28), so all years line up."""
    md = pd.to_datetime(dates).dt.strftime("%m-%d").replace("02-29", "02-28")
    return pd.to_datetime("2021-" + md).dt.dayofyear.to_numpy()


def score_against(values, history):
    """0-100: share of `history` below each value (ties count half). NaN values stay NaN."""
    history = np.sort(history[~np.isnan(history)])
    if len(history) == 0:
        return np.full(len(values), np.nan)
    below = np.searchsorted(history, values, side="left")
    at_or_below = np.searchsorted(history, values, side="right")
    out = (below + at_or_below) / 2 / len(history) * 100
    return np.where(np.isnan(values), np.nan, out)


def add_scores(df, split_date=SPLIT_DATE, verbose=True, rows_start=TRAIN_START):
    """hw_score (daily vs all days) and dry_score (7-day average vs the same +/-15-day season),
    plus today's labels. The ruler is every year from the start of the data up to the split date,
    minus the day's own year. Only days from shortly before `rows_start` on are scored (older days
    are only part of the ruler)."""
    df = df.sort_values(["monitoring_location_id", "date"]).reset_index(drop=True)
    split = pd.Timestamp(split_date)
    score_from = pd.Timestamp(rows_start) - pd.Timedelta(days=45)   # a little early, for the 30-day features
    # Rounded so the same 7-day average always gives the same number (a running average picks up
    # tiny rounding errors that depend on where it started, which would change how ties count).
    df["level_7d"] = (df.groupby("monitoring_location_id")["level_value"]
                      .transform(lambda s: s.rolling(SMOOTH_DAYS, min_periods=5).mean())).round(6)
    doy = day_of_year_365(df["date"])
    year = df["date"].dt.year.to_numpy()
    on_ruler = (df["date"] < split).to_numpy()
    to_score = (df["date"] >= score_from).to_numpy()
    level_all = df["level_value"].to_numpy(dtype=float)
    level_7d_all = df["level_7d"].to_numpy(dtype=float)

    # Without the 2007-2019 history the ruler only has the 2020+ years, so the minimum can't apply.
    has_history = df["date"].min() < pd.Timestamp(TRAIN_START)
    min_years = MIN_RULER_YEARS if has_history else 1

    hw = np.full(len(df), np.nan)
    dry = np.full(len(df), np.nan)
    too_short = 0
    for _, idx in df.groupby("monitoring_location_id").indices.items():
        level, level_7d = level_all[idx], level_7d_all[idx]
        ruler, score, d, yr = on_ruler[idx], to_score[idx], doy[idx], year[idx]

        # Count ruler years with real data (at least ~10 months of readings).
        has_value = ~np.isnan(level)
        years_with_data = pd.Series(yr[ruler & has_value]).value_counts()
        if (years_with_data >= 300).sum() < min_years:
            too_short += 1
            continue

        # (days to score, the ruler they are compared against)
        groups = [(score & ~ruler, ruler)]                                     # test days: all ruler years
        groups += [(score & ruler & (yr == y), ruler & (yr != y))              # ruler-period days: all other years
                   for y in np.unique(yr[score & ruler])]

        for rows, against in groups:
            if rows.any():
                hw[idx[rows]] = score_against(level[rows], level[against])
        for day in range(1, 366):
            on_day = d == day
            if not (on_day & score).any():
                continue
            dist = np.abs(d - day)
            near = np.minimum(dist, 365 - dist) <= WINDOW_DAYS              # Dec 25 and Jan 5 are 11 days apart
            for rows, against in groups:
                today = rows & on_day
                if today.any():
                    dry[idx[today]] = score_against(level_7d[today], level_7d[against & near])

    if verbose and too_short:
        print(f"  Left out {too_short} gauge(s) with fewer than {min_years} years on the ruler (split {split_date}).")
    df["hw_score"] = hw
    df["dry_score"] = dry
    df["is_high"] = (df["hw_score"] >= HIGH_PCT).astype(float).where(df["hw_score"].notna())
    drought = (df["dry_score"] < LOW_PCT) & ~(df["hw_score"] >= HIGH_PCT)        # tie rule: high water wins
    df["is_drought"] = drought.astype(float).where(df["dry_score"].notna())
    return df


# ------------------------------------------------ Step 3: weather features
def load_inputs():
    """The site -> grid square map, daily weather, and site details."""
    cell_map = read_dataset("weather_cell_map", dtype={"monitoring_location_id": str, "cell_id": str})
    weather = read_dataset("weather_daily", dtype={"cell_id": str})
    try:
        old = read_dataset("weather_history", dtype={"cell_id": str})
        weather = pd.concat([old, weather], ignore_index=True).drop_duplicates(["cell_id", "date"], keep="last")
    except FileNotFoundError:
        print("  NOTE: no 2007-2019 weather history found, so training rows start in 2020.")
        print("        Run: python -m src.flowcast.ingestion.weather_history_backfill")
    weather["date"] = pd.to_datetime(weather["date"])
    sites = read_dataset("usgs_active_sites", dtype={"monitoring_location_id": str})
    return cell_map, weather, sites


def weather_features(weather):
    """Rolling weather totals per grid square (weather has no missing days)."""
    w = weather.sort_values(["cell_id", "date"]).reset_index(drop=True)
    g = w.groupby("cell_id")

    def rolling(col, days, how="sum"):
        r = g[col].transform(lambda s: getattr(s.rolling(days, min_periods=days), how)())
        return r

    out = w[["cell_id", "date"]].copy()
    out["precip_1d"] = w["precipitation_sum"]
    for days in (3, 7, 30):
        out[f"precip_{days}d"] = rolling("precipitation_sum", days)
    out["et0_7d"] = rolling("et0_fao_evapotranspiration", 7)
    out["et0_30d"] = rolling("et0_fao_evapotranspiration", 30)
    out["water_balance_30d"] = out["precip_30d"] - out["et0_30d"]
    out["temp_max_mean_7d"] = rolling("temperature_2m_max", 7, "mean")
    # Rain over the NEXT h days (tomorrow through h days from now). Training only: the app uses the forecast.
    for h, col in FUTURE_RAIN_COLUMNS.items():
        out[col] = g["precipitation_sum"].transform(lambda s, h=h: s.rolling(h, min_periods=h).sum().shift(-h))
    return out


# ------------------------------------------------ Step 4: features and targets
def add_features_and_targets(df, cell_map, weather, sites):
    df = df.sort_values(["monitoring_location_id", "date"]).reset_index(drop=True)
    g = df.groupby("monitoring_location_id")

    df["hw_score_change_1d"] = df["hw_score"] - g["hw_score"].shift(1)
    df["hw_score_change_3d"] = df["hw_score"] - g["hw_score"].shift(3)
    df["hw_score_mean_30d"] = g["hw_score"].transform(lambda s: s.rolling(30, min_periods=20).mean())
    df["dry_score_change_7d"] = df["dry_score"] - g["dry_score"].shift(7)

    df = df.merge(cell_map[["monitoring_location_id", "cell_id"]], on="monitoring_location_id", how="left")
    df = df.merge(weather_features(weather), on=["cell_id", "date"], how="left")

    df["month"] = df["date"].dt.month
    df = df.merge(sites[["monitoring_location_id", "drainage_area", "county_name"]],
                  on="monitoring_location_id", how="left")
    df["log_drainage_area"] = np.log1p(pd.to_numeric(df["drainage_area"], errors="coerce"))

    # TARGETS: today's label, looked up h days later. Empty when that day has no reading.
    df = df.sort_values(["monitoring_location_id", "date"]).reset_index(drop=True)
    g = df.groupby("monitoring_location_id")
    for h in HIGH_WATER_HORIZONS:
        df[f"target_high_{h}d"] = g["is_high"].shift(-h)
    for h in DROUGHT_HORIZONS:
        df[f"target_drought_{h}d"] = g["is_drought"].shift(-h)
    return df


def build_table(level, split_date=SPLIT_DATE, cell_map=None, weather=None, sites=None, verbose=True):
    """Everything after loading: scores for this split date, features, targets, and the train/test label.
    `level` is the output of load_level(). Cross-validation calls this once per split date."""
    if cell_map is None or weather is None or sites is None:
        cell_map, weather, sites = load_inputs()
    # Training rows start when the weather starts (2007 with the weather history, else 2020).
    rows_start = max(weather["date"].min(), level["date"].min())
    df = add_scores(level.copy(), split_date, verbose=verbose, rows_start=rows_start)
    df = add_features_and_targets(df, cell_map, weather, sites)

    before = len(df)
    df = df[df["date"] >= rows_start]                  # older days (if any) were only part of the ruler
    df = df[df["hw_score"].notna()].copy()        # need today's reading to make a prediction
    df["split"] = np.where(df["date"] < pd.Timestamp(split_date), "train", "test")
    if verbose:
        print(f"  Training rows start {rows_start.date()}. Kept {len(df):,} of {before:,} gauge-days "
              f"(dropped days with no reading, and ruler-only days).")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[ID_COLUMNS + FEATURE_COLUMNS + list(FUTURE_RAIN_COLUMNS.values())
              + TODAY_LABEL_COLUMNS + TARGET_COLUMNS]


# ------------------------------------------------ helpers for the training scripts
def train_test_masks(df, target, split_date=SPLIT_DATE):
    """(train_mask, test_mask) for one target, with the GAP applied:
    a training row is used only if its answer day is still before the split date.
    Rows with no answer (missing reading on the answer day) are left out of both."""
    horizon = int(target.rsplit("_", 1)[1].rstrip("d"))
    dates = pd.to_datetime(df["date"])
    split = pd.Timestamp(split_date)
    has_answer = df[target].notna()
    train = has_answer & (dates + pd.Timedelta(days=horizon) < split)
    test = has_answer & (dates >= split)
    return train, test


def features_for(target, with_future_rain=False):
    """The input columns a model for `target` may use; with_future_rain adds the rain over the
    model's own horizon (tomorrow for high_1d, the next 7 days for drought_7d)."""
    cols = list(FEATURE_COLUMNS)
    if with_future_rain:
        horizon = int(target.rsplit("_", 1)[1].rstrip("d"))
        cols.append(FUTURE_RAIN_COLUMNS[horizon])
    return cols


# ------------------------------------------------ run
def main():
    print("Step 1: choosing one measure per gauge...")
    level = load_level()
    print(f"  {level['date'].min().date()} to {level['date'].max().date()}")

    print("Steps 2-4: scores, weather features and targets (this takes a minute)...")
    df = build_table(level, SPLIT_DATE)

    print("\nSaving...")
    write_dataset("training_data", df)

    print(f"\n{len(df):,} rows x {len(df.columns)} columns  |  gauges: {df['monitoring_location_id'].nunique()}  |  "
          f"train rows: {(df['split'] == 'train').sum():,}  |  test rows: {(df['split'] == 'test').sum():,}")

    print("\n  How often each target is 'yes' (train can be above 10%: dry or wet years now show up as such):")
    rates = pd.DataFrame({t: df.groupby("split")[t].mean() * 100 for t in TARGET_COLUMNS}).T.round(1)
    print(rates.astype(str).add("%").to_string())

    print("\n  Missing values per input column:")
    missing = df[FEATURE_COLUMNS + list(FUTURE_RAIN_COLUMNS.values())].isna().mean() * 100
    missing = missing[missing > 0].round(1)
    print(missing.astype(str).add("%").to_string() if len(missing) else "  none")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="read/write the local data/ folder only (no S3)")
    if parser.parse_args().local:
        os.environ["FLOWCAST_DATA_SOURCE"] = "local"
    main()
