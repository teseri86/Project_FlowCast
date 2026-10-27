"""
*** LEGACY / DEPRECATED - not part of the current pipeline. ***
This checks the OLD labels (from the old training_table, built by the removed
build_training_table.py). It still runs only while that old table exists in S3.
Kept for its NWS flood-stage and Drought Monitor downloads and its county codes
(COUNTY_FIPS, used by notebooks/eda_raw_data.ipynb), which may be reused to
check the new labels later.

Check our flood/drought labels against official sources:
  FLOODS   -> National Weather Service flood stages for each gauge (in feet)
  DROUGHTS -> U.S. Drought Monitor weekly county maps

Our labels come from each gauge's own 0-100 score (see build_training_table.py).
This script asks two questions:
  1. On days when a river was OFFICIALLY at flood level, did our labels say "flood"?
  2. In weeks when the Drought Monitor had a county IN DROUGHT, did our gauges there
     score low?

HOW TO RUN (from the repo root, after build_training_table.py)
    python -m src.flowcast.model.validate_labels            # read from S3 (default)
    python -m src.flowcast.model.validate_labels --local    # local data/ folder only

OUTPUTS (data/validation/, also uploaded to S3: processed/validation/<today>/)
    nws_flood_stages.csv        official flood stages for each gauge that has them
    flood_check_by_gauge.csv    per gauge: official flood days vs our flood labels
    official_flood_days.csv     every gauge-day at or above an official flood stage
    usdm_county_weekly.csv      Drought Monitor % of each county in drought, weekly
    drought_check_weekly.csv    per county-week: Drought Monitor vs our scores

LIMITATIONS (worth mentioning in the write-up)
  - We only have DAILY AVERAGE gauge height. A river can pass flood stage for a few
    hours and still average below it, so official flood days are undercounted.
  - Not every USGS gauge has NWS flood stages (mostly larger rivers do).
  - The Drought Monitor describes whole counties and blends many signals (soil,
    rain, rivers); one creek can disagree with its county.
"""

import io
import os
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

from src.flowcast.storage import read_dataset, write_dataset

DATA_DIR = "data"
OUT_DIR = os.path.join(DATA_DIR, "validation")

NWPS_URL = "https://api.water.noaa.gov/nwps/v1/gauges/{}"
USDM_URL = ("https://usdmdataservices.unl.edu/api/CountyStatistics/"
            "GetDroughtSeverityStatisticsByAreaPercent")
START_DATE = "2020-01-01"

FLOOD_CATEGORIES = ["action", "minor", "moderate", "major"]  # NWS order, low to high

# County name -> 5-digit FIPS code (state code + county code).
COUNTY_FIPS = {
    ("North Carolina", "Mecklenburg County"): "37119", ("North Carolina", "Union County"): "37179",
    ("North Carolina", "Cabarrus County"): "37025", ("North Carolina", "Gaston County"): "37071",
    ("North Carolina", "Iredell County"): "37097", ("North Carolina", "Rowan County"): "37159",
    ("North Carolina", "Lincoln County"): "37109", ("North Carolina", "Anson County"): "37007",
    ("North Carolina", "Alexander County"): "37003", ("North Carolina", "Catawba County"): "37035",
    ("North Carolina", "Cleveland County"): "37045", ("North Carolina", "Stanly County"): "37167",
    ("South Carolina", "York County"): "45091", ("South Carolina", "Lancaster County"): "45057",
    ("South Carolina", "Chester County"): "45023", ("South Carolina", "Chesterfield County"): "45025",
}


def load_scores():
    """Our daily 0-100 score per gauge (level_pct = that day's score)."""
    t = read_dataset("training_table", dtype={"monitoring_location_id": str},
                     usecols=["monitoring_location_id", "date", "level_pct", "county_name"])
    sites = read_dataset("usgs_active_sites", dtype={"monitoring_location_id": str})
    return t.merge(sites[["monitoring_location_id", "state_name", "monitoring_location_name"]],
                   on="monitoring_location_id", how="left")


# ======================================================== FLOOD CHECK
def find_categories(obj):
    """Find the NWS 'categories' block wherever it sits in the response."""
    if isinstance(obj, dict):
        if "categories" in obj and isinstance(obj["categories"], dict):
            return obj["categories"]
        for value in obj.values():
            found = find_categories(value)
            if found:
                return found
    return None


def get_flood_stages(site_ids):
    rows = []
    print(f"Flood check: asking NWS for flood stages at {len(site_ids)} gauges...")
    for n, site in enumerate(site_ids, start=1):
        usgs_number = site.replace("USGS-", "")
        try:
            resp = requests.get(NWPS_URL.format(usgs_number), timeout=30)
        except requests.RequestException as e:
            print(f"  {site}: request failed ({e})")
            continue
        if resp.status_code == 404:
            continue  # normal: many small gauges aren't NWS forecast points
        if resp.status_code != 200:
            print(f"  {site}: HTTP {resp.status_code}")
            continue
        body = resp.json()
        cats = find_categories(body) or {}
        row = {"monitoring_location_id": site, "nws_lid": body.get("lid"),
               "nws_name": body.get("name")}
        for cat in FLOOD_CATEGORIES:
            stage = (cats.get(cat) or {}).get("stage")
            row[f"{cat}_stage_ft"] = stage if stage not in (None, -9999) else np.nan
        if any(pd.notna(row[f"{c}_stage_ft"]) for c in FLOOD_CATEGORIES):
            rows.append(row)
        time.sleep(0.2)
    stages = pd.DataFrame(rows)
    print(f"  {len(stages)} of {len(site_ids)} gauges have official NWS flood stages.")
    return stages


def official_category(df):
    """Highest NWS category each day's gauge height reached ('none' if below action)."""
    reached = pd.Series("none", index=df.index)
    for cat in FLOOD_CATEGORIES:  # low to high, so higher categories overwrite lower
        stage = df[f"{cat}_stage_ft"]
        reached[stage.notna() & (df["gauge_height_ft"] >= stage)] = cat
    return reached


def flood_check(scores):
    site_ids = sorted(scores["monitoring_location_id"].unique())
    stages = get_flood_stages(site_ids)
    write_dataset("nws_flood_stages", stages)
    if stages.empty:
        print("  No gauges with flood stages - skipping flood check.")
        return

    usgs = read_dataset("usgs_daily", dtype={"monitoring_location_id": str, "parameter_code": str})
    gh = (usgs[usgs["parameter_code"] == "00065"]
          .groupby(["monitoring_location_id", "time"])["value"].mean()
          .rename("gauge_height_ft").reset_index().rename(columns={"time": "date"}))

    df = (scores.merge(gh, on=["monitoring_location_id", "date"], how="inner")
                .merge(stages, on="monitoring_location_id", how="inner"))
    df["official"] = official_category(df)
    df["our_label"] = pd.cut(df["level_pct"], [0, 90, 95, 99, 100.01], right=False,
                             labels=["normal_or_dry", "above_normal_flow", "high_flow", "very_high_flow"])

    # Every official high-water day, for looking up real events.
    events = df[df["official"] != "none"][
        ["date", "monitoring_location_id", "monitoring_location_name", "gauge_height_ft",
         "official", "level_pct", "our_label"]].sort_values(["date", "monitoring_location_id"])
    write_dataset("official_flood_days", events)

    print(f"\n  Compared {len(df):,} gauge-days at {df['monitoring_location_id'].nunique()} gauges.")
    print("\n  QUESTION 1: On official high-water days, what did our labels say?")
    print("  (rows = NWS category that day, columns = our label, % of each row)")
    table = pd.crosstab(df["official"], df["our_label"], normalize="index") * 100
    table = table.reindex(["none"] + FLOOD_CATEGORIES).dropna(how="all")
    counts = df["official"].value_counts().reindex(table.index)
    table.insert(0, "gauge-days", counts)
    print(table.round(1).to_string())

    print("\n  QUESTION 2: On days WE called very high flow (score 99+), how high was the river officially?")
    severe = df[df["our_label"] == "very_high_flow"]["official"].value_counts(normalize=True) * 100
    print((severe.reindex(["none"] + FLOOD_CATEGORIES).fillna(0).round(1).astype(str) + "%").to_string())

    per_gauge = df.groupby(["monitoring_location_id", "monitoring_location_name"]).agg(
        max_gauge_height_ft=("gauge_height_ft", "max"),
        action_stage_ft=("action_stage_ft", "first"),
        minor_flood_stage_ft=("minor_stage_ft", "first"),
        official_action_or_higher_days=("official", lambda s: (s != "none").sum()),
        our_flood_level_days=("level_pct", lambda s: (s >= 90).sum()),
        our_severe_days=("level_pct", lambda s: (s >= 99).sum()),
    ).reset_index()
    write_dataset("flood_check_by_gauge", per_gauge)

    if not events.empty:
        print("\n  Biggest official flood events (most gauges at action stage or higher on one day):")
        top = events.groupby("date").size().sort_values(ascending=False).head(5)
        print(top.rename("gauges_at_action_or_higher").to_string())


# ======================================================= DROUGHT CHECK
USDM_DOWNLOAD_DIR = os.path.join(OUT_DIR, "usdm_downloads")  # manual CSV downloads go here


def get_usdm(fips):
    """Download one county's weekly history from the Drought Monitor data service."""
    today = date.today()
    params = {"aoi": fips, "startdate": "1/1/2020",
              "enddate": f"{today.month}/{today.day}/{today.year}",
              "statisticsType": 1}
    resp = requests.get(USDM_URL, params=params, headers={"Accept": "text/csv"}, timeout=60)
    resp.raise_for_status()
    return parse_usdm(pd.read_csv(io.StringIO(resp.text)), fips=fips)


def load_usdm_downloads(wanted_fips):
    """Read CSV files downloaded by hand from droughtmonitor.unl.edu (used when
    the data service is down). One file can hold one county or many."""
    if not os.path.isdir(USDM_DOWNLOAD_DIR):
        return []
    frames = []
    for name in sorted(os.listdir(USDM_DOWNLOAD_DIR)):
        if not name.lower().endswith(".csv"):
            continue
        raw = pd.read_csv(os.path.join(USDM_DOWNLOAD_DIR, name), dtype=str)
        try:
            parsed = parse_usdm(raw)
        except ValueError as e:
            print(f"  {name}: {e}")
            continue
        parsed = parsed[parsed["fips"].isin(wanted_fips)]
        print(f"  {name}: {parsed['fips'].nunique()} of our counties, {len(parsed):,} weeks")
        frames.append(parsed)
    return frames


def parse_usdm(df, fips=None):
    """Turn a Drought Monitor table (from the service or a manual download) into
    one row per county per week with CUMULATIVE percents d0..d4."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower = {}
    for c in df.columns:
        key = c.lower().replace(" ", "").replace("_", "")
        if len(key) >= 2 and key[0] == "d" and key[1] in "01234":
            key = key[:2]  # "D1-D4" or "D1" -> "d1"
        lower.setdefault(key, c)
    needed = ["validstart", "validend", "d0", "d1", "d2", "d3", "d4"] + ([] if fips else ["fips"])
    missing = [c for c in needed if c not in lower]
    if missing:
        raise ValueError(f"Unexpected Drought Monitor columns {list(df.columns)}; missing {missing}. "
                         "Paste this message so the parser can be adjusted.")
    out = pd.DataFrame({
        "fips": fips if fips else df[lower["fips"]].astype(str).str.split(".").str[0].str.zfill(5),
        "valid_start": pd.to_datetime(df[lower["validstart"]]),
        "valid_end": pd.to_datetime(df[lower["validend"]]),
    })
    ds = ["d0", "d1", "d2", "d3", "d4"]
    for d in ds + (["none"] if "none" in lower else []):
        out[d] = pd.to_numeric(df[lower[d]].astype(str).str.replace(",", ""), errors="coerce")

    # We want CUMULATIVE percents ("% of county in D1 or worse"). Tell which kind
    # the service sent: cumulative -> None + D0 = 100; separate -> None + D0..D4 = 100.
    separate = False
    if "none" in out.columns:
        cum_err = (out["none"] + out["d0"] - 100).abs().median()
        sep_err = (out["none"] + out[ds].sum(axis=1) - 100).abs().median()
        separate = sep_err < cum_err
    else:
        separate = (out[ds].diff(axis=1).iloc[:, 1:] > 0.5).any().any()
    if separate:
        cumulative = out[ds[::-1]].cumsum(axis=1)   # d4, d4+d3, ... d4+...+d0
        for d in ds:
            out[d] = cumulative[d]
    return out.drop(columns=["none"], errors="ignore")


def usdm_category(row):
    """Worst drought category covering at least half the county."""
    for d, label in [("d4", "D4"), ("d3", "D3"), ("d2", "D2"), ("d1", "D1"), ("d0", "D0")]:
        if row[d] >= 50:
            return label
    return "none"


def drought_check(scores):
    scores = scores.copy()
    scores["fips"] = [COUNTY_FIPS.get((s, c)) for s, c in zip(scores["state_name"], scores["county_name"])]
    fips_list = sorted(scores["fips"].dropna().unique())
    print(f"\nDrought check: getting Drought Monitor history for {len(fips_list)} counties...")
    frames = load_usdm_downloads(fips_list)
    have = set(pd.concat(frames)["fips"]) if frames else set()
    service_down = False
    for fips in [f for f in fips_list if f not in have]:
        if service_down:
            break
        try:
            frames.append(get_usdm(fips))
        except requests.exceptions.SSLError:
            print("  The Drought Monitor data service has a certificate problem (their side).")
            service_down = True
        except (requests.RequestException, ValueError) as e:
            print(f"  {fips}: {e}")
        time.sleep(0.5)
    got = set(pd.concat(frames)["fips"]) if frames else set()
    missing = [f for f in fips_list if f not in got]
    if missing:
        print(f"  Missing {len(missing)} county(ies): {', '.join(missing)}")
        print(f"  Workaround: download them as CSV from droughtmonitor.unl.edu (Data > Comprehensive")
        print(f"  Statistics, county level, percent area, 2020 to today) into {USDM_DOWNLOAD_DIR}/")
        print("  and run this again.")
    if not frames:
        print("  No Drought Monitor data - skipping drought check.")
        return
    usdm = pd.concat(frames, ignore_index=True)
    usdm["dsci"] = usdm[["d0", "d1", "d2", "d3", "d4"]].sum(axis=1)  # 0 (none) to 500 (all D4)
    usdm["usdm_category"] = usdm.apply(usdm_category, axis=1)
    write_dataset("usdm_county_weekly", usdm)

    # Our side: average score of each county's gauges over each Drought Monitor week.
    scores["date"] = pd.to_datetime(scores["date"])
    rows = []
    for _, wk in usdm.iterrows():
        mask = (scores["fips"] == wk["fips"]) & scores["date"].between(wk["valid_start"], wk["valid_end"])
        week = scores.loc[mask, "level_pct"]
        if len(week):
            rows.append({"fips": wk["fips"], "week_start": wk["valid_start"].date(),
                         "our_mean_score": week.mean(), "our_share_drought_days": (week < 10).mean() * 100,
                         "usdm_category": wk["usdm_category"], "usdm_pct_d1_or_worse": wk["d1"],
                         "usdm_dsci": wk["dsci"]})
    weekly = pd.DataFrame(rows)
    write_dataset("drought_check_weekly", weekly)

    print(f"  Compared {len(weekly):,} county-weeks.")
    print("\n  QUESTION: When the Drought Monitor said a county was in drought, did our gauges score low?")
    print("  (average of our 0-100 scores; lower = drier. D1+ = in drought, D0 = abnormally dry)")
    summary = weekly.groupby("usdm_category").agg(
        county_weeks=("our_mean_score", "size"),
        our_avg_score=("our_mean_score", "mean"),
        our_pct_days_in_drought_levels=("our_share_drought_days", "mean"),
    ).reindex(["none", "D0", "D1", "D2", "D3", "D4"]).dropna(how="all")
    print(summary.round(1).to_string())
    corr = weekly["our_mean_score"].corr(weekly["usdm_dsci"])
    print(f"\n  Correlation between our weekly score and Drought Monitor severity: {corr:.2f}")
    print("  (expect a NEGATIVE number: drier rivers <-> more drought. Closer to -1 = stronger agreement)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scores = load_scores()
    flood_check(scores)
    drought_check(scores)
    print(f"\nDetail files saved in {OUT_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="read/write the local data/ folder only (no S3)")
    if parser.parse_args().local:
        os.environ["FLOWCAST_DATA_SOURCE"] = "local"
    main()