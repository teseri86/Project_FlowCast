"""
ONE-TIME S3 cleanup (2026-10-27). Deletes only the files listed below: old-pipeline outputs and
older dated copies that a newer copy replaces. Delete this script after running it (don't commit it).

HOW TO RUN (from the repo root, in the flowcast_env terminal)
    python s3_cleanup.py            # PREVIEW: lists what would be deleted; deletes nothing
    python s3_cleanup.py --delete   # actually delete

SAFETY
    - Only the exact file paths below are touched (no wildcards).
    - Before deleting an older copy, it checks the newer copy being KEPT exists. If any is missing,
      it stops and deletes nothing.
    - Files already gone are reported and skipped.
"""

import argparse

from src.flowcast.storage import _s3

# 1. Old pipeline, replaced by the new scripts. (processed/training_table/ is NOT here: its newest copy
#    stays for the legacy validate_labels.py; only its older duplicate is in section 2.)
OLD_PIPELINE = [
    "models/decision_trees/2026-09-27/decision_trees.joblib",
    "models/gradient_boosting/2026-09-27/gradient_boosting.joblib",
    "processed/model/2026-09-27/cv_results.csv",
    "processed/model/2026-09-27/cv_summary.csv",
    "processed/model/2026-09-27/gradient_boosting_feature_importance.csv",
    "processed/model/2026-09-27/gradient_boosting_level_report.csv",
    "processed/model/2026-09-27/gradient_boosting_metrics.csv",
    "processed/model/2026-09-27/gradient_boosting_scenarios.csv",
]

# 2. Older dated copies -> the newer copy that is KEPT (must exist, or nothing is deleted).
def copies(folder, file, old_dates, keep_date):
    return [(f"{folder}/{d}/{file}", f"{folder}/{keep_date}/{file}") for d in old_dates]

SEPT = ["2026-09-24", "2026-09-25", "2026-09-27"]
SUPERSEDED = (
    copies("raw/usgs/daily_values", "usgs_daily_2020_present.csv", SEPT, "2026-10-25")
    + copies("raw/openmeteo/daily_weather", "weather_daily_2020_present.csv", SEPT, "2026-10-25")
    + copies("raw/usgs/active_sites", "usgs_active_sites.csv", SEPT, "2026-10-25")
    + copies("raw/usgs/daily_series", "usgs_daily_series.csv", SEPT, "2026-10-25")
    + copies("raw/usgs/monitoring_locations", "usgs_monitoring_locations.csv", SEPT, "2026-10-25")
    + copies("raw/openmeteo/site_cell_map", "site_weather_cell_map.csv", SEPT, "2026-10-25")
    + copies("processed/training_data", "training_data.csv", ["2026-10-25"], "2026-10-26")
    + copies("processed/training_table", "training_table.csv", ["2026-09-25"], "2026-09-27")
    + copies("processed/model", "compare_results.csv", ["2026-10-26"], "2026-10-27")
    + copies("processed/model", "compare_summary.csv", ["2026-10-26"], "2026-10-27")
    + [pair for f in ["drought_check_weekly.csv", "flood_check_by_gauge.csv", "nws_flood_stages.csv",
                      "official_flood_days.csv", "usdm_county_weekly.csv"]
       for pair in copies("processed/validation", f, ["2026-09-25"], "2026-09-27")]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="actually delete (default: preview only)")
    args = parser.parse_args()

    s3, bucket = _s3()
    sizes = {o["Key"]: o["Size"] for p in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket)
             for o in p.get("Contents", [])}

    missing_keep = sorted({keep for _, keep in SUPERSEDED if keep not in sizes})
    if missing_keep:
        print("STOPPED - these newer copies (to be kept) are missing, so nothing was deleted:")
        for k in missing_keep:
            print("   ", k)
        return

    targets = OLD_PIPELINE + [old for old, _ in SUPERSEDED]
    present = [k for k in targets if k in sizes]
    gone = [k for k in targets if k not in sizes]
    total_mb = sum(sizes[k] for k in present) / 1e6

    print(f"{'DELETING' if args.delete else 'PREVIEW - would delete'} {len(present)} files ({total_mb:,.1f} MB) "
          f"from s3://{bucket}:")
    for k in present:
        print(f"  {sizes[k] / 1e6:8.1f} MB  {k}")
    for k in gone:
        print(f"  (already gone)  {k}")

    if not args.delete:
        print("\nNothing deleted. Run again with --delete to delete these files.")
        return
    for i in range(0, len(present), 1000):                 # S3 deletes up to 1,000 keys per request
        resp = s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in present[i:i + 1000]]})
        for err in resp.get("Errors", []):
            print(f"  FAILED: {err['Key']}: {err['Message']}")
    print(f"\nDone. Deleted {len(present)} files ({total_mb:,.1f} MB).")


if __name__ == "__main__":
    main()
