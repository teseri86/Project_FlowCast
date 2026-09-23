"""
Pull USGS monitoring location and daily streamflow data for a list of counties.

Reads credentials from a .env file (see .env.example at the repo root).
Run from the repo root with: python -m src.flowcast.ingestion.pull_usgs
"""

import ast
import os
import re
from dotenv import load_dotenv
from dataretrieval import waterdata
import pandas as pd

load_dotenv()  # reads .env and sets environment variables for this process

# dataretrieval automatically picks up API_USGS_PAT from the environment,
# so no explicit auth call is needed here as long as .env is loaded first.

pd.set_option("display.max_columns", None)

# Counties surrounding Mecklenburg County, NC
COUNTIES = [
    ("Mecklenburg", "North Carolina"),
    ("Union", "North Carolina"),
    ("Cabarrus", "North Carolina"),
    ("Gaston", "North Carolina"),
    ("Iredell", "North Carolina"),
    ("Rowan", "North Carolina"),
    ("Lincoln", "North Carolina"),
    ("Anson", "North Carolina"),
    ("York", "South Carolina"),
    ("Lancaster", "South Carolina"),
    ("Chester", "South Carolina"),
    ("Alexander", "North Carolina"),
    ("Catawba", "North Carolina"),
    ("Cleveland", "North Carolina"),
    ("Stanly", "North Carolina"),
    ("Chesterfield", "South Carolina"),
]


def extract_lat_lon(geometry_value):
    """
    USGS returns geometry in one of two formats depending on the environment:
      - WKT: "POINT (-80.77 35.47)"
      - a GeoJSON-style list: [-80.77, 35.47]
    Both are longitude first, then latitude. This normalizes either one into
    plain (lat, lon) floats, so every downstream script (NOAA matching, address
    geocoding, etc.) works with simple lat/lon columns instead of a geometry string.
    """
    if pd.isna(geometry_value):
        return None, None

    text = str(geometry_value).strip()

    wkt_match = re.match(r"POINT\s*\(\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*\)", text, re.IGNORECASE)
    if wkt_match:
        lon, lat = float(wkt_match.group(1)), float(wkt_match.group(2))
        return lat, lon

    try:
        lon, lat = ast.literal_eval(text)
        return lat, lon
    except (ValueError, SyntaxError):
        return None, None


def get_monitoring_locations(counties=COUNTIES):
    """Fetch USGS stream monitoring locations for each (county, state) pair."""
    all_sites = []

    for county, state in counties:
        county_full = f"{county} County"
        print(f"Fetching sites for {county_full}, {state}...")
        try:
            sites, meta = waterdata.get_monitoring_locations(
                state_name=state,
                county_name=county_full,
                site_type="Stream",
                properties=[
                    "monitoring_location_id",
                    "monitoring_location_name",
                    "state_name",
                    "county_name",
                    "drainage_area",
                    "geometry",
                ],
            )
            if not sites.empty:
                all_sites.append(sites)
            print(f"  -> found {len(sites)} sites")
        except Exception as e:
            print(f"  -> error fetching {county_full}, {state}: {e}")

    if not all_sites:
        return pd.DataFrame()

    return pd.concat(all_sites, ignore_index=True)


def get_daily_values(site_id, parameter_code="00060", start_date=None, end_date=None):
    """Fetch daily values (default: discharge/streamflow) for a single site."""
    daily, meta = waterdata.get_daily(
        monitoring_location_id=site_id,
        parameter_code=parameter_code,
        time=(start_date, end_date) if start_date and end_date else None,
    )
    return daily


if __name__ == "__main__":
    sites = get_monitoring_locations()

    if not sites.empty and "geometry" in sites.columns:
        lat_lon = sites["geometry"].apply(extract_lat_lon)
        sites["latitude"] = lat_lon.apply(lambda t: t[0])
        sites["longitude"] = lat_lon.apply(lambda t: t[1])
        sites = sites.drop(columns=["geometry"])

    print(f"\nTotal sites across all counties: {len(sites)}")
    print(sites.head(20))

    output_path = os.path.join("resources", "usgs_monitoring_locations.csv")
    sites.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")