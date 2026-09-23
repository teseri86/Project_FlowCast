"""
Find the nearest NOAA GHCND weather station to each USGS site and pull
daily precipitation data for it.

Reads credentials from a .env file (see .env.example at the repo root).
Run from the repo root with: python -m src.flowcast.ingestion.pull_noaa
"""

import os
import time
from math import radians, sin, cos, sqrt, atan2

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

NOAA_TOKEN = os.environ["NOAA_TOKEN"]
BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
HEADERS = {"token": NOAA_TOKEN}


def haversine_miles(lat1, lon1, lat2, lon2):
    """you feed it two GPS coordinates, and it hands back "how many miles apart these two points actually are on Earth's curved surface"""

    R = 3958.8 # Earth radius in miles

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2]) # converts from degrees to radians
    dlat, dlon = lat2 - lat1, lon2 - lon1 # how far apart the points are in radians
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2 # haversine formula. how far apart two points are on a curved surface.
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def find_nearest_station(lat, lon, radius_deg=0.5):
    """Search a bounding box around (lat, lon) for GHCND stations reporting precipitation."""
    extent = f"{lat - radius_deg},{lon - radius_deg},{lat + radius_deg},{lon + radius_deg}"
    resp = requests.get(
        f"{BASE_URL}/stations",
        headers=HEADERS,
        params={"datasetid": "GHCND", "extent": extent, "limit": 50, "datatypeid": "PRCP"},
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return None

    closest = min(results, key=lambda s: haversine_miles(lat, lon, s["latitude"], s["longitude"]))
    closest["distance_miles"] = haversine_miles(lat, lon, closest["latitude"], closest["longitude"])
    return closest


def get_precipitation(station_id, start_date, end_date):
    """Pull daily precipitation (PRCP, metric units) for one station between two dates."""
    resp = requests.get(
        f"{BASE_URL}/data",
        headers=HEADERS,
        params={
            "datasetid": "GHCND",
            "stationid": station_id,
            "datatypeid": "PRCP",
            "startdate": start_date,
            "enddate": end_date,
            "units": "metric",
            "limit": 1000,
        },
    )
    resp.raise_for_status()
    return pd.DataFrame(resp.json().get("results", []))


def pull_precipitation_for_sites(sites_df, start_date="2023-01-01", end_date="2024-01-01"):
    """For each USGS site (with latitude/longitude columns), find the nearest NOAA station and pull precipitation."""
    precip_frames = []

    for _, row in sites_df.iterrows():
        lat, lon = row.get("latitude"), row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue

        station = find_nearest_station(lat, lon)
        if station is None:
            print(f"No NOAA station found near {row['monitoring_location_id']}")
            continue

        print(f"{row['monitoring_location_id']} -> {station['name']} ({station['distance_miles']:.1f} mi away)")

        precip = get_precipitation(station["id"], start_date, end_date)
        if not precip.empty:
            precip["monitoring_location_id"] = row["monitoring_location_id"]
            precip["noaa_station_id"] = station["id"]
            precip["distance_miles"] = station["distance_miles"]
            precip_frames.append(precip)

        time.sleep(0.5)  # stay under NOAA's rate limits

    return pd.concat(precip_frames, ignore_index=True) if precip_frames else pd.DataFrame()


if __name__ == "__main__":
    sites = pd.read_csv(os.path.join("resources", "usgs_monitoring_locations.csv"))
    precip_data = pull_precipitation_for_sites(sites)

    output_path = os.path.join("resources", "noaa_precipitation_by_site.csv")
    precip_data.to_csv(output_path, index=False)
    print(f"\nSaved {len(precip_data)} precipitation records to {output_path}")