"""
Restaurant Dataset Builder
==========================
Uses Google Places API (Nearby Search) with a grid-based approach
to collect restaurant coordinates across an entire city.

Outputs a deduplicated CSV with: place_id, name, latitude, longitude

Usage:
    1. Set your API key below (or as env var GOOGLE_MAPS_API_KEY)
    2. Set your city center coordinates and grid parameters
    3. Run:  python restaurant_scraper.py
"""

import requests
import csv
import time
import os
import itertools

# ──────────────────────────────────────────────
# CONFIGURATION — Edit these values
# ──────────────────────────────────────────────

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "YOUR_API_KEY_HERE")

# City center coordinates (default: Zurich, Switzerland)
CITY_CENTER_LAT = 47.3769
CITY_CENTER_LNG = 8.5417

# Grid settings
GRID_STEPS = 5          # Number of grid points per axis (5 = 5x5 = 25 search circles)
GRID_SPACING = 0.01     # Degrees between grid points (~1.1 km)
SEARCH_RADIUS = 1500    # Meters — radius for each Nearby Search circle

# Output file
OUTPUT_FILE = "restaurants.csv"

# ──────────────────────────────────────────────
# STEP 1: Generate grid points
# ──────────────────────────────────────────────

def generate_grid(center_lat, center_lng, steps, spacing):
    """Create a grid of (lat, lng) points centered on the city."""
    half = steps // 2
    offsets = [i * spacing for i in range(-half, half + 1)]
    grid = [
        (center_lat + dlat, center_lng + dlng)
        for dlat, dlng in itertools.product(offsets, offsets)
    ]
    print(f"[Grid] Generated {len(grid)} search points "
          f"covering ~{steps * spacing * 111:.1f} km x {steps * spacing * 111:.1f} km")
    return grid


# ──────────────────────────────────────────────
# STEP 2: Fetch restaurants for a single point
# ──────────────────────────────────────────────

def fetch_restaurants_at(lat, lng, radius, api_key):
    """
    Fetch all restaurants near a point using Google Places Nearby Search.
    Handles pagination (up to 3 pages / 60 results per point).
    Returns a list of dicts with place_id, name, lat, lng.
    """
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "type": "restaurant",
        "key": api_key,
    }

    results = []
    page = 1

    while True:
        response = requests.get(url, params=params)
        data = response.json()

        status = data.get("status")
        if status not in ("OK", "ZERO_RESULTS"):
            print(f"  [Warning] API returned status: {status} — "
                  f"{data.get('error_message', 'no details')}")
            break

        for place in data.get("results", []):
            loc = place["geometry"]["location"]
            results.append({
                "place_id": place["place_id"],
                "name": place["name"],
                "latitude": loc["lat"],
                "longitude": loc["lng"],
            })

        next_token = data.get("next_page_token")
        if not next_token:
            break

        # Google requires a short delay before the next_page_token becomes valid
        time.sleep(2)
        params = {"pagetoken": next_token, "key": api_key}
        page += 1

    return results


# ──────────────────────────────────────────────
# STEP 3: Grid search — fetch across all points
# ──────────────────────────────────────────────

def grid_search(grid_points, radius, api_key):
    """Run Nearby Search for every grid point, collecting raw results."""
    all_results = []
    total = len(grid_points)

    for i, (lat, lng) in enumerate(grid_points, 1):
        print(f"[Search] Point {i}/{total}  ({lat:.4f}, {lng:.4f})", end="")
        batch = fetch_restaurants_at(lat, lng, radius, api_key)
        print(f"  → {len(batch)} results")
        all_results.extend(batch)

        # Small delay between grid points to stay within rate limits
        if i < total:
            time.sleep(0.3)

    print(f"[Search] Total raw results: {len(all_results)}")
    return all_results


# ──────────────────────────────────────────────
# STEP 4: Deduplicate and combine
# ──────────────────────────────────────────────

def deduplicate(results):
    """Remove duplicate restaurants using place_id as the unique key."""
    seen = {}
    for r in results:
        pid = r["place_id"]
        if pid not in seen:
            seen[pid] = r
    unique = list(seen.values())
    print(f"[Dedup] {len(results)} raw → {len(unique)} unique restaurants")
    return unique


# ──────────────────────────────────────────────
# STEP 5: Save to CSV
# ──────────────────────────────────────────────

def save_csv(restaurants, filename):
    """Write the final dataset to a CSV file."""
    fieldnames = ["place_id", "name", "latitude", "longitude"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(restaurants)
    print(f"[Save] Written to {filename}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Set your Google Maps API key first!")
        print("  Option 1: Edit API_KEY in this script")
        print("  Option 2: export GOOGLE_MAPS_API_KEY='your-key'")
        return

    print("=" * 55)
    print("  Restaurant Dataset Builder")
    print("=" * 55)
    print(f"  City center : {CITY_CENTER_LAT}, {CITY_CENTER_LNG}")
    print(f"  Grid        : {GRID_STEPS}x{GRID_STEPS} points, {GRID_SPACING}° spacing")
    print(f"  Radius      : {SEARCH_RADIUS}m per point")
    print(f"  Output      : {OUTPUT_FILE}")
    print("=" * 55)

    # Step 1: Generate grid
    grid = generate_grid(CITY_CENTER_LAT, CITY_CENTER_LNG, GRID_STEPS, GRID_SPACING)

    # Step 2-3: Search all grid points
    raw_results = grid_search(grid, SEARCH_RADIUS, API_KEY)

    # Step 4: Deduplicate
    unique = deduplicate(raw_results)

    # Step 5: Save
    save_csv(unique, OUTPUT_FILE)

    print(f"\nDone! {len(unique)} restaurants saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()