
import sys
import git

PROJECT_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(PROJECT_ROOT)

import pandas as pd

from geopy.distance import geodesic
from pprint import pprint

NUMERIC_VARS = [
    "building_area",
    "year_alt_1",
    "year_alt_2",
    "lot_area",
    "n_floors",
]
OUTPUTS = f"{PROJECT_ROOT}/outputs"
CENTER = (40.765983, -73.977208)  # (LAT, LNG)
RADIUS = 1


def buildings_within_radius(data):
    return data[data.apply(lambda r: km_diff(r) <= RADIUS, axis=1)]


def km_diff(r):
    return geodesic(CENTER, (r.lat, r.lng)).km


def statistics(data):
    return {v: numeric_statistics(data, v) for v in NUMERIC_VARS}


def numeric_statistics(data, var):
    return {
        "min": data[var].min(),
        "mean": data[var].mean(),
        "median": data[var].median(),
        "max": data[var].max(),
        "std": data[var].std(),
    }


if __name__ == "__main__":
    data = pd.read_csv(f"{OUTPUTS}/data.csv")
    data = data[~data.lat.isnull()]
    print(f"[+] FINDING BUILDINGS IN AREA ({3.14 * RADIUS ** 2} KM2)...")
    data_subset = buildings_within_radius(data)
    print(f"[+] GETTING STATISTICS FOR BUILDINGS IN AREA...")
    stats = statistics(data_subset)
    pprint(stats)
    print("[+] DONE")
