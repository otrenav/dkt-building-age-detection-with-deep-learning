
import sys
import git

PROJECT_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(PROJECT_ROOT)

import os
import json
import requests
import subprocess

import pandas as pd

from PIL import Image
from tqdm import tqdm
from pprint import pprint

# N_SAMPLES = 10
N_SAMPLES = 10000
# N_SAMPLES = 25000
INPUTS = f"{PROJECT_ROOT}/inputs"
OUTPUTS = f"{PROJECT_ROOT}/outputs"
META_DATA = ["dataset", "country", "state", "municipality"]
ZERO_TO_NAN = ["year", "year_alt_1", "year_alt_2", "lot_area", "building_area"]

META_DATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata?"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview?"
KEY = "AIzaSyCdYwmVKMEYztqPmsP26Pqc3aV1C5mzij8"
IMAGE_BOX = (0, 0, 500, 500)
SIZE = "500x600"
PITCH = 0
FOV = 50

pitch = f"pitch={PITCH}"
size = f"size={SIZE}"
fov = f"fov={FOV}"
key = f"key={KEY}"

META_DATA_REQUEST = f"{META_DATA_URL}&{size}&{pitch}&{fov}&{key}"
IMAGE_REQUEST = f"{IMAGE_URL}&{size}&{pitch}&{fov}&{key}"


def remove_previous_results():
    subprocess.Popen(["rm", f"{OUTPUTS}/data.csv"])
    subprocess.Popen(["rm", "-r", f"{OUTPUTS}/images/"])
    subprocess.Popen(["mkdir", f"{OUTPUTS}/images/"])


def standarize_variable_names(data, dataset):
    data = data.rename(columns=dataset["vars"])
    return data


def keep_required_variables(data, dataset):
    data = data[list(dataset["vars"].values())]
    return data


def add_metadata(data, dataset):
    for v in META_DATA:
        data[v] = dataset.get(v)
    return data


def codify_nans(data):
    for v in ZERO_TO_NAN:
        data.loc[data[v] == 0, v] = None
    return data


def complete_cases(data):
    data = data[~data.year.isnull()]
    data = data[~data.address.isnull()]
    return data


def load(fname, dataset):
    data = pd.read_csv(fname)
    data = standarize_variable_names(data, dataset)
    data = keep_required_variables(data, dataset)
    data = add_metadata(data, dataset)
    data = codify_nans(data)
    data = complete_cases(data)
    return data


def ingest():
    data = pd.DataFrame()
    for dir, subdirs, filenames in os.walk(f"{INPUTS}/datasets/"):
        for subdir in sorted(subdirs):
            with open(f"{dir}{subdir}/metadata.json", "r") as f:
                metadata = json.load(f)
            for dataset in metadata["datasets"]:
                if dataset.get("ignore_reason") is None:
                    fname = f"{dir}{subdir}/{dataset['file']}"
                    print(f"    - {fname}")
                    new_data = load(fname, dataset)
                    data = data.append(new_data)
    return data


def random_sample(data, n_samples):
    return data.sample(n=n_samples)


def reset_index_name(data):
    data = data.reset_index().iloc[:, 1:]
    data.index = data.index.set_names(["id"])
    return data


def coordinates_and_images(data):
    for r in tqdm(list(data.index)):
        location = row_location(r, data)
        resp = requests.get(f"{META_DATA_REQUEST}&{location}").json()
        if resp["status"] != "OK":
            data.loc[r, "pano_error"] = resp["status"]
            print(data.loc[r,])
            pprint(resp)
            continue
        data = update_with_pano_metadata(r, data, resp)
        fname = f"{OUTPUTS}/images/{r}.jpeg"
        pano = f"pano={resp['pano_id']}"
        with open(fname, "wb") as f:
            f.write(requests.get(f"{IMAGE_REQUEST}&{pano}").content)
        crop_image_to_remove_logo(fname)
    return data


def row_location(row, data):
    location = f"location={data.loc[row, 'address']}, "
    location += f"{data.loc[row, 'municipality']}, "
    location += f"{data.loc[row, 'state']}, "
    location += f"{data.loc[row, 'country']}"
    location = location.replace(" ", "%20")
    return location


def update_with_pano_metadata(row, data, resp):
    data.loc[row, "pano_date"] = resp["date"]
    data.loc[row, "pano_id"] = resp["pano_id"]
    data.loc[row, "lat"] = resp["location"]["lat"]
    data.loc[row, "lng"] = resp["location"]["lng"]
    return data


def crop_image_to_remove_logo(fname):
    Image.open(fname).crop(IMAGE_BOX).save(fname)


def remove_duplicate_images(data):
    dups = data.duplicated("pano_id")
    for i in list(data.index[dups]):
        subprocess.Popen(["rm", f"{OUTPUTS}/images/{i}.jpeg"])
    return data[~dups]


if __name__ == "__main__":
    remove_previous_results()
    print(f"[+] INGESTING...")
    data = ingest()
    print(f"[+] RANDOM SAMPLE (N_OBS: {N_SAMPLES})...")
    data = random_sample(data, N_SAMPLES)
    print(f"[+] RESETING INDEX NAME...")
    data = reset_index_name(data)
    print(f"[+] GETTING COORDNINATES AND IMAGES...")
    data = coordinates_and_images(data)
    print(f"[+] REMOVING DUPLICATED PANOS...")
    data = remove_duplicate_images(data)
    print(data.head())
    print(f"[+] STORING UNIFIED DATA...")
    data.to_csv(f"{OUTPUTS}/data.csv")
    print("[+] DONE")
