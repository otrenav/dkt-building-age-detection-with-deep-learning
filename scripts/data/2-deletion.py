import sys
import git

PROJECT_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(PROJECT_ROOT)

import os

import pandas as pd

OUTPUTS = f"{PROJECT_ROOT}/outputs"
IMAGES = f"{OUTPUTS}/images/"


def delete_observations_without_images(data):
    images = existing_images_indexes()
    select = data.index.isin(images)
    return data[select]


def existing_images_indexes():
    return [int(i.replace(".jpeg", "")) for i in os.listdir(IMAGES)]


def delete_observations_with_errors(data):
    return data[~data.lat.isnull()]


if __name__ == "__main__":
    data = pd.read_csv(f"{OUTPUTS}/data.csv")
    n_start = data.shape[0]
    print(f"[+] DELETING OBSERVATIONS WITH IMAGE RETRIEVAL ERRORS...")
    data = delete_observations_with_errors(data)
    print(f"[+] DELETING OBSERVATIONS WITHOUT CORRESPONDING IMAGES...")
    data = delete_observations_without_images(data)
    n_end = data.shape[0]
    print(f"    - Kept {n_end}/{n_start} observations ({n_end/n_start})")
    print("[+] DONE")
    data.to_csv(f"{OUTPUTS}/data.csv", index=False)
