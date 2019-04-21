
import sys
import git

PROJECT_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(PROJECT_ROOT)

import logging
import subprocess

import pandas as pd

from statistics import mean, variance
from tqdm import tqdm
from PIL import Image

from shared import directory_contents
from labeler import Labeler

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

INPUTS = f"{PROJECT_ROOT}/outputs"
OUTPUTS = f"{PROJECT_ROOT}/outputs/patches"

LABELS = pd.read_csv(f"{INPUTS}/labels.csv")
LABELER = Labeler(PROJECT_ROOT)

KEEP_TOP_N = 10


def delete_patch(fname):
    subprocess.Popen(["rm", fname])


def delete_patches_using_labels():
    for subdir in tqdm(directory_contents(INPUTS)):
        for fname in tqdm(directory_contents(subdir)):
            label = LABELER.labels(fname, top_n_labels=1)[0]
            if label in LABELS.label.unique():
                if LABELS.loc[LABELS.label == label, "action"] == "delete":
                    delete_patch(fname)
            else:
                LOGGER.warning("WARNING: Previously unseen label: {label}")


def select_patches_using_contrast():
    #
    # TODO: Before using contrast, kMeans with normalized SIFT, and from
    #       within each cluster, select top n images (using contrast)
    #
    for subdir in tqdm(directory_contents(f"{INPUTS}/patches")):
        data = pd.DataFrame()
        for fname in tqdm(directory_contents(subdir)):
            score = rgb_score(rgb_histograms(fname))
            data = data.append(new_row(fname, score), ignore_index=True)
        data = keep_top_n(data)
        for fname in directory_contents(subdir):
            if fname not in list(data["image"]):
                delete_patch(fname)


def rgb_histograms(fname):
    image = Image.open(fname)
    histogram = image.histogram()
    red = histogram[0:256]
    green = histogram[256:512]
    blue = histogram[512:769]
    return red, green, blue


def rgb_score(red, green, blue):
    return mean([variance(red), variance(green), variance(blue)])


def new_row(fname, score):
    return {"image": fname, "score": score}


def keep_top_n(data):
    return data.sort_values(by=["score"], ascending=False).iloc[:KEEP_TOP_N]


if __name__ == "__main__":
    print(f"[+] DELETING PATCHES USING LABELS...")
    delete_patches_using_labels()
    print(f"[+] SELECTING PATCHES USING CONTRAST...")
    select_patches_using_contrast()
    print("[+] DONE")
