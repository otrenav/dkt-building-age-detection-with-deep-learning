import sys
import git

PROJECT_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(PROJECT_ROOT)

import tqdm
import numpy as np
import pandas as pd

from PIL import Image

from shared import directory_contents

INPUTS = f"{PROJECT_ROOT}/outputs"
OUTPUTS = f"{PROJECT_ROOT}/outputs"

WIDTH = 20
HEIGHT = 20


def downsample(image, width, height):
    return image.resize((width, height))


def grayscale(image):
    return image.convert("L")


def numeric_data_from_image(image_fname):
    image = Image.open(image_fname)
    image = downsample(image, WIDTH, HEIGHT)
    image = grayscale(image)
    return np.array(image).flatten()


def prepare_data():
    new_data = pd.DataFrame()
    data = pd.read_csv(f"{INPUTS}/data.csv")
    pixels = list(range(1, WIDTH * HEIGHT + 1))
    for i in tqdm.tqdm(range(data.shape[0])):
        row = data.iloc[i]
        try:
            contents = directory_contents(f"{INPUTS}/patches/{row['id']}")
        except FileNotFoundError:
            print(f"Directory not found: {INPUTS}/patches/{row['id']}")
            continue
        for img_fname in contents:
            numeric_data = numeric_data_from_image(img_fname)
            row_copy = row.append(pd.Series(numeric_data, index=pixels))
            new_data = new_data.append(row_copy, ignore_index=True)
        if i % 10 == 0:
            new_data.to_csv(f"{OUTPUTS}/data_with_pixels.csv", index=False)


if __name__ == "__main__":
    print(f"[+] PREPARING DATA...")
    prepare_data()
    print("[+] DONE")
