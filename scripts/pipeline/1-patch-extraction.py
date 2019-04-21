
import sys
import git

PROJECT_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(PROJECT_ROOT)

import os
import time
import subprocess

from PIL import Image

INPUTS = f"{PROJECT_ROOT}/outputs/images"
OUTPUTS = f"{PROJECT_ROOT}/outputs/patches"

PARAMS = [
    # {
    #     "size": {"name": "small", "width": 50, "height": 50},
    #     "skip": {"width": 50, "height": 50},
    #     "resize": {"width": 50, "height": 50},
    # },
    {
        "size": {"name": "large", "width": 100, "height": 100},
        "skip": {"width": 100, "height": 100},
        "resize": {"width": 50, "height": 50},
    }
]


def remove_previous_results():
    subprocess.Popen(["rm", "-fr", OUTPUTS])
    time.sleep(3)
    subprocess.Popen(["mkdir", OUTPUTS])
    time.sleep(3)


def file_names_in_directory(directory):
    return [f"{directory}/{x}" for x in os.listdir(directory)]


def building_images():
    file_names = file_names_in_directory(INPUTS)
    for i, fname in enumerate(file_names):
        yield i, len(file_names), fname, Image.open(fname)


def patch_extraction():
    for i, n, fname, image in building_images():
        print(f"- [{i}/{n}] Image: {fname}")
        subdir_created = False
        for params in PARAMS:
            if not subdir_created:
                create_subdirectory(fname)
                subdir_created = True
            extract_and_save_patches(fname, image, params)


def subdirectory(fname):
    return fname.replace("images", "patches").replace(".jpeg", "")


def create_subdirectory(fname):
    subprocess.Popen(["mkdir", subdirectory(fname)])


def extract_and_save_patches(fname, image, params):
    size_name = params["size"]["name"]
    rh = params["resize"]["height"]
    rw = params["resize"]["width"]
    subdir = subdirectory(fname)
    image_height = image.height
    image_width = image.width
    y1 = 0
    y2 = y1 + params["size"]["height"]
    while y2 <= image_height:
        x1 = 0
        x2 = x1 + params["size"]["width"]
        while x2 <= image_width:
            patch = image.crop((x1, y1, x2, y2)).resize((rw, rh))
            try:
                patch.save(f"{subdir}/{size_name}_{x1}_{y1}_{x2}_{y2}.jpeg")
            except OSError:
                time.sleep(2)
                patch.save(f"{subdir}/{size_name}_{x1}_{y1}_{x2}_{y2}.jpeg")
            x1 += params["skip"]["width"]
            x2 = x1 + params["size"]["width"]
        y1 += params["skip"]["height"]
        y2 = y1 + params["size"]["height"]


if __name__ == "__main__":
    print(f"[+] REMOVING PREVIOUS RESULTS...")
    remove_previous_results()
    print(f"[+] EXTRACTING PATCHES...")
    patch_extraction()
    print("[+] DONE")
