
import sys
import git

PROJECT_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(PROJECT_ROOT)

import os

import numpy as np

from skimage.measure import compare_ssim, compare_nrmse, compare_mse
from skimage.color import rgb2gray
from skimage import io

INPUTS = f"{PROJECT_ROOT}/inputs"


def find_image_similarities():
    for dir, subdirs, filenames in os.walk(f"{INPUTS}/similarity/"):
        for subdir in subdirs:
            imgs = []
            for fname in os.listdir(f"{dir}{subdir}/"):
                if "diff" in fname or "skip" in fname:
                    continue
                print(fname)
                imgs.append(rgb2gray(io.imread(f"{dir}{subdir}/{fname}")))
            mssim, diff = compare_ssim(imgs[0], imgs[1], full=True)
            nrmse = compare_nrmse(imgs[0], imgs[1])
            mse = compare_mse(imgs[0], imgs[1])
            diff = np.array([[max(min(i, 1), -1) for i in l] for l in diff])
            print(f"MMSIM: {mssim}")
            print(f"NRMSE: {nrmse}")
            print(f"MSE: {mse}")
            io.imsave(f"{dir}{subdir}/diff.jpeg", diff)


if __name__ == "__main__":
    find_image_similarities()
    print("[+] DONE")
