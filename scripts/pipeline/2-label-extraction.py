
import sys
import git

PROJECT_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(PROJECT_ROOT)

import pandas as pd
from tqdm import tqdm

from shared import directory_contents
from labeler import Labeler

INPUTS = f"{PROJECT_ROOT}/outputs/patches"
OUTPUTS = f"{PROJECT_ROOT}/outputs/labels.csv"

LABELER = Labeler(PROJECT_ROOT)


def save_labels(labels):
    labels = [{"label": l, "action": ""} for l in labels]
    data = pd.DataFrame.from_records(labels)
    data.to_csv(OUTPUTS, index=False)


def label_extraction():
    labels = []
    for subdir in tqdm(directory_contents(INPUTS)):
        for fname in tqdm(directory_contents(subdir)):
            new_labels = LABELER.labels(fname)
            labels += new_labels
    return labels


if __name__ == "__main__":
    print(f"[+] EXTRACTING LABELS...")
    labels = label_extraction()
    print(f"[+] SAVING LABELS...")
    save_labels(sorted(list(set(labels))))
    print("[+] DONE")
