import sys
import git

PROJECT_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(PROJECT_ROOT)

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

OUTPUTS = f"{PROJECT_ROOT}/outputs"

SAMPLE_PROPORTION = 0.7
CROSS_VALIDATION = 3
CATEGORY_BINS = []
N_CATEGORIES = 2
N_TREES = 20
VERBOSE = 1
N_JOBS = 4

OUTPUT_VAR = "year"
INPUT_VARS = [
    "municipality",
    "building_area",
    "year_alt_1",
    "year_alt_2",
    "lot_area",
    "n_floors",
    "lat",
    "lng",
    "zip",
]
MISSINGS_RECODING = {
    "building_area": -1000,
    "year_alt_1": -1000,
    "year_alt_2": -1000,
    "lot_area": -1000,
    "n_floors": -1000,
    "zip": -1000,
}
CATEGORICALS_RECODING = {
    "municipality": {
        "State Island": 1,
        "Manhattan": 2,
        "Brooklyn": 3,
        "Queens": 4,
        "Bronx": 5,
    }
}
GRID = {
    "max_depth": [None, 5, 10, 20, 40, 80],
    "max_features": [2, 4, 6, 8],
    "bootstrap": [True, False],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
}
DECADES = {}


def recode_missing_values(data):
    for k, v in MISSINGS_RECODING.items():
        data.loc[data[k].isnull(), k] = v
    return data


def identify_categories(data):
    if N_CATEGORIES is None:
        identify_decade_categories(data)
    else:
        identify_custom_categories(data)


def identify_decade_categories(data):
    decades = data.year.apply(decade)
    decades = sorted(list(set(decades)))
    globals()["DECADES"] = {d: i for i, d in enumerate(decades)}


def decade(num):
    return int(float(str(float(num))[:3] + "0"))


def identify_custom_categories(data):
    _, bins = pd.qcut(
        data.year, N_CATEGORIES, labels=False, retbins=True, duplicates="drop"
    )
    globals()["CATEGORY_BINS"] = bins


def recode_categorical_values(data):
    data = data.replace(CATEGORICALS_RECODING)
    data[OUTPUT_VAR] = data[OUTPUT_VAR].apply(category)
    return data


def categories(list_of_nums):
    return [category(num) for num in list_of_nums]


def category(num):
    if N_CATEGORIES is None:
        return DECADES.get(decade(num), -1)
    return custom_category(num)


def custom_category(num):
    for i in range(1, len(CATEGORY_BINS)):
        if CATEGORY_BINS[i - 1] < num <= CATEGORY_BINS[i]:
            return i - 1
    return -1


def split_train_validate_samples(data):
    select = np.random.rand(data.shape[0]) < SAMPLE_PROPORTION
    return data[select], data[~select]


def train_and_find_best_predictor(data):
    # classifier = RandomForestRegressor(n_estimators=N_TREES)
    classifier = RandomForestClassifier(n_estimators=N_TREES)
    search = GridSearchCV(
        classifier,
        cv=CROSS_VALIDATION,
        param_grid=GRID,
        verbose=VERBOSE,
        n_jobs=N_JOBS,
    )
    search.fit(data[INPUT_VARS], data[OUTPUT_VAR])
    return search.best_estimator_


def build_confusion_matrix(predictor, data):
    yp = list(predictor.predict(data[INPUT_VARS]))
    y = list(data[OUTPUT_VAR])
    cm = confusion_matrix(y, yp)
    print(cm)
    return cm


def save_confusion_matrix_graph(cm, data):
    classes = sorted(list(set(categories(data.year.unique()))))
    cmap = plt.cm.Blues
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(f"{OUTPUTS}/confusion_matrix.png")


def evaluate_confusion_matrix(cm, data):
    c = 0
    t = data.shape[0]
    for i, _ in enumerate(cm):
        c += cm[i][i]
    print(f"    - CORRECT PREDICTIONS: {c}/{t} ({c/t})")


if __name__ == "__main__":
    data_ = pd.read_csv(f"{OUTPUTS}/data.csv")
    data_ = data_[~data_.lat.isnull()]
    print(f"[+] IDENTIFYING CATEGORIES...")
    identify_categories(data_)
    print(f"[+] RECODING MISSING VALUES...")
    data_ = recode_missing_values(data_)
    print(f"[+] RECODING CATEGORICAL VALUES...")
    data_ = recode_categorical_values(data_)
    print(f"[+] SPLITTING INTO TRAININGN AND VALIDATION SAMPLES...")
    data_train, data_validate = split_train_validate_samples(data_)
    print(f"[+] TRAINING WITH GRID SEARCH CV FOR RANDOM FORESTS...")
    best_predictor = train_and_find_best_predictor(data_train)
    print(f"[+] FORMING CONFUSION MATRIX...")
    cm_ = build_confusion_matrix(best_predictor, data_validate)
    print(f"[+] GRAPHING CONFUSION MATRIX...")
    save_confusion_matrix_graph(cm_, data_validate)
    print(f"[+] EVALUATING PREDCITOR...")
    evaluate_confusion_matrix(cm_, data_validate)
    print("[+] DONE")
