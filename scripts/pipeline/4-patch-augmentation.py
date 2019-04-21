
import sys
import git

PROJECT_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(PROJECT_ROOT)

from PIL import Image, ImageEnhance
from random import randint
from tqdm import tqdm

from shared import directory_contents


INPUTS = f"{PROJECT_ROOT}/outputs/patches"
OUTPUTS = f"{PROJECT_ROOT}/outputs/patches"

N_RANDOMIZATIONS = 3


def select_function(method):
    if method == "brightness":
        return random_brightness
    elif method == "contrast":
        return random_contrast
    elif method == "rotation":
        return random_rotation


def save_new_patch(fname, image, method, i):
    base = fname.split(".jpeg")[0]
    select_function(method)(image).save(f"{base}_{method}_{i}.jpeg")


def random_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(randint(5, 15) / 10)


def random_contrast(image):
    level = randint(100, 200)
    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c):
        value = 128 + factor * (c - 128)
        return max(0, min(255, value))

    return image.point(contrast)


def random_rotation(image):
    return image.rotate(randint(0, 360))


def patch_augmentation():
    for subdir in tqdm(directory_contents(INPUTS)):
        for fname in tqdm(directory_contents(subdir)):
            image = Image.open(fname)
            for i in range(N_RANDOMIZATIONS):
                save_new_patch(fname, image.copy(), "brightness", i)
                save_new_patch(fname, image.copy(), "contrast", i)
                save_new_patch(fname, image.copy(), "rotation", i)


if __name__ == "__main__":
    print(f"[+] AUGMENTING PATCHES...")
    patch_augmentation()
    print("[+] DONE")
