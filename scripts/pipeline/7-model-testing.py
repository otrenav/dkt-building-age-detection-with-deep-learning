import sys
import git

PROJECT_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(PROJECT_ROOT)


INPUTS = f"{PROJECT_ROOT}/outputs/patches"
OUTPUTS = f"{PROJECT_ROOT}/outputs/patches"


def test_models():
    pass


if __name__ == "__main__":
    print(f"[+] TRAINING MODELS...")
    test_models()
    print("[+] DONE")
