"""Point the index files at a local audio folder.

The index files (dataset_*.csv in the repository root and their copies in model/) store the paths used on
Google Colab, e.g. /content/drive/MyDrive/NewProject/codes/experiments/KAMPA/data_test/construction/x.wav.
The training and evaluation code opens these paths as written, so outside that Colab layout run

    python scripts/set_data_root.py                  # audio under this repository: <repo>/data_test/...
    python scripts/set_data_root.py --root /data/kampa

to replace the prefix in place. `git checkout -- dataset_*.csv model/dataset_*.csv` restores the originals.
Run it before training: the data lists are part of each run's identity, so checkpoints trained with one
prefix are retrained after changing it (evaluating a released checkpoint with model/eval.py is not affected).
"""
import argparse
import glob
import os

COLAB_PREFIX = "/content/drive/MyDrive/NewProject/codes/experiments/KAMPA/"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=REPO, help="folder that holds data_labeled/, data_unlabeled/, data_val/, data_test/ (default: this repository)")
    ap.add_argument("--old", default=COLAB_PREFIX, help="prefix to replace (default: the Colab prefix)")
    args = ap.parse_args()
    new = os.path.abspath(args.root).replace("\\", "/").rstrip("/") + "/"
    files = sorted(glob.glob(os.path.join(REPO, "dataset_*.csv")) + glob.glob(os.path.join(REPO, "model", "dataset_*.csv")))
    total = 0
    for path in files:
        with open(path, encoding="utf-8", newline="") as handle:
            text = handle.read()
        n = text.count(args.old)
        if n:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(text.replace(args.old, new))
        total += n
        print(f"{os.path.relpath(path, REPO)}: {n} paths")
    if not total:
        raise SystemExit(f"[STOP] no path starts with {args.old}")
    print(f"[OK] {total} paths now start with {new}")


if __name__ == "__main__":
    main()
