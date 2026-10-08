"""Problem 5: the most confident mistakes of saved models.

Usage:
    uv run python src/problem5_mistakes.py --tags clean run --k 10

For each tag, runs worst_examples on that run's own test set (the paths saved
in results/<tag>_settings.json) and on data/new_images, and prints the mistakes
with the model's answer and its softmax probability.

The test paths were saved when the model was trained. Images removed by
clean.py since then are no longer in data/clean, so every image is read from
data/raw, which never changes, and marked if it was removed later.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import prepare_image
from evaluate import worst_examples

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
RAW = ROOT / "data" / "raw"
NEW = ROOT / "data" / "new_images"
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load(paths):
    X = np.zeros((len(paths), 3, 224, 224), dtype=np.float32)
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            X[i] = prepare_image(image)
    return X


def show(title, mistakes, class_names, label):
    print(title)
    if not mistakes:
        print("  no mistakes")
    for rank, m in enumerate(mistakes, start=1):
        print(f"  {rank:2d}. {label(m['path']):42s} said {class_names[m['predicted']]:18s} "
              f"{m['confidence']:6.1%}   really {class_names[m['true']]}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=["clean", "run"])
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    for tag in args.tags:
        settings = json.loads((ROOT / "results" / f"{tag}_settings.json").read_text())
        class_names = settings["class_names"]
        model = torch.load(ROOT / "results" / f"{tag}_model.pt",
                           map_location="cpu", weights_only=False).eval()
        print(f"=== model results/{tag}_model.pt, trained on {settings['n_images']} images\n")

        # its own test set, read from data/raw
        names = [Path(p) for p in settings["test_paths"]]
        raw = [RAW / p.parent.name / p.name for p in names]
        y = [class_names.index(p.parent.name) for p in names]
        mistakes = worst_examples(model, load(raw), y, raw, k=args.k)

        def test_label(path):
            name = f"{path.parent.name}/{path.name}"
            return name + ("" if (CLEAN / name).exists() else "  (removed later)")

        show(f"test set ({len(raw)} images), most confident mistakes:",
             mistakes, class_names, test_label)

        # the new images
        new, y_new = [], []
        for c, name in enumerate(class_names):
            for p in sorted((NEW / name).iterdir()):
                if p.suffix.lower() in EXTENSIONS:
                    new.append(p)
                    y_new.append(c)
        mistakes = worst_examples(model, load(new), y_new, new, k=args.k)
        show(f"new images ({len(new)} images), most confident mistakes:",
             mistakes, class_names, lambda p: f"new_images/{p.parent.name}/{p.name}")


if __name__ == "__main__":
    main()
