"""Problem 7: what a saved model says about original and changed images.

Usage:
    uv run python src/problem7_predict.py --tag clean \
        --folders data/new_images/back_double_biceps data/changed/bdb_arms_covered

For every image name found in the first folder, prints the model's softmax
probability for each class on that image in every folder, prepared with
prepare_image exactly as in training. At the end, one line per folder with the
mean probability of each class over the images in it, and per folder how many
images got a different answer than the same image in the first folder.

    --leave-out data/changed/some_folder/IMG_1.PNG
        prints that image's answers but leaves it out of its folder's means
    --also-without IMG_1.PNG
        also prints the means over the images without that name, in every folder
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import prepare_image
from evaluate import predict_logits

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="clean")
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--leave-out", nargs="*", default=[],
                        help="folder/image paths to leave out of the means")
    parser.add_argument("--also-without", nargs="*", default=[],
                        help="image names; also print the means without them")
    args = parser.parse_args()

    settings = json.loads((ROOT / "results" / f"{args.tag}_settings.json").read_text())
    class_names = settings["class_names"]
    model = torch.load(ROOT / "results" / f"{args.tag}_model.pt",
                       map_location="cpu", weights_only=False).eval()

    folders = [Path(f) for f in args.folders]
    left_out = {Path(p) for p in args.leave_out}
    names = sorted(p.name for p in folders[0].iterdir()
                   if p.suffix.lower() in EXTENSIONS)

    print(f"model: results/{args.tag}_model.pt")
    print("folders:")
    for folder in folders:
        print(f"  {folder}")
    if left_out:
        print("left out of the means (answers still printed, marked *):")
        for path in sorted(left_out):
            print(f"  {path}")
    print()

    width = max(len(str(f)) for f in folders)
    header = f"  {'folder':{width}s}  " + "  ".join(f"{c:>18s}" for c in class_names) + "  answer"
    collected = {folder: [] for folder in folders}
    answers = {folder: {} for folder in folders}
    for name in names:
        print(name)
        print(header)
        for folder in folders:
            path = folder / name
            if not path.exists():
                print(f"  {str(folder):{width}s}  missing")
                continue
            with Image.open(path) as image:
                x = prepare_image(image)[None]
            logits = predict_logits(model, x)[0].astype(np.float64)
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            mark = " *" if path in left_out else ""
            answers[folder][name] = int(probs.argmax())
            if not mark:
                collected[folder].append((name, probs))
            cells = "  ".join(f"{p:18.1%}" for p in probs)
            print(f"  {str(folder):{width}s}  {cells}  {class_names[int(probs.argmax())]}{mark}")
        print()

    print_means("mean probability over the images in each folder"
                + (" (without the ones marked *)" if left_out else ""),
                collected, class_names, width, set())
    print()
    print_changes(answers, folders, width, left_out)
    if args.also_without:
        print()
        print_means("mean probability without " + ", ".join(args.also_without),
                    collected, class_names, width, set(args.also_without))


def print_changes(answers, folders, width, left_out):
    first = answers[folders[0]]
    print(f"images whose answer differs from the answer on the same image in {folders[0]}")
    print(f"  {'folder':{width}s}  changed  of")
    for folder in folders[1:]:
        names = [n for n in answers[folder] if n in first]
        changed = [n for n in names if answers[folder][n] != first[n]]
        marked = sum(folder / n in left_out for n in names)
        note = f"   (includes {marked} marked *)" if marked else ""
        print(f"  {str(folder):{width}s}  {len(changed):7d}  {len(names):2d}{note}"
              + (f"   {', '.join(changed)}" if changed else ""))


def print_means(title, collected, class_names, width, without):
    print(title)
    print(f"  {'folder':{width}s}  " + "  ".join(f"{c:>18s}" for c in class_names) + "  images")
    for folder, rows in collected.items():
        rows = [probs for name, probs in rows if name not in without]
        if not rows:
            print(f"  {str(folder):{width}s}  no images")
            continue
        cells = "  ".join(f"{p:18.1%}" for p in np.mean(rows, axis=0))
        print(f"  {str(folder):{width}s}  {cells}  {len(rows):6d}")


if __name__ == "__main__":
    main()
