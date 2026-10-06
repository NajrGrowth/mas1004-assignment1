"""Problem 7: how big the grey blocks of guess 1 and guess 2 are.

Usage:
    uv run python src/problem7_block_sizes.py

The block positions below are the ones used to make data/changed/bdb_arms_covered
(guess 1) and data/changed/bdb_upperback_covered (guess 2), as fractions of the
image width and height. Each block was filled with PIL's ImageDraw.rectangle on
the rounded pixel corners, which paints both edges, so a block from x0 to x1 is
x1 - x0 + 1 pixels wide.

As a check, for PNG images it also counts the pixels that really differ between
the original and the changed copy. JPEG copies were saved again with lossy
compression, so every pixel differs a little and that check is skipped.
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ORIGINALS = ROOT / "data" / "new_images" / "back_double_biceps"
ARMS_FOLDER = ROOT / "data" / "changed" / "bdb_arms_covered"
BACK_FOLDER = ROOT / "data" / "changed" / "bdb_upperback_covered"

# guess 1: left arm block, right arm block, as (x0, y0, x1, y1)
ARMS = {
    "IMG_0512.jpeg": ((0.30, 0.25, 0.49, 0.39), (0.75, 0.25, 0.90, 0.39)),
    "IMG_0513.PNG":  ((0.19, 0.29, 0.36, 0.40), (0.53, 0.29, 0.75, 0.40)),
    "IMG_1481.PNG":  ((0.05, 0.41, 0.37, 0.58), (0.63, 0.41, 0.91, 0.58)),
    "IMG_2234.PNG":  ((0.09, 0.31, 0.40, 0.46), (0.64, 0.31, 0.98, 0.46)),
    "IMG_2306.PNG":  ((0.16, 0.36, 0.41, 0.51), (0.62, 0.36, 0.95, 0.51)),
    "IMG_2359.PNG":  ((0.03, 0.38, 0.35, 0.52), (0.53, 0.38, 0.82, 0.52)),
    "IMG_9293.jpeg": ((0.07, 0.06, 0.40, 0.37), (0.64, 0.06, 0.95, 0.37)),
}

# guess 2: the back block, from the left armpit to the right armpit and from the
# base of the neck to the waist, as (x0, y0, x1, y1)
BACK = {
    "IMG_0512.jpeg": (0.500, 0.330, 0.770, 0.460),
    "IMG_0513.PNG":  (0.360, 0.345, 0.580, 0.448),
    "IMG_1481.PNG":  (0.300, 0.520, 0.690, 0.684),
    "IMG_2234.PNG":  (0.414, 0.409, 0.766, 0.579),
    "IMG_2306.PNG":  (0.400, 0.443, 0.708, 0.600),
    "IMG_2359.PNG":  (0.242, 0.467, 0.632, 0.625),
    "IMG_9293.jpeg": (0.362, 0.242, 0.655, 0.505),
}


def painted(box, width, height):
    """How many pixels ImageDraw.rectangle fills for this box."""
    x0, y0 = round(box[0] * width), round(box[1] * height)
    x1, y1 = round(box[2] * width), round(box[3] * height)
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def changed_pixels(original, changed):
    a = np.asarray(Image.open(original).convert("RGB"))
    b = np.asarray(Image.open(changed).convert("RGB"))
    return int((a != b).any(axis=2).sum())


def main():
    print("grey block areas in pixels: guess 1 (both arm blocks together) and "
          "guess 2 (back block)\n")
    print(f"  {'image':14s} {'size':>11s} {'arms':>10s} {'back':>10s} "
          f"{'back/arms':>10s}   check on the changed files")
    totals = [0, 0]
    for name in ARMS:
        with Image.open(ORIGINALS / name) as image:
            width, height = image.size
        arms = sum(painted(box, width, height) for box in ARMS[name])
        back = painted(BACK[name], width, height)
        totals[0] += arms
        totals[1] += back
        if name.lower().endswith(".png"):
            got_arms = changed_pixels(ORIGINALS / name, ARMS_FOLDER / name)
            got_back = changed_pixels(ORIGINALS / name, BACK_FOLDER / name)
            check = f"{got_arms:,} and {got_back:,} pixels differ"
        else:
            check = "JPEG, not checked"
        print(f"  {name:14s} {width:>5d}x{height:<5d} {arms:>10,d} {back:>10,d} "
              f"{back / arms:>10.2f}   {check}")
    print(f"\n  back/arms over all 7 images together: {totals[1] / totals[0]:.2f}")


if __name__ == "__main__":
    main()
