"""Problem 7, guess 3: a block with the same area as the two guess 1 arm blocks.

Usage:
    uv run python src/problem7_samearea.py                     print the table
    uv run python src/problem7_samearea.py --preview out.png   draw the outlines
    uv run python src/problem7_samearea.py --write             save the copies

The covered block is centred on the spine and runs from the top of the head to
the waist; its width is chosen so that its area equals the two arm blocks of
guess 1 (ARMS in problem7_block_sizes.py). It may overlap those arm blocks.

The control block has exactly the same size and shape, on trousers or
background, at least GAP away from the head, the torso (armpit to armpit, top of
the head to the waist) and the guess 1 arm blocks.

Body positions are fractions of the image width and height, read by eye off a
2% grid. Blocks are filled with ImageDraw.rectangle, which paints both edges.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from problem7_block_sizes import ARMS, ORIGINALS, painted

ROOT = Path(__file__).resolve().parents[1]
COVERED_FOLDER = ROOT / "data" / "changed" / "bdb_samearea_covered"
CONTROL_FOLDER = ROOT / "data" / "changed" / "bdb_samearea_control"
GREY = (128, 128, 128)
GAP = 0.02

# top of head, spine, waist, left armpit, right armpit, trousers left, trousers right
BODY = {
    "IMG_0512.jpeg": (0.262, 0.650, 0.460, 0.500, 0.770, 0.530, 0.740),
    "IMG_0513.PNG":  (0.280, 0.470, 0.448, 0.360, 0.580, 0.384, 0.568),
    "IMG_1481.PNG":  (0.408, 0.495, 0.684, 0.300, 0.690, 0.320, 0.660),
    "IMG_2234.PNG":  (0.308, 0.580, 0.579, 0.414, 0.766, 0.445, 0.700),
    "IMG_2306.PNG":  (0.364, 0.555, 0.600, 0.400, 0.708, 0.455, 0.686),
    "IMG_2359.PNG":  (0.382, 0.435, 0.625, 0.242, 0.632, 0.302, 0.566),
    "IMG_9293.jpeg": (0.052, 0.510, 0.505, 0.362, 0.655, 0.380, 0.660),
}

# where the control block goes in each image
CONTROL_PLACE = {
    "IMG_0512.jpeg": "below the waist",
    "IMG_0513.PNG":  "below the waist",
    "IMG_1481.PNG":  "below the waist",
    "IMG_2234.PNG":  "left of the body",
    "IMG_2306.PNG":  "left of the body",
    "IMG_2359.PNG":  "below the waist",
    "IMG_9293.jpeg": "below the waist",
}


def px(box, width, height):
    return (round(box[0] * width), round(box[1] * height),
            round(box[2] * width), round(box[3] * height))


def area(b):
    return (b[2] - b[0] + 1) * (b[3] - b[1] + 1)


def overlap(a, b):
    return (max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
            * max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1))


def view(width, height):
    """The centre square the model sees, in pixels (inclusive)."""
    side = min(width, height)
    x0, y0 = (width - side) // 2, (height - side) // 2
    return (x0, y0, x0 + side - 1, y0 + side - 1)


def blocks(name, width, height):
    top, spine, waist, lpit, rpit, tl, tr = BODY[name]
    arms = [px(b, width, height) for b in ARMS[name]]
    target = sum(painted(b, width, height) for b in ARMS[name])

    y0, y1 = round(top * height), round(waist * height)
    h = y1 - y0 + 1
    w = round(target / h)
    x0 = round(spine * width - w / 2)
    covered = (x0, y0, x0 + w - 1, y1)

    gx, gy = round(GAP * width), round(GAP * height)
    place = CONTROL_PLACE[name]
    if place == "below the waist":
        cx = min(max(x0, 0), width - w)
        cy = round((waist + GAP) * height) + 1
    elif place == "left of the body":
        cx = round(min(lpit, tl) * width) - gx - w - 1
        cy = max(b[3] for b in arms) + gy + 1
    else:
        raise ValueError(place)
    control = (cx, cy, cx + w - 1, cy + h - 1)

    # the control must lie inside the image and clear of the body by GAP
    torso = px((lpit, top, rpit, waist), width, height)
    grown = (control[0] - gx, control[1] - gy, control[2] + gx, control[3] + gy)
    assert control[0] >= 0 and control[1] >= 0 and control[2] < width and control[3] < height, name
    assert all(overlap(grown, part) == 0 for part in arms + [torso]), name
    return target, arms, covered, control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    print("guess 3: covered block (same area as both arm blocks, spine-centred, "
          "top of head to waist) and control block\n")
    print(f"  {'image':14s} {'arm area':>10s} {'block':>10s} {'area':>10s} {'diff':>7s} "
          f"{'covered in view':>16s} {'control in view':>16s} {'overlap with arms':>22s}  control position")
    tiles = []
    for name in BODY:
        image = Image.open(ORIGINALS / name).convert("RGB")
        width, height = image.size
        target, arms, covered, control = blocks(name, width, height)
        v = view(width, height)
        shared = sum(overlap(covered, a) for a in arms)
        size = f"{covered[2] - covered[0] + 1}x{covered[3] - covered[1] + 1}"
        print(f"  {name:14s} {target:>10,d} {size:>10s} {area(covered):>10,d} "
              f"{area(covered) / target - 1:>+7.2%} "
              f"{overlap(covered, v) / area(covered):>16.0%} {overlap(control, v) / area(control):>16.0%} "
              f"{shared:>11,d} px {shared / target:>6.1%}  {CONTROL_PLACE[name]}")

        if args.preview:
            draw = ImageDraw.Draw(image)
            line = max(5, width // 150)
            draw.rectangle(v, outline=(255, 0, 255), width=line)
            for a in arms:
                draw.rectangle(a, outline=(255, 0, 0), width=max(3, line // 2))
            draw.rectangle(covered, outline=(0, 255, 0), width=line)
            draw.rectangle(control, outline=(255, 255, 0), width=line)
            tiles.append(image.resize((round(width * 900 / height), 900)))
        if args.write:
            for folder, box in ((COVERED_FOLDER, covered), (CONTROL_FOLDER, control)):
                folder.mkdir(parents=True, exist_ok=True)
                changed = image.copy()
                ImageDraw.Draw(changed).rectangle(box, fill=GREY)
                if name.lower().endswith((".jpg", ".jpeg")):
                    changed.save(folder / name, quality=95)
                else:
                    changed.save(folder / name)

    print("\n  overlap with arms: pixels of the covered block inside the guess 1 arm "
          "blocks, and that as a share of the arm area")
    if args.preview:
        sheet = Image.new("RGB", (sum(t.width + 8 for t in tiles), 900), "white")
        x = 0
        for tile in tiles:
            sheet.paste(tile, (x, 0))
            x += tile.width + 8
        sheet.save(args.preview)
        print(f"\nwrote {args.preview}")


if __name__ == "__main__":
    main()
