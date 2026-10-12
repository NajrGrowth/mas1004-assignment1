"""Print the numbers in the long report that no other output prints.

Usage:
    uv run python src/report_numbers.py | tee results/report_numbers.txt

Everything here is worked out again from files on disk: the run outputs and
saved models in results/, results/cleaning/suspects/suspects.csv,
data/removed/log.csv, data/extra_download/ and the images themselves.
Sections follow the Problems of the long report.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import prepare_image
from evaluate import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
NEW = ROOT / "data" / "new_images"
CHANGED = ROOT / "data" / "changed"
EXTRA = ROOT / "data" / "extra_download"
LOG = ROOT / "data" / "removed" / "log.csv"
SUSPECTS = ROOT / "results" / "cleaning" / "suspects" / "suspects.csv"
RESULTS = ROOT / "results"
EXTENSIONS = {".jpg", ".jpeg", ".png"}
SAME = 0.95


def title(text):
    print(f"\n{text}\n{'-' * len(text)}")


def images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)


def number(name):
    """0151 from back_lat_spread/0151.jpg"""
    return int(Path(name).stem)


def read_log():
    with open(LOG) as handle:
        return list(csv.DictReader(handle))


def run_output(tag):
    """Per-epoch values and the final lines of results/<tag>_output.txt."""
    text = (RESULTS / f"{tag}_output.txt").read_text()
    epochs = [tuple(float(v) for v in m) for m in re.findall(
        r"epoch\s+\d+/\d+\s+train loss ([\d.]+) acc ([\d.]+)%\s+test loss ([\d.]+) acc ([\d.]+)%", text)]
    train = float(re.search(r"train accuracy ([\d.]+)%", text).group(1))
    test = float(re.search(r"test\s+accuracy ([\d.]+)%", text).group(1))
    trainable = int(re.search(r"([\d,]+) of them will be trained", text).group(1).replace(",", ""))
    total = int(re.search(r"model has ([\d,]+) parameters", text).group(1).replace(",", ""))
    return epochs, train, test, trainable, total


def view_share(box, width, height):
    """Share of a pixel box (x0, y0, x1, y1, inclusive) inside the centre square."""
    side = min(width, height)
    vx, vy = (width - side) // 2, (height - side) // 2
    ix = max(0, min(box[2], vx + side - 1) - max(box[0], vx) + 1)
    iy = max(0, min(box[3], vy + side - 1) - max(box[1], vy) + 1)
    return ix * iy / ((box[2] - box[0] + 1) * (box[3] - box[1] + 1))


def problem1():
    title("Problem 1: images per category")
    added = sum(1 for _ in open(EXTRA / "added.csv")) - 1
    print(f"  {'category':20s} {'data/raw':>9s} {'first download':>15s} {'data/clean':>11s}")
    raw_total = clean_total = 0
    for folder in sorted(RAW.iterdir()):
        if not folder.is_dir():
            continue
        raw = len(images(folder))
        clean = len(images(CLEAN / folder.name))
        first = sum(1 for p in images(folder) if number(p.name) <= 150)
        raw_total += raw
        clean_total += clean
        print(f"  {folder.name:20s} {raw:9d} {first:15d} {clean:11d}")
    print(f"  {'total':20s} {raw_total:9d} {'':15s} {clean_total:11d}")
    print(f"  images added to back_lat_spread from the extra download "
          f"(data/extra_download/added.csv): {added}")


def problem2():
    title("Problem 2: what the centre crop keeps of back_lat_spread/0090.jpg")
    w, h = Image.open(RAW / "back_lat_spread" / "0090.jpg").size
    scale = 256 / min(w, h)
    rw, rh = round(w * scale), round(h * scale)
    print(f"  original {w}x{h}, shorter side resized to 256: {rw}x{rh}")
    print(f"  the 224x224 crop keeps {224 * 224 / (rw * rh):.1%} of the area")


def problem3():
    title("Problem 3: from the run outputs")
    runs = {tag: run_output(tag) for tag in ("run", "scratch", "frozen", "epochs20")}
    epochs, train, test, trainable, total = runs["run"]
    print(f"  run, train loss: epoch 1 {epochs[0][0]:.4f}, epoch 2 {epochs[1][0]:.4f}, "
          f"epoch {len(epochs)} {epochs[-1][0]:.4f}")
    later = epochs[1:]
    print(f"  run, test loss epochs 2-{len(epochs)}: from {min(e[2] for e in later):.4f} "
          f"to {max(e[2] for e in later):.4f}")
    print(f"  run, test accuracy epochs 3-{len(epochs)}: from {min(e[3] for e in epochs[2:]):.1f}% "
          f"to {max(e[3] for e in epochs[2:]):.1f}%")
    e20 = runs["epochs20"][0]
    print(f"  epochs20, test loss: epoch 3 {e20[2][2]:.4f}, epoch {len(e20)} {e20[-1][2]:.4f}, "
          f"highest {max(e[2] for e in e20[2:]):.4f}")
    print(f"  epochs20 minus run, test accuracy: {runs['epochs20'][2]:.1f} - {test:.1f} = "
          f"{runs['epochs20'][2] - test:.1f} points")
    print(f"  run, train minus test accuracy: {train:.1f} - {test:.1f} = {train - test:.1f} points")
    sc = runs["scratch"][2]
    print(f"  run minus scratch, test accuracy: {test:.1f} - {sc:.1f} = {test - sc:.1f} points")
    fr = runs["frozen"]
    print(f"  frozen trains {fr[3]:,} of {fr[4]:,} parameters = {fr[3] / fr[4]:.3%}")
    print(f"  frozen recovers ({fr[2]:.1f} - {sc:.1f}) / ({test:.1f} - {sc:.1f}) = "
          f"{(fr[2] - sc) / (test - sc):.1%} of the gap between scratch and run")

    for tag in ("run", "scratch"):
        settings = json.loads((RESULTS / f"{tag}_settings.json").read_text())
        names = settings["class_names"]
        paths = [RAW / Path(p).parent.name / Path(p).name for p in settings["test_paths"]]
        X = np.stack([prepare_image(Image.open(p)) for p in paths])
        y = np.array([names.index(p.parent.name) for p in paths])
        model = torch.load(RESULTS / f"{tag}_model.pt", map_location="cpu", weights_only=False)
        matrix = confusion_matrix(model, X, y, len(names))
        print(f"\n  {tag}: confusion matrix on its {len(paths)} test images "
              f"(rows true, columns predicted)")
        print(f"  {'':20s}" + "".join(f"{n[:9]:>10s}" for n in names) + "     total")
        for i, name in enumerate(names):
            print(f"  {name:20s}" + "".join(f"{v:10d}" for v in matrix[i]) + f"{matrix[i].sum():10d}")


def copy_pairs():
    with open(SUSPECTS) as handle:
        return [(r[0], r[2], float(r[3])) for r in csv.reader(handle)
                if len(r) > 3 and r[1] == "copy of"]


def groups(pairs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, _ in pairs:
        parent[find(a)] = find(b)
    out = {}
    for x in list(parent):
        out.setdefault(find(x), set()).add(x)
    return list(out.values())


def problem4():
    title("Problem 4: near copies in results/cleaning/suspects/suspects.csv")
    pairs = copy_pairs()
    log = read_log()
    reason = {r["file"]: r["reason"] for r in log}
    category = lambda f: f.split("/")[0]
    same = [p for p in pairs if category(p[0]) == category(p[1])]
    cross = [p for p in pairs if category(p[0]) != category(p[1])]
    print(f"  near-copy pairs: {len(pairs)}")
    for label, part in (("same category", same), ("cross category", cross)):
        high = sum(s >= 0.99 for _, _, s in part)
        print(f"    {label}: {len(part)} ({high} at >= 0.99, {len(part) - high} at 0.95-0.99)")
    for a, b in (("back_double_biceps/0002.jpg", "back_double_biceps/0052.jpg"),
                 ("side_chest/0032.jpg", "side_chest/0046.jpg")):
        s = next(s for x, y, s in pairs if {x, y} == {a, b})
        print(f"  similarity {a} ~ {b}: {s:.4f}")

    other = lambda f: f in reason and not reason[f].startswith("near-duplicate")
    both_other = [p for p in same if other(p[0]) and other(p[1])]
    rest = [p for p in same if p not in both_other]
    rest_groups = groups(rest)
    kept = Counter(sum((CLEAN / f).exists() for f in g) for g in rest_groups)
    print(f"  same-category pairs with both images removed for reasons other than "
          f"near-duplicate: {len(both_other)}")
    print(f"  the other {len(rest)} same-category pairs form {len(rest_groups)} groups; "
          f"images kept per group now: " + ", ".join(f"{k} kept in {v}" for k, v in sorted(kept.items())))
    print(f"  removed with reason 'near-duplicate ...': "
          f"{sum(r['reason'].startswith('near-duplicate') for r in log)}")

    cross_groups = [g for g in groups(pairs) if len({category(f) for f in g}) > 1]
    some_kept = sum(any((CLEAN / f).exists() for f in g) for g in cross_groups)
    print(f"  groups that span more than one category: {len(cross_groups)}, "
          f"sizes {sorted(len(g) for g in cross_groups)}")
    print(f"    with one image still in data/clean: {some_kept}; "
          f"with every image removed: {len(cross_groups) - some_kept}")

    title("Problem 4: the first 20 suspects of each class")
    with open(SUSPECTS) as handle:
        rows = [r for r in csv.reader(handle) if len(r) > 3 and r[1] != "copy of" and r[0] != "file"]
    for name in sorted({r[1] for r in rows}):
        top = [r[0] for r in rows if r[1] == name and int(r[2]) <= 20]
        print(f"  {name:20s} {sum(f in reason for f in top):2d} of {len(top)} removed")

    title("Problem 4: removals in data/removed/log.csv, by pass")
    new_first = min(r["when"] for r in log if number(r["file"]) > 150)
    passes = {
        "first pass, 2 October": [r for r in log if r["when"].startswith("2026-10-02")],
        "second pass, 3 October (images 0001-0150)": [
            r for r in log if r["when"].startswith("2026-10-03") and number(r["file"]) <= 150
            and not r["reason"].startswith("near-duplicate")],
        "extra download (back_lat_spread 0151-0209)": [r for r in log if number(r["file"]) > 150],
        "near-duplicate pass": [r for r in log if r["reason"].startswith("near-duplicate")],
    }
    for label, rows in passes.items():
        print(f"  {label:45s} {len(rows):4d}")
    print(f"  {'total':45s} {sum(len(r) for r in passes.values()):4d} (log has {len(log)} rows)")
    second = passes["second pass, 3 October (images 0001-0150)"]
    for wording in ("front lat spread instead of back lat spread",
                    "back double biceps instead of back lat spread"):
        n = sum(r["class"] == "back_lat_spread" and wording in r["reason"] for r in second)
        print(f"  second pass, back_lat_spread, reason with '{wording}': {n}")
    before = [r for r in log if r["class"] == "back_lat_spread" and r["when"] < new_first
              and number(r["file"]) <= 150]
    print(f"  back_lat_spread before the extra download: 150 - {len(before)} removed = "
          f"{150 - len(before)} left")
    extra = passes["extra download (back_lat_spread 0151-0209)"]
    added = sum(1 for _ in open(EXTRA / "added.csv")) - 1
    print(f"  extra download: {added} added, {len(extra)} removed, {added - len(extra)} left")

    title("Problem 4: the extra download, data/extra_download/")
    import clean
    staged = images(next(p for p in EXTRA.iterdir() if p.is_dir()))
    old = [p for c in sorted(RAW.iterdir()) if c.is_dir() for p in images(c)
           if not (c.name == "back_lat_spread" and number(p.name) > 150)]
    removed_then = {r["file"] for r in log if r["when"] < new_first}
    f = clean.describe([p.resolve() for p in staged] + [p.resolve() for p in old])
    unit = f / np.linalg.norm(f, axis=1, keepdims=True)
    n = len(staged)
    to_old = unit[:n] @ unit[n:].T
    best = to_old.argmax(1)
    copies = [i for i in range(n) if to_old[i, best[i]] > SAME]
    where = Counter()
    for i in copies:
        m = old[best[i]]
        status = "removed" if f"{m.parent.name}/{m.name}" in removed_then else "still in data/clean"
        where[status] += 1
    rest = [i for i in range(n) if i not in set(copies)]
    within = unit[:n] @ unit[:n].T
    dup = set()
    for i in rest:
        if any(j < i and j not in dup and within[i, j] > SAME for j in rest):
            dup.add(i)
    fresh = {staged[i].name for i in rest if i not in dup}
    listed = {line.split(",")[0] for line in open(EXTRA / "added.csv").read().split()[1:]}
    print(f"  downloaded: {n}")
    print(f"  near copy (> {SAME}) of an earlier download: {len(copies)} "
          f"({where['still in data/clean']} of an image still in data/clean at the time, "
          f"{where['removed']} of an image already removed)")
    print(f"  near copy of another image in the new download: {len(dup)}")
    print(f"  left out in total: {len(copies) + len(dup)}; not a near copy of anything: {len(fresh)}")
    print(f"  these {len(fresh)} are exactly the files listed in added.csv: {fresh == listed}")


def problem4_compare():
    title("Problem 4: before and after cleaning, on the new images")
    text = (ROOT / "results" / "compare.txt").read_text()
    got = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(r"\n\s+(\w+)\s+[\d.]+%\s+[\d.]+%\s+\((\d+)/(\d+)\)", text)}
    (a, n), (b, _) = got["run"], got["clean"]
    print(f"  run {a}/{n}, clean {b}/{n}: difference {b - a} image(s)")


def problem5():
    title("Problem 5: the new images")
    kinds = Counter()
    for p in sorted(NEW.rglob("*")):
        if p.suffix.lower() in EXTENSIONS:
            im = Image.open(p)
            kinds[(im.format, im.size)] += 1
    for (fmt, size), count in sorted(kinds.items()):
        print(f"  {count:3d} {fmt:5s} {size[0]}x{size[1]}")
    print(f"  total {sum(kinds.values())}")
    test = run_output("clean")[2]
    m = re.search(r"accuracy on your new images: (\d+)/(\d+) = ([\d.]+)%",
                  (ROOT / "results" / "check.txt").read_text())
    print(f"  clean model: test set {test:.1f}%, new images {m.group(3)}%: "
          f"difference {test - float(m.group(3)):.1f} points")
    print(f"  one image of {m.group(2)} is {100 / int(m.group(2)):.2f} points")


def problem7():
    title("Problem 7, guess 2: control blocks inside the centre square")
    print("  measured from the changed pixels between original and control (PNG only)")
    for name in ("IMG_1481.PNG", "IMG_2359.PNG"):
        a = np.asarray(Image.open(NEW / "back_double_biceps" / name).convert("RGB"))
        b = np.asarray(Image.open(CHANGED / "bdb_upperback_control" / name).convert("RGB"))
        ys, xs = np.nonzero((a != b).any(axis=2))
        box = (xs.min(), ys.min(), xs.max(), ys.max())
        print(f"  {name}: block x {box[0]}-{box[2]}, y {box[1]}-{box[3]}, "
              f"{view_share(box, a.shape[1], a.shape[0]):.0%} inside")

    title("Problem 7, guess 3: covered block outside the centre square")
    import problem7_samearea
    for name in ("IMG_9293.jpeg",):
        w, h = Image.open(NEW / "back_double_biceps" / name).size
        _, _, covered, _ = problem7_samearea.blocks(name, w, h)
        print(f"  {name}: {1 - view_share(covered, w, h):.0%} of the covered block is outside")


def main():
    print("Numbers for the long report that no other output prints.")
    problem1()
    problem2()
    problem3()
    problem4()
    problem4_compare()
    problem5()
    problem7()


if __name__ == "__main__":
    main()
