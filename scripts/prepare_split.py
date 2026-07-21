"""Build a LEAKAGE-FREE train/val/test split for the driver-behaviour dataset.

Why this exists
---------------
The training images come from three different sources ("domains") mixed together:

  * statefarm : original Kaggle frames (img_*.jpg) in classes c0-c9. These are
                consecutive VIDEO FRAMES of only 20 drivers, so a random split
                leaks near-identical frames of the same driver into both train
                and validation -> fake ~99% accuracy (memorisation, not learning).
  * whatsapp  : custom phone photos (WhatsApp Image ....jpeg) in c0-c9. Each
                "burst" (same timestamp) is a set of near-duplicate shots.
  * drowsy    : a SEPARATE drowsiness dataset in class c10 (jpg + png, numbered).

To get an honest evaluation we must guarantee that no *group* of near-duplicates
is ever shared across splits:

  * statefarm -> split by SUBJECT (driver id from driver_imgs_list.csv)
  * whatsapp  -> split by TIMESTAMP BURST
  * drowsy    -> split by CONTIGUOUS BLOCK (adjacent frames stay together)
  * misc      -> handful of odd renamed files -> train only

Output: dataset/splits/{train,val,test}.csv with columns filepath,label,domain,group
plus a printed summary and a hard assertion that no group leaks across splits.

Run:  python scripts/prepare_split.py
"""

import os
import csv
import random
import collections

# Reproducible: fixed seed so the split is identical on every machine / rerun.
SEED = 42
random.seed(SEED)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

TRAIN_DIR = "dataset/train/imgs/train"
CSV_PATH = "dataset/train/driver_imgs_list.csv"
OUT_DIR = "dataset/splits"

# Fractions apply at the GROUP level, not the image level.
VAL_FRAC = 0.15
TEST_FRAC = 0.15
DROWSY_CLASS = "c10"

IMG_EXTS = (".jpg", ".jpeg", ".png")


def load_subject_map():
    """img filename -> subject id (p0xx) for the State Farm images."""
    m = {}
    if not os.path.exists(CSV_PATH):
        print(f"WARNING: {CSV_PATH} not found; State Farm images fall back to "
              f"per-file grouping (weaker leakage protection).")
        return m
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            m[row["img"]] = row["subject"]
    return m


def classify(filename, label, subject_map):
    """Return (domain, group_key) for one image, namespaced so keys never collide."""
    lower = filename.lower()
    if label == DROWSY_CLASS:
        # numbered frames from a separate dataset -> group by contiguous block later
        return "drowsy", f"drowsy/{filename}"
    if filename.startswith("img_") and lower.endswith(".jpg"):
        subj = subject_map.get(filename)
        if subj:
            return "statefarm", f"statefarm/subj:{subj}"
        return "statefarm", f"statefarm/orphan:{filename}"
    if filename.startswith("WhatsApp"):
        # "WhatsApp Image 2026-04-14 at 2.47.31 AM (4).jpeg" -> burst key w/o the (n)
        key = filename.split(" (")[0]
        key = os.path.splitext(key)[0]
        return "whatsapp", f"whatsapp/{key}"
    return "misc", f"misc/{filename}"


def assign_by_image_count(groups, images_of, val_frac, test_frac, block):
    """Assign group keys to train/val/test, balancing by cumulative image count.

    block=False -> seeded shuffle of groups (subjects, bursts).
    block=True  -> keep sorted order, slice contiguously (adjacent frames).
    """
    groups = sorted(groups)
    if not block:
        random.shuffle(groups)  # seeded
    total = sum(len(images_of[g]) for g in groups) or 1
    val_cut = total * val_frac
    test_cut = total * (val_frac + test_frac)
    assign, running = {}, 0
    for g in groups:
        if running < val_cut:
            assign[g] = "val"
        elif running < test_cut:
            assign[g] = "test"
        else:
            assign[g] = "train"
        running += len(images_of[g])
    return assign


def main():
    subject_map = load_subject_map()

    # 1) Enumerate every image, tag with (label, domain, group).
    rows = []                                   # (filepath, label, domain, group)
    images_of = collections.defaultdict(list)   # group -> [filepaths]
    for label in sorted(os.listdir(TRAIN_DIR)):
        cdir = os.path.join(TRAIN_DIR, label)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if not fn.lower().endswith(IMG_EXTS):
                continue
            domain, group = classify(fn, label, subject_map)
            fp = os.path.join(cdir, fn).replace("\\", "/")
            rows.append((fp, label, domain, group))
            images_of[group].append(fp)

    # 2) Assign groups to splits, PER DOMAIN, with the right strategy per domain.
    by_domain = collections.defaultdict(set)
    for _, _, domain, group in rows:
        by_domain[domain].add(group)

    group_split = {}
    group_split.update(assign_by_image_count(          # drivers
        by_domain["statefarm"], images_of, VAL_FRAC, TEST_FRAC, block=False))
    group_split.update(assign_by_image_count(          # timestamp bursts
        by_domain["whatsapp"], images_of, VAL_FRAC, TEST_FRAC, block=False))
    group_split.update(assign_by_image_count(          # contiguous frame blocks
        by_domain["drowsy"], images_of, VAL_FRAC, TEST_FRAC, block=True))
    for g in by_domain["misc"]:                         # odd files -> train
        group_split[g] = "train"

    # 3) Materialise the split.
    split_rows = {"train": [], "val": [], "test": []}
    for fp, label, domain, group in rows:
        split_rows[group_split.get(group, "train")].append((fp, label, domain, group))

    # 4) HARD CHECK: no group key may appear in more than one split.
    group_to_splits = collections.defaultdict(set)
    for split, rws in split_rows.items():
        for _, _, _, group in rws:
            group_to_splits[group].add(split)
    leaked = {g: s for g, s in group_to_splits.items() if len(s) > 1}
    assert not leaked, f"LEAKAGE: {len(leaked)} groups span splits: {list(leaked)[:5]}"

    # 5) Write CSVs.
    os.makedirs(OUT_DIR, exist_ok=True)
    for split, rws in split_rows.items():
        with open(os.path.join(OUT_DIR, f"{split}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "label", "domain", "group"])
            w.writerows(sorted(rws))

    # 6) Summary.
    labels = sorted({r[1] for r in rows}, key=lambda c: (len(c), c))
    print(f"\nSEED={SEED}  |  no group crosses splits (checked {len(group_to_splits)} groups)\n")
    print(f"{'split':<6} {'images':>7} " + " ".join(f"{c:>5}" for c in labels))
    for split in ("train", "val", "test"):
        cnt = collections.Counter(r[1] for r in split_rows[split])
        print(f"{split:<6} {len(split_rows[split]):>7} " +
              " ".join(f"{cnt.get(c, 0):>5}" for c in labels))
    print("\nDomain x split (images):")
    dsplit = collections.Counter((r[2], s) for s in split_rows for r in split_rows[s])
    for dom in sorted({r[2] for r in rows}):
        print(f"  {dom:<10} " +
              " ".join(f"{s}={dsplit.get((dom, s), 0)}" for s in ("train", "val", "test")))
    subj_assign = collections.Counter(
        group_split[g].upper()[:4] for g in by_domain["statefarm"])
    print(f"\nState Farm driver-groups per split: "
          f"{dict(collections.Counter(group_split[g] for g in by_domain['statefarm']))}")
    print(f"Wrote train/val/test.csv to {OUT_DIR}/\n")


if __name__ == "__main__":
    main()
