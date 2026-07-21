"""Visual sanity check: run the deployed model on RANDOM held-out test images
(never seen in training) and render a grid of image + predicted label + true
label, green if correct / red if wrong. Also prints overall accuracy on the
sampled set. Pure honesty check - samples are random (seed=42), not cherry-picked.

Writes models/sample_predictions.png
"""

import os
import csv
import json
import random
import numpy as np
import cv2
import tensorflow as tf

SEED = 42
random.seed(SEED)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

PER_CLASS = 4
IMG = 224
OUT = "models/sample_predictions.png"

BEHAVIOUR = {"c0": "Safe driving", "c1": "Texting-R", "c2": "Phone-R",
             "c3": "Texting-L", "c4": "Phone-L", "c5": "Radio", "c6": "Drinking",
             "c7": "Reaching", "c8": "Hair/makeup", "c9": "Passenger", "c10": "Drowsy"}

candidates = ["models/driver_model_v3.keras", "models/driver_model_v2.keras"]
model_path = next(p for p in candidates if os.path.exists(p))
model = tf.keras.models.load_model(model_path)
print(f"Model: {model_path}")

with open("models/class_indices.json") as f:
    class_to_idx = json.load(f)
idx_to_class = {i: c for c, i in class_to_idx.items()}

# group test images by label, sample PER_CLASS each (seeded)
by_label = {}
with open("dataset/splits/test.csv", newline="") as f:
    for row in csv.DictReader(f):
        by_label.setdefault(row["label"], []).append(row["filepath"])
classes = sorted(class_to_idx, key=lambda c: class_to_idx[c])

picks = []  # (filepath, true_label)
for c in classes:
    files = by_label.get(c, [])
    for fp in random.sample(files, min(PER_CLASS, len(files))):
        picks.append((fp, c))


def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (IMG, IMG)).astype(np.float32)
    mean = img.mean()
    std = max(img.std(), 1.0 / np.sqrt(img.size))
    return (img - mean) / std


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ncols = PER_CLASS
nrows = len(classes)
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6))
correct = 0
total = 0
for ax in axes.ravel():
    ax.axis("off")

for k, (fp, true_c) in enumerate(picks):
    bgr = cv2.imread(fp)
    if bgr is None:
        continue
    probs = model.predict(preprocess(bgr)[None], verbose=0)[0]
    pi = int(np.argmax(probs))
    pred_c = idx_to_class[pi]
    conf = float(probs[pi])
    ok = pred_c == true_c
    correct += ok
    total += 1

    r = classes.index(true_c)
    c = k % ncols
    ax = axes[r, c]
    ax.imshow(cv2.cvtColor(cv2.resize(bgr, (IMG, IMG)), cv2.COLOR_BGR2RGB))
    ax.axis("off")
    color = "#1a9850" if ok else "#d73027"
    ax.set_title(f"pred: {BEHAVIOUR[pred_c]} {conf*100:.0f}%\ntrue: {BEHAVIOUR[true_c]}",
                 fontsize=8, color=color)
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(color); s.set_linewidth(3)
    ax.patch.set_edgecolor(color)

acc = correct / total if total else 0
fig.suptitle(f"Held-out test predictions ({model_path.split('/')[-1]}) - "
             f"sampled accuracy {correct}/{total} = {acc*100:.0f}%",
             fontsize=13, y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(OUT, dpi=110)
print(f"Sampled accuracy: {correct}/{total} = {acc*100:.1f}%")
print(f"Saved -> {OUT}")
