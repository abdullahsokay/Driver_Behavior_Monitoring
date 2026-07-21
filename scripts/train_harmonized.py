"""Harmonize the drowsy (c10) dataset into c0-c9 and retrain, so the model
cannot lean on 'which dataset' cues (the thing that made drowsy over-fire and
score an unrealistic 99.6%).

Two harmonisation levers, applied UNIFORMLY to all 11 classes:
  1. Per-image standardisation (zero-mean, unit-variance) at load time -> removes
     global brightness / colour / contrast fingerprints that differ between the
     State Farm, WhatsApp and drowsy sources.
  2. A regularised, calibrated head (L2 + heavy dropout + label smoothing +
     balanced class weights) -> stops any single class dominating and reduces
     over-confidence, so the confidence threshold at inference is meaningful.

It also runs a DOMAIN-SEPARABILITY PROBE: a logistic-regression that tries to
tell 'drowsy vs not-drowsy' from raw features. If harmonisation works, this
probe gets *harder* (lower accuracy) -> quantitative evidence the drowsy set
blends in instead of being trivially separable.

Reads dataset/splits/{train,val,test}.csv (run prepare_split.py first).
Writes models/driver_model_v3.keras, models/metrics_v3.json.
"""

import os
import csv
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

SPLIT_DIR = "dataset/splits"
CACHE_DIR = "models/_feature_cache"
RAW_PREFIX = ""            # non-harmonised caches: train_feats.npy etc.
HARM_PREFIX = "harm_"      # harmonised caches: harm_train_feats.npy etc.
MODEL_OUT = "models/driver_model_v3.keras"
METRICS_OUT = "models/metrics_v3.json"
IMG, BATCH = 224, 64

BEHAVIOUR = {"c0": "Safe driving", "c1": "Texting - right",
             "c2": "Talking on phone - right", "c3": "Texting - left",
             "c4": "Talking on phone - left", "c5": "Operating the radio",
             "c6": "Drinking", "c7": "Reaching behind", "c8": "Hair and makeup",
             "c9": "Talking to passenger", "c10": "Drowsy"}


def read_split(name):
    fps, labels = [], []
    with open(os.path.join(SPLIT_DIR, f"{name}.csv"), newline="") as f:
        for row in csv.DictReader(f):
            fps.append(row["filepath"]); labels.append(row["label"])
    return fps, labels


def load_harmonized(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3,
                             expand_animations=False)
    img = tf.image.resize(img, [IMG, IMG])
    img = tf.image.per_image_standardization(img)   # <-- harmonisation
    return img


def extract(base, name, fps, prefix):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{prefix}{name}_feats.npy")
    if os.path.exists(cache):
        f = np.load(cache)
        if len(f) == len(fps):
            print(f"  [{prefix}{name}] cached {f.shape}"); return f
    print(f"  [{prefix}{name}] extracting {len(fps)} imgs...")
    ds = (tf.data.Dataset.from_tensor_slices(fps)
          .map(load_harmonized, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH).prefetch(tf.data.AUTOTUNE))
    f = base.predict(ds, verbose=0)
    np.save(cache, f)
    print(f"  [{prefix}{name}] -> {cache} {f.shape}"); return f


def domain_probe(Xtr, ytr_drowsy, Xte, yte_drowsy, tag):
    """How easily can a linear model tell drowsy from not-drowsy? Lower = merged."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr_drowsy)
    acc = clf.score(Xte, yte_drowsy)
    print(f"  domain-separability probe ({tag}): {acc*100:.1f}% "
          f"(100% = trivially separate, ~50% = indistinguishable)")
    return acc


def main():
    train_fps, train_lab = read_split("train")
    val_fps, val_lab = read_split("val")
    test_fps, test_lab = read_split("test")
    classes = sorted(set(train_lab) | set(val_lab) | set(test_lab))
    c2i = {c: i for i, c in enumerate(classes)}
    n = len(classes)

    def y(lab):
        return tf.keras.utils.to_categorical([c2i[l] for l in lab], n)
    ytr, yva, yte = y(train_lab), y(val_lab), y(test_lab)

    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(IMG, IMG, 3), pooling="avg")
    base.trainable = False

    print("\nExtracting HARMONISED features...")
    Xtr = extract(base, "train", train_fps, HARM_PREFIX)
    Xva = extract(base, "val", val_fps, HARM_PREFIX)
    Xte = extract(base, "test", test_fps, HARM_PREFIX)

    # Domain-separability: before (raw [0,1] feats, if present) vs after (harmonised)
    print("\nDrowsy-vs-rest separability (evidence of merge):")
    dtr = np.array([l == "c10" for l in train_lab])
    dte = np.array([l == "c10" for l in test_lab])
    raw_tr = os.path.join(CACHE_DIR, "train_feats.npy")
    raw_te = os.path.join(CACHE_DIR, "test_feats.npy")
    if os.path.exists(raw_tr) and os.path.exists(raw_te):
        domain_probe(np.load(raw_tr), dtr, np.load(raw_te), dte, "BEFORE / raw [0,1]")
    domain_probe(Xtr, dtr, Xte, dte, "AFTER  / harmonised")

    # Regularised, calibrated head + balanced class weights
    print("\nTraining regularised head...")
    cw_counts = np.bincount([c2i[l] for l in train_lab], minlength=n)
    class_weight = {i: len(train_lab) / (n * c) if c else 0.0
                    for i, c in enumerate(cw_counts)}
    head = models.Sequential([
        layers.Input(shape=(Xtr.shape[1],)),
        layers.Dense(128, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.5),
        layers.Dense(n, activation="softmax"),
    ])
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                 metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                          restore_best_weights=True, verbose=1)
    head.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=60, batch_size=64,
             class_weight=class_weight, callbacks=[es], verbose=2)

    full = models.Model(base.input, head(base.output))
    full.compile(optimizer="adam", loss="categorical_crossentropy",
                 metrics=["accuracy"])
    full.save(MODEL_OUT)
    print(f"Saved -> {MODEL_OUT}")

    # Evaluate on held-out test
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    ypred = np.argmax(head.predict(Xte, verbose=0), axis=1)
    ytrue = np.argmax(yte, axis=1)
    acc = accuracy_score(ytrue, ypred)
    names = [f"{c} {BEHAVIOUR[c]}" for c in classes]
    print(f"\nHARMONISED held-out test accuracy: {acc*100:.2f}%\n")
    print(classification_report(ytrue, ypred, target_names=names, digits=3,
                                zero_division=0))
    cm = confusion_matrix(ytrue, ypred, labels=list(range(n)))

    # false-drowsy rate
    c10 = classes.index("c10")
    false_drowsy = sum(cm[r][c10] for r in range(n) if r != c10)
    nondrowsy = sum(int(sum(cm[r])) for r in range(n) if r != c10)
    print(f"False-drowsy: {false_drowsy}/{nondrowsy} = "
          f"{false_drowsy/nondrowsy*100:.2f}%")

    # per-domain
    dom = read_domains("test")
    per = {}
    for p, t, d in zip(ypred, ytrue, dom):
        per.setdefault(d, [0, 0]); per[d][1] += 1; per[d][0] += int(p == t)
    print("Per-domain accuracy:")
    for d, (c, tot) in sorted(per.items()):
        print(f"  {d:<10} {c}/{tot} = {c/tot*100:.1f}%")

    with open(METRICS_OUT, "w") as f:
        json.dump({"seed": SEED, "test_accuracy": float(acc),
                   "false_drowsy_rate": false_drowsy / nondrowsy,
                   "per_domain": {d: v[0] / v[1] for d, v in per.items()},
                   "confusion_matrix": cm.tolist(), "classes": classes}, f, indent=2)
    print(f"Saved metrics -> {METRICS_OUT}")


def read_domains(name):
    doms = []
    with open(os.path.join(SPLIT_DIR, f"{name}.csv"), newline="") as f:
        for row in csv.DictReader(f):
            doms.append(row["domain"])
    return doms


if __name__ == "__main__":
    main()
