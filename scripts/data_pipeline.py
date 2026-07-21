"""Shared tf.data input pipeline built on the LEAKAGE-FREE split.

All training/eval scripts load images through here so that:
  * they read the exact same train/val/test split produced by prepare_split.py
    (no near-duplicate frames leaking across splits), and
  * preprocessing is identical everywhere and matches realtime_detect.py
    (resize to 224x224, scale to [0, 1]).

Run prepare_split.py first to generate dataset/splits/{train,val,test}.csv.
"""

import os
import csv
import tensorflow as tf

IMG = 224
SPLIT_DIR = "dataset/splits"


def read_split(name):
    """Return (filepaths, labels) for split 'train' | 'val' | 'test'."""
    path = os.path.join(SPLIT_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/prepare_split.py")
    fps, labels = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fps.append(row["filepath"])
            labels.append(row["label"])
    return fps, labels


def get_classes():
    """Union of labels across all splits, sorted like Keras (c0,c1,c10,c2,...)."""
    labs = set()
    for name in ("train", "val", "test"):
        labs |= set(read_split(name)[1])
    return sorted(labs)


def make_augmenter(seed):
    """Light on-the-fly augmentation (Phase 1). Applied only to the train set."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.05, seed=seed),
        tf.keras.layers.RandomZoom(0.1, seed=seed),
        tf.keras.layers.RandomTranslation(0.1, 0.1, seed=seed),
    ], name="augment")


def make_strong_augmenter(seed):
    """Stronger augmentation for Phase 2 fine-tuning (brightness/contrast added)."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.08, seed=seed),
        tf.keras.layers.RandomZoom(0.15, seed=seed),
        tf.keras.layers.RandomTranslation(0.15, 0.15, seed=seed),
        tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0), seed=seed),
        tf.keras.layers.RandomContrast(0.15, seed=seed),
    ], name="augment_strong")


def _load(path, label_idx, num_classes):
    data = tf.io.read_file(path)
    img = tf.io.decode_image(data, channels=3, expand_animations=False)  # RGB
    img = tf.image.resize(img, [IMG, IMG])
    # Per-image standardisation (zero-mean, unit-variance): normalises brightness/
    # colour/contrast uniformly across the State Farm, WhatsApp and drowsy sources.
    # Must match realtime_detect.py's preprocessing and the v3 model exactly.
    img = tf.image.per_image_standardization(img)
    return img, tf.one_hot(label_idx, num_classes)


def make_dataset(split, class_to_idx, batch=32, shuffle=False,
                 augmenter=None, seed=42):
    """Build a batched, prefetched tf.data.Dataset for one split.

    Returns (dataset, n_images). augmenter is a Keras Sequential (or None).
    """
    fps, labels = read_split(split)
    num_classes = len(class_to_idx)
    y = [class_to_idx[l] for l in labels]

    ds = tf.data.Dataset.from_tensor_slices((fps, y))
    if shuffle:
        ds = ds.shuffle(len(fps), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: _load(p, l, num_classes),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch)
    if augmenter is not None:
        ds = ds.map(lambda x, y_: (augmenter(x, training=True), y_),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE), len(fps)
