"""CPU-friendly honest training + evaluation on the leakage-free split.

This machine has no GPU (native-Windows TensorFlow is CPU-only), so training the
full MobileNetV2 end-to-end for many epochs is impractical. Instead we use the
standard "frozen-base transfer learning" trick made fast:

  1. Run every image through the frozen MobileNetV2 base ONCE and cache the
     1280-d feature vector to disk  (the slow part, ~one forward pass over data).
  2. Train a small classification head on the cached features (seconds/epoch).
  3. Re-assemble base + head into ONE end-to-end .keras model that takes raw
     224x224 [0,1] images, so realtime_detect.py / test_model.py work unchanged.
  4. Evaluate on the HELD-OUT test split (drivers/bursts/blocks never seen in
     training) and print an honest classification report + confusion matrix.

Reads: dataset/splits/{train,val,test}.csv  (run prepare_split.py first)
Writes: models/driver_model_v2.keras, models/class_indices.json,
        models/confusion_matrix.png, models/metrics.json
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
CACHE_DIR = os.environ.get("FEATURE_CACHE_DIR", "models/_feature_cache")
MODEL_OUT = "models/driver_model_v2.keras"
CLASS_IDX_OUT = "models/class_indices.json"
CM_OUT = "models/confusion_matrix.png"
METRICS_OUT = "models/metrics.json"

IMG = 224
BATCH = 64

BEHAVIOUR_NAMES = {
    "c0": "Safe driving", "c1": "Texting - right", "c2": "Talking on phone - right",
    "c3": "Texting - left", "c4": "Talking on phone - left", "c5": "Operating the radio",
    "c6": "Drinking", "c7": "Reaching behind", "c8": "Hair and makeup",
    "c9": "Talking to passenger", "c10": "Drowsy",
}


def read_split(name):
    fps, labels = [], []
    with open(os.path.join(SPLIT_DIR, f"{name}.csv"), newline="") as f:
        for row in csv.DictReader(f):
            fps.append(row["filepath"])
            labels.append(row["label"])
    return fps, labels


def load_image(path):
    data = tf.io.read_file(path)
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG, IMG])
    img = tf.cast(img, tf.float32) / 255.0   # match realtime_detect.py preprocessing
    return img


def extract_features(base, name, fps):
    """Return cached 1280-d features for a split, computing + caching on first run."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{name}_feats.npy")
    if os.path.exists(cache):
        feats = np.load(cache)
        if len(feats) == len(fps):
            print(f"  [{name}] loaded cached features {feats.shape}")
            return feats
    print(f"  [{name}] extracting features for {len(fps)} images ...")
    ds = (tf.data.Dataset.from_tensor_slices(fps)
          .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH).prefetch(tf.data.AUTOTUNE))
    feats = base.predict(ds, verbose=1)
    np.save(cache, feats)
    print(f"  [{name}] cached features {feats.shape} -> {cache}")
    return feats


def main():
    # ---- classes (sorted like Keras: c0,c1,c10,c2,...c9) ----
    train_fps, train_lab = read_split("train")
    val_fps, val_lab = read_split("val")
    test_fps, test_lab = read_split("test")

    classes = sorted(set(train_lab) | set(val_lab) | set(test_lab))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    with open(CLASS_IDX_OUT, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Classes ({num_classes}): {class_to_idx}")

    def to_y(labels):
        return tf.keras.utils.to_categorical(
            [class_to_idx[l] for l in labels], num_classes)

    y_train, y_val, y_test = to_y(train_lab), to_y(val_lab), to_y(test_lab)

    # ---- frozen base + cached features ----
    print("\nBuilding frozen MobileNetV2 base and extracting features...")
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(IMG, IMG, 3), pooling="avg")
    base.trainable = False
    Xtr = extract_features(base, "train", train_fps)
    Xva = extract_features(base, "val", val_fps)
    Xte = extract_features(base, "test", test_fps)

    # ---- train the head on cached features (fast) ----
    print("\nTraining classification head on cached features...")
    head = models.Sequential([
        layers.Input(shape=(Xtr.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),                     # regularise -> less overfit
        layers.Dense(num_classes, activation="softmax"),
    ])
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss="categorical_crossentropy", metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1)
    head.fit(Xtr, y_train, validation_data=(Xva, y_val),
             epochs=60, batch_size=64, callbacks=[es], verbose=2)

    # ---- assemble end-to-end model (image -> prediction) and save ----
    full = models.Model(inputs=base.input, outputs=head(base.output))
    full.compile(optimizer="adam", loss="categorical_crossentropy",
                 metrics=["accuracy"])   # so evaluate() works after reload
    full.save(MODEL_OUT)
    print(f"\nSaved end-to-end model -> {MODEL_OUT}")

    # ---- honest evaluation on held-out test ----
    print("\nEvaluating on HELD-OUT test split (unseen drivers/bursts/blocks)...")
    probs = head.predict(Xte, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score)
    target_names = [f"{c} {BEHAVIOUR_NAMES.get(c, c)}" for c in classes]
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names,
                                   digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nHELD-OUT TEST ACCURACY: {acc:.4f} ({acc*100:.2f}%)\n")
    print(report)
    print("Confusion matrix:\n", cm)

    # sanity check: also score on TRAIN to expose the train/test gap
    train_acc = accuracy_score(np.argmax(y_train, axis=1),
                               np.argmax(head.predict(Xtr, verbose=0), axis=1))
    print(f"\nTrain accuracy: {train_acc:.4f}  |  Held-out test accuracy: {acc:.4f}"
          f"  |  generalisation gap: {train_acc-acc:.4f}")

    with open(METRICS_OUT, "w") as f:
        json.dump({"seed": SEED, "classes": classes,
                   "train_accuracy": float(train_acc),
                   "test_accuracy": float(acc),
                   "n_train": len(train_fps), "n_val": len(val_fps),
                   "n_test": len(test_fps),
                   "confusion_matrix": cm.tolist()}, f, indent=2)

    # ---- confusion matrix figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
        ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Held-out test confusion matrix (acc={acc*100:.1f}%)")
        thresh = cm.max() / 2 if cm.max() else 0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)
        fig.colorbar(im); fig.tight_layout(); fig.savefig(CM_OUT, dpi=130)
        print(f"Saved confusion matrix figure -> {CM_OUT}")
    except Exception as e:
        print(f"(Could not render confusion matrix PNG: {e})")


if __name__ == "__main__":
    main()
