"""Evaluate a trained model on the HELD-OUT TEST split.

The test split (dataset/splits/test.csv) contains drivers, photo-bursts and
frame-blocks that appear in NO other split, so these numbers reflect real
generalisation. (The old version evaluated on the training folder, which is why
it always printed a near-perfect, identical confusion matrix - the model had
simply memorised those images.)

Prerequisite:  python scripts/prepare_split.py
Output: prints accuracy + per-class report + confusion matrix,
        saves models/confusion_matrix.png
"""

import os
import json
import numpy as np
import tensorflow as tf

import data_pipeline as dp

SEED = 42
tf.keras.utils.set_random_seed(SEED)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

class_indices_path = "models/class_indices.json"
cm_out = "models/confusion_matrix.png"

# Prefer the harmonised model (v3), then v2, then fine-tuned, then Phase 1.
candidates = ["models/driver_model_v3.keras",
              "models/driver_model_v2.keras",
              "models/driver_model_finetuned.keras",
              "models/driver_model.keras"]
model_path = next((p for p in candidates if os.path.exists(p)), None)
if model_path is None:
    raise FileNotFoundError("No trained model found in models/. Train one first.")

behaviour_names = {
    "c0": "Safe driving", "c1": "Texting - right", "c2": "Talking on phone - right",
    "c3": "Texting - left", "c4": "Talking on phone - left", "c5": "Operating the radio",
    "c6": "Drinking", "c7": "Reaching behind", "c8": "Hair and makeup",
    "c9": "Talking to passenger", "c10": "Drowsy",
}

with open(class_indices_path) as f:
    class_to_idx = json.load(f)              # {'c0':0,'c1':1,'c10':2,...}
idx_to_class = {i: c for c, i in class_to_idx.items()}
num_classes = len(class_to_idx)
classes = [idx_to_class[i] for i in range(num_classes)]
target_names = [f"{c} {behaviour_names.get(c, c)}" for c in classes]

print(f"Evaluating: {model_path}")
model = tf.keras.models.load_model(model_path)
# Some models are saved for inference only (not compiled); compile so evaluate()
# can report loss/accuracy. This does not change the learned weights.
model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

# Held-out test set, no shuffle -> stable, reproducible, and (unlike before)
# genuinely unseen data.
test_ds, n_test = dp.make_dataset("test", class_to_idx, batch=32, shuffle=False)
print(f"Held-out test images: {n_test}")

print("\nEvaluating on held-out test data...")
loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Predictions (recompute in the same fixed order for the report).
probs = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probs, axis=1)
y_true = np.array([class_to_idx[l] for l in dp.read_split("test")[1]])

from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report (held-out test):")
print(classification_report(y_true, y_pred, target_names=target_names,
                            digits=3, zero_division=0))
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
print("Confusion Matrix:\n", cm)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Held-out test confusion matrix (acc={accuracy*100:.1f}%)")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(cm_out, dpi=130)
    print(f"\nSaved confusion matrix figure -> {cm_out}")
except Exception as e:
    print(f"(Could not render confusion matrix PNG: {e})")
