import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

dataset_path = "dataset/train/imgs/train"
finetuned_path = "models/driver_model_finetuned.keras"
phase1_path = "models/driver_model.keras"
model_path = finetuned_path if os.path.exists(finetuned_path) else phase1_path
class_indices_path = "models/class_indices.json"
img_height, img_width = 224, 224
batch_size = 32

# ---- 1. Raw image counts per class folder ----
print("=" * 50)
print("IMAGE COUNT PER CLASS FOLDER")
print("=" * 50)
counts = {}
for folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        n = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        counts[folder] = n
        print(f"  {folder}: {n} images")

total = sum(counts.values())
print(f"\nTotal: {total} images")
if counts:
    max_class = max(counts, key=counts.get)
    max_pct = counts[max_class] / total * 100
    print(f"Largest class: {max_class} ({max_pct:.1f}% of all data)")
    if max_pct > 20:  # 11 balanced classes = ~9% each
        print("  -> WARNING: this class is over-represented relative to a balanced ~9% share.")

# ---- 2. Per-class accuracy on the validation split ----
print("\n" + "=" * 50)
print("PER-CLASS VALIDATION ACCURACY")
print("=" * 50)

with open(class_indices_path, "r") as f:
    class_indices = json.load(f)
idx_to_folder = {v: k for k, v in class_indices.items()}

model = tf.keras.models.load_model(model_path)
print(f"Loaded model: {model_path}")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical',
    shuffle=False
)

preds = model.predict(val_data, verbose=1)
pred_classes = np.argmax(preds, axis=1)
true_classes = val_data.classes

# Confusion-style breakdown: for each true class, what did the model predict?
print("\nFor each TRUE class, distribution of PREDICTED classes:")
for true_idx in sorted(set(true_classes)):
    mask = true_classes == true_idx
    n_samples = mask.sum()
    predicted_for_this_class = pred_classes[mask]
    unique, cnts = np.unique(predicted_for_this_class, return_counts=True)
    breakdown = ", ".join(
        f"{idx_to_folder[u]}={c}" for u, c in sorted(zip(unique, cnts), key=lambda x: -x[1])
    )
    acc = (predicted_for_this_class == true_idx).mean() * 100
    print(f"  True={idx_to_folder[true_idx]} (n={n_samples}, acc={acc:.1f}%): predicted as [{breakdown}]")

overall_acc = (pred_classes == true_classes).mean() * 100
print(f"\nOverall validation accuracy: {overall_acc:.1f}%")

# What accuracy would a trivial "always predict the majority class" model get?
majority_idx = np.bincount(true_classes).argmax()
trivial_acc = (true_classes == majority_idx).mean() * 100
print(f"'Always predict {idx_to_folder[majority_idx]}' baseline would get: {trivial_acc:.1f}%")
if overall_acc <= trivial_acc + 5:
    print("  -> WARNING: model barely beats the trivial majority-class baseline. Likely collapsed to that class.")