"""Phase 2 - fine-tune the top of MobileNetV2 for the driver-behaviour domain.

Loads the Phase 1 model, unfreezes the last 30 base layers, and continues
training at a low learning rate on the SAME leakage-free split (stronger
augmentation). Because train/val never share a driver, burst or frame block,
the reported val accuracy is honest.

Prerequisites:  scripts/prepare_split.py  then  scripts/train_model.py
Note: fine-tuning backprops through the conv base - practical on GPU, slow on
CPU-only TensorFlow.

Output: models/driver_model_finetuned.keras
"""

print("Phase 2: fine-tuning started")
import os
import json
import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau)

import data_pipeline as dp

SEED = 42
tf.keras.utils.set_random_seed(SEED)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

model_load_path = "models/driver_model.keras"
model_save_path = "models/driver_model_finetuned.keras"
class_indices_path = "models/class_indices.json"

batch_size = 16
epochs = 15

if not os.path.exists(model_load_path):
    raise FileNotFoundError(
        f"Phase 1 model not found at {model_load_path}. Run train_model.py first.")

with open(class_indices_path) as f:
    class_to_idx = json.load(f)
print(f"Classes ({len(class_to_idx)}): {class_to_idx}")

train_ds, n_train = dp.make_dataset(
    "train", class_to_idx, batch=batch_size, shuffle=True,
    augmenter=dp.make_strong_augmenter(SEED), seed=SEED)
val_ds, n_val = dp.make_dataset(
    "val", class_to_idx, batch=batch_size, shuffle=False)
print(f"Train images: {n_train}  |  Val images: {n_val}")

model = tf.keras.models.load_model(model_load_path)
print("Phase 1 model loaded.")

# Freeze everything, then unfreeze only the last 30 layers for domain adaptation.
for layer in model.layers:
    layer.trainable = False
for layer in model.layers[-30:]:
    layer.trainable = True
trainable = sum(1 for l in model.layers if l.trainable)
print(f"Trainable layers: {trainable}/{len(model.layers)}")

# Low LR is critical when fine-tuning so pretrained weights aren't destroyed.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    ModelCheckpoint(model_save_path, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2,
                      min_lr=1e-7, verbose=1),
]

print("\nStarting Phase 2 fine-tuning...")
model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

model.save(model_save_path)
print(f"\nFine-tuned model saved at {model_save_path}")
print("Evaluate on the held-out test set with: python scripts/test_model.py")
