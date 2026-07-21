"""Phase 1 - transfer learning with a FROZEN MobileNetV2 base.

Trains only a small classification head on top of ImageNet features, using the
leakage-free split from prepare_split.py (train/val are separated by driver,
photo-burst and frame-block, so validation accuracy reflects real generalisation
- not memorised near-duplicate frames).

Prerequisite:  python scripts/prepare_split.py
Note: on a GPU this is fast. On CPU-only TensorFlow it is slow because every
epoch runs full images through MobileNetV2 - use scripts/train_fast_cpu.py for a
quick CPU run that caches features instead.

Output: models/driver_model.keras, models/class_indices.json
"""

print("Phase 1: training started")
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import data_pipeline as dp

# Reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

model_save_path = "models/driver_model.keras"
class_indices_path = "models/class_indices.json"

img_height = img_width = dp.IMG   # 224
batch_size = 32
epochs = 15

os.makedirs("models", exist_ok=True)

# ---- classes & datasets from the leakage-free split ----
classes = dp.get_classes()                       # ['c0','c1','c10','c2',...,'c9']
class_to_idx = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)
with open(class_indices_path, "w") as f:
    json.dump(class_to_idx, f, indent=2)
print(f"Classes ({num_classes}): {class_to_idx}")

train_ds, n_train = dp.make_dataset(
    "train", class_to_idx, batch=batch_size, shuffle=True,
    augmenter=dp.make_augmenter(SEED), seed=SEED)
val_ds, n_val = dp.make_dataset(
    "val", class_to_idx, batch=batch_size, shuffle=False)
print(f"Train images: {n_train}  |  Val images: {n_val}")

# ---- frozen MobileNetV2 base + custom head ----
base_model = MobileNetV2(weights="imagenet", include_top=False,
                         input_shape=(img_height, img_width, 3))
base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)                        # regularise -> reduce overfit
output = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs=base_model.input, outputs=output)

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    ModelCheckpoint(model_save_path, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=4,
                  restore_best_weights=True, verbose=1),
]

model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

model.save(model_save_path)
print(f"Model saved at {model_save_path}")
print("Next: python scripts/finetune_model.py  (or evaluate with test_model.py)")
