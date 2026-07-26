print("Script started successfully")
import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Ensure we always run relative to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# ========================
# 1️⃣ Paths and Parameters
# ========================

dataset_path = "dataset/train/imgs/train"  # Folder with class subfolders c0-c10
model_save_path = "models/driver_model.keras"
class_indices_path = "models/class_indices.json"

img_height, img_width = 224, 224  # Input size for MobileNetV2
batch_size = 32
num_classes = 11
epochs = 10

# ========================
# 1.5️⃣ Blur augmentation
# ========================
# Randomly softens ~50% of training images. This stops the model from being
# able to use raw sharpness/resolution/compression artifacts as a shortcut
# for telling classes apart (which is what was happening: c10 images came
# from a different, sharper source than c0-c9, so the model learned "sharp
# close-up" == Drowsy instead of learning actual behaviour cues).
def blur_augment(img):
    if np.random.rand() < 0.5:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        ksize = int(np.random.choice([3, 5]))
        img_uint8 = cv2.GaussianBlur(img_uint8, (ksize, ksize), 0)
        img = img_uint8.astype(np.float32)
    return img

# ========================
# 2️⃣ Check if paths exist
# ========================

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder not found at {dataset_path}")

if not os.path.exists("models"):
    os.makedirs("models")

# ========================
# 3️⃣ Image Preprocessing
# ========================

# Create ImageDataGenerator with validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% validation
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    preprocessing_function=blur_augment,  # runs on 0-255 range, before rescale
)

# Training generator
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

# Validation generator
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)

# Save class index mapping (folder name -> numeric index assigned by Keras)
# This lets test_model.py and realtime_detect.py reverse-map predictions correctly,
# even though Keras sorts folders alphabetically (c0, c1, c10, c2, ...).
with open(class_indices_path, "w") as f:
    json.dump(train_data.class_indices, f, indent=2)
print(f"Class indices saved: {train_data.class_indices}")

# ========================
# 4️⃣ Load MobileNetV2 Base Model
# ========================

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# ========================
# 5️⃣ Build Custom Model
# ========================

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# ========================
# 6️⃣ Compile Model
# ========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()  # Optional: prints model architecture

# ========================
# 7️⃣ Callbacks
# ========================

checkpoint = ModelCheckpoint(
    model_save_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    restore_best_weights=True
)

callbacks = [checkpoint, early_stop]

# ========================
# 8️⃣ Train Model
# ========================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=callbacks
)

# ========================
# 9️⃣ Save Final Model (if not already saved by checkpoint)
# ========================

model.save(model_save_path)
print(f"Model saved at {model_save_path}")