print("Phase 2: Fine-tuning started")
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Ensure we always run relative to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# ========================
# Paths and Parameters
# ========================

dataset_path = "dataset/train/imgs/train"
model_load_path = "models/driver_model.keras"
model_save_path = "models/driver_model_finetuned.keras"

img_height, img_width = 224, 224
batch_size = 16  # Smaller batch size for fine-tuning (uses more memory)
epochs = 5

# ========================
# Blur augmentation (same as Phase 1 - keeps the model from leaning on
# sharpness/source artifacts instead of real behaviour cues)
# ========================
def blur_augment(img):
    if np.random.rand() < 0.5:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        ksize = int(np.random.choice([3, 5]))
        img_uint8 = cv2.GaussianBlur(img_uint8, (ksize, ksize), 0)
        img = img_uint8.astype(np.float32)
    return img

# ========================
# Check paths
# ========================

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder not found at {dataset_path}")

if not os.path.exists(model_load_path):
    raise FileNotFoundError(f"Phase 1 model not found at {model_load_path}. Run train_model.py first.")

# ========================
# Image Preprocessing
# ========================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    preprocessing_function=blur_augment,
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)

# ========================
# Load Phase 1 Trained Model
# ========================

model = tf.keras.models.load_model(model_load_path)
print("Phase 1 model loaded successfully.")

# ========================
# Unfreeze last 30 layers for fine-tuning
# ========================

for layer in model.layers:
    layer.trainable = False

for layer in model.layers[-30:]:
    layer.trainable = True

trainable_count = sum(1 for layer in model.layers if layer.trainable)
total_count = len(model.layers)
print(f"Trainable layers: {trainable_count}/{total_count}")

# ========================
# Recompile with lower learning rate
# ========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ========================
# Callbacks
# ========================

checkpoint = ModelCheckpoint(
    model_save_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stop, reduce_lr]

# ========================
# Fine-tune Model
# ========================

print("\nStarting Phase 2 fine-tuning...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=callbacks
)

# ========================
# Save Final Model
# ========================

model.save(model_save_path)
print(f"\nFine-tuned model saved at {model_save_path}")
print("Phase 2 complete! Use this model for real-time detection.")