import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure we always run relative to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# ========================
# Paths and Parameters
# ========================

# Use fine-tuned model if available, otherwise fall back to Phase 1 model
finetuned_path = "models/driver_model_finetuned.keras"
phase1_path = "models/driver_model.keras"
model_path = finetuned_path if os.path.exists(finetuned_path) else phase1_path
class_indices_path = "models/class_indices.json"
test_dataset_path = "dataset/train/imgs/train"  # Update this if you have a separate test set

img_height, img_width = 224, 224
batch_size = 32

# Map dataset folder names to human-readable behaviour labels
behaviour_names = {
    "c0":  "Safe driving",
    "c1":  "Texting - right",
    "c2":  "Talking on phone - right",
    "c3":  "Texting - left",
    "c4":  "Talking on phone - left",
    "c5":  "Operating the radio",
    "c6":  "Drinking",
    "c7":  "Reaching behind",
    "c8":  "Hair and makeup",
    "c9":  "Talking to passenger",
    "c10": "Drowsy",
}

# Load the folder->index mapping that Keras assigned during training,
# then invert it to build class_labels (index -> human label).
if not os.path.exists(class_indices_path):
    raise FileNotFoundError(
        f"{class_indices_path} not found. Run train_model.py first to generate it."
    )

with open(class_indices_path, "r") as f:
    class_indices = json.load(f)  # e.g. {"c0": 0, "c1": 1, "c10": 2, "c2": 3, ...}

class_labels = {idx: behaviour_names[folder] for folder, idx in class_indices.items()}
num_classes = len(class_labels)
print(f"Loaded {num_classes} classes: {class_labels}")

# ========================
# Load Model
# ========================

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# ========================
# Prepare Test Data
# ========================

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ========================
# Evaluate Model
# ========================

print("\nEvaluating model on test data...")
loss, accuracy = model.evaluate(test_data)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ========================
# Per-Class Predictions
# ========================

print("\nGenerating per-class predictions...")
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

# Classification report
from sklearn.metrics import classification_report, confusion_matrix

target_names = [class_labels[i] for i in range(num_classes)]
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=target_names))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)
