import os
import json
import time
from collections import deque
import cv2
import numpy as np
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, db as rtdb

# Ensure we always run relative to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# ========================
# Paths and Parameters
# ========================

# Prefer the harmonised model (v3), then v2, then fine-tuned, then Phase 1.
_model_candidates = [
    "models/driver_model_v3.keras",         # harmonised + regularised (best)
    "models/driver_model_v2.keras",         # leakage-free split
    "models/driver_model_finetuned.keras",
    "models/driver_model.keras",
]
model_path = next((p for p in _model_candidates if os.path.exists(p)),
                  "models/driver_model.keras")
class_indices_path = "models/class_indices.json"
img_height, img_width = 224, 224

# Confidence threshold: predictions below this show "Uncertain"
confidence_threshold = 0.60

# Temporal smoothing: average predictions over the last N frames to reduce flicker
smoothing_window = 5

# ========================
# Firebase Configuration
# ========================

firebase_key_path = "firebase_key.json"
firebase_db_url = "https://sotms-abdullah-new-2026-default-rtdb.asia-southeast1.firebasedatabase.app"
alerts_path = "driverBehaviour/cameraAlerts"  # SOTMS Driver Behaviour page reads from here
driver_id = "driver_001"        # TODO: replace with the actual logged-in driver's ID
driver_name = "Test Driver"     # TODO: replace from your auth/session
tanker_id = "tanker_001"        # TODO: replace with the tanker this camera is mounted in
tanker_name = "TNK-001"         # TODO: replace with the tanker's display name
alert_cooldown_seconds = 5  # don't fire the same behaviour alert more than once per N seconds

# High-severity behaviours: phone use, texting, and drowsiness
high_severity_codes = {"c1", "c2", "c3", "c4", "c10"}
medium_severity_codes = {"c5", "c6", "c8"}  # radio, drinking, hair/makeup

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

# Load the folder->index mapping that Keras assigned during training
if not os.path.exists(class_indices_path):
    raise FileNotFoundError(
        f"{class_indices_path} not found. Run train_model.py first to generate it."
    )

with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

class_labels = {idx: behaviour_names[folder] for folder, idx in class_indices.items()}

# Class index for "Safe driving" (used to color predictions green vs red)
safe_idx = class_indices["c0"]

# ========================
# Load Model
# ========================

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# ========================
# Initialize Firebase
# ========================

firebase_enabled = False
alerts_ref = None
if os.path.exists(firebase_key_path):
    try:
        cred = credentials.Certificate(firebase_key_path)
        firebase_admin.initialize_app(cred, {"databaseURL": firebase_db_url})
        alerts_ref = rtdb.reference(alerts_path)
        firebase_enabled = True
        print(f"Firebase initialized. Alerts will be pushed to '/{alerts_path}'.")
    except Exception as e:
        print(f"Firebase init failed: {e}. Continuing without alerts.")
else:
    print(f"{firebase_key_path} not found. Continuing without alerts.")

last_alert_time = {}  # {class_idx: last_sent_timestamp}


def get_severity(behaviour_code):
    if behaviour_code in high_severity_codes:
        return "high"
    if behaviour_code in medium_severity_codes:
        return "medium"
    return "low"


def send_alert(behaviour_label, behaviour_code, confidence_value):
    """Push a distracted-driving alert to Realtime Database."""
    if not firebase_enabled:
        return
    try:
        alerts_ref.push({
            "driverId": driver_id,
            "driverName": driver_name,
            "tankerId": tanker_id,
            "tankerName": tanker_name,
            "behaviour": behaviour_label,
            "behaviourCode": behaviour_code,
            "confidence": float(confidence_value),
            "severity": get_severity(behaviour_code),
            "timestamp": {".sv": "timestamp"},  # Realtime DB server-side timestamp
            "acknowledged": False,
        })
        print(f"Alert sent: {behaviour_label} ({confidence_value*100:.1f}%)")
    except Exception as e:
        print(f"Failed to send alert: {e}")

# ========================
# Start Webcam
# ========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check your camera connection.")

print("Camera started. Press 'q' to quit.")

prediction_history = deque(maxlen=smoothing_window)

# Reverse map: numeric idx -> folder code (e.g. 0 -> "c0")
idx_to_code = {idx: code for code, idx in class_indices.items()}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_status_bar(frame, label, confidence, accent):
    """Minimal, clean overlay: one translucent bottom bar with a status dot,
    the behaviour label, and the confidence. Nothing else."""
    h, w = frame.shape[:2]
    bar_h = 64
    cy = h - bar_h // 2

    # translucent dark bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # coloured status dot
    cv2.circle(frame, (30, cy), 10, accent, -1, cv2.LINE_AA)

    # behaviour label (white)
    cv2.putText(frame, label, (54, cy + 8), FONT, 0.9, (245, 245, 245), 2, cv2.LINE_AA)

    # confidence, right-aligned, in the accent colour
    conf_text = f"{confidence * 100:.0f}%"
    (tw, _), _ = cv2.getTextSize(conf_text, FONT, 0.9, 2)
    cv2.putText(frame, conf_text, (w - tw - 24, cy + 8), FONT, 0.9, accent, 2, cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess frame for prediction. Two things must match training exactly:
    #  1) colour order: the model was trained on RGB, but OpenCV frames are BGR.
    #  2) per-image standardisation (same as data_pipeline.py / the v3 model).
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (img_width, img_height)).astype(np.float32)
    mean = img.mean()
    std = max(img.std(), 1.0 / np.sqrt(img.size))   # tf.image.per_image_standardization
    img_array = np.expand_dims((img - mean) / std, axis=0)

    # Predict and apply temporal smoothing
    predictions = model.predict(img_array, verbose=0)[0]
    prediction_history.append(predictions)
    smoothed = np.mean(prediction_history, axis=0)

    class_idx = int(np.argmax(smoothed))
    confidence = float(smoothed[class_idx])

    # Apply confidence threshold
    if confidence < confidence_threshold:
        label = "Uncertain"
        accent = (0, 190, 235)   # amber (BGR)
    else:
        label = class_labels[class_idx]
        if class_idx == safe_idx:
            accent = (90, 200, 90)   # green
        else:
            accent = (60, 60, 235)   # red

            # Push alert to Firebase (with cooldown to avoid spamming)
            now = time.time()
            if now - last_alert_time.get(class_idx, 0) >= alert_cooldown_seconds:
                last_alert_time[class_idx] = now
                send_alert(label, idx_to_code[class_idx], confidence)

    # Minimal overlay: one clean status line + confidence
    draw_status_bar(frame, label, confidence, accent)

    # Show frame
    cv2.imshow("Driver Behavior Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera stopped.")
