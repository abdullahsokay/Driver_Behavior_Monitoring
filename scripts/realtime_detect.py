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

# Use fine-tuned model if available, otherwise fall back to Phase 1 model
finetuned_path = "models/driver_model_finetuned.keras"
phase1_path = "models/driver_model.keras"
model_path = finetuned_path if os.path.exists(finetuned_path) else phase1_path
class_indices_path = "models/class_indices.json"
img_height, img_width = 224, 224

# Confidence threshold: predictions below this show "Uncertain"
confidence_threshold = 0.60

# Temporal smoothing: average predictions over the last N frames to reduce flicker
smoothing_window = 5

# Drowsy is enabled, but GATED: it is only accepted as the prediction when the
# model is at least this confident about it. Below this bar, drowsy is ignored and
# the best of the OTHER behaviours is shown - so drowsy no longer covers everything
# while still firing for a genuinely drowsy driver.
# Raise drowsy_min_confidence -> drowsy fires less;  lower it -> fires more easily.
# (Set skip_drowsy = True to disable drowsy completely again.)
skip_drowsy = False
drowsy_min_confidence = 0.97

# Sustained-detection: how many consecutive smoothed frames must agree on the
# SAME class before we fire an alert for it. Different severities get different
# bars — "critical" (drowsy) requires the longest sustained run, since a single
# blink or head-turn shouldn't trigger it, but a genuinely nodding-off driver will
# hold that prediction for many consecutive frames.
required_consecutive_frames = {
    "critical": 8,
    "high": 3,
    "medium": 3,
    "low": 3,
}

# ========================
# Resolution normalization (MUST match train_model.py / finetune_model.py
# exactly - this is what stops the model from using "this looks sharper/
# lower-res than what I was trained on" as a Drowsy shortcut)
# ========================
NORM_SIZE = (320, 240)

def normalize_resolution(img):
    h, w = img.shape[:2]
    img = cv2.resize(img, NORM_SIZE, interpolation=cv2.INTER_LINEAR)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img

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

# Severity tiers. Drowsy (c10) is its own "critical" tier, separate from the
# other high-severity distraction behaviours (phone use / texting).
critical_severity_codes = {"c10"}
high_severity_codes = {"c1", "c2", "c3", "c4"}
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
    if behaviour_code in critical_severity_codes:
        return "critical"
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

# Sustained-detection state: tracks how many consecutive smoothed frames
# the CURRENT top class has held, so a single flickery frame can't fire an alert.
consecutive_class_idx = None
consecutive_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess frame for prediction.
    # IMPORTANT: cv2 reads frames as BGR, but the training data was loaded via
    # Keras' ImageDataGenerator (PIL), which reads images as RGB. Feeding BGR
    # frames into a model trained on RGB is a consistent color-channel mismatch
    # that biases predictions toward whichever class's features best match the
    # inverted colors — this is what was causing "Drowsy" to dominate regardless
    # of actual behaviour. Converting to RGB here fixes that mismatch.
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = normalize_resolution(img)
    img = cv2.resize(img, (img_width, img_height))
    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict and apply temporal smoothing
    predictions = model.predict(img_array, verbose=0)[0]
    prediction_history.append(predictions)
    smoothed = np.mean(prediction_history, axis=0)

    # Gate drowsy: only let it win when it is genuinely confident, otherwise
    # suppress it so a frontal face isn't labelled Drowsy over every other class.
    if "c10" in class_indices:
        c10_idx = class_indices["c10"]
        drowsy_confident = smoothed[c10_idx] >= drowsy_min_confidence
        if skip_drowsy or not drowsy_confident:
            smoothed = smoothed.copy()
            smoothed[c10_idx] = 0.0
            total = smoothed.sum()
            if total > 0:
                smoothed = smoothed / total   # renormalise over the remaining classes

    class_idx = int(np.argmax(smoothed))
    confidence = float(smoothed[class_idx])

    # Track how many consecutive frames this class has been the top prediction
    # (above threshold). Any change of class, or dropping below threshold,
    # resets the streak.
    if confidence >= confidence_threshold and class_idx == consecutive_class_idx:
        consecutive_count += 1
    elif confidence >= confidence_threshold:
        consecutive_class_idx = class_idx
        consecutive_count = 1
    else:
        consecutive_class_idx = None
        consecutive_count = 0

    # Apply confidence threshold
    if confidence < confidence_threshold:
        label = "Uncertain"
        color = (0, 255, 255)  # Yellow for uncertain
    else:
        label = class_labels[class_idx]
        if class_idx == safe_idx:
            color = (0, 255, 0)  # Green for safe driving
        else:
            color = (0, 0, 255)  # Red for distracted behavior

            behaviour_code = idx_to_code[class_idx]
            severity = get_severity(behaviour_code)
            needed = required_consecutive_frames.get(severity, 3)

            # Only alert once the behaviour has been sustained for long enough
            # (critical/drowsy needs the longest sustained run) AND we're past
            # the per-class cooldown, so alerts don't spam.
            if consecutive_count >= needed:
                now = time.time()
                if now - last_alert_time.get(class_idx, 0) >= alert_cooldown_seconds:
                    last_alert_time[class_idx] = now
                    send_alert(label, behaviour_code, confidence)

    # Display main prediction
    text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display top-2 predictions in bright white for visibility
    top2_idx = np.argsort(smoothed)[-2:][::-1]
    cv2.putText(frame, "Top-2:", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, idx in enumerate(top2_idx):
        debug_text = f"{i+1}. {class_labels[idx]}: {smoothed[idx]*100:.1f}%"
        cv2.putText(frame, debug_text, (10, 120 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Driver Behavior Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera stopped.")