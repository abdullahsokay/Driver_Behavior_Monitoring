"""Create the `driverBehaviourAlerts` collection in Cloud Firestore.

Firestore creates a collection lazily the first time a document is written to it,
so this script seeds one sample alert document. Run it once; afterwards, the
collection will be visible in the Firebase console and ready for the realtime
detector (or a backend service) to write further alerts.
"""

import os
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

FIREBASE_KEY_PATH = "firebase_key.json"
COLLECTION_NAME = "driverBehaviourAlerts"


def init_firestore():
    if not os.path.exists(FIREBASE_KEY_PATH):
        raise FileNotFoundError(
            f"{FIREBASE_KEY_PATH} not found in {os.getcwd()}. "
            "Place your Firebase service-account key there before running."
        )
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return firestore.client()


def seed_sample_alert(db):
    sample = {
        "alertId": "ALERT-001",
        "driverId": "driver_001",
        "driverName": "Test Driver",
        "tankerId": "tanker_001",
        "tankerName": "TNK-001",
        "behaviour": "Safe driving",
        "behaviourCode": "c0",
        "confidence": 1.0,
        "severity": "low",
        "acknowledged": True,
        "notes": "Seed document created to initialize the collection.",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "timestamp": firestore.SERVER_TIMESTAMP,
    }
    doc_ref = db.collection(COLLECTION_NAME).document()
    doc_ref.set(sample)
    return doc_ref.id


def main():
    db = init_firestore()
    doc_id = seed_sample_alert(db)
    print(f"Created collection '{COLLECTION_NAME}' with seed document '{doc_id}'.")


if __name__ == "__main__":
    main()
