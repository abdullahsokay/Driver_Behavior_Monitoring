# Driver Behaviour Monitoring

Real-time driver-distraction detection for the SOTMS road-safety system. A
MobileNetV2 classifier labels the driver's activity into 11 behaviours and pushes
distracted-driving alerts to Firebase, which the SOTMS web + mobile apps display
as live notifications.

## Behaviour classes

| Code | Behaviour | Severity |
|------|-----------|----------|
| c0 | Safe driving | low |
| c1 | Texting - right | high |
| c2 | Talking on phone - right | high |
| c3 | Texting - left | high |
| c4 | Talking on phone - left | high |
| c5 | Operating the radio | medium |
| c6 | Drinking | medium |
| c7 | Reaching behind | low |
| c8 | Hair and makeup | medium |
| c9 | Talking to passenger | low |
| c10 | Drowsy | high |

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

Native-Windows TensorFlow is **CPU-only**. For GPU training use WSL2 or Linux.

## Pipeline

```bash
python scripts/prepare_split.py     # 1. build the leakage-free train/val/test split
python scripts/train_fast_cpu.py    # 2a. FAST honest training on CPU (feature-cached)
# --- or, on a GPU machine, the full end-to-end pipeline: ---
python scripts/train_model.py       # 2b. Phase 1 (frozen base)
python scripts/finetune_model.py    # 2c. Phase 2 (fine-tune top layers)

python scripts/test_model.py        # 3. evaluate on the HELD-OUT test set
python scripts/realtime_detect.py   # 4. live webcam detection + Firebase alerts
```

## Data leakage — read this before trusting any accuracy number

The images come from **three different sources** mixed together:

- **State Farm** Kaggle video frames (`img_*.jpg`, classes c0-c9) — consecutive
  frames of only 20 drivers.
- **WhatsApp** custom phone photos (classes c0-c9) — near-duplicate bursts.
- A **separate drowsiness dataset** (class c10) — different image style entirely.

A naive random 80/20 split leaks near-identical frames of the same driver into
both train and validation, producing a fake ~99% accuracy (memorisation). It also
lets the model separate c10 by *dataset style* instead of by drowsiness.

`prepare_split.py` fixes this by splitting **by group**: whole drivers, whole
photo-bursts, and contiguous frame-blocks never span splits. Uniform per-image
standardisation + regularisation then lifts the honest held-out accuracy from
**60.5%** (leakage-free baseline, `driver_model_v2`) to **76.1%**
(`driver_model_v3`), with **73.5%** on the real behaviour classes and a **0%
false-drowsy rate**. Drowsy (c10) still scores ~100% because it is a *separate
dataset* (a domain shortcut, not validated drowsiness detection) — see
[CHAPTER_REPORT.md](CHAPTER_REPORT.md) §9.4 for the separability probe and next steps.

## Firebase

`realtime_detect.py` and `create_firestore_alerts_collection.py` need a
service-account key at `firebase_key.json` (git-ignored — never commit it). Set
the database URL and driver/tanker identifiers at the top of `realtime_detect.py`.
