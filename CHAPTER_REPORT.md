# Chapter: Driver Behaviour Monitoring Module

---

## 1. Introduction

Driver distraction is among the leading causes of road traffic accidents worldwide. According to global road-safety reports, a significant percentage of fatal collisions involve some form of driver inattention — including phone use, eating, drinking, talking to passengers, or operating in-car systems. To address this safety issue, the **Driver Behaviour Monitoring Module** has been developed as part of the larger road-safety system.

The module uses **deep learning** and **computer vision** to automatically detect eleven distinct driver behaviours in real time using a vehicle-mounted camera. The system is built on the **MobileNetV2** convolutional neural network, fine-tuned on a curated driver-behaviour image dataset, and deployed through a Python-based pipeline that captures live frames from a webcam, classifies them, applies temporal smoothing and a confidence threshold, and pushes distracted-driving alerts to **Firebase** for display on the SOTMS web and mobile apps.

This chapter presents the design, dataset, methodology, training pipeline, and real-time deployment details of the module.

---

## 2. Objectives

The objectives of the Driver Behaviour Monitoring module are:

1. To classify driver activity into one of ten predefined behaviour categories.
2. To distinguish between **safe driving** and **distracted driving** in real time.
3. To leverage **transfer learning** with MobileNetV2 to achieve high accuracy with limited training data.
4. To deploy the trained model on a real-time webcam feed for in-vehicle monitoring.
5. To provide clear, on-screen feedback to the driver so distracted behaviour can be flagged immediately.

---

## 3. Driver Behaviour Classes

The system classifies driver activity into **eleven classes** based on the standard driver-distraction taxonomy, extended with a drowsiness class.

### Table 3.1 — Behaviour Class Definitions

| Class ID | Behaviour | Risk Category | Alert Severity |
|----------|--------------------------------|---------------|----------------|
| c0       | Safe driving                   | Safe          | low            |
| c1       | Texting — right hand           | Distracted    | high           |
| c2       | Talking on phone — right hand  | Distracted    | high           |
| c3       | Texting — left hand            | Distracted    | high           |
| c4       | Talking on phone — left hand   | Distracted    | high           |
| c5       | Operating the radio            | Distracted    | medium         |
| c6       | Drinking                       | Distracted    | medium         |
| c7       | Reaching behind                | Distracted    | low            |
| c8       | Hair and makeup                | Distracted    | medium         |
| c9       | Talking to passenger           | Distracted    | low            |
| c10      | Drowsy                         | Distracted    | high           |

> **[Insert Figure 3.1 — Sample images from each behaviour class here]**

---

## 4. Dataset

### 4.1 Dataset Description

The training dataset consists of labelled driver images organised into eleven class subfolders (`c0` through `c10`), stored under `dataset/train/imgs/train/`. Crucially, the images come from **three different sources ("domains")** mixed together, which has major implications for evaluation (Section 4.4):

- **State Farm** — original Kaggle *Distracted Driver Detection* video frames (`img_*.jpg`) in classes c0–c9. These are consecutive frames from short videos of only **20 drivers**. All 17,462 of these images map to a driver ID via `driver_imgs_list.csv`.
- **WhatsApp** — custom phone photos (`WhatsApp Image ...jpeg`) added to classes c0–c9 (~472 images), collected in near-duplicate "bursts" sharing a timestamp.
- **Drowsiness dataset** — a **separate** dataset forming class c10 (723 jpg + 1,087 png), with none of the State Farm framing or subjects.

### 4.2 Dataset Distribution

### Table 4.1 — Image Count per Class

| Class | Behaviour                       | Image Count |
|-------|---------------------------------|-------------|
| c0    | Safe driving                    | 2,031       |
| c1    | Texting — right                 | 1,811       |
| c2    | Talking on phone — right        | 1,839       |
| c3    | Texting — left                  | 1,834       |
| c4    | Talking on phone — left         | 1,867       |
| c5    | Operating the radio             | 1,807       |
| c6    | Drinking                        | 1,857       |
| c7    | Reaching behind                 | 1,600       |
| c8    | Hair and makeup                 | 1,546       |
| c9    | Talking to passenger            | 1,742       |
| c10   | Drowsy                          | 1,810       |
| **Total** |                             | **19,744**  |

> **[Insert Figure 4.1 — Bar chart of image distribution across classes here]**

### 4.3 Leakage-Free Dataset Split

A naive random 80/20 split (as produced by `ImageDataGenerator(validation_split=0.2)`) is **unsafe** for this dataset: because State Farm images are consecutive video frames of the same 20 drivers, near-identical frames of one driver land in both the training and validation sets. The model then *memorises drivers* and reports a fake ~99% validation accuracy that does not generalise to unseen drivers.

To obtain an honest evaluation, `scripts/prepare_split.py` builds the split at the **group level**, guaranteeing no near-duplicate ever spans two splits:

| Domain | Grouping unit | Split strategy |
|--------|---------------|----------------|
| State Farm | Driver subject (`p0xx`) | Whole drivers assigned to one split |
| WhatsApp | Timestamp burst | Whole bursts assigned to one split |
| Drowsy (c10) | Contiguous frame block | Adjacent frames kept together |

The resulting split (seed = 42, ~70/15/15 by group) is:

| Split | Images | State Farm drivers |
|-------|--------|--------------------|
| Train | 13,445 | 14 |
| Validation | 3,096 | 3 |
| Test | 3,203 | 3 |

No driver, burst or frame block appears in more than one split — the script asserts this before writing the files.

### 4.4 Data-Overlap Caveat (c10)

Because c10 comes from a *different dataset* than c0–c9, a classifier can separate "drowsy vs not-drowsy" from **image style alone** (resolution, framing, background) rather than from genuine signs of drowsiness. This inflates c10's metrics and any overall-accuracy figure. It is a form of dataset leakage and is discussed further in Sections 9 and 10.

---

## 5. System Architecture

The module is organised into four pipeline stages:

1. **Data preparation** — image loading, resizing, normalisation, augmentation
2. **Phase 1 training** — transfer learning with frozen MobileNetV2 base
3. **Phase 2 fine-tuning** — unfreezing top layers for domain adaptation
4. **Real-time inference** — live webcam classification with on-screen feedback

> **[Insert Figure 5.1 — System architecture / data-flow diagram here]**

### 5.1 Project Structure

```text
driver_Behaviour/
├── dataset/
│   ├── train/
│   │   ├── imgs/train/       # Class folders c0–c10
│   │   └── driver_imgs_list.csv   # State Farm image -> driver subject map
│   └── splits/               # generated: train/val/test.csv (leakage-free)
├── models/
│   ├── driver_model.keras            # Phase 1 model
│   ├── driver_model_finetuned.keras  # Phase 2 model
│   ├── driver_model_v2.keras         # honest model (leakage-free split)
│   ├── class_indices.json            # folder -> index mapping
│   └── confusion_matrix.png          # held-out test confusion matrix
├── scripts/
│   ├── prepare_split.py      # build the leakage-free train/val/test split
│   ├── data_pipeline.py      # shared tf.data loader for the split
│   ├── train_model.py        # Phase 1 training (frozen base)
│   ├── finetune_model.py     # Phase 2 fine-tuning
│   ├── train_fast_cpu.py     # fast CPU training via cached features
│   ├── test_model.py         # honest held-out evaluation
│   ├── realtime_detect.py    # live webcam detection + Firebase alerts
│   └── create_firestore_alerts_collection.py
├── requirements.txt
├── firebase_key.json         # git-ignored service-account key (never commit)
└── README.md
```

---

## 6. Methodology

The module follows a **two-phase transfer-learning** strategy. This approach is widely accepted for image-classification tasks where labelled data is limited and class boundaries are subtle (e.g. distinguishing right-hand texting from left-hand texting).

### 6.1 Phase 1 — Transfer Learning with Frozen Base

In Phase 1, **MobileNetV2** pre-trained on ImageNet is used as a feature extractor. The convolutional base is fully frozen, and only a small custom classification head is trained.

**Architecture:**

| Layer                       | Output Shape         | Trainable |
|-----------------------------|----------------------|-----------|
| MobileNetV2 (base)          | (7, 7, 1280)         | No        |
| GlobalAveragePooling2D      | (1280,)              | —         |
| Dense (128 units, ReLU)     | (128,)               | Yes       |
| Dense (10 units, Softmax)   | (10,)                | Yes       |

**Hyperparameters:**

| Parameter           | Value                          |
|---------------------|--------------------------------|
| Input size          | 224 × 224 × 3                  |
| Batch size          | 32                             |
| Optimiser           | Adam (default LR = 0.001)      |
| Loss function       | Categorical cross-entropy      |
| Epochs              | 10                             |
| Validation split    | 0.2                            |

**Data augmentation:**
- Rotation: ±10°
- Zoom: ±10%
- Width / height shift: ±10%
- Horizontal flip: enabled
- Rescale: 1/255

**Callbacks used:**
- `ModelCheckpoint` — saves the best model based on validation accuracy
- `EarlyStopping` — patience of 3 epochs, restores best weights

The trained model is saved as `models/driver_model.keras`.

> **[Insert Figure 6.1 — Phase 1 training accuracy / loss curves here]**
>
> **[Insert Figure 6.2 — Screenshot of Phase 1 training console output here]**

### 6.2 Phase 2 — Fine-Tuning

After Phase 1 converges, Phase 2 unfreezes the **last 30 layers** of the MobileNetV2 base, allowing the network to learn domain-specific features (driver poses, in-car backgrounds, hand positions) instead of generic ImageNet features.

**Hyperparameters:**

| Parameter           | Value                              |
|---------------------|------------------------------------|
| Optimiser           | Adam                               |
| Learning rate       | 1 × 10⁻⁵ (much lower than Phase 1) |
| Batch size          | 16                                 |
| Epochs              | 15                                 |
| Layers unfrozen     | Last 30 of MobileNetV2             |

**Stronger augmentation in Phase 2:**
- Rotation: ±15°
- Zoom: ±15%
- Width / height shift: ±15%
- Brightness range: [0.8, 1.2]
- Shear range: 0.1
- Horizontal flip: enabled

**Additional callback:**
- `ReduceLROnPlateau` — halves the learning rate if validation accuracy plateaus for 2 epochs (minimum LR: 1 × 10⁻⁷)

The fine-tuned model is saved as `models/driver_model_finetuned.keras`.

> **[Insert Figure 6.3 — Phase 2 fine-tuning accuracy / loss curves here]**
>
> **[Insert Figure 6.4 — Screenshot of Phase 2 console output here]**

---

## 7. Tools and Technologies

### Table 7.1 — Software Stack

| Component       | Tool / Library                  | Purpose                                    |
|-----------------|---------------------------------|--------------------------------------------|
| Language        | Python 3.10                     | Primary development language               |
| Deep learning   | TensorFlow / Keras              | Model building, training, inference        |
| Backbone model  | MobileNetV2 (ImageNet weights)  | Pre-trained feature extractor              |
| Computer vision | OpenCV (`cv2`)                  | Webcam capture and on-screen rendering     |
| Numerical       | NumPy                           | Array operations and tensor preparation    |
| Evaluation      | scikit-learn                    | Classification report, confusion matrix    |
| Environment     | Python `venv`                   | Isolated dependency management             |
| Version control | Git / GitHub                    | Source code management                     |

---

## 8. Implementation

### 8.1 Phase 1 Training Script

The Phase 1 training pipeline is implemented in [scripts/train_model.py](scripts/train_model.py). Key sections:

- **Lines 18–24:** Path and hyperparameter configuration
- **Lines 41–67:** ImageDataGenerator setup with augmentation
- **Lines 73–88:** MobileNetV2 base + custom classification head
- **Lines 94–98:** Model compilation
- **Lines 106–120:** ModelCheckpoint and EarlyStopping callbacks
- **Lines 126–131:** Model fitting

### 8.2 Phase 2 Fine-Tuning Script

The fine-tuning pipeline is in [scripts/finetune_model.py](scripts/finetune_model.py). Key features:

- Loads the Phase 1 model from disk
- Selectively unfreezes the last 30 layers
- Recompiles with a low learning rate (1 × 10⁻⁵)
- Adds `ReduceLROnPlateau` for adaptive learning-rate scheduling
- Saves to a separate file so the original Phase 1 model is preserved

### 8.3 Offline Evaluation Script

The evaluation pipeline is in [scripts/test_model.py](scripts/test_model.py). It:

- Loads the trained model
- Runs predictions on the dataset
- Reports overall test loss and accuracy
- Generates a per-class classification report (precision, recall, F1-score)
- Generates a confusion matrix

### 8.4 Real-Time Detection Script

Real-time webcam detection is implemented in [scripts/realtime_detect.py](scripts/realtime_detect.py). The script:

1. Automatically selects the best available model (leakage-free `driver_model_v2`, then fine-tuned, then Phase 1)
2. Opens the default webcam via OpenCV
3. Captures each frame, resizes it to 224 × 224, normalises pixel values to [0, 1]
4. Runs the model's `predict()` to obtain class probabilities
5. Applies **temporal smoothing** (averaging the last 5 frames) to reduce flicker
6. Applies a **confidence threshold** (0.60): low-confidence frames are shown as "Uncertain" instead of guessing a class
7. Overlays the predicted behaviour, confidence percentage, and the top-3 predictions on the frame
8. Colour codes the output: **green** for safe driving, **red** for distracted behaviour, **yellow** for uncertain
9. Pushes a distracted-driving **alert to Firebase Realtime Database** (`driverBehaviour/cameraAlerts`) with driver/tanker metadata, behaviour code, confidence and severity — subject to a per-behaviour cooldown so the same alert is not spammed. The SOTMS web and mobile apps read this path to raise live notifications.
10. Quits cleanly when the user presses **q**

> **[Insert Figure 8.1 — Screenshot of real-time detection window here]**

### 8.5 Firebase Alert Integration

The realtime detector is the bridge between the model and the SOTMS apps. Each distracted-driving event is written as a document under `driverBehaviour/cameraAlerts` containing `driverId`, `driverName`, `tankerId`, `tankerName`, `behaviour`, `behaviourCode`, `confidence`, `severity` (high/medium/low), a server timestamp, and an `acknowledged` flag. The service-account key lives in `firebase_key.json`, which is **git-ignored and must never be committed**. `scripts/create_firestore_alerts_collection.py` seeds an equivalent Cloud Firestore collection for backends that read from Firestore instead of RTDB.

---

## 9. Results

All results below are measured on the **held-out test split** (3,203 images from 3 drivers, plus held-out drowsy blocks and WhatsApp bursts) that the model never saw during training. This is the key correction over earlier reporting, which evaluated on the *training* folder and therefore showed a meaningless near-perfect confusion matrix.

### 9.1 From Leaked "99%" to an Honest, Stronger Model

The headline result is the progression across three stages, all judged on the same held-out test set:

| Stage | Test accuracy | What it means |
|-------|---------------|----------------|
| Original random split (leaked) | ~99% (train-eval) | Memorised near-duplicate frames — meaningless |
| Leakage-free split (v2) | **60.5%** | First *honest* number; exposes the real difficulty |
| Harmonised + regularised (v3) | **76.1%** | Uniform per-image standardisation + L2/dropout/label-smoothing/class-weights |

Training accuracy for v3 is ~97% versus 76.1% on unseen data — a real but far healthier gap than the leaked pipeline. The 15.6-point jump from v2 to v3 came from **treating all three data sources identically** (per-image standardisation removes brightness/colour fingerprints) plus regularisation — i.e. from *fixing the data handling*, not from leakage.

### 9.2 Accuracy by Data Domain (v3, held-out test)

| Domain | Accuracy | Interpretation |
|--------|----------|----------------|
| Drowsy (c10, separate dataset) | **100.0%** | Still trivially separable — see 9.4 |
| WhatsApp (custom photos) | 90.3% | Small, burst-correlated; optimistic |
| State Farm (real behaviours, unseen drivers) | **73.5%** | The genuine behaviour-classification skill (was 56.2%) |
| **Overall** | **76.1%** | — |

Critically, the model **never falsely predicts drowsy**: 0 of 2,932 non-drowsy test images were labelled c10 (0.00% false-drowsy rate). The over-prediction seen with the earlier leaked model is resolved.

### 9.3 Per-Class Classification Report (v3, held-out test)

| Class | Behaviour | Precision | Recall | F1 |
|-------|-----------|-----------|--------|-----|
| c0 | Safe driving | 0.940 | **0.621** | 0.748 |
| c1 | Texting — right | 0.745 | 0.546 | 0.630 |
| c2 | Talking on phone — right | 0.863 | 0.746 | 0.800 |
| c3 | Texting — left | 0.807 | 0.949 | 0.872 |
| c4 | Talking on phone — left | 0.936 | 0.920 | 0.928 |
| c5 | Operating the radio | 0.561 | 0.828 | 0.669 |
| c6 | Drinking | 0.672 | 0.692 | 0.682 |
| c7 | Reaching behind | 0.826 | 0.964 | 0.890 |
| c8 | Hair and makeup | 0.776 | 0.609 | 0.683 |
| c9 | Talking to passenger | 0.456 | 0.566 | 0.505 |
| c10 | Drowsy | 1.000 | 1.000 | 1.000 |
| | **Macro average** | 0.780 | 0.767 | 0.764 |

Safe-driving recall recovered from **0.096 (v2) to 0.621 (v3)** — the safety-critical failure is largely fixed. The remaining weak spot is c9 (talking to passenger), still confused with safe driving. (Figure: `models/confusion_matrix.png`, from `test_model.py`.)

### 9.4 Domain-Separability Probe — the Drowsy Overlap, Quantified

To test whether the drowsy set can be *hidden* among c0–c9, a logistic-regression classifier was trained to predict "drowsy vs not-drowsy" from MobileNetV2 features:

| Features | Drowsy-vs-rest separability |
|----------|------------------------------|
| Raw ([0,1]) | 100.0% |
| Harmonised (per-image standardised) | 100.0% |

Harmonisation removes *technical* fingerprints but not the fact that drowsy images are **different scenes**, so the class remains perfectly separable (hence c10's 100% accuracy). This is reported transparently: drowsiness is a **supplementary class** whose high score reflects a domain shortcut, not validated drowsiness detection. The honest fix (Section 10.2) is to collect drowsy footage from the same in-vehicle camera as the other classes.

### 9.4 Real-Time Performance

On CPU (no GPU available on native Windows), MobileNetV2 inference runs at roughly 3–8 FPS per frame with 5-frame temporal smoothing. GPU or a quantised/edge model would raise this substantially (Section 10.2).

> **[Insert Figure 9.3 — Sample webcam predictions for various behaviours here]**

---

## 10. Discussion

The two-phase training strategy provides a strong trade-off between training time and model quality. Phase 1 produces a baseline model rapidly because only a small classification head is updated. Phase 2 then adapts the deeper convolutional features to the specifics of the driver-behaviour domain, which is essential for distinguishing visually similar classes such as **texting — right** vs **talking on phone — right**.

### 10.1 Observed Challenges

- **Data leakage (the dominant issue):** Random splitting of State Farm video frames caused the same driver to appear in train and validation, yielding a fake ~99% score. Fixing this with group-aware splitting (`prepare_split.py`) dropped honest accuracy to 60.5% — the true starting point — after which harmonisation and regularisation raised it to 76.1% *legitimately*. See Sections 4.3 and 9.1.
- **Dataset-domain overlap:** c10 (drowsiness) comes from a different dataset than c0–c9. A separability probe (Section 9.4) shows it stays 100% distinguishable even after harmonisation, so its 100% score reflects a domain shortcut rather than validated drowsiness detection. It does not transfer to a single in-vehicle camera.
- **Few training subjects:** Only 14 drivers are available for training. With so little person-diversity, generalisation to unseen drivers is limited; the previously catastrophic *Safe driving* recall (0.096) recovered to 0.621 after harmonisation but the ceiling is still set by driver count.
- **Class similarity:** Behaviours involving the same hand (e.g. texting vs phone-call on the right), and *safe driving vs talking to passenger* (c0/c9), share strong visual cues and remain the hardest to separate.
- **Real-time latency:** Prediction runs sequentially with frame capture; FPS is capped by CPU inference speed.

### 10.2 Possible Improvements

Already implemented in the current pipeline: group-aware leakage-free splitting, fixed random seeds, uniform per-image standardisation across all data sources, regularisation (L2 + dropout + label smoothing + balanced class weights), an RGB colour-order fix for the webcam, temporal smoothing, and a confidence threshold with an "Uncertain" state.

Recommended next steps to raise honest accuracy:

- **Unify the capture domain.** Re-collect *all* classes — including drowsiness — from the same in-vehicle camera setup so the model cannot shortcut on dataset style. This is the highest-impact fix for the c10 overlap.
- **Increase subject diversity.** Add many more distinct drivers (and lighting/angle conditions). Person-diversity, not raw image count, is what improves generalisation to unseen drivers.
- **Fine-tune on the corrected split.** Run Phase 2 (`finetune_model.py`) on a GPU; unfreezing the top layers on leakage-free data should recover several points over the frozen-base baseline.
- **Address the c0/c9 confusion** (safe driving vs talking to passenger) with targeted data and possibly a two-head design (distracted/not-distracted, then fine behaviour).
- **Deploy a quantised / edge model** (e.g. TFLite) for higher FPS in-vehicle.

---

## 11. Conclusion

The Driver Behaviour Monitoring module uses **transfer learning** on **MobileNetV2** to classify eleven distinct driver activities in real time and pushes distracted-driving alerts to Firebase for the SOTMS web and mobile apps. A two-phase training pipeline (frozen-base training followed by fine-tuning) is implemented, complemented by a group-aware split generator, an honest held-out evaluation script, and a live OpenCV detector with temporal smoothing, a confidence threshold, and colour-coded feedback.

The critical lesson of this work is methodological: the module's earlier ~99% accuracy was an artefact of **data leakage** (random splitting of video frames and mixing of separate datasets), not real performance. After enforcing leakage-free, group-aware splitting, the honest baseline was **60.5%**; uniform per-image standardisation and regularisation then raised it *legitimately* to **76.1%**, recovering the safety-critical *Safe driving* class in the process. The drowsiness class is reported transparently as a supplementary class whose score reflects a domain shortcut (Section 9.4). The result is an accurate, defensible foundation, with the clearest remaining gains available from unifying the capture domain and increasing driver diversity (Section 10.2).

---
