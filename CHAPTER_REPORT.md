# Chapter: Driver Behaviour Monitoring Module

---

## 1. Introduction

Driver distraction is among the leading causes of road traffic accidents worldwide. According to global road-safety reports, a significant percentage of fatal collisions involve some form of driver inattention — including phone use, eating, drinking, talking to passengers, or operating in-car systems. To address this safety issue, the **Driver Behaviour Monitoring Module** has been developed as part of the larger road-safety system.

The module uses **deep learning** and **computer vision** to automatically detect ten distinct driver behaviours in real time using a vehicle-mounted camera. The system is built on the **MobileNetV2** convolutional neural network, fine-tuned on a custom-curated driver-behaviour image dataset, and deployed through a Python-based pipeline that captures live frames from a webcam, classifies them, and provides immediate visual feedback.

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

The system classifies driver activity into **ten classes** based on the standard driver-distraction taxonomy.

### Table 3.1 — Behaviour Class Definitions

| Class ID | Behaviour | Risk Category |
|----------|--------------------------------|---------------|
| c0       | Safe driving                   | Safe          |
| c1       | Texting — right hand           | Distracted    |
| c2       | Talking on phone — right hand  | Distracted    |
| c3       | Texting — left hand            | Distracted    |
| c4       | Talking on phone — left hand   | Distracted    |
| c5       | Operating the radio            | Distracted    |
| c6       | Drinking                       | Distracted    |
| c7       | Reaching behind                | Distracted    |
| c8       | Hair and makeup                | Distracted    |
| c9       | Talking to passenger           | Distracted    |

> **[Insert Figure 3.1 — Sample images from each behaviour class here]**

---

## 4. Dataset

### 4.1 Dataset Description

The training dataset consists of labelled driver images organised into ten class subfolders (`c0` through `c9`). Each folder contains images representing one specific driver behaviour. The dataset is stored under `dataset/train/imgs/train/` in the project directory.

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
| **Total** |                             | **17,934**  |

> **[Insert Figure 4.1 — Bar chart of image distribution across classes here]**

### 4.3 Dataset Split

The dataset is split using a **stratified 80/20 ratio** at training time using Keras's `ImageDataGenerator(validation_split=0.2)` mechanism:

- **Training set:** 80% (~14,347 images)
- **Validation set:** 20% (~3,587 images)

---

## 5. System Architecture

The module is organised into four pipeline stages:

1. **Data preparation** — image loading, resizing, normalisation, augmentation
2. **Phase 1 training** — transfer learning with frozen MobileNetV2 base
3. **Phase 2 fine-tuning** — unfreezing top layers for domain adaptation
4. **Real-time inference** — live webcam classification with on-screen feedback

> **[Insert Figure 5.1 — System architecture / data-flow diagram here]**

### 5.1 Project Structure

```
driver_Behaviour/
├── dataset/
│   └── train/
│       └── imgs/
│           └── train/        # Class folders c0–c9
├── models/
│   ├── driver_model.keras            # Phase 1 model
│   └── driver_model_finetuned.keras  # Phase 2 model
├── scripts/
│   ├── train_model.py        # Phase 1 training
│   ├── finetune_model.py     # Phase 2 fine-tuning
│   ├── test_model.py         # Offline evaluation
│   └── realtime_detect.py    # Live webcam detection
├── venv/                     # Python virtual environment
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

1. Automatically selects the fine-tuned model if available; otherwise falls back to the Phase 1 model
2. Opens the default webcam via OpenCV
3. Captures each frame, resizes it to 224 × 224, normalises pixel values to [0, 1]
4. Runs the model's `predict()` to obtain class probabilities
5. Overlays the predicted behaviour, confidence percentage, and **live FPS counter** on the frame
6. Colour codes the output: **green** for safe driving, **red** for distracted behaviour
7. Quits cleanly when the user presses **q**

> **[Insert Figure 8.1 — Screenshot of real-time detection window here]**

---

## 9. Results

### 9.1 Phase 1 (Frozen Base) — Performance

> **[Insert Table 9.1 — Phase 1 final training and validation metrics here]**
>
> **[Insert Figure 9.1 — Phase 1 confusion matrix here]**

### 9.2 Phase 2 (Fine-Tuned) — Performance

> **[Insert Table 9.2 — Phase 2 final training and validation metrics here]**
>
> **[Insert Figure 9.2 — Phase 2 confusion matrix here]**

### 9.3 Per-Class Classification Report

> **[Insert Table 9.3 — Per-class precision, recall, F1-score from `test_model.py` output here]**

### 9.4 Real-Time Performance

> **[Insert Table 9.4 — Real-time FPS measurements (model load time, average FPS, peak FPS) here]**
>
> **[Insert Figure 9.3 — Sample webcam predictions for various behaviours here]**

---

## 10. Discussion

The two-phase training strategy provides a strong trade-off between training time and model quality. Phase 1 produces a baseline model rapidly because only a small classification head is updated. Phase 2 then adapts the deeper convolutional features to the specifics of the driver-behaviour domain, which is essential for distinguishing visually similar classes such as **texting — right** vs **talking on phone — right**.

### 10.1 Observed Challenges

- **Class similarity:** Behaviours involving the same hand (e.g. texting vs phone-call on the right) share strong visual cues; Phase 2 fine-tuning is critical for separating these.
- **Lighting variation:** Real webcam conditions differ from training images, so brightness augmentation is added in Phase 2 to improve robustness.
- **Real-time latency:** Each prediction runs sequentially with frame capture; FPS is capped by inference speed on CPU.

### 10.2 Possible Improvements

- Replace MobileNetV2 with a smaller / quantised model for higher FPS on edge devices
- Apply temporal smoothing across consecutive frames to reduce flicker
- Add a confidence threshold so low-confidence predictions trigger an "uncertain" output rather than a possibly wrong class
- Collect additional in-vehicle images under varied lighting / camera angles to further boost real-world performance

---

## 11. Conclusion

The Driver Behaviour Monitoring module successfully uses **transfer learning** on **MobileNetV2** to classify ten distinct driver activities in real time. A two-phase training pipeline (frozen-base training followed by fine-tuning) is implemented, complemented by an offline evaluation script and a live OpenCV-based detection interface that overlays predicted behaviour, confidence, and FPS on the camera feed.

The module integrates cleanly into the broader road-safety system and provides a foundation that can be extended with temporal smoothing, confidence thresholds, and edge-device deployment in future iterations.

---
