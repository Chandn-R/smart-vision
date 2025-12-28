# Technical Architecture & Calculations

This document details the technical implementation, data flow, and calculations used in the SmartVision threat detection pipeline.

## System Overview

The pipeline processes video frames sequentially through five main stages:
1.  **Object Detection (YOLO)**
2.  **Object Tracking (ByteTrack)**
3.  **Pose Estimation (MediaPipe)**
4.  **Action Recognition (LSTM)**
5.  **Threat Logic (Rule-Based Matrix)**

---

## 1. Object Detection (YOLOv11)

*   **Model**: Custom fine-tuned YOLOv11s (`yolo11s_fine_tune_v3.pt`).
*   **Input**: Raw BGR Frame (Resolution Varies).
*   **Output**: Bounding boxes `[x1, y1, x2, y2]`, Confidence Score `conf`, Class ID `cls`.
*   **Configuration**:
    *   `CONFIDENCE_THRESHOLD`: **0.45** (Configurable in `src/config.py`).
    *   **Classes**:
        *   `0`: ATM
        *   `1`: Backpack
        *   `2`: Gun
        *   `3`: Handbag
        *   `4`: Knife
        *   `5`: Person

### Calculations
*   **Non-Maximum Suppression (NMS)**: Performed internally by YOLO to remove duplicate boxes.
*   **Filtering**: Only objects with `conf >= 0.45` are passed to downstream stages.

---

## 2. Object Tracking (ByteTrack)

*   **Library**: `supervision.ByteTrack`.
*   **Input**: List of Person Detections `([x1, y1, x2, y2], conf, class_id=5)`.
*   **Output**: Unique `track_id` for each person across frames.

### Parameters & Calculations
*   **Track Activation Threshold**: **0.25**. Detections below this confidence can still be associated with existing tracks but won't start new ones.
*   **Lost Track Buffer**: **60 Frames**. If a person is occluded, their ID is preserved for ~2 seconds (at 30 FPS) before being discarded.
*   **Matching Threshold**: **0.8**. Controls how strictly detections must overlap (IoU) with predicted track locations.
*   **Kalman Filter**: Used internally to predict the next position of a track based on velocity.

---

## 3. Pose Estimation (MediaPipe)

*   **Model**: `MediaPipe Pose` (Google).
*   **Input**: Cropped image of the **Person** (defined by Tracker Bounding Box).
*   **Output**: 33 Skeletal Landmarks.

### Calculations
*   **Data Layout**: Each landmark has `(x, y, z, visibility)`.
    *   `x, y`: Normalized coordinates [0.0, 1.0].
    *   `z`: Approximate depth.
    *   `visibility`: Confidence score [0.0, 1.0].
*   **Feature Vector**: Flattened array of size **132** (`33 landmarks * 4 values`).
    *   If no pose is detected, a zero-vector `[0.0] * 132` is used to maintain sequence continuity.

---

## 4. Action Recognition (LSTM)

*   **Model**: Bidirectional LSTM (PyTorch).
*   **Input**: Sequence of **30 Frames** of Pose Feature Vectors.
    *   Tensor Shape: `(1, 30, 132)`
*   **Output**: Probability distribution over Action Classes.

### Architecture
*   **Input Size**: 132
*   **Hidden Size**: Defined in checkpoint (typically 64 or 128).
*   **Layers**: Defined in checkpoint (typically 2).
*   **Bidirectional**: Yes (Effective Hidden Size = `Hidden * 2`).
*   **Classifier**: Fully Connected Layer mapping to Action Classes.

### Calculations
*   **Sliding Window**: A `deque` (`maxlen=30`) maintains the history for each `track_id`. Newest frame is appended; oldest is dropped.
*   **Softmax**: Applied to model output logits to get probabilities.
*   **Confidence Threshold**: `ACTION_CONFIDENCE_THRESHOLD = 0.70`.
    *   If `max(probs) > 0.70`, the action is accepted.
    *   Otherwise, defaults to `normal` or retains previous valid state (implementation dependent).

---

## 5. Threat Logic Manager

*   **Input**:
    *   **Action Label** (from LSTM).
    *   **Visible Objects** (List of Class IDs from current frame).
    *   **Duration** (Time elapsed since `track_id` first appeared).
    *   **Unattended Bag Timer** (Time duration where `Bags > 0` AND `Persons == 0`).

### Logic Matrix
(See `docs/system_behavior.md` for the full matrix).

### Calculations
*   **Person Duration**: `current_time - person_start_time`.
*   **Unattended Bag Timer**: Increments by `dt` (frame time delta) whenever `has_bag` is True and `num_persons` is 0. Resets to 0 otherwise.
*   **Loitering Check**: `has_atm` AND `duration > 60.0s`.
*   **Unattended Check**: `unattended_bag_timer > 60.0s`.
