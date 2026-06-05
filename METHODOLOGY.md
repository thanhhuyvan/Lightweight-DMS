# Methodology: Lightweight Drowsiness Detection System

This document provides a detailed technical explanation of the methodologies, formulas, and algorithms used in the Lightweight-DMS project.

---

## 1. Pipeline Overview: The 5-Stage Architecture
The system follows a modular, feed-forward architecture designed to transform raw pixels into high-level physiological insights:

1.  **Stage 1: Computer Vision & Preprocessing**: Enhancing raw video and extracting geometric landmarks.
2.  **Stage 2: Data Engineering & Signal Refinement**: Cleaning noise and calibrating individual baselines.
3.  **Stage 3: Duration Logic & Behavioral Stats**: Extracting frame-level behavioral indicators (e.g., micro-sleeps vs. blinks).
4.  **Stage 4: Statistical Aggregation (Sliding Window)**: Compiling temporal vectors over 60-second intervals.
5.  **Stage 5: Machine Learning Classification**: Using behavioral patterns to predict the final drowsiness state.

---

## 2. Phase 1: Computer Vision & Preprocessing

### 2.1. Frame Extraction & Enhancement
*   **Sampling**: Videos are processed at **4 FPS** to balance temporal resolution with computational efficiency.
*   **Enhancement (CLAHE)**: Contrast Limited Adaptive Histogram Equalization is applied to normalize lighting conditions.
    *   *Clip Limit*: 2.0 | *Tile Grid*: 8x8

### 2.2. Geometric Feature Extraction
We use **MediaPipe Face Mesh** (468 3D landmarks) to calculate:
*   **Eye Aspect Ratio (EAR)**: $EAR = \frac{\|P_2 - P_6\| + \|P_3 - P_5\|}{2 \|P_1 - P_4\|}$
*   **Mouth Aspect Ratio (MAR)**: Used to detect yawning.
*   **Head Pose (solvePnP)**: Euler angles (Yaw, Pitch, Roll) mapped to a generic 3D face model.

---

## 3. Phase 2: Data Engineering & Calibration

### 3.1. Signal Refinement
*   **Interpolation**: Polynomial interpolation (Order 2) handles gaps up to 1 second (4 frames) to maintain signal continuity.
*   **Smoothing**: A **Savitzky-Golay filter** (Window=5, Order=2) removes high-frequency jitter while preserving the depth of blinks.

### 3.2. Dynamic Calibration (Personalization)
*   **Baseline**: The first 5 seconds of the video establish the user's "awake" state.
*   **Threshold ($\alpha$)**: $\alpha = 0.75 \times \text{85th percentile of EAR_{awake}}$.
*   **Physical State**: A frame is marked as "Closed" ($eye\_state=1$) if $EAR < \alpha$.

---

## 4. Ground Truth & Labeling Strategy

We distinguish between **Physical States** (deterministic) and **Physiological States** (behavioral):

### 4.1. Frame-level State (Physical)
*   **Status**: `eye_state` (0: Open, 1: Closed).
*   **Role**: A raw signal feature, **not** an ML target.

### 4.2. Video-level Ground Truth (Physiological)
*   **Target (Label)**: Derived from dataset source.
*   **Levels**: `0` (Alert), `5` (Drowsy), `10` (Sleeping).
*   **Role**: The primary goal for Machine Learning classification.

---

## 5. Phase 3: Machine Learning & Behavioral Analysis

### 5.1. Shifting to Behavioral Classification
To avoid "circular logic," the model does not predict if an eye is closed. Instead, it classifies the driver's drowsiness level by analyzing **patterns of movement** over time.

### 5.2. Statistical Aggregation (60s Sliding Window)
We aggregate data into vectors to capture trends:
*   **PERCLOS**: Percentage of time eyes are closed ($eye\_state=1$).
*   **Blink Frequency**: Number of state transitions per minute.
*   **EAR Variance**: Standard deviation of EAR (indicates stability of gaze).
*   **Pose Stability Index**: Variance in Pitch/Yaw to detect "nodding off."

### 5.3. Model: Behavioral Classifier (Random Forest)
*   **Input**: Aggregated temporal vectors.
*   **Logic**: The model distinguishes an Alert driver (who blinks frequently but briefly) from a Drowsy driver (who has prolonged closures and erratic head movements).
*   **Validation**: **GroupKFold** (Participant-based) to ensure the model generalizes to new faces.

---

## 6. System Robustness: The 4 Protective Layers

1.  **Signal Integrity Layer**: Uses interpolation to prevent face-loss from corrupting temporal statistics.
2.  **Duration Logic Layer**: Pre-filters data by separating biological blinks from physiological drowsiness at the processing level.
3.  **Temporal Persistence Layer**: 60s windows prevent "flickering" alerts caused by momentary noise.
4.  **Contextual Fusion Layer**: Cross-verifies EAR signals with Head Pose data to increase detection confidence.

---

## 7. Architectural Standards
*   **Cross-Platform**: Strict use of `pathlib` and `src/core_config.py`.
*   **Reproducibility**: Global seed set to 42.
*   **Traceability**: All pipeline steps logged in `logs/` with detailed durations.
