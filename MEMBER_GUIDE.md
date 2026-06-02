# 📘 Lightweight-DMS: Master Member Guide

This document provides technical instructions and implementation standards for all active and upcoming tasks in the Drowsiness Detection Pipeline.

---

## 🟢 Phase 1: Computer Vision (Vision Specialist)

### **#2: Eye Landmark Extraction & EAR Calculation**
*   **Goal**: Extract 6 landmarks per eye and calculate the stable Eye Aspect Ratio (EAR).
*   **MediaPipe Indices**:
    *   **Right Eye**: `[33, 160, 158, 133, 153, 144]` (P1 to P6)
    *   **Left Eye**: `[362, 385, 386, 263, 374, 380]`
*   **Formula**: $EAR = \frac{\|P_2 - P_6\| + \|P_3 - P_5\|}{2 \|P_1 - P_4\|}$
*   **Standard**: Save both `left_EAR`, `right_EAR`, and `mean_EAR` to `landmarks_full.csv`.

### **#5 & #6: 3D Model & solvePnP (Head Pose)**
*   **Goal**: Transition from simple movement to 3D Euler angles (Yaw, Pitch, Roll).
*   **3D Model Points (Generic)**:
    *   Nose: `(0.0, 0.0, 0.0)`
    *   Chin: `(0.0, -330.0, -65.0)`
    *   Left Eye Corner: `(-225.0, 170.0, -135.0)`
    *   Right Eye Corner: `(225.0, 170.0, -135.0)`
*   **Implementation**: Use `cv2.solvePnP` with a camera matrix approximated by image width as focal length.
*   **Output**: Save `yaw`, `pitch`, and `roll` to the CSV.

---

## 🟡 Phase 2: Data Engineering (Feature Engineer)

### **#4: PERCLOS Sliding Window Setup (60s)**
*   **Goal**: Calculate the Percentage of Eye Closure over a 60-second temporal window.
*   **Formula**: $\text{PERCLOS} = \frac{\text{Frames where Eye State == 1}}{\text{Total Frames in Window}} \times 100$
*   **Parameters**:
    *   Window Size: 60 seconds (at 4 FPS, this is 240 frames).
    *   Stride: 1 second (4 frames).
*   **Standard**: Use `pandas.rolling()` to calculate this efficiently on `features_summary.csv`.

### **#9: Contextual Fusion (Rule-based Logic)**
*   **Goal**: Combine EAR and Head Pose to detect "Compensatory Behaviors".
*   **Logic Example**: If EAR indicates closure AND Head Pose indicates a sudden "nod" (pitch change), increase the Drowsiness Confidence Score.
*   **Standard**: Create a new column `fusion_score` (0.0 to 1.0).

---

## 🔵 Phase 3: Machine Learning (ML Lead)

### **#14 (ML-03): Participant-Based Data Splitting**
*   **Goal**: Partition data so that a participant in 'Train' never appears in 'Test'.
*   **Implementation**: Use `sklearn.model_selection.GroupKFold` or `StratifiedGroupKFold`.
*   **Groups**: Set `groups=df['participant_id']`.
*   **Standard**: Export a metadata file `data_splits.json` containing the participant IDs assigned to each fold.

### **ML-04: Train Random Forest Baseline**
*   **Goal**: Train the first predictive model using engineered features.
*   **Features**: `mean_EAR_smooth`, `MAR_smooth`, `head_dx_smooth`, `head_dy_smooth`, `PERCLOS`.
*   **Target**: `eye_state` (or a ground truth label if available).
*   **Metric**: Focus on **Recall** (minimizing false negatives for drowsiness).

---

## 🛠️ General Implementation Standards
1.  **Paths**: Use `from src.core_config import PROJECT_ROOT` for all file references.
2.  **Logging**: Use `logging.info()` instead of `print()`.
3.  **Error Handling**: Wrap CSV reading in `try-except` to handle missing face detections (NaNs).
