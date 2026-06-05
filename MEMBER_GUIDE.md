# 📘 Lightweight-DMS: Master Member Guide

This document provides technical specifications and implementation standards for the 5-Stage Behavioral Pipeline.

---

## 🟢 Phase 1: Computer Vision (Vision Specialist)

### **#2: Landmark Extraction (EAR/MAR)**
*   **Goal**: Extract 6 landmarks per eye and calculate stable ratios.
*   **Right Eye**: `[33, 160, 158, 133, 153, 144]` | **Left Eye**: `[362, 385, 386, 263, 374, 380]`
*   **Formula**: $EAR = \frac{\|P_2 - P_6\| + \|P_3 - P_5\|}{2 \|P_1 - P_4\|}$

### **#5, #6, #7: 3D Head Pose (Euler Angles)**
*   **Implementation**: Use `cv2.solvePnP` with a generic 3D face model.
*   **Output**: Save `yaw` (horizontal), `pitch` (vertical/nodding), and `roll` (tilt) to the CSV.

---

## 🟡 Phase 2: Data Engineering (Feature Engineer)

### **#17: [Stage 3] Duration Logic (Blinks vs. Micro-sleeps)**
*   **Goal**: Segment continuous eye closure events.
*   **Standard**:
    *   **Blink**: `eye_state == 1` for 1-2 frames ($< 0.5s$ at 4 FPS).
    *   **Micro-sleep**: `eye_state == 1` for $> 4$ consecutive frames ($> 1.0s$).
*   **Task**: Create a script that calculates `closure_duration` and flags micro-sleep events.

### **#4: [Stage 4] Statistical Aggregation (60s Window)**
*   **Parameters**: Window = 240 frames (60s) | Stride = 4 frames (1s).
*   **Features to Calculate**:
    *   `PERCLOS`: (Frames where $eye\_state=1$) / 240.
    *   `Blink_Rate`: Number of blinks per minute.
    *   `EAR_Std`: Standard deviation of EAR (High std = gaze instability).
    *   `Pose_Jitter`: Variance of Pitch/Yaw angles.

---

## 🔵 Phase 3: Machine Learning (ML Lead)

### **#14: [Stage 5] Participant-Based Data Splitting**
*   **Logic**: Use `sklearn.model_selection.GroupKFold`.
*   **Groups**: `df['participant_id']`.
*   **Standard**: Export `data_splits.json` to track which participants are in Train/Val/Test.

### **ML-04: Behavioral Random Forest Classifier**
*   **Input**: Statistical vectors from Stage 4 (PERCLOS, EAR_Std, Blink_Rate, etc.).
*   **Target**: `video_id` (0: Alert, 5: Drowsy, 10: Sleeping).
*   **Objective**: Train the model to recognize "Behavioral Signatures" of drowsiness rather than frame-by-frame closure.
*   **Metric**: Prioritize **Recall** on label 10 (Critical Sleep detection).

---

## 🛠️ General Implementation Standards
1.  **Paths**: Use `from src.core_config import PROJECT_ROOT` for all file references.
2.  **Failure Handling**: All Stage 3-5 scripts must handle `face_detected == False` using interpolation or by marking the window as "High Uncertainty."
3.  **Centralization**: All final features MUST be saved into `frame/csv/features_summary.csv`.
