# 📘 Lightweight-DMS: Master Member Guide

This guide provides technical specifications for implementing each stage of the drowsiness detection pipeline.

---

## 🟢 Stage 1: Computer Vision (Vision Specialist)

### **Landmark Calculation (EAR/MAR)**
*   **Indices**: Right Eye `[33, 160, 158, 133, 153, 144]`, Left Eye `[362, 385, 386, 263, 374, 380]`.
*   **Geometric Formula**: $EAR = \frac{\|P_2 - P_6\| + \|P_3 - P_5\|}{2 \|P_1 - P_4\|}$

### **3D Pose Estimation**
*   **Implementation**: Use `cv2.solvePnP`.
*   **Required Outputs**: `yaw`, `pitch`, and `roll` angles saved to `landmarks_full.csv`.

---

## 🟡 Stage 2 & 3: Data Engineering (Feature Engineer)

### **Stage 2: Signal Integrity**
*   **Interpolation**: Use `pandas.interpolate(method='polynomial', order=2)` for gaps $\le 4$ frames.
*   **Smoothing**: Apply `savgol_filter(window_length=5, polyorder=2)`.

### **Stage 3: Duration Logic (Filtering)**
*   **Goal**: Distinguish physical blinks from physiological closures.
*   **Classification Rules**:
    *   `blink`: $eye\_state=1$ for $\le 2$ frames ($0.5s$).
    *   `micro_sleep`: $eye\_state=1$ for $\ge 4$ consecutive frames ($1.0s$).
*   **Metric**: Calculate `closure_frequency` and `average_closure_duration` per video segment.

---

## 🟡 Stage 4: Statistical Aggregation (Feature Engineer)

### **Behavioral Vector Generation**
*   **Window Size**: 240 frames (60 seconds).
*   **Stride**: 4 frames (1 second).
*   **Key Features**:
    *   `PERCLOS`: Frames with $eye\_state=1$ per window.
    *   `EAR_Var`: Variance of the EAR signal (Indicates gaze stability).
    *   `Pose_Jitter`: Sum of variances of Yaw, Pitch, and Roll.

---

## 🔵 Stage 5: Machine Learning (ML Lead)

### **Training Protocol**
*   **Target Variable**: `video_id` (0: Alert, 5: Drowsy, 10: Sleep).
*   **Model**: Random Forest Classifier with `n_estimators=100`.
*   **Splitting**: `GroupKFold` using `participant_id` as the grouping factor.
*   **Optimization**: Prioritize **Recall** for the 'Sleeping' (10) class.

---

## 🛠️ Operational Standards
1.  **Centralized Config**: Never hardcode paths. Import `PROJECT_ROOT` from `src.core_config`.
2.  **Failure Analysis**: Log the percentage of `face_detected == False` for every processed video.
3.  **Output Integrity**: Final features must be merged into `frame/csv/features_summary.csv`.
