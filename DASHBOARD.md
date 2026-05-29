# 📊 Lightweight-DMS Live Dashboard

This is the interactive progress tracker. Members should check the boxes as they complete steps.

---

## 🟢 Phase 1: Computer Vision (Vision Specialist)
*Files: `4fps.py`, `Mesh_apply.py`*

- [x] **#8: [Postprocessing] Preprocessing (CLAHE) & Face Mesh** - *Ensure frames are clear and lighting is balanced.*
- [ ] **#5: [Layer 1] 3D Model Setup & Core Landmarks** - *Verify the 6 core landmarks are extracted correctly.*
- [ ] **#6: [Layer 1] solvePnP Config & Camera Matrix** - *Calibrate the camera matrix for head pose estimation.*
- [ ] **#7: [Layer 1] Euler Angles (Yaw, Pitch, Roll)** - *Finalize Yaw, Pitch, and Roll extraction.*

---

## 🟡 Phase 2: Data Engineering (Feature Engineer)
*Files: `to_csv.py`, `calibration.py`*

- [x] **#13: [ML-02] Feature Engineering: Sliding windows** - *Implement the temporal windowing logic.*
- [x] **#1: [Logic] Calibration Alpha (Dynamic Thresholds)** - *Set up dynamic thresholds based on initial frames.*
- [ ] **#9: [Logic] Contextual Fusion (Rule-based Logic)** - *Integrate head pose and EAR signals.*
- [ ] **#10: [Logic] Context Filter (Behavioral Safety)** - *Apply behavioral safety rules to the final signal.*

---

## 🔵 Phase 3: Machine Learning (ML Lead)
*Files: `train_baseline.py`, `eye_state.py`*

- [ ] **#14: [ML-03] Participant-Based Data Splitting** - *Implement participant-based GroupKFold.*
- [ ] **#15: [Dataset] Train/Val/Test Partitioning** - *Create final Train/Val/Test sets.*
- [ ] **ML-04: [ML-04] Train Random Forest Baseline** - *Train and evaluate the baseline model.*
- [ ] **#16: [DEP-01] Real-time Camera Inference** - *Integrate the model into the real-time main.py.*

---

## 🛠 Management Notes
- **Verification**: Only check a box after the PR has been merged into `main`.
- **Blocked**: If a task is waiting on another phase, leave it unchecked.
