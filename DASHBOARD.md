# 📊 Lightweight-DMS Live Dashboard

This is the interactive progress tracker. Members should check the boxes as they complete steps.

---

## 🟢 Phase 1: Computer Vision (Vision Specialist)
*Files: `4fps.py`, `Mesh_apply.py`*

- [x] **#8: Preprocessing (CLAHE)** - *Ensure frames are clear and lighting is balanced.*
- [ ] **#5: 3D Model Setup** - *Verify the 6 core landmarks are extracted correctly.*
- [ ] **#6: solvePnP Config** - *Calibrate the camera matrix for head pose estimation.*
- [ ] **#7: Euler Angles** - *Finalize Yaw, Pitch, and Roll extraction.*

---

## 🟡 Phase 2: Data Engineering (Feature Engineer)
*Files: `to_csv.py`, `calibration.py`*

- [x] **#13: Sliding Windows** - *Implement the temporal windowing logic.*
- [x] **#1: Calibration Alpha** - *Set up dynamic thresholds based on initial frames.*
- [ ] **#9: Contextual Fusion** - *Integrate head pose and EAR signals.*
- [ ] **#10: Context Filter** - *Apply behavioral safety rules to the final signal.*

---

## 🔵 Phase 3: Machine Learning (ML Lead)
*Files: `train_baseline.py`, `eye_state.py`*

- [ ] **#14: ML-03 Data Splitting** - *Implement participant-based GroupKFold.*
- [ ] **#15: Data Partition** - *Create final Train/Val/Test sets.*
- [ ] **ML-04: Random Forest** - *Train and evaluate the baseline model.*
- [ ] **#16: DEP-01 Inference** - *Integrate the model into the real-time main.py.*

---

## 🛠 Management Notes
- **Verification**: Only check a box after the PR has been merged into `main`.
- **Blocked**: If a task is waiting on another phase, leave it unchecked.
