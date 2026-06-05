# Lightweight-DMS Task Board (5-Stage Behavioral Architecture)

---

## 🟢 Phase 1: Computer Vision (Vision Specialist)
*Goal: Stage 1 - Ensure high-quality frame extraction and accurate landmark/pose data.*

| Order | Issue | Task Description | Status | Branch |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **#8** | [Stage 1] Preprocessing (CLAHE) & Face Mesh | ✅ Done | `main` |
| **2** | **#2** | [Stage 1] Eye & Mouth Landmark Extraction (EAR/MAR) | ✅ Done | `main` |
| **3** | **#5** | [Stage 1] 3D Model Setup & Core Landmarks | 🚀 In Progress | `feature/pnp-head-pose` |
| **4** | **#6** | [Stage 1] solvePnP Config & Camera Matrix | 🚀 In Progress | `feature/pnp-head-pose` |
| **5** | **#7** | [Stage 1] Euler Angles (Yaw, Pitch, Roll) | 📋 Todo | `feature/pnp-head-pose` |

---

## 🟡 Phase 2: Data Engineering (Feature Engineer)
*Goal: Stages 2, 3 & 4 - Signal refinement, Duration logic, and Statistical aggregation.*

| Order | Issue | Task Description | Status | Branch |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **#1** | [Stage 2] Calibration Alpha (Dynamic Thresholds) | ✅ Done | `main` |
| **2** | **#13** | [Stage 2] Signal Refinement (Interpolation & Smoothing) | ✅ Done | `main` |
| **3** | **#17** | [Stage 3] Duration Logic (Blinks vs. Micro-sleeps) | 📋 Todo | `feature/duration-logic` |
| **4** | **#4** | [Stage 4] Statistical Aggregation (60s Sliding Window) | 🚀 In Progress | `feature/perclos-stats` |
| **5** | **#9** | [Stage 4] Contextual Fusion (EAR + Head Pose) | 📋 Todo | `feature/contextual-fusion` |
| **6** | **#10** | [Stage 4] Behavioral Safety Filter (Post-processing) | 📋 Todo | `main` |

---

## 🔵 Phase 3: Machine Learning (ML Lead)
*Goal: Stage 5 - Behavioral Classification and System Integration.*

| Order | Issue | Task Description | Status | Branch |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **#14** | [Stage 5] Participant-Based Data Splitting (GroupKFold) | 🚀 In Progress | `main` |
| **2** | **#15** | [Stage 5] Dataset Partitioning (Train/Val/Test) | 📋 Todo | `main` |
| **3** | **ML-04** | [Stage 5] Train Behavioral Random Forest Classifier | 🚀 In Progress | `feature/ML-04-behavioral` |
| **4** | **#16** | [DEP-01] Real-time Pipeline Integration | 📋 Todo | `main` |

---

## ✅ Legend
- **Order**: Follow this sequence strictly to maintain Stage dependencies.
- **Done**: Merged into `main`.
- **In Progress**: Coding currently happening.
- **Todo**: Waiting for the previous step to finish.
