# Lightweight-DMS Task Board (Phased Sequence)

This board is organized by **Phase Order**. Members should complete tasks in the numerical order specified for their role.

---

## 🟢 Phase 1: Computer Vision (Vision Specialist)
*Goal: Ensure high-quality frame extraction and accurate landmark/pose data.*

| Order | Issue | Task Description | Status | Branch |
| :--- | :--- | :--- | :--- | :--- |
| **1 (Begin)** | **#8** | [Postprocessing] Preprocessing (CLAHE) & Face Mesh | ✅ Done | `feature/preprocessing-clahe` |
| **2** | **#5** | [Layer 1] 3D Model Setup & Core Landmarks | 🚀 In Progress | `feature/pnp-head-pose` |
| **3** | **#6** | [Layer 1] solvePnP Config & Camera Matrix | 🚀 In Progress | `feature/pnp-head-pose` |
| **4 (End)** | **#7** | [Layer 1] Euler Angles (Yaw, Pitch, Roll) | 📋 Todo | `feature/pnp-head-pose` |

---

## 🟡 Phase 2: Data Engineering (Feature Engineer)
*Goal: Convert raw landmarks into meaningful signals and cleaned CSVs.*

| Order | Issue | Task Description | Status | Branch |
| :--- | :--- | :--- | :--- | :--- |
| **1 (Begin)** | **#13** | [ML-02] Feature Engineering: Sliding windows | ✅ Done | `main` |
| **2** | **#1** | [Logic] Calibration Alpha (Dynamic Thresholds) | ✅ Done | `feature/calibration-alpha-3` |
| **3** | **#9** | [Logic] Contextual Fusion (Rule-based Logic) | 📋 Todo | `feature/contextual-fusion` |
| **4 (End)** | **#10** | [Logic] Context Filter (Behavioral Safety) | 📋 Todo | `main` |

---

## 🔵 Phase 3: Machine Learning (ML Lead)
*Goal: Build the intelligence to detect drowsiness with high accuracy.*

| Order | Issue | Task Description | Status | Branch |
| :--- | :--- | :--- | :--- | :--- |
| **1 (Begin)** | **#14** | [ML-03] Participant-Based Data Splitting | 🚀 In Progress | `main` |
| **2** | **#15** | [Dataset] Train/Val/Test Partitioning | 📋 Todo | `main` |
| **3** | **ML-04** | [ML-04] Train Random Forest Baseline | 🚀 In Progress | `feature/ML-04-random-forest` |
| **4 (End)** | **#16** | [DEP-01] Real-time Camera Inference | 📋 Todo | `main` |

---

## ✅ Legend
- **Order**: Follow this sequence strictly.
- **Done**: Merged into `main`.
- **In Progress**: Coding currently happening.
- **Todo**: Waiting for the previous step to finish.
