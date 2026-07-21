# Technical Plan & Hyperparameter Tracker

This document tracks the detailed technical specifications and hyperparameters for each phase of the project. It is a **living document** and should be updated whenever a configuration change is made.

---

## 🛠️ Phase 1: Data Engineering (Pre-processing)
*Objective: High-fidelity feature extraction and normalization.*

| Category | Hyperparameter | Current Value | Note |
| :--- | :--- | :--- | :--- |
| **Video** | Target FPS | `4` | Balanced for real-time & motion. |
| **Image** | CLAHE Clip Limit | `2.0` | Controls contrast enhancement. |
| **Image** | CLAHE Tile Grid | `(8, 8)` | Local contrast area. |
| **Ocular** | EAR Smoothing Window | `3` (frames) | Moving average to reduce noise. |
| **Ocular** | Normalization Window | `10s` (40 frames) | Dynamic Min-Max scaling. |
| **Pose** | Anchor Points | `6` points | [1, 152, 33, 263, 61, 291]. |
| **Patches**| Isotropic Resizing | `24x24` | 1:1 Aspect Ratio + Black Padding. |

---

## 🧠 Phase 2: Hybrid Model Architecture (FiLM + GRU)
*Objective: Fusing geometric context with visual appearance.*

| Module | Component | Hyperparameter | Value |
| :--- | :--- | :--- | :--- |
| **CNN** | Backbone | MobileNetV3-Small | Width Multiplier: `0.5`. |
| **CNN** | Input Channels | `1` (Grayscale) | Reduced CPU/Memory footprint. |
| **FiLM** | MLP Hidden Layers | `[64, 32]` | Vector -> $(\gamma, \beta)$ mapping. |
| **FiLM** | Activation | `ReLU` | Non-linearity for modulation. |
| **GRU** | Hidden Size | `128` | Temporal state memory. |
| **GRU** | Layers | `1` | Sufficient for 10s window. |
| **GRU** | Dropout | `0.2` | Regularization. |

---

## 🚀 Phase 3: Training Strategy
*Objective: Robust convergence and class separation.*

| Category | Hyperparameter | Current Value | Note |
| :--- | :--- | :--- | :--- |
| **Optimizer** | Type | `AdamW` | Better weight decay handling. |
| **Optimizer** | Learning Rate | `1e-4` | Validated via Mini-Experiment (Stable convergence). |
| **Loss** | Base Loss | `CrossEntropy` | Primary classification. |
| **Batch** | Batch Size | `32` | Optimized for CPU; can scale to `128` on GPU. |
| **Training**| Epochs | `50` | With Early Stopping (patience=7). |
| **Model** | Complexity | `~9.3k params` | Extremely lightweight for real-time inference. |

---

## 🛡️ Phase 4: Residual Fallback System
*Objective: Performance guarantee via deterministic baseline.*

| Component | Logic | Hyperparameter | Value |
| :--- | :--- | :--- | :--- |
| **Baseline** | XGBoost $S_{base}$ | F1 Target: `0.5422` | Constant baseline from Geo. |
| **Delta** | DL Refinement $\Delta S$ | Max Range | `± 0.15` |
| **Activation**| Scaling | `Tanh` | $S = S_{base} + \text{Tanh}(\Delta S) \times 0.15$. |

---

## 📊 Performance Log (Incremental)
| Date | Change Description | F1-Score | Status |
| :--- | :--- | :--- | :--- |
| 2026-06-17 | Established Baseline (XGB) | 0.5422 | 🟢 Baseline |
| 2026-06-17 | Mini-Experiment (CNN 24x24) | N/A (Sanity) | 🟡 Arch Stable |
| | | | |

| | | | |
