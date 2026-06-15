# Lightweight-DMS Task Board (Hybrid Evolution)

---

## 🔵 Module 1: Data Engineering (Data Squad)
*Goal: Generate the Hybrid Dataset (CSV + Image Patches).*

| ID | Task | Status | Assignee |
| :--- | :--- | :---: | :--- |
| **D-01** | Implement **Isotropic Padding** in image extraction. | Todo | |
| **D-02** | Update `Mesh_apply.py` with 6-anchor **solvePnP**. | Done | |
| **D-03** | Script to generate **10s Sliding Windows** (Sequences). | In Progress | |
| **D-04** | Build **HybridDataLoader** (syncing CSV rows with images). | Todo | |

## 🔴 Module 2: Model Architecture (Model Squad)
*Goal: Build the FiLM-CNN-GRU Network.*

| ID | Task | Status | Assignee |
| :--- | :---: | :---: | :--- |
| **M-01** | Setup **MobileNetV3-Small** backbone for $24 \times 24$ patches. | Todo | |
| **M-02** | Implement **FiLM Layer** (MLP + Modulation). | Todo | |
| **M-03** | Implement **Gated GRU** with [MASK] embedding. | Todo | |
| **M-04** | Integrate **Residual Fallback** layer ($S = Base + \Delta S$). | Todo | |

## 🟡 Module 3: Advanced Training (F1 Squad)
*Goal: Maximize F1-Score via Contrastive Learning.*

| ID | Task | Status | Assignee |
| :--- | :--- | :---: | :--- |
| **T-01** | Implement **Triplet Loss** module for PyTorch. | Todo | |
| **T-02** | Setup **Class-Weighted Attention** mechanism. | Todo | |
| **T-03** | Run **GroupKFold** cross-validation on Hybrid model. | Todo | |

## 🟢 Module 4: Benchmarking (Leader)
*Goal: Final evaluation and paper charts.*

| ID | Task | Status | Assignee |
| :--- | :--- | :---: | :--- |
| **B-01** | Generate **Precision-Recall Curves** (Hybrid vs Base). | Todo | Huy |
| **B-02** | Measure **Inference Latency** on CPU (ms/frame). | Todo | Huy |
| **B-03** | Visualize **FiLM Feature Maps** (Geometry influence). | Todo | Huy |

---

## ✅ Progress Summary
*   **Baseline (Geometry Only):** F1 = $0.5422$ (XGBoost).
*   **Target (Hybrid):** F1 > $0.80$.
