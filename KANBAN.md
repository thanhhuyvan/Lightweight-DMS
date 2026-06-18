# Lightweight-DMS Task Board (Hybrid Evolution)

---

## 🔵 Module 1: Data Engineering (Data Squad)
*Goal: Prepare the Multimodal Dataset (Syncing Math + Image Patches).*

| ID | Task | Status | Assignee |
| :--- | :--- | :---: | :--- |
| **D-01.1** | Create `patch_extractor.py` to define eye/mouth BBoxes from landmarks. | Done | |
| **D-01.2** | Implement **Isotropic Padding** logic (1:1 aspect ratio) with black pixels. | Done | |
| **D-01.3** | Implement grayscale conversion and $24 \times 24$ patch resizing. | Done | |
| **D-01.4** | Batch process all frames to `frame/patches/`. | Done | |
| **D-04.1** | Build `HybridDataset` class to sync CSV rows with Image Patches. | Done | |

## 🔴 Module 2: Model Architecture (Model Squad)
*Goal: Build the FiLM-CNN-GRU Network.*

| ID | Task | Status | Assignee |
| :--- | :--- | :---: | :--- |
| **M-01.1** | Setup **MobileNetV3-Small** backbone with 1-channel input modification. | Done | |
| **M-02.1** | Build **MLP Generator** (Input: 12D Geo Vector -> Output: $\gamma, \beta$). | Done | |
| **M-02.2** | Implement **FiLM Modulation Layer** in PyTorch. | Done | |
| **M-03.1** | Setup **GRU Sequential Layer** with hidden state persistence. | Done | |
| **M-04.1** | Implement **Residual Fallback Logic** ($S_{final} = S_{base} + \Delta S$). | Todo | |

## 🟡 Module 3: Advanced Training (F1 Squad)
*Goal: Maximize F1-Score via Contrastive Learning.*

| ID | Task | Status | Assignee |
| :--- | :--- | :---: | :--- |
| **T-01.1** | Implement **TripletMarginLoss** with Hard Negative Mining. | Todo | |
| **T-02.1** | Setup **Class-Weighted Attention** for temporal weighting. | Todo | |
| **T-03.1** | Run **GroupKFold** (5 folds) with F1-Macro monitoring. | Todo | |

## 🟢 Module 4: Benchmarking & Visualization (Leader)
*Goal: Academic reporting and validation.*

| ID | Task | Status | Assignee |
| :--- | :--- | :---: | :--- |
| **B-01** | Generate **Precision-Recall Curves** (Hybrid vs. Baseline). | Todo | Huy |
| **B-02** | Measure **Inference Latency** on CPU (Goal: < 20ms). | Todo | Huy |
| **B-03** | **Qualitative Analysis**: Visualize what the CNN sees after FiLM. | Todo | Huy |

---

## ✅ Completed Milestones
- [x] **D-02:** Real 3D Head Pose (PnP) integration.
- [x] **D-03:** 10s Sliding Windows aggregation.
- [x] **ML-01:** Established 0.5422 F1-Score Geometry Baseline.
