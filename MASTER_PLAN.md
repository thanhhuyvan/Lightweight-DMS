# Master Project Plan: Robust Hybrid Drowsiness Detection

## 1. Project Vision
To build a high-performance, real-time Driver Monitoring System (DMS) that fuses geometric "math" with visual "context" using a **Hybrid Residual** architecture. The system prioritizes safety by ensuring a deterministic baseline performance while leveraging Deep Learning for state-of-the-art accuracy.

**Primary Goal:** Achieve **F1-Score > 0.80** using FiLM + GRU with a **Residual Fallback** mechanism.

---

## 2. Phase 1: Multimodal Data Engineering
*Goal: Transform raw frames and landmarks into synchronized hybrid features.*

- **Task 1.1: Isotropic Patch Extraction**
    - Implement `patch_extractor.py` for Eye and Mouth regions.
    - Apply **Isotropic Rule**: Pad patches to 1:1 ratio before resizing to $24 \times 24$.
    - Standardize to Grayscale to minimize CPU load.
- **Task 1.2: Geometric Feature Refinement**
    - Validate **solvePnP** Head Pose (Pitch, Yaw, Roll).
    - Implement **Asymmetry EAR** ($|EAR_L - EAR_R|$).
    - Ensure **Per-Participant Scaling**: 10s window continuous Min-Max normalization.
- **Task 1.3: Hybrid Dataset Synchronization**
    - Build `HybridDataset` (PyTorch) to map CSV geometric vectors to corresponding image patches.
    - Implement temporal windowing (10s segments) for GRU consumption.

## 3. Phase 2: Hybrid Model Architecture (FiLM + GRU)
*Goal: Create a "Geometry-Steered" neural network.*

- **Task 2.1: CNN Backbone**
    - Customize **MobileNetV3-Small** for 1-channel $24 \times 24$ inputs.
- **Task 2.2: FiLM Modulation**
    - Build **MLP Parameter Generator**: Geometry Vector $\rightarrow (\gamma, \beta)$.
    - Integrate **FiLM Layers** into MobileNet blocks to modulate features based on head pose and EAR.
- **Task 2.3: Temporal Persistence**
    - Implement **Single-Layer GRU** to handle video sequence dynamics.
    - Add **Confidence Gating**: Use [MASK] embeddings when MediaPipe confidence $< 0.4$.

## 4. Phase 3: Residual Fallback System
*Goal: Implement the "Safety Net" that prevents performance degradation.*

- **Task 3.1: Baseline Lock-in**
    - Freeze the **XGBoost Geometry Baseline** (F1: 0.5422) as the $S_{base}$ provider.
- **Task 3.2: Residual Summation Layer**
    - Implement $S_{final} = S_{base} + \text{Tanh}(\Delta S) \times 0.15$.
    - Ensure the Deep Learning branch only provides a "correction" delta.

## 5. Phase 4: Advanced Training & Optimization
*Goal: Maximize class separation and robustness.*

- **Task 4.1: Semantic Shaping (Triplet Loss)**
    - Train the embedding space using **TripletMarginLoss** to cluster "Drowsy" states away from "Alert".
- **Task 4.2: Temporal Attention**
    - Implement **Class-Weighted Attention** to prioritize recall for "Drowsy" and "Low Vigilant" states.
- **Task 4.3: Robust Validation**
    - Execute **GroupKFold (5-Fold)** cross-validation to ensure zero participant leakage.

## 6. Phase 5: Benchmarking & Diagnostic reporting
*Goal: Quantify success and verify real-time viability.*

- **Task 5.1: Performance Analytics**
    - Generate Confusion Matrices, F1-Macro reports, and Precision-Recall curves.
- **Task 5.2: Latency & Profiling**
    - Benchmark inference speed on CPU (Target: $< 20ms$).
- **Task 5.3: Qualitative Viz**
    - Use Grad-CAM or similar to visualize what the CNN focuses on under different head poses (modulated by FiLM).

---

## 📅 Roadmap Milestones
- [x] **MS-01: Baseline established** (F1 0.5422).
- [x] **MS-02: Isotropic Data Engine Ready** (Syncing Patches + Geo).
- [x] **MS-03: FiLM Core Integration** (Geometry steers CNN).
- [ ] **MS-04: Temporal Alpha Test** (GRU sequence modeling).
- [ ] **MS-05: Final Hybrid Delivery** (F1 > 0.80).
