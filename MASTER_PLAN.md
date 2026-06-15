# Master Project Plan: Behavioral Drowsiness Detection (Hybrid)

## 1. Executive Summary
This project implements a multi-stage monitoring pipeline that combines **Geometric Landmark Analysis** with **Deep Contextual Learning**. By using a "Residual Fallback" architecture, we ensure that the system remains reliable even when one data stream (pixels or math) is compromised.

## 2. Strategic Objectives
*   **Accuracy & Recall:** Reach F1-Score > 0.80 on unseen participants (GroupKFold).
*   **Edge Efficiency:** Maintain > 10 FPS execution on standard CPU hardware.
*   **Reproducibility:** Global seed 42, strictly documented normalization and padding.

## 3. The 5-Stage Pipeline
1.  **Extraction:** Dual-stream trích xuất (Geometry + Image Patches).
2.  **Modulation:** FiLM-based feature fusion (Math steers Pixels).
3.  **Persistence:** Gated GRU modeling to handle lost tracking.
4.  **Semantic Shaping:** Contrastive Learning (Triplet Loss) for class separation.
5.  **Residual Guard:** $S = S_{base} + \Delta S$ (Guaranteed $0.5422$ performance).

## 4. Engineering Standards
*   **Paths:** All paths must resolve via `src/core_config.py`.
*   **Validation:** ALWAYS use `GroupKFold` to prevent participant data leakage.
*   **Structure:** Logic in `src/`, processing in `frame/`, results in `report/`.
