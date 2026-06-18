# Risk Assessment & Mitigation Strategy

Before moving to Phase 3 (Full Training), we have identified the following technical and operational risks.

## 🔴 High Risk: Technical & Data

| Risk | Impact | Mitigation Strategy |
| :--- | :---: | :--- |
| **MediaPipe "Jitter"** | High | Use the **Temporal Smoothing (Window=3)** already implemented in `src/to_csv.py` and the **Confidence Gating ([MASK] token)** for values < 0.15. |
| **Extreme Class Imbalance** | High | Apply **WeightedRandomSampler** in PyTorch and use **Triplet Margin Loss** to force separation even with fewer "Drowsy" samples. |
| **FiLM Overpowering CNN** | Med | If the MLP learns faster than the CNN, the model might ignore image patches. Mitigation: Use a **smaller Learning Rate for the MLP** than the CNN backbone. |
| **Residual Fallback Ceiling** | Med | If the XGBoost baseline (0.54) is too weak, a ±0.15 delta only reaches 0.69. Mitigation: If F1 < 0.80, we may need to increase the delta range to ±0.25 after validating stability. |

## 🟡 Medium Risk: Performance & Deployment

| Risk | Impact | Mitigation Strategy |
| :--- | :---: | :--- |
| **CPU Inference Latency** | Med | MobileNetV3-Small with `width_mult=0.5` is designed for this. If latency > 20ms, we will apply **Post-Training Quantization (INT8)**. |
| **Lighting Sensitivity** | Med | The use of **CLAHE** in Phase 1 mitigates IR video contrast issues, but we should add **Random Brightness Augmentation** during training. |
| **Sequence Length Error** | Med | If 10s (40 frames) is too long for the GRU to keep hidden state relevant, we will test **Temporal Attention Layers** to focus on the most "informative" frames. |

## 🟢 Low Risk: Validation & Engineering

| Risk | Impact | Mitigation Strategy |
| :--- | :---: | :--- |
| **Participant Leakage** | Low | Strictly enforce **GroupKFold (By ParticipantID)**. Never shuffle frames between participants during the train/test split. |
| **Unicode Path Failures** | Low | Already addressed with `imread_unicode` and `imwrite_unicode` wrappers in Phase 1. |

---

## 🛡️ The "Plan B" Protocols

1. **If FiLM fails to improve accuracy:** We will pivot to a **Multi-Stage Concatenation** (Feature Fusion) instead of Modulation.
2. **If GRU is too slow:** We will switch to a **1D-CNN Temporal Backbone** which is more parallelizable on CPU.
3. **If F1 Score plateaus at 0.70:** We will introduce **Synthetic Data Augmentation** (GAN or geometric eye-closing) to balance the "Drowsy" class.

---
*Last Updated: 2026-06-17*
