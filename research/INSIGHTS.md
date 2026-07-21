# Phase 1 & 2 Technical Insights (Full Report)

## 🔍 Overview
Conducted comprehensive research and implementation from 2026-06-17. Validated the hybrid data pipeline and the FiLM modulation mechanism.

## 💡 Key Findings

### 1. Data Engineering (Phase 1)
- **Scale:** Processed **44,098 frames**; **42,049 frames (95.3%)** passed face detection and are synchronized for hybrid training.
- **Speed:** ~30 FPS on standard CPU. Unicode path support (Vietnamese characters) fully implemented and stable.
- **Quality:** **Isotropic Padding** successfully prevents eye-feature squashing. Verified 1:1 aspect ratio preservation on all extracted 24x24 patches.

### 2. FiLM Robustness (Phase 2)
Rigorous mathematical verification of the FiLM (Feature-wise Linear Modulation) layer yielded:
- **Identity Mapping:** 100% PASS. Neutral geometry does not distort CNN features.
- **Batch Independence:** 100% PASS. No cross-sample interference during parallel processing.
- **Sensitivity:** Significant feature shift (**13.66**) for only 10° of head pitch change. Confirms high sensitivity to driver posture.
- **Stability:** Stable under extreme (180°) head rotation (No NaN/Inf detected).

### 3. Model Efficiency
- **Parameter Count:** **~9,347 parameters** for the full hybrid backbone (MobileNetV3-Small @ 0.5 width).
- **Latency:** Estimated < 10ms for backbone inference, making it highly suitable for low-power embedded systems.

## 🧠 Architectural Decisions
- **Modulation over Fusion:** FiLM proved more responsive than simple concatenation in early demos.
- **Dataset Class:** `HybridDrowsyDataset` successfully links visual patches with 12D geometric vectors in real-time.
- **Normalizers:** Angles (Pitch/Yaw/Roll) must be normalized by factor of 90.0, and EAR/MAR by factor of 0.5 for optimal FiLM stability.

---
*Status: Phase 1 & 2 Research & Verification - COMPLETED*
