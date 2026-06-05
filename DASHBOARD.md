# 📊 Lightweight-DMS Live Dashboard

This is the interactive progress tracker. Members should check the boxes as they complete steps.

[👉 View/Edit Diagram in Mermaid Live Editor](https://mermaid.live/edit#pako:eNptkctuwjAQRX_FmhowpPCh6iIhhIQUtatU7KIszCQmMTYenmAnRPnvnRAnfXvunTtzL_Y-9D4NIsA7v2mE1uD2n1S3YJvG-62Gz5vX9_N0-Ytq_M8U72Zp1Yp8S7Z_Jp9E6iXzEOnFIsZ6idSizNisLNb_xN7zGOmZ5vYQ6clizvR8NGe-HiyfRer5eM78pM-ZrxYzN-19YvWSuT9EeqTYmY470-3Z6f_E5_OIsUonZ53KclO5O8X27_TTmF_T8UUnO_u8XFX9mO5ZzI8Uq5O5P5_5D2xOByv2-vGg-U-2-XF96p-v1e88O1Os97yM9Z4Z_u3M7p-vzeuV7Xf-8Y_T_6D_F-wXfD8)

```mermaid
graph TD
    subgraph Stage1 [Stage 1: Computer Vision]
        S1_1[#8 Preprocessing]
        S1_2[#2 Eye/Mouth Landmarks]
        S1_3[#5-7 Head Pose]
    end

    subgraph Stage23 [Stage 2 & 3: Signal & Logic]
        S2_1[#1 Calibration]
        S2_2[#13 Interpolation/Smoothing]
        S2_3[#17 Duration Logic]
    end

    subgraph Stage4 [Stage 4: Stats Aggregation]
        S4_1[#4 60s Sliding Window]
        S4_2[#9 Contextual Fusion]
        S4_3[#10 Safety Filters]
    end

    subgraph Stage5 [Stage 5: Machine Learning]
        S5_1[#14 Participant Splitting]
        S5_2[ML-04 RF Classifier]
        S5_3[#16 Real-time Integration]
    end

    Stage1 --> Stage23
    Stage23 --> Stage4
    Stage4 --> Stage5
```

---

## 🟢 Phase 1: Computer Vision (Vision Specialist)
- [x] **#8: [Stage 1] Preprocessing (CLAHE) & Face Mesh**
- [x] **#2: [Stage 1] Eye & Mouth Landmark Extraction (EAR/MAR)**
- [ ] **#5: [Stage 1] 3D Model Setup & Core Landmarks**
- [ ] **#6: [Stage 1] solvePnP Config & Camera Matrix**
- [ ] **#7: [Stage 1] Euler Angles (Yaw, Pitch, Roll)**

## 🟡 Phase 2: Data Engineering (Feature Engineer)
- [x] **#1: [Stage 2] Calibration Alpha (Dynamic Thresholds)**
- [x] **#13: [Stage 2] Signal Refinement (Interpolation & Smoothing)**
- [ ] **#17: [Stage 3] Duration Logic (Blinks vs. Micro-sleeps)**
- [ ] **#4: [Stage 4] Statistical Aggregation (60s Sliding Window)**
- [ ] **#9: [Stage 4] Contextual Fusion (EAR + Head Pose)**
- [ ] **#10: [Stage 4] Behavioral Safety Filter (Post-processing)**

## 🔵 Phase 3: Machine Learning (ML Lead)
- [ ] **#14: [Stage 5] Participant-Based Data Splitting**
- [ ] **#15: [Stage 5] Dataset Partitioning (Train/Val/Test)**
- [ ] **ML-04: [Stage 5] Train Behavioral Random Forest Classifier**
- [ ] **#16: [DEP-01] Real-time Pipeline Integration**

---

## 🛠 Management Notes
- **Verification**: Only check a box after the PR has been merged into `main`.
- **Reference**: Consult [METHODOLOGY.md](METHODOLOGY.md) for technical formulas.
