# Project Kanban - Buồn ngủ (Lightweight-DMS)

## Overview
Tracking architectural refactoring, data quality, and modeling progress for the Drowsiness Detection pipeline.

## Backlog

### Data Quality & Processing
| Task ID | Task Description | Status | Priority | Category |
| :--- | :--- | :--- | :--- | :--- |
| FD-02 | Manually audit sample frames in `failed_detections/` | Todo | Medium | Data Quality |
| FD-04 | Test additional preprocessing (sharpening) for failed frames | Todo | Low | Preprocessing |

### Modeling (ML)
| Task ID | Task Description | Status | Priority | Category |
| :--- | :--- | :--- | :--- | :--- |
| ML-01 | Research initial model architectures (Random Forest/LSTM) | Done | Medium | Modeling |
| ML-02 | **Feature Engineering**: Sliding windows & temporal deltas | In Progress | High | Modeling |
| ML-03 | **Data Splitting**: Group split by participant (Train/Val/Test) | Todo | High | Modeling |
| ML-04 | Train Baseline Model (Random Forest) | Todo | Medium | Modeling |
| ML-05 | Train Temporal Model (LSTM or GRU) | Todo | Medium | Modeling |
| ML-06 | **Evaluation**: Metrics report (F1, Confusion Matrix) | Todo | High | Modeling |

### Deployment & UI
| Task ID | Task Description | Status | Priority | Category |
| :--- | :--- | :--- | :--- | :--- |
| DEP-01 | Create Real-time Camera Inference script | Todo | High | Deployment |
| DEP-02 | Implement Visual Feedback/Alert System | Todo | Medium | Deployment |

## In Progress

| Task ID | Task Description | Started | Priority |
| :--- | :--- | :--- | :--- |
| ML-02 | **Feature Engineering**: Sliding windows & temporal deltas | 2026-05-28 | High |

## Completed

| Task ID | Task Description | Completed | Notes |
| :--- | :--- | :--- | :--- |
| ML-01 | Research initial model architectures (Random Forest/LSTM) | 2026-05-28 | Found Random Forest best for baseline; LSTM for temporal |
| SETUP-01 | Project structure and Master Plan established | 2026-05-28 | |
| SETUP-02 | Initial Pipeline (Frames -> Mesh -> CSV) functional | 2026-05-28 | |
| ARCH-02 | Centralize paths/constants in `src/core_config.py` | 2026-05-28 | Unified all scripts; removed redundant configs |
| ARCH-01 | Refactor `to_csv.py` for memory efficiency (chunked processing) | 2026-05-28 | Optimized via `usecols` (99% memory save) |
| FD-01 | Create `src/analyze_failures.py` to find failure clusters | 2026-05-28 | Found 7.88% global failure rate; Participant1 is outlier (31%) |
| FD-03 | **Retry Logic**: Implement low-confidence retry in `mesh_apply.py` | 2026-05-28 | Added 0.15 threshold retry and 'method' logging |
| FD-06 | Log confidence scores (method) in CSV output | 2026-05-28 | Integrated into Mesh_apply.py as 'detection_method' |
| FD-05 | Refine interpolation (e.g., cubic spline) in `to_csv.py` | 2026-05-28 | Upgraded to Polynomial (ord 2) + Savitzky-Golay filter |
| FEAT-03 | **Calibration & Dynamic Alpha**: Implement T_calib & alpha thresholds | 2026-05-28 | Calculated 85th percentile EAR; dynamic eye-state B(t) |
