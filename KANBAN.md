# Project Kanban - Buồn ngủ (Lightweight-DMS)

## Overview
Tracking architectural refactoring, data quality, and modeling progress for the Drowsiness Detection pipeline.

## Backlog

### Data Quality & Processing
| Task ID | Task Description | Status | Priority | Category | Branch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PRE-01 | **CLAHE Preprocessing**: Enhance frame contrast | Done | High | Preprocessing | `feature/preprocessing-clahe` |
| FD-02 | Manually audit sample frames in `failed_detections/` | Todo | Medium | Data Quality | `main` |
| FD-04 | Test additional preprocessing (sharpening) for failed frames | Todo | Low | Preprocessing | `main` |

### Modeling (ML)
| Task ID | Task Description | Status | Priority | Category | Branch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| L1-01 | **3D Model & 6 Landmarks**: PnP Head Pose baseline | In Progress | High | Modeling | `feature/pnp-head-pose` |
| FEAT-01 | **EAR Calculation**: Eye Aspect Ratio implementation | Done | High | Modeling | `feature/calculate-ear` |
| FEAT-02 | **PERCLOS**: Sliding window eye closure percentage | Done | High | Modeling | `feature/perclos-window` |
| ML-01 | Research initial model architectures (Random Forest/LSTM) | Done | Medium | Modeling | `main` |
| ML-02 | **Feature Engineering**: Sliding windows & temporal deltas | Done | High | Modeling | `main` |
| ML-03 | **Data Splitting**: Group split by participant (Train/Val/Test) | In Progress | High | Modeling | `main` |
| ML-04 | Train Baseline Model (Random Forest) | Todo | Medium | Modeling | `main` |
| ML-05 | Train Temporal Model (LSTM or GRU) | Todo | Medium | Modeling | `main` |
| ML-06 | **Evaluation**: Metrics report (F1, Confusion Matrix) | Todo | High | Modeling | `main` |
| FUSE-01 | **Contextual Fusion**: Multi-feature integration | Todo | Medium | Modeling | `feature/contextual-fusion` |

### Deployment & UI
| Task ID | Task Description | Status | Priority | Category | Branch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DEP-01 | Create Real-time Camera Inference script | Todo | High | Deployment | `main` |
| DEP-02 | Implement Visual Feedback/Alert System | Todo | Medium | Deployment | `main` |

## In Progress

| Task ID | Task Description | Started | Priority | Branch |
| :--- | :--- | :--- | :--- | :--- |
| L1-01 | **3D Model & 6 Landmarks** | 2026-05-29 | High | `feature/pnp-head-pose` |
| ML-03 | **Data Splitting**: Group split by participant | 2026-05-29 | High | `main` |

## Completed

| Task ID | Task Description | Completed | Notes | Branch |
| :--- | :--- | :--- | :--- | :--- |
| FEAT-03 | **Calibration & Dynamic Alpha** | 2026-05-29 | Calculated 85th percentile EAR | `feature/calibration-alpha-3` |
| PRE-01 | **CLAHE Preprocessing** | 2026-05-29 | Enhanced frame contrast | `feature/preprocessing-clahe` |
| FEAT-01 | **EAR Calculation** | 2026-05-29 | Baseline EAR features | `feature/calculate-ear` |
| FEAT-02 | **PERCLOS** | 2026-05-29 | Temporal eye closure | `feature/perclos-window` |
| SETUP-01 | Project structure and Master Plan established | 2026-05-28 | |
| SETUP-02 | Initial Pipeline (Frames -> Mesh -> CSV) functional | 2026-05-28 | |
| ARCH-02 | Centralize paths/constants in `src/core_config.py` | 2026-05-28 | Unified all scripts; removed redundant configs |
| ARCH-01 | Refactor `to_csv.py` for memory efficiency (chunked processing) | 2026-05-28 | Optimized via `usecols` (99% memory save) |
| FD-01 | Create `src/analyze_failures.py` to find failure clusters | 2026-05-28 | Found 7.88% global failure rate; Participant1 is outlier (31%) |
| FD-03 | **Retry Logic**: Implement low-confidence retry in `mesh_apply.py` | 2026-05-28 | Added 0.15 threshold retry and 'method' logging |
| FD-06 | Log confidence scores (method) in CSV output | 2026-05-28 | Integrated into Mesh_apply.py as 'detection_method' |
| FD-05 | Refine interpolation (e.g., cubic spline) in `to_csv.py` | 2026-05-28 | Upgraded to Polynomial (ord 2) + Savitzky-Golay filter |
