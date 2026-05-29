# Project Kanban - Buồn ngủ (Lightweight-DMS)

## Overview
Tracking architectural refactoring, data quality, and modeling progress for the Drowsiness Detection pipeline.

## In Progress (UPLOAD HERE)

| Task ID | Task Description | Started | Priority | **Branch / GitHub Location** |
| :--- | :--- | :--- | :--- | :--- |
| **L1-01** | **3D Model & 6 Landmarks** | 2026-05-29 | High | 🚩 **UPLOAD TO:** `feature/pnp-head-pose` |
| **ML-03** | **Data Splitting** | 2026-05-29 | High | 🚩 **UPLOAD TO:** `main` |

## Backlog (Planned Tasks)

### Modeling (ML)
| Task ID | Task Description | Status | **Branch / GitHub Location** |
| :--- | :--- | :--- | :--- |
| **FUSE-01** | **Contextual Fusion** | Todo | 🚩 **USE BRANCH:** `feature/contextual-fusion` |
| **ML-04** | Train Baseline Model (RF) | Todo | → `main` |
| **ML-05** | Train Temporal Model (LSTM) | Todo | → `main` |

### Data Quality & Processing
| Task ID | Task Description | Status | Branch |
| :--- | :--- | :--- | :--- |
| FD-02 | Manually audit sample frames | Todo | `main` |
| FD-04 | Test sharpening for failed frames | Todo | `main` |

## Completed

| Task ID | Task Description | Completed | Notes | Branch |
| :--- | :--- | :--- | :--- | :--- |
| FEAT-03 | **Calibration & Dynamic Alpha** | 2026-05-29 | Calculated 85th percentile EAR | `feature/calibration-alpha-3` |
| PRE-01 | **CLAHE Preprocessing** | 2026-05-29 | Enhanced frame contrast | `feature/preprocessing-clahe` |
| FEAT-01 | **EAR Calculation** | 2026-05-29 | Baseline EAR features | `feature/calculate-ear` |
| FEAT-02 | **PERCLOS** | 2026-05-29 | Temporal eye closure | `feature/perclos-window` |
| SETUP-01 | Project structure and Master Plan established | 2026-05-28 | | |
| ARCH-02 | Centralize paths/constants in `src/core_config.py` | 2026-05-28 | Unified all scripts | |
| FD-01 | Create `src/analyze_failures.py` | 2026-05-28 | Found 7.88% failure rate | |
