# Drowsiness Detection Project Instructions

This project implements a pipeline for drowsiness detection using facial landmark analysis (EAR, MAR, head movement).

## Project Overview

The pipeline processes videos of participants to extract frames, apply face mesh, and calculate features like Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to detect drowsiness.

## Pipeline Architecture

The workflow is orchestrated by `main.py` and consists of three main steps:

1.  **Frame Extraction (`Frame_exrtaction/4fps.py`)**:
    *   Reads videos from `Video_container/`.
    *   Extracts frames at 4 FPS.
    *   Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for better visibility.
    *   Saves raw and processed frames to `frame/frames_raw/` and `frame/frames_clahe/`.

2.  **Face Mesh Processing (`frame/mesh_apply.py`)**:
    *   Uses MediaPipe's Face Mesh to detect landmarks on processed frames.
    *   Calculates EAR, MAR, and head movement features.
    *   Saves landmark data to `frame/csv/landmarks.csv`.
    *   Optionally saves visualization frames to `frame/frames_mesh/`.

3.  **CSV Feature Export (`to_csv.py`)**:
    *   Reads `landmarks.csv`.
    *   Exports a summarized feature set to `frame/csv/features_summary.csv`.

4.  **Calibration (`src/calibration.py`)**:
    *   Uses the first 5 seconds of data to establish a baseline.
    *   Calculates `EAR_open` (85th percentile) and sets a dynamic threshold `alpha = 0.75 * EAR_open`.

5.  **Eye State Calculation (`src/eye_state.py`)**:
    *   Applies the dynamic threshold to calculate the binary eye state `B(t)`.
    *   Saves the results back to `features_summary.csv`.

## Directory Structure

*   `Video_container/`: Input MP4 videos organized by participant.
*   `frame/`: Central storage for processed assets.
    *   `frames_raw/`: Original extracted frames.
    *   `frames_clahe/`: Frames with CLAHE applied.
    *   `frames_mesh/`: Frames with landmarks drawn.
    *   `csv/`: Results (landmarks and summarized features).
*   `Frame_exrtaction/`: Scripts and configuration for frame processing.
*   `report/`: Generated reports and visualizations.
*   `.venv/`: Python virtual environment (recommended for execution).

## Setup and Execution

1.  **Environment**: Ensure the virtual environment is activated.
    ```powershell
    .\.venv\Scripts\activate
    ```
2.  **Configuration**: Global paths and parameters (FPS, dimensions, thresholds) are managed in `Frame_exrtaction/config.py`.
3.  **Running the Pipeline**: Execute the full process via the main script:
    ```powershell
    python main.py
    ```

## Development Guidelines

*   **Path Management**: Always use `Path` from `pathlib` for cross-platform compatibility. Reference `PROJECT_ROOT` from `config.py`.
*   **Logging**: Use the logging configuration in `main.py` for pipeline-wide tracking. Logs are stored in `logs/`.
*   **Task Model**: The project uses `face_landmarker.task` for MediaPipe landmarks. Ensure this file is present in the root.
