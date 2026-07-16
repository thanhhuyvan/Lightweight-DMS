# Docker Deployment Pipeline
## Lightweight Driver Monitoring System — Stage E (FiLM+GRU)

---

## Overview

This folder contains everything needed to build and run the DMS inference pipeline
as a self-contained Docker container. The container accepts a video file and outputs
per-window drowsiness predictions as CSV + JSON. No webcam, no GUI, fully headless.

```
Video file (.mp4 / .avi)
        |
        v
+------------------------------------------+
|           Docker Container               |
|                                          |
|  1. CLAHE Preprocessing                  |
|        |                                 |
|  2. MediaPipe Face Mesh (478 landmarks)  |
|        |                                 |
|  3. Isotropic Patch Extraction           |
|     Left Eye / Right Eye / Mouth         |
|     24x24 px grayscale                   |
|        |                                 |
|  4. Geometry Feature Computation         |
|     EAR, PERCLOS, MAR, Pose Jitter ...   |
|        |                                 |
|  5. FiLM+GRU+Attention Inference         |
|     Stage E  |  9,347 params  |  CPU     |
|        |                                 |
|  6. Per-window Prediction Output         |
+------------------------------------------+
        |
        v
 predictions.csv  +  summary.json
```

---

## Folder Contents

```
docker/
├── DOCKER_PIPELINE.md        <- This file
├── Dockerfile                <- Container build instructions
├── requirements-docker.txt   <- Pinned CPU-only dependencies
├── .dockerignore             <- Excludes large data folders from build context
└── predict.py                <- Headless inference entrypoint
```

Files copied INTO the container at build time (from project root):
```
models/film_gru_fold3.onnx    <- Best single-fold checkpoint in ONNX format
face_landmarker.task          <- MediaPipe landmark model (3.5 MB)
src/core_config.py            <- Pipeline configuration constants
docker/predict.py             <- Inference script
```

---

## Prerequisites

- Docker Desktop installed and running
- `models/film_gru_fold3.onnx` exists in project root (run `python src/export_onnx.py` to generate it)
- `face_landmarker.task` exists in project root
- A video file to run inference on (.mp4 or .avi)

---

## Build

Run from the **project root** (not from inside docker/):

```bash
docker build -f docker/Dockerfile -t dms-inference:latest .
```

First build: ~1-2 minutes (downloads ONNX Runtime and standard packages).
Cached rebuild: < 5 seconds.
Expected image size: **~250 MB** (extremely lightweight!).

---

## Run

### 1. Batch Video File Inference (Headless)

**Linux / Mac:**
```bash
docker run --rm \
  -v "$(pwd)/Video_container:/app/input" \
  -v "$(pwd)/output:/app/output" \
  dms-inference:latest \
  --input /app/input/video.mp4 \
  --output /app/output/predictions.csv \
  --output-video /app/output/annotated_output.mp4
```

**Windows PowerShell:**
```powershell
docker run --rm `
  -v "${PWD}/Video_container:/app/input" `
  -v "${PWD}/output:/app/output" `
  dms-inference:latest `
  --input /app/input/video.mp4 `
  --output /app/output/predictions.csv `
  --output-video /app/output/annotated_output.mp4
```

### 2. Live Webcam Capture & Recording (Headless Docker)

To run inference on your webcam inside a container, you can pass the camera device (e.g. `/dev/video0`) and capture for a fixed duration, saving the annotated HUD video.

**Linux (or WSL2 with attached USB webcam):**
```bash
docker run --rm \
  --device /dev/video0:/dev/video0 \
  -v "$(pwd)/output:/app/output" \
  dms-inference:latest \
  --input 0 \
  --duration 10.0 \
  --output /app/output/predictions.csv \
  --output-video /app/output/webcam_annotated.mp4
```

Results will appear in `output/predictions.csv`, `output/summary.json`, and the annotated video in `output/webcam_annotated.mp4` showing the active face mesh, EAR/MAR gauges, and alertness state overlays.

---

## CLI Options

```
python predict.py [OPTIONS]

  --input          STR     Input video file path OR webcam device index (e.g. 0) [required]
  --output         PATH    Output CSV path  [default: /app/output/predictions.csv]
  --json           PATH    Summary JSON path [default: /app/output/summary.json]
  --output-video   PATH    Path to save annotated output video (.mp4) [default: None]
  --duration       FLOAT   In seconds, only used when input is webcam [default: None]
  --model          PATH    Model checkpoint [default: models/film_gru_fold3.onnx]
  --xgb-model      PATH    XGBoost model checkpoint for Residual Fallback [default: None]
  --no-xgb                 Disable Residual Fallback (use raw DL model logits only)
  --fps            INT     Sampling rate Hz [default: 4]
  --threshold      FLOAT   Drowsy probability threshold [default: 0.5]
  --no-clahe               Disable CLAHE preprocessing
  --show-mesh      BOOL    Render face mesh landmarks in output video [default: True]
  --save-insets    BOOL    Include visual cropped patch insets in output video [default: True]
  --webcam-width   INT     Webcam input frame width [default: 640]
  --webcam-height  INT     Webcam input frame height [default: 480]
```

---

## Output Format

### predictions.csv

| Column        | Description                                        |
|---------------|----------------------------------------------------|
| window_idx    | Sequential window number (0-indexed)               |
| start_frame   | First frame index of this window                   |
| end_frame     | Last frame index of this window                    |
| timestamp_sec | Time in seconds at window center                   |
| label         | ALERT or DROWSY                                    |
| drowsy_prob   | Model confidence 0.0 to 1.0                        |
| ear_mean      | Mean Eye Aspect Ratio over window                  |
| mar_mean      | Mean Mouth Aspect Ratio over window                |
| perclos       | Percentage of frames with closed eyes (0-100)      |
| valid_rate    | Fraction of frames with successful mesh detection  |

### summary.json

```json
{
  "total_windows": 45,
  "drowsy_windows": 12,
  "alert_windows": 33,
  "drowsy_ratio": 0.267,
  "mean_drowsy_prob": 0.41,
  "model": "film_gru_fold3.onnx",
  "inference_device": "cpu",
  "total_inference_ms": 1240.5,
  "avg_latency_per_window_ms": 27.6
}
```

---

## Image Size Optimizations Applied

| Technique                              | Saving      |
|----------------------------------------|-------------|
| python:3.11-slim base (not full)       | ~1.0 GB     |
| ONNX Runtime CPU-only (vs PyTorch)     | ~1.5 GB     |
| opencv-python-headless (no Qt/GUI)     | ~50 MB      |
| Removed ipython, matplotlib, seaborn   | ~100 MB     |
| .dockerignore blocks frame/, logs/     | build ctx   |
| Combined RUN commands (fewer layers)   | ~50 MB      |
| **Estimated final image size**         | **~250 MB** |

---

## Architecture Reference

```
Input patches  (B=1, T=40, 3, 24, 24)
        |
FrameCNNEncoder  [shared weights across 40 frames]
  Conv2d(3->16)  -> BN -> ReLU -> MaxPool(2)
  Conv2d(16->32) -> BN -> ReLU -> MaxPool(2)
  Conv2d(32->64) -> BN -> ReLU -> AdaptiveAvgPool(1)
  Linear(64->64) -> ReLU -> Dropout(0.3)
        |  (B, 40, 64)
        |
FiLM Layer  <-- geo_encoder(11 features -> 32-dim) -> gamma, beta
  frame_emb = gamma * frame_emb + beta   [per-frame conditioning]
        |
  concat [FiLM(cnn) | geo_replicated]  -> 96-dim input
        |
GRU(input=96, hidden=64, layers=1, batch_first=True)
        |
Temporal Attention  [learned weighted sum over valid frames]
        |
Dropout(0.3) -> Linear(64->2) -> Softmax
        |
P(ALERT),  P(DROWSY)
```

Total parameters:  9,347
Inference latency: < 10 ms per window on CPU
Best LOPO-CV F1:   0.8269 (5-fold, participant1 excluded)

---

## Troubleshooting

**face_landmarker.task not found during build**
```
Ensure face_landmarker.task is in the project root before running docker build.
```

**Video produces 0 windows**
```
MediaPipe could not detect a face in the video.
Use a well-lit, front-facing video clip.
Try --fps 2 to reduce sampling rate for longer videos.
```

**Permission denied on output folder (Linux/Mac)**
```bash
mkdir -p output && chmod 777 output
```

**Rebuild after code changes**
```bash
docker build --no-cache -f docker/Dockerfile -t dms-inference:latest .
```
