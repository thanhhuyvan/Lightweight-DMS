# Dockerfile — DMS Inference Container
# Stage E: FiLM+GRU Drowsiness Detection
#
# Build from project root:
#   docker build -f docker/Dockerfile -t dms-inference:latest .
#
# Run:
#   docker run --rm \
#     -v "$(pwd)/Video_container:/app/input" \
#     -v "$(pwd)/output:/app/output" \
#     dms-inference:latest \
#     --input /app/input/video.mp4 \
#     --output /app/output/predictions.csv

# ── Base image ────────────────────────────────────────────────────────────
# python:3.11-slim saves ~1GB vs python:3.11 (no dev tools, no docs)
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────
LABEL maintainer="DMS Project"
LABEL description="Lightweight Driver Monitoring System — Stage E FiLM+GRU Inference"
LABEL version="1.0"

# ── System dependencies ───────────────────────────────────────────────────
# libgl1       : OpenCV requires libGL.so.1
# libglib2.0-0 : Required by OpenCV and MediaPipe
# Combined into one RUN to minimize layers.
# rm -rf cleans apt cache in the SAME layer so the cache doesn't persist.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgles2 \
        libegl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching.
# If requirements don't change, this layer is cached even if code changes.
COPY docker/requirements-docker.txt /app/requirements-docker.txt

# Install all dependencies from requirements-docker.txt.
# --no-cache-dir avoids storing pip cache inside the image layer.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements-docker.txt

# ── Copy model weights ────────────────────────────────────────────────────
# Only the best single-fold checkpoint is included (ONNX format).
# Other folds can be mounted at runtime via -v if needed.
COPY models/models/film_gru_fold3.onnx      /app/models/film_gru_fold3.onnx
COPY models/models/film_gru_fold3.onnx.data /app/models/film_gru_fold3.onnx.data
COPY models/models/final_scaler.joblib      /app/models/final_scaler.joblib
COPY models/models/final_xgb_model.joblib   /app/models/final_xgb_model.joblib

# ── Copy MediaPipe face landmark model ───────────────────────────────────
COPY face_landmarker.task           /app/face_landmarker.task

# ── Copy inference script ─────────────────────────────────────────────────
COPY docker/predict.py              /app/predict.py

# ── Output directory ──────────────────────────────────────────────────────
# Created here so the container can write output even without a -v mount.
RUN mkdir -p /app/output /app/input

# ── Default entrypoint ────────────────────────────────────────────────────
ENTRYPOINT ["python", "/app/predict.py"]

# Default arguments — overridden by user at runtime
CMD ["--help"]
