"""
predict.py — Headless Inference Entrypoint
-------------------------------------------
Runs the full DMS pipeline on a video file or webcam input.
No GUI display required — fully compatible with Docker headless environments.
Supports saving annotated video output with face mesh and DMS HUD.

Pipeline:
  Video/Webcam -> CLAHE -> MediaPipe Face Mesh -> Patch Extraction
  -> Geometry Features -> FiLM+GRU+Attention -> CSV + JSON + MP4 output

Usage:
    # Run on a video file
    python predict.py --input video.mp4 --output predictions.csv --output-video output.mp4
    
    # Run on webcam 0 for 10 seconds and save annotated video
    python predict.py --input 0 --duration 10 --output-video output.mp4
"""

import argparse
import json
import math
import os
import sys
import tempfile
import shutil
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        class Module:
            def __init__(self, *args, **kwargs): pass
            
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

import joblib
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Paths — all relative to /app inside the container
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

# Check if we are running locally in the host workspace or inside the container
if (APP_DIR.parent / "models" / "models").exists():
    # Outside the container in host workspace (e.g. E:\Buồn_ngủ\docker\..)
    MODEL_ONNX_DEFAULT = APP_DIR.parent / "models" / "models" / "film_gru_fold3.onnx"
    MODEL_PTH_DEFAULT  = APP_DIR.parent / "models" / "models" / "film_gru_fold3.pth"
    SCALER_DEFAULT     = APP_DIR.parent / "models" / "models" / "final_scaler.joblib"
    LANDMARKER_PATH    = APP_DIR.parent / "face_landmarker.task"
else:
    # Inside the container, where folders are flattened relative to /app
    MODEL_ONNX_DEFAULT = APP_DIR / "models" / "film_gru_fold3.onnx"
    MODEL_PTH_DEFAULT  = APP_DIR / "models" / "film_gru_fold3.pth"
    SCALER_DEFAULT     = APP_DIR / "models" / "final_scaler.joblib"
    LANDMARKER_PATH    = APP_DIR / "face_landmarker.task"

MODEL_DEFAULT = MODEL_ONNX_DEFAULT if MODEL_ONNX_DEFAULT.exists() else MODEL_PTH_DEFAULT

# ---------------------------------------------------------------------------
# FiLM+GRU Model Definition (matches training architecture exactly)
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, feature_dim)
        self.beta_net  = nn.Linear(cond_dim, feature_dim)
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.ones_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, x, cond):
        gamma = self.gamma_net(cond).unsqueeze(1)
        beta  = self.beta_net(cond).unsqueeze(1)
        return gamma * x + beta


class FrameCNNEncoder(nn.Module):
    def __init__(self, embedding_dim=64, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, patches):
        B, seq_len = patches.shape[:2]
        x   = patches.reshape(B * seq_len, 3, 24, 24)
        emb = self.proj(self.encoder(x))
        return emb.reshape(B, seq_len, -1)


class FiLMGRUModel(nn.Module):
    def __init__(self, num_classes=2, cnn_dim=64, geo_dim=11,
                 geo_hidden=32, gru_hidden=64, gru_layers=1,
                 dropout=0.3, use_film=True):
        super().__init__()
        self.use_film    = use_film
        self.cnn_encoder = FrameCNNEncoder(embedding_dim=cnn_dim, dropout=dropout)
        self.geo_encoder = nn.Sequential(
            nn.Linear(geo_dim, geo_hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        if self.use_film:
            self.film = FiLMLayer(cond_dim=geo_hidden, feature_dim=cnn_dim)
        self.gru = nn.GRU(
            input_size=cnn_dim + geo_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.use_attention = True
        self.attn = nn.Linear(gru_hidden, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, num_classes),
        )

    def forward(self, patches, valid_mask, geo, confidence=None):
        B, seq_len = patches.shape[:2]
        frame_emb  = self.cnn_encoder(patches)
        geo_cond   = self.geo_encoder(geo)
        mask       = valid_mask.unsqueeze(-1)

        if self.use_film:
            frame_emb = frame_emb * mask
            frame_emb = self.film(frame_emb, geo_cond)
            frame_emb = frame_emb * mask

        geo_rep   = geo_cond.unsqueeze(1).expand(-1, seq_len, -1)
        gru_input = torch.cat([frame_emb, geo_rep], dim=-1)

        if confidence is not None:
            gru_input = gru_input * confidence.unsqueeze(-1)

        gru_out, _ = self.gru(gru_input)

        if self.use_attention:
            scores  = self.attn(gru_out).squeeze(-1)
            scores  = scores.masked_fill(valid_mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            last_h  = (gru_out * weights).sum(dim=1)
        else:
            lengths  = valid_mask.sum(dim=1).long().clamp(min=1)
            last_idx = (lengths - 1).clamp(0, seq_len - 1)
            last_h   = gru_out[torch.arange(B, device=gru_out.device), last_idx]

        return self.head(last_h)


# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
LEFT_EYE_IDXS  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33,  160, 158, 133, 153, 144]
MOUTH_IDXS     = [61,  291, 13,  14]
LEFT_IRIS_IDXS  = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDXS = [473, 474, 475, 476, 477]

GEO_FEATURES = [
    "PERCLOS", "Blink_Rate", "Blink_Avg_Duration",
    "EAR_Mean", "EAR_Std",
    "MAR_Mean", "MAR_Max",
    "Pitch_Jitter", "Yaw_Jitter", "Roll_Jitter", "Pose_Jitter",
]

# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def eye_aspect_ratio(landmarks, eye_idxs, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idxs]
    v1 = euclidean(pts[1], pts[5])
    v2 = euclidean(pts[2], pts[4])
    hz = euclidean(pts[0], pts[3])
    return (v1 + v2) / (2.0 * hz + 1e-6)


def mouth_aspect_ratio(landmarks, w, h):
    top    = (landmarks[13].x * w,  landmarks[13].y * h)
    bottom = (landmarks[14].x * w,  landmarks[14].y * h)
    left   = (landmarks[61].x * w,  landmarks[61].y * h)
    right  = (landmarks[291].x * w, landmarks[291].y * h)
    return euclidean(top, bottom) / (euclidean(left, right) + 1e-6)


def get_head_pose(landmarks, w, h):
    model_pts = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
    ], dtype=np.float32)
    img_pts = np.array([
        (landmarks[1].x*w,   landmarks[1].y*h),
        (landmarks[152].x*w, landmarks[152].y*h),
        (landmarks[33].x*w,  landmarks[33].y*h),
        (landmarks[263].x*w, landmarks[263].y*h),
        (landmarks[61].x*w,  landmarks[61].y*h),
        (landmarks[291].x*w, landmarks[291].y*h),
    ], dtype=np.float32)
    fl  = w
    cam = np.array([[fl, 0, w/2], [0, fl, h/2], [0, 0, 1]], dtype=np.float32)
    ok, rvec, _ = cv2.solvePnP(model_pts, img_pts, cam,
                                np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    pitch = math.degrees(math.asin(-rmat[1, 2]))
    yaw   = math.degrees(math.atan2(rmat[0, 2], rmat[2, 2]))
    roll  = math.degrees(math.atan2(rmat[1, 0], rmat[1, 1]))
    return pitch, yaw, roll


def crop_isotropic(img, landmarks, idxs, w, h, size=(24, 24), pad=1.2):
    pts  = np.array([(landmarks[i].x*w, landmarks[i].y*h) for i in idxs])
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    cx, cy     = (xmin+xmax)/2, (ymin+ymax)/2
    side       = max(xmax-xmin, ymax-ymin) * pad
    nx1, ny1   = int(cx-side/2), int(cy-side/2)
    nx2, ny2   = int(cx+side/2), int(cy+side/2)
    pv         = int(side)
    padded     = cv2.copyMakeBorder(img, pv, pv, pv, pv,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
    roi = padded[ny1+pv:ny2+pv, nx1+pv:nx2+pv]
    if roi.size == 0:
        return np.zeros(size, dtype=np.uint8)
    resized = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized


def apply_clahe(frame, clip=2.0, tile=(8, 8)):
    lab    = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl     = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    lab_eq = cv2.merge([cl.apply(l), a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def count_blinks(states):
    if len(states) < 2:
        return 0
    return int(np.sum(np.diff(states) == 1))


def mean_blink_duration(states, fps):
    durations, in_b, start = [], False, 0
    for i, s in enumerate(states):
        if s == 1 and not in_b:
            in_b, start = True, i
        elif s == 0 and in_b:
            in_b = False
            durations.append((i - start) / fps)
    if in_b:
        durations.append((len(states) - start) / fps)
    return float(np.mean(durations)) if durations else 0.0


def aggregate_features(window, fps):
    valid = [f for f in window if f["valid"]]
    if not valid:
        return np.zeros(11, dtype=np.float32)
    eye_states = np.array([f["eye_state"] for f in window])
    ear_v  = np.array([f["ear"] for f in valid])
    mar_v  = np.array([f["mar"] for f in valid])
    p_v    = np.array([f["pitch"] for f in valid])
    y_v    = np.array([f["yaw"]   for f in valid])
    r_v    = np.array([f["roll"]  for f in valid])
    n      = len(window)
    perc   = (np.sum(eye_states == 1) / n) * 100.0
    br     = (count_blinks(eye_states) / (n / fps)) * 60.0
    bd     = mean_blink_duration(eye_states, fps)
    jit_p  = np.var(p_v, ddof=1) if len(p_v) > 1 else 0.0
    jit_y  = np.var(y_v, ddof=1) if len(y_v) > 1 else 0.0
    jit_r  = np.var(r_v, ddof=1) if len(r_v) > 1 else 0.0
    return np.array([
        perc, br, bd,
        np.mean(ear_v), np.std(ear_v, ddof=1) if len(ear_v) > 1 else 0.0,
        np.mean(mar_v), np.max(mar_v),
        jit_p, jit_y, jit_r, jit_p + jit_y + 0.5 * jit_r,
    ], dtype=np.float32)

# ---------------------------------------------------------------------------
# HUD Rendering functions (similar to webcam_live_fps.py)
# ---------------------------------------------------------------------------

def draw_custom_mesh(img, landmarks, w, h):
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 1, (0, 200, 0), -1)
    if len(landmarks) >= 478:
        for idx in LEFT_IRIS_IDXS + RIGHT_IRIS_IDXS:
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 1, (255, 255, 255), -1)
    for idx in LEFT_EYE_IDXS + RIGHT_EYE_IDXS:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 2, (0, 165, 255), -1)
    for idx in MOUTH_IDXS:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 2, (255, 50, 50), -1)


def draw_hud_overlay(frame, metrics, prob, state, ear_thresh, patches, show_insets):
    h, w = frame.shape[:2]
    
    # Left overlay panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, h), (18, 18, 27), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Title
    cv2.putText(frame, "DMS DOCKER INFERENCE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 244), 2)
    cv2.line(frame, (15, 40), (300, 40), (49, 50, 68), 1)
    
    # State Banner
    alert_bg = (30, 58, 47) if state == "ALERT" else (30, 30, 58)
    alert_border = (46, 204, 113) if state == "ALERT" else (231, 76, 60)
    cv2.rectangle(frame, (15, 55), (300, 105), alert_bg, -1)
    cv2.rectangle(frame, (15, 55), (300, 105), alert_border, 2)
    cv2.putText(frame, state, (90 if state == "ALERT" else 75, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, alert_border, 3)

    # Gauges
    y_offset = 140
    cv2.putText(frame, "REAL-TIME METRICS", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (108, 112, 134), 1)
    y_offset += 25
    
    def draw_gauge(label, val, max_val, y_pos, color_bar):
        val_str = "N/A" if (val is None or np.isnan(val)) else f"{val:.3f}"
        cv2.putText(frame, f"{label}: {val_str}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 214, 244), 1)
        cv2.rectangle(frame, (15, y_pos + 8), (300, y_pos + 18), (30, 30, 46), -1)
        if val is not None and not np.isnan(val) and max_val > 0:
            val_w = int(285 * min(val / max_val, 1.0))
            cv2.rectangle(frame, (15, y_pos + 8), (15 + val_w, y_pos + 18), color_bar, -1)

    # EAR
    ear = metrics.get('ear', np.nan)
    ear_color = (46, 204, 113) if (not np.isnan(ear) and ear >= ear_thresh) else (231, 76, 60)
    draw_gauge("EAR", ear, 0.45, y_offset, ear_color)
    tick_pos = 15 + int(285 * min(ear_thresh / 0.45, 1.0))
    cv2.line(frame, (tick_pos, y_offset + 5), (tick_pos, y_offset + 21), (255, 255, 255), 2)
    
    y_offset += 40
    # MAR
    mar = metrics.get('mar', np.nan)
    mar_color = (249, 226, 175) if mar < 0.25 else (231, 76, 60)
    draw_gauge("MAR (Mouth)", mar, 0.6, y_offset, mar_color)
    
    y_offset += 40
    # PERCLOS
    perclos = metrics.get('perclos', 0.0)
    perclos_color = (46, 204, 113) if perclos < 15.0 else ((249, 226, 175) if perclos < 30.0 else (231, 76, 60))
    draw_gauge(f"PERCLOS (10s)", perclos / 100.0, 1.0, y_offset, perclos_color)
    cv2.putText(frame, f"{perclos:.1f}%", (250, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (166, 173, 200), 1)
    
    y_offset += 45
    cv2.putText(frame, f"Blink Rate: {metrics.get('blink_rate', 0.0):.1f} /min", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 214, 244), 1)
    y_offset += 20
    cv2.putText(frame, f"Blink Duration: {metrics.get('blink_duration', 0.0):.2f} s", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 214, 244), 1)
    
    y_offset += 30
    cv2.putText(frame, "DROWSY CONFIDENCE", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (108, 112, 134), 1)
    y_offset += 18
    cv2.rectangle(frame, (15, y_offset), (300, y_offset + 18), (30, 30, 46), -1)
    prob_color = (231, 76, 60) if prob > 0.5 else (46, 204, 113)
    prob_w = int(285 * prob)
    cv2.rectangle(frame, (15, y_offset), (15 + prob_w, y_offset + 18), prob_color, -1)
    cv2.putText(frame, f"{prob*100:.1f}%", (140, y_offset + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (205, 214, 244), 2)

    # Inset patches
    if show_insets and patches is not None:
        y_start = h - 110
        cv2.putText(frame, "INPUT PATCHES (24x24)", (15, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (108, 112, 134), 1)
        
        # Left Eye (rescaled to 60x60)
        l_eye = cv2.resize((patches[0] * 255).astype(np.uint8), (60, 60), interpolation=cv2.INTER_NEAREST)
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_GRAY2BGR)
        frame[y_start:y_start+60, 15:75] = l_eye
        cv2.rectangle(frame, (15, y_start), (75, y_start+60), (49, 50, 68), 1)
        cv2.putText(frame, "L EYE", (15, y_start + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (166, 173, 200), 1)
        
        # Right Eye
        r_eye = cv2.resize((patches[1] * 255).astype(np.uint8), (60, 60), interpolation=cv2.INTER_NEAREST)
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_GRAY2BGR)
        frame[y_start:y_start+60, 95:155] = r_eye
        cv2.rectangle(frame, (95, y_start), (155, y_start+60), (49, 50, 68), 1)
        cv2.putText(frame, "R EYE", (95, y_start + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (166, 173, 200), 1)
        
        # Mouth
        m_patch = cv2.resize((patches[2] * 255).astype(np.uint8), (60, 60), interpolation=cv2.INTER_NEAREST)
        m_patch = cv2.cvtColor(m_patch, cv2.COLOR_GRAY2BGR)
        frame[y_start:y_start+60, 175:235] = m_patch
        cv2.rectangle(frame, (175, y_start), (235, y_start+60), (49, 50, 68), 1)
        cv2.putText(frame, "MOUTH", (175, y_start + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (166, 173, 200), 1)

# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_inference(args):
    device_str = "CPU (PyTorch)"
    model_is_onnx = Path(args.model).suffix.lower() == ".onnx"
    
    print(f"[DMS] Input:  {args.input}")
    print(f"[DMS] Model:  {args.model}")

    # ── Load model ──────────────────────────────────────────────────────────
    model = None
    ort_session = None
    
    if model_is_onnx:
        if not HAS_ORT:
            print("[DMS] Error: Model is ONNX but onnxruntime is not installed.")
            sys.exit(1)
        print(f"[DMS] Loading ONNX model: {args.model}")
        ort_session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
        device_str = "CPU (ONNXRuntime)"
        print(f"[DMS] ONNX session initialized successfully.")
    else:
        if not HAS_TORCH:
            print("[DMS] Error: Model is PyTorch (.pth) but torch is not installed.")
            sys.exit(1)
        device = torch.device("cpu")
        print(f"[DMS] Device: {device}")
        print(f"[DMS] Loading PyTorch model: {args.model}")
        model = FiLMGRUModel().to(device)
        state = torch.load(args.model, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"[DMS] PyTorch model loaded: {Path(args.model).name}")

    # ── Load scaler ──────────────────────────────────────────────────────────
    scaler_path = Path(args.model).parent / "final_scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f"[DMS] Scaler loaded: {scaler_path.name}")
    else:
        class _Identity:
            def transform(self, x): return x
        scaler = _Identity()
        print("[DMS] Warning: scaler not found, using identity transform")

    # ── Load XGBoost fallback model if enabled ──────────────────────────────
    xgb_model = None
    if not args.no_xgb:
        xgb_path = Path(args.xgb_model) if args.xgb_model else Path(args.model).parent / "final_xgb_model.joblib"
        if xgb_path.exists():
            try:
                import xgboost as xgb
                xgb_model = joblib.load(xgb_path)
                print(f"[DMS] XGBoost fallback loaded: {xgb_path.name}")
            except Exception as e:
                print(f"[DMS] Warning: failed to load XGBoost: {e}")
        else:
            print(f"[DMS] Warning: XGBoost model not found at {xgb_path}")

    # ── MediaPipe setup ──────────────────────────────────────────────────────
    lm_path = str(LANDMARKER_PATH)
    if not Path(lm_path).exists():
        print(f"[DMS] Error: face_landmarker.task not found at {lm_path}")
        sys.exit(1)

    base_opts = mp_python.BaseOptions(model_asset_path=lm_path)
    lm_opts   = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
    )
    landmarker = vision.FaceLandmarker.create_from_options(lm_opts)

    # ── Open video / Webcam ──────────────────────────────────────────────────
    is_webcam = args.input.isdigit()
    if is_webcam:
        cam_idx = int(args.input)
        cap = cv2.VideoCapture(cam_idx)
        w = int(args.webcam_width)
        h = int(args.webcam_height)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = 30.0
        total_frames = -1
        print(f"[DMS] Webcam: Device index {cam_idx} @ {w}x{h}")
        if args.duration is None:
            print("[DMS] Warning: --duration not specified for webcam. Recording for 10 seconds default.")
            args.duration = 10.0
    else:
        cap = cv2.VideoCapture(str(args.input))
        if not cap.isOpened():
            print(f"[DMS] Error: cannot open video: {args.input}")
            sys.exit(1)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[DMS] Video: {w}x{h} @ {video_fps:.1f} FPS, {total_frames} frames")

    # Video Writer setup if output-video is requested
    out_writer = None
    if args.output_video:
        out_video_path = Path(args.output_video)
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(out_video_path), fourcc, video_fps, (w, h))
        print(f"[DMS] Output video writer enabled: {out_video_path}")

    # Sampling interval
    sample_interval = max(1, round(video_fps / args.fps))
    seq_len         = 40
    ear_thresh      = 0.225

    # ── Main extraction loop ──────────────────────────────────────────────────
    window_history = deque(maxlen=seq_len)
    patch_history  = deque(maxlen=seq_len)
    results_rows   = []
    frame_idx      = 0
    window_idx     = 0
    t_start        = time.time()

    print(f"[DMS] Sampling every {sample_interval} frames (~{args.fps} fps)")
    print("[DMS] Processing...")

    try:
        while True:
            # Check duration limits if running on webcam
            if is_webcam and args.duration is not None:
                if time.time() - t_start >= args.duration:
                    print(f"[DMS] Duration limit ({args.duration}s) reached. Exiting capture.")
                    break

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % sample_interval != 0:
                # Write original frame if saving output video (to keep video timing aligned)
                if out_writer is not None:
                    out_writer.write(frame)
                continue

            # Progress report
            if not is_webcam and total_frames > 0 and frame_idx % (sample_interval * 100) == 0:
                pct = (frame_idx / total_frames * 100)
                print(f"[DMS]   {pct:.0f}% ({frame_idx}/{total_frames} frames)")

            # Preprocessing
            enhanced = apply_clahe(frame) if not args.no_clahe else frame.copy()
            img_rgb  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # Landmark detection
            detection = landmarker.detect(mp_img)
            frame_stat = {
                "ear": np.nan, "mar": np.nan,
                "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                "eye_state": 0, "valid": False,
            }
            patches = np.zeros((3, 24, 24), dtype=np.float32)

            if detection.face_landmarks:
                lms = detection.face_landmarks[0]
                ear = eye_aspect_ratio(lms, LEFT_EYE_IDXS,  w, h)
                mar = mouth_aspect_ratio(lms, w, h)
                pit, yaw, rol = get_head_pose(lms, w, h)

                le = crop_isotropic(enhanced, lms, LEFT_EYE_IDXS,  w, h)
                re = crop_isotropic(enhanced, lms, RIGHT_EYE_IDXS, w, h)
                mo = crop_isotropic(enhanced, lms, MOUTH_IDXS,     w, h)
                patches = np.stack([
                    le.astype(np.float32) / 255.0,
                    re.astype(np.float32) / 255.0,
                    mo.astype(np.float32) / 255.0,
                ], axis=0)

                frame_stat.update({
                    "ear": ear, "mar": mar,
                    "pitch": pit, "yaw": yaw, "roll": rol,
                    "eye_state": 1 if ear < ear_thresh else 0,
                    "valid": True,
                })

            window_history.append(frame_stat)
            patch_history.append(patches)

            # Draw mesh if requested
            annotated_frame = frame.copy()
            if args.show_mesh and detection.face_landmarks:
                draw_custom_mesh(annotated_frame, detection.face_landmarks[0], w, h)

            prob   = 0.0
            label  = "ALERT"

            # Run inference when window is full
            if len(window_history) >= seq_len:
                # Build tensors
                win_list   = list(window_history)
                patch_list = list(patch_history)
                vmask      = [float(f["valid"]) for f in win_list]

                # Confidence decay: forward-fill missing frames
                valid_idx = np.where(np.array(vmask) > 0)[0]
                conf_seq  = np.ones(seq_len, dtype=np.float32)
                if len(valid_idx) > 0:
                    for t in range(seq_len):
                        if vmask[t] > 0:
                            continue
                        prev = valid_idx[valid_idx < t]
                        nxt  = valid_idx[valid_idx > t]
                        if len(prev) and len(nxt):
                            dist = min(t - prev[-1], nxt[0] - t)
                            patch_list[t] = patch_list[prev[-1]]
                        elif len(prev):
                            dist = t - prev[-1]
                            patch_list[t] = patch_list[prev[-1]]
                        else:
                            dist = nxt[0] - t
                            patch_list[t] = patch_list[nxt[0]]
                        conf_seq[t] = 0.85 ** dist
                else:
                    conf_seq[:] = 0.0

                geo_vec    = aggregate_features(win_list, args.fps)
                scaled_geo = scaler.transform(geo_vec.reshape(1, -1)).flatten().astype(np.float32)

                # Convert grayscale patches (1-channel) to 3-channel RGB for CNN
                patches_rgb = []
                for p in patch_list:
                    if p.shape[0] == 3:
                        patches_rgb.append(p)
                    else:
                        patches_rgb.append(np.stack([p[0], p[0], p[0]], axis=0))

                if model_is_onnx:
                    patches_np = np.stack(patches_rgb)[np.newaxis].astype(np.float32)
                    vmask_np = np.array(vmask, dtype=np.float32)[np.newaxis]
                    geo_np = scaled_geo[np.newaxis].astype(np.float32)
                    conf_np = conf_seq[np.newaxis].astype(np.float32)

                    outputs = ort_session.run(
                        ["logits"],
                        {
                            "patches": patches_np,
                            "valid_mask": vmask_np,
                            "geo": geo_np,
                            "confidence": conf_np
                        }
                    )
                    logits = outputs[0]
                    
                    if xgb_model is not None:
                        s_base = xgb_model.predict_proba(scaled_geo.reshape(1, -1))[0, 1]
                        delta_s = math.tanh(logits[0, 1] - logits[0, 0]) * 0.15
                        prob = float(np.clip(s_base + delta_s, 0.0, 1.0))
                    else:
                        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                        prob = float(probs[0, 1])
                    label  = "DROWSY" if prob >= args.threshold else "ALERT"
                else:
                    patches_t  = torch.from_numpy(np.stack(patches_rgb)).unsqueeze(0).float()
                    vmask_t    = torch.tensor(vmask, dtype=torch.float32).unsqueeze(0)
                    geo_t      = torch.from_numpy(scaled_geo).unsqueeze(0)
                    conf_t     = torch.from_numpy(conf_seq).unsqueeze(0)

                    with torch.no_grad():
                        logits = model(patches_t, vmask_t, geo_t, conf_t)
                        if xgb_model is not None:
                            # Residual Fallback Formula: S_final = S_base + Delta_S
                            s_base = xgb_model.predict_proba(scaled_geo.reshape(1, -1))[0, 1]
                            delta_s = math.tanh(logits[0, 1] - logits[0, 0]) * 0.15
                            prob = float(np.clip(s_base + delta_s, 0.0, 1.0))
                        else:
                            prob   = float(torch.softmax(logits, dim=1)[0, 1].item())
                        label  = "DROWSY" if prob >= args.threshold else "ALERT"

                valid_count = sum(1 for f in win_list if f["valid"])
                ear_vals    = [f["ear"] for f in win_list if f["valid"] and not np.isnan(f["ear"])]
                mar_vals    = [f["mar"] for f in win_list if f["valid"] and not np.isnan(f["mar"])]

                results_rows.append({
                    "window_idx":    window_idx,
                    "start_frame":   frame_idx - seq_len * sample_interval,
                    "end_frame":     frame_idx,
                    "timestamp_sec": round(frame_idx / video_fps, 2),
                    "label":         label,
                    "drowsy_prob":   round(prob, 4),
                    "ear_mean":      round(float(np.mean(ear_vals)), 4) if ear_vals else 0.0,
                    "mar_mean":      round(float(np.mean(mar_vals)), 4) if mar_vals else 0.0,
                    "perclos":       round(float(geo_vec[0]), 2),
                    "valid_rate":    round(valid_count / seq_len, 3),
                })
                window_idx += 1

            # Render HUD on the annotated frame
            metrics_display = {
                'ear': window_history[-1]['ear'] if len(window_history) > 0 else np.nan,
                'mar': window_history[-1]['mar'] if len(window_history) > 0 else np.nan,
                'blink_rate': 0.0,
                'blink_duration': 0.0,
                'perclos': 0.0
            }
            if len(window_history) >= seq_len:
                metrics_display['perclos'] = float(geo_vec[0])
                metrics_display['blink_rate'] = float(geo_vec[1])
                metrics_display['blink_duration'] = float(geo_vec[2])

            latest_patches = patches if detection.face_landmarks else None
            draw_hud_overlay(annotated_frame, metrics_display, prob, label, ear_thresh, latest_patches, args.save_insets)

            # Write annotated frame to video
            if out_writer is not None:
                out_writer.write(annotated_frame)

    except KeyboardInterrupt:
        print("[DMS] Interrupted by user.")
    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
        landmarker.close()

    elapsed_ms = (time.time() - t_start) * 1000.0
    print(f"[DMS] Done. {window_idx} windows processed in {elapsed_ms:.0f} ms")

    # ── Write outputs ────────────────────────────────────────────────────────
    if not results_rows:
        print("[DMS] Warning: no windows produced. Check that faces are detectable in the video.")
        sys.exit(0)

    df = pd.DataFrame(results_rows)

    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[DMS] CSV saved: {out_path}")

    # Summary JSON
    drowsy_wins = int((df["label"] == "DROWSY").sum())
    alert_wins  = int((df["label"] == "ALERT").sum())
    summary = {
        "total_windows":            window_idx,
        "drowsy_windows":           drowsy_wins,
        "alert_windows":            alert_wins,
        "drowsy_ratio":             round(drowsy_wins / window_idx, 3) if window_idx else 0.0,
        "mean_drowsy_prob":         round(float(df["drowsy_prob"].mean()), 4),
        "model":                    Path(args.model).name,
        "inference_device":         device_str,
        "total_inference_ms":       round(elapsed_ms, 1),
        "avg_latency_per_window_ms": round(elapsed_ms / window_idx, 1) if window_idx else 0.0,
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DMS] JSON saved: {json_path}")
    print(f"[DMS] Result: {alert_wins} ALERT / {drowsy_wins} DROWSY "
          f"({summary['drowsy_ratio']*100:.1f}% drowsy)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="DMS inference (Video or Webcam) — FiLM+GRU Stage E"
    )
    p.add_argument("--input",     required=True,
                   help="Input video file (.mp4, .avi) or webcam device index (e.g. 0)")
    p.add_argument("--output",    default="/app/output/predictions.csv",
                   help="Output CSV path")
    p.add_argument("--model",     default=str(MODEL_DEFAULT),
                   help="Path to model .pth checkpoint")
    p.add_argument("--fps",       type=int,   default=4,
                   help="Sampling rate in Hz (default: 4)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Drowsy probability threshold (default: 0.5)")
    p.add_argument("--no-clahe",  action="store_true",
                   help="Disable CLAHE preprocessing")
    p.add_argument("--json",      default="/app/output/summary.json",
                   help="Output summary JSON path")
    p.add_argument("--output-video", default=None,
                   help="Path to save annotated output video (.mp4)")
    p.add_argument("--xgb-model", default=None,
                   help="Path to XGBoost model checkpoint for Residual Fallback")
    p.add_argument("--no-xgb",    action="store_true",
                   help="Disable Residual Fallback (use raw deep learning logits only)")
    p.add_argument("--duration", type=float, default=None,
                   help="In seconds, only used when input is webcam")
    p.add_argument("--show-mesh", type=bool, default=True,
                   help="Render face mesh landmarks in output video")
    p.add_argument("--save-insets", type=bool, default=True,
                   help="Include visual cropped patch insets in output video")
    p.add_argument("--webcam-width", type=int, default=640,
                   help="Webcam input frame width")
    p.add_argument("--webcam-height", type=int, default=480,
                   help="Webcam input frame height")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
