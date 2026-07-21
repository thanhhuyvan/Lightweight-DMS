"""
live_demo.py
------------------
Stage E: FiLM + GRU Live Demo for Driver Monitoring System (DMS)
Supports real-time webcam feed or video file input.

Performs:
  1. Real-time face mesh landmark detection using MediaPipe.
  2. Calibration phase (first 5 seconds) to set dynamic eye state threshold alpha.
  3. Real-time eye/mouth patch cropping (isotropic 24x24 grayscale stacked to 3-channel).
  4. Real-time geometry features calculation over a 40-frame sliding window (10s at 4 FPS).
  5. Drowsiness prediction using FiLM+GRU (Stage E) model with optional Residual Fallback (XGBoost).
  6. Visual HUD overlay (Face mesh, EAR/MAR gauges, PERCLOS, status flags, Alert/Drowsy state).

Usage:
  python live_demo.py
  python live_demo.py --video path/to/video.mp4
  python live_demo.py --model models/film_gru_fold3.pth --residual
"""

import os
import sys
import time
import math
import argparse
import tempfile
import shutil
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Setup Project Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try importing model definition from train_film_gru, fallback to local definition if import fails
try:
    from src.s4_training.train_film_gru import FiLMGRUModel
    print("Loaded FiLMGRUModel from project source.")
except ImportError:
    print("Warning: Could not import FiLMGRUModel from src.s4_training. Defining locally...")
    
    class FiLMLayer(nn.Module):
        def __init__(self, cond_dim: int, feature_dim: int):
            super().__init__()
            self.gamma_net = nn.Linear(cond_dim, feature_dim)
            self.beta_net  = nn.Linear(cond_dim, feature_dim)
            nn.init.zeros_(self.gamma_net.weight)
            nn.init.ones_(self.gamma_net.bias)
            nn.init.zeros_(self.beta_net.weight)
            nn.init.zeros_(self.beta_net.bias)

        def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            gamma = self.gamma_net(cond).unsqueeze(1)
            beta  = self.beta_net(cond).unsqueeze(1)
            return gamma * x + beta

    class FrameCNNEncoder(nn.Module):
        def __init__(self, embedding_dim: int = 64, dropout: float = 0.2):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def forward(self, patches: torch.Tensor) -> torch.Tensor:
            B, seq_len = patches.shape[:2]
            x = patches.reshape(B * seq_len, 3, 24, 24)
            emb = self.proj(self.encoder(x))
            return emb.reshape(B, seq_len, -1)

    class FiLMGRUModel(nn.Module):
        def __init__(
            self,
            num_classes:  int = 2,
            cnn_dim:      int = 64,
            geo_dim:      int = 11,
            geo_hidden:   int = 32,
            gru_hidden:   int = 64,
            gru_layers:   int = 1,
            dropout:      float = 0.3,
            use_film:     bool = True,
        ):
            super().__init__()
            self.use_film = use_film
            self.cnn_encoder = FrameCNNEncoder(embedding_dim=cnn_dim, dropout=dropout)
            self.geo_encoder  = nn.Sequential(
                nn.Linear(geo_dim, geo_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if self.use_film:
                self.film = FiLMLayer(cond_dim=geo_hidden, feature_dim=cnn_dim)
            gru_input_size = cnn_dim + geo_hidden

            self.gru  = nn.GRU(
                input_size=gru_input_size,
                hidden_size=gru_hidden,
                num_layers=gru_layers,
                batch_first=True,
                dropout=dropout if gru_layers > 1 else 0.0,
            )
            self.use_attention = False
            self.attn = nn.Linear(gru_hidden, 1)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(gru_hidden, num_classes),
            )

        def forward(
            self,
            patches:    torch.Tensor,
            valid_mask: torch.Tensor,
            geo:        torch.Tensor,
            confidence: torch.Tensor = None,
        ) -> torch.Tensor:
            B, seq_len = patches.shape[:2]
            frame_emb = self.cnn_encoder(patches)
            geo_cond = self.geo_encoder(geo)
            mask = valid_mask.unsqueeze(-1)
            
            if self.use_film:
                frame_emb = frame_emb * mask
                frame_emb = self.film(frame_emb, geo_cond)
                frame_emb = frame_emb * mask
                geo_replicated = geo_cond.unsqueeze(1).expand(-1, seq_len, -1)
                gru_input = torch.cat([frame_emb, geo_replicated], dim=-1)
            else:
                frame_emb = frame_emb * mask
                geo_replicated = geo_cond.unsqueeze(1).expand(-1, seq_len, -1)
                gru_input = torch.cat([frame_emb, geo_replicated], dim=-1)

            if confidence is not None:
                gru_input = gru_input * confidence.unsqueeze(-1)

            gru_out, _ = self.gru(gru_input)

            if self.use_attention:
                scores = self.attn(gru_out).squeeze(-1)
                scores = scores.masked_fill(valid_mask == 0, float("-inf"))
                weights = torch.softmax(scores, dim=1).unsqueeze(-1)
                last_h = (gru_out * weights).sum(dim=1)
            else:
                lengths  = valid_mask.sum(dim=1).long().clamp(min=1)
                last_idx = (lengths - 1).clamp(0, seq_len - 1)
                last_h   = gru_out[torch.arange(B, device=gru_out.device), last_idx]

            return self.head(last_h)

# ---------------------------------------------------------------------------
# Global Configs / Landmarks
# ---------------------------------------------------------------------------
LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
MOUTH_IDXS = [61, 291, 13, 14]
LEFT_IRIS_IDXS  = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDXS = [473, 474, 475, 476, 477]

GEO_FEATURES = [
    "PERCLOS", "Blink_Rate", "Blink_Avg_Duration",
    "EAR_Mean", "EAR_Std",
    "MAR_Mean", "MAR_Max",
    "Pitch_Jitter", "Yaw_Jitter", "Roll_Jitter", "Pose_Jitter",
]

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def eye_aspect_ratio(landmarks, eye_idxs, img_w, img_h):
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_idxs]
    v1 = euclidean(pts[1], pts[5])
    v2 = euclidean(pts[2], pts[4])
    h  = euclidean(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def mouth_aspect_ratio(landmarks, img_w, img_h):
    top    = (landmarks[13].x * img_w, landmarks[13].y * img_h)
    bottom = (landmarks[14].x * img_w, landmarks[14].y * img_h)
    left   = (landmarks[78].x * img_w, landmarks[78].y * img_h) if len(landmarks) > 78 else (landmarks[61].x * img_w, landmarks[61].y * img_h)
    right  = (landmarks[308].x * img_w, landmarks[308].y * img_h) if len(landmarks) > 308 else (landmarks[291].x * img_w, landmarks[291].y * img_h)
    vert  = euclidean(top, bottom)
    horiz = euclidean(left, right)
    return vert / (horiz + 1e-6)

def get_head_pose(landmarks, img_w, img_h):
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float32)

    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),
        (landmarks[152].x * img_w, landmarks[152].y * img_h),
        (landmarks[33].x * img_w, landmarks[33].y * img_h),
        (landmarks[263].x * img_w, landmarks[263].y * img_h),
        (landmarks[61].x * img_w, landmarks[61].y * img_h),
        (landmarks[291].x * img_w, landmarks[291].y * img_h)
    ], dtype=np.float32)

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rot_vec)
    pitch = math.asin(-rmat[1, 2])
    yaw = math.atan2(rmat[0, 2], rmat[2, 2])
    roll = math.atan2(rmat[1, 0], rmat[1, 1])

    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

def crop_isotropic(img, landmarks, idxs, img_w, img_h, target_size=(24, 24), padding_factor=1.2):
    pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in idxs])
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    w = x_max - x_min
    h = y_max - y_min
    cx, cy = x_min + w/2, y_min + h/2
    side = max(w, h) * padding_factor
    nx1 = int(cx - side/2)
    ny1 = int(cy - side/2)
    nx2 = int(cx + side/2)
    ny2 = int(cy + side/2)
    pad_val = int(side)
    img_padded = cv2.copyMakeBorder(img, pad_val, pad_val, pad_val, pad_val, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    roi = img_padded[ny1+pad_val:ny2+pad_val, nx1+pad_val:nx2+pad_val]
    if roi.size == 0:
        return np.zeros(target_size, dtype=np.uint8)
    resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized

def apply_clahe(frame_bgr: np.ndarray, clip_limit=2.0, tile_grid=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

# ---------------------------------------------------------------------------
# Feature Extraction over sliding window
# ---------------------------------------------------------------------------
def count_blinks(eye_state_array):
    if len(eye_state_array) < 2:
        return 0
    transitions = np.diff(eye_state_array)
    return int(np.sum(transitions == 1))

def mean_blink_duration(eye_state_array, fps):
    if len(eye_state_array) < 2:
        return 0.0
    durations = []
    in_blink = False
    blink_start = 0
    for i, state in enumerate(eye_state_array):
        if state == 1 and not in_blink:
            in_blink = True
            blink_start = i
        elif state == 0 and in_blink:
            in_blink = False
            durations.append((i - blink_start) / fps)
    if in_blink:
        durations.append((len(eye_state_array) - blink_start) / fps)
    return float(np.mean(durations)) if durations else 0.0

def aggregate_window_features(window_history, fps):
    """
    Computes the 11 geometry features from a window history list.
    Each item in window_history is a dict containing:
      'ear', 'mar', 'pitch', 'yaw', 'roll', 'eye_state', 'valid'
    """
    win_len = len(window_history)
    valid_frames = [f for f in window_history if f['valid']]
    
    if len(valid_frames) == 0:
        return np.zeros(11, dtype=np.float32)
        
    eye_states = np.array([f['eye_state'] for f in window_history])
    ear_vals = np.array([f['ear'] for f in valid_frames])
    mar_vals = np.array([f['mar'] for f in valid_frames])
    p_vals = np.array([f['pitch'] for f in valid_frames])
    y_vals = np.array([f['yaw'] for f in valid_frames])
    r_vals = np.array([f['roll'] for f in valid_frames])
    
    perclos = (np.sum(eye_states == 1) / win_len) * 100.0
    n_blinks = count_blinks(eye_states)
    blink_rate = (n_blinks / (win_len / fps)) * 60.0
    blink_avg_dur = mean_blink_duration(eye_states, fps)
    
    ear_mean = np.mean(ear_vals)
    ear_std = np.std(ear_vals, ddof=1) if len(ear_vals) > 1 else 0.0
    
    mar_mean = np.mean(mar_vals)
    mar_max = np.max(mar_vals)
    
    jit_p = np.var(p_vals, ddof=1) if len(p_vals) > 1 else 0.0
    jit_y = np.var(y_vals, ddof=1) if len(y_vals) > 1 else 0.0
    jit_r = np.var(r_vals, ddof=1) if len(r_vals) > 1 else 0.0
    pose_jitter = jit_p + jit_y + (0.5 * jit_r)
    
    return np.array([
        perclos, blink_rate, blink_avg_dur,
        ear_mean, ear_std,
        mar_mean, mar_max,
        jit_p, jit_y, jit_r, pose_jitter
    ], dtype=np.float32)

# ---------------------------------------------------------------------------
# Visual HUD Drawing
# ---------------------------------------------------------------------------
def draw_hud(frame, metrics, prob, state, is_calibrating, calib_progress, alpha_thresh, fps_curr):
    h, w = frame.shape[:2]
    
    # Overlay transparent background panel on the left (w: 300)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, h), (18, 18, 27), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Draw Title
    cv2.putText(frame, "STAGE E: DMS LIVE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 244), 2)
    cv2.line(frame, (15, 40), (300, 40), (49, 50, 68), 1)
    
    # Status Panel
    if is_calibrating:
        cv2.putText(frame, "STATUS: CALIBRATING", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (137, 180, 250), 2)
        # Draw Calibration Progress Bar
        cv2.rectangle(frame, (15, 80), (300, 95), (30, 30, 46), -1)
        progress_w = int(285 * calib_progress)
        cv2.rectangle(frame, (15, 80), (15 + progress_w, 95), (137, 180, 250), -1)
        cv2.putText(frame, f"{int(calib_progress*100)}%", (140, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (17, 17, 27), 2)
    else:
        # Prediction State Alert Banner
        alert_bg = (30, 58, 47) if state == "ALERT" else (30, 30, 58)
        alert_border = (46, 204, 113) if state == "ALERT" else (231, 76, 60)
        cv2.rectangle(frame, (15, 55), (300, 105), alert_bg, -1)
        cv2.rectangle(frame, (15, 55), (300, 105), alert_border, 2)
        
        cv2.putText(frame, state, (90 if state == "ALERT" else 75, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, alert_border, 3)
        
    # Metrics
    y_offset = 140
    cv2.putText(frame, "REAL-TIME GAUGES", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (108, 112, 134), 1)
    y_offset += 25
    
    def draw_gauge(label, val, max_val, y_pos, color_bar):
        val_str = "N/A" if (val is None or np.isnan(val)) else f"{val:.3f}"
        cv2.putText(frame, f"{label}: {val_str}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 214, 244), 1)
        cv2.rectangle(frame, (15, y_pos + 8), (300, y_pos + 18), (30, 30, 46), -1)
        if val is not None and not np.isnan(val) and max_val > 0:
            val_w = int(285 * min(val / max_val, 1.0))
            cv2.rectangle(frame, (15, y_pos + 8), (15 + val_w, y_pos + 18), color_bar, -1)
        
    # EAR Gauge
    ear = metrics.get('ear', np.nan)
    if np.isnan(ear):
        ear_color = (108, 112, 134)
    else:
        ear_color = (46, 204, 113) if ear >= alpha_thresh else (231, 76, 60)
    draw_gauge("Mean EAR", ear, 0.45, y_offset, ear_color)
    # Draw Threshold Tick
    tick_pos = 15 + int(285 * min(alpha_thresh / 0.45, 1.0))
    cv2.line(frame, (tick_pos, y_offset + 5), (tick_pos, y_offset + 21), (249, 226, 175), 2)
    
    y_offset += 40
    
    # MAR Gauge
    mar = metrics.get('mar', np.nan)
    if np.isnan(mar):
        mar_color = (108, 112, 134)
    else:
        mar_color = (249, 226, 175) if mar < 0.25 else (231, 76, 60)
    draw_gauge("MAR (Mouth Open)", mar, 0.6, y_offset, mar_color)
    
    y_offset += 40
    
    # PERCLOS Gauge
    perclos = metrics.get('perclos', 0.0)
    perclos_color = (46, 204, 113) if perclos < 15.0 else ((249, 226, 175) if perclos < 30.0 else (231, 76, 60))
    draw_gauge(f"PERCLOS (10s)", perclos / 100.0, 1.0, y_offset, perclos_color)
    cv2.putText(frame, f"{perclos:.1f}%", (250, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (166, 173, 200), 1)
    
    y_offset += 45
    
    # Other metrics list
    cv2.putText(frame, f"Blink Rate: {metrics.get('blink_rate', 0.0):.1f} /min", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (205, 214, 244), 1)
    y_offset += 22
    cv2.putText(frame, f"Blink Avg Dur: {metrics.get('blink_duration', 0.0):.2f} s", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (205, 214, 244), 1)
    y_offset += 22
    cv2.putText(frame, f"Pose Jitter: {metrics.get('pose_jitter', 0.0):.4f}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (205, 214, 244), 1)
    
    y_offset += 30
    cv2.line(frame, (15, y_offset), (300, y_offset), (49, 50, 68), 1)
    y_offset += 20
    
    # Head Pose values
    cv2.putText(frame, "HEAD POSE", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (108, 112, 134), 1)
    y_offset += 20
    
    pitch = metrics.get('pitch', np.nan)
    yaw = metrics.get('yaw', np.nan)
    roll = metrics.get('roll', np.nan)
    pitch_str = "N/A" if np.isnan(pitch) else f"{pitch:.1f}°"
    yaw_str = "N/A" if np.isnan(yaw) else f"{yaw:.1f}°"
    roll_str = "N/A" if np.isnan(roll) else f"{roll:.1f}°"
    cv2.putText(frame, f"Pitch: {pitch_str}  Yaw: {yaw_str}  Roll: {roll_str}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 214, 244), 1)
    
    y_offset += 30
    cv2.line(frame, (15, y_offset), (300, y_offset), (49, 50, 68), 1)
    y_offset += 20
    
    # Model Output Probability
    if not is_calibrating:
        cv2.putText(frame, "DROWSY PROBABILITY", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (108, 112, 134), 1)
        y_offset += 20
        cv2.rectangle(frame, (15, y_offset), (300, y_offset + 18), (30, 30, 46), -1)
        prob_color = (231, 76, 60) if prob > 0.5 else (46, 204, 113)
        prob_w = int(285 * prob)
        cv2.rectangle(frame, (15, y_offset), (15 + prob_w, y_offset + 18), prob_color, -1)
        cv2.putText(frame, f"{prob*100:.1f}%", (140, y_offset + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (205, 214, 244), 2)
        
    # Frame stats bottom right corner
    cv2.putText(frame, f"FPS: {fps_curr:.1f}", (w - 100, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (166, 173, 200), 1)

def draw_custom_mesh(img, landmarks, w, h):
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 1, (0, 180, 0), -1)
    if len(landmarks) >= 478:
        for idx in LEFT_IRIS_IDXS + RIGHT_IRIS_IDXS:
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 1, (255, 255, 255), -1)
    for idx in LEFT_EYE_IDXS + RIGHT_EYE_IDXS:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 2, (255, 100, 0), -1)
    for idx in MOUTH_IDXS:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 2, (0, 128, 255), -1)

# ---------------------------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Live DMS Video Demo (FiLM + GRU)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (if None, uses webcam 0)")
    parser.add_argument("--model", type=str, default="models/models/film_gru_fold3.pth", help="Path to PyTorch model weights")
    parser.add_argument("--scaler", type=str, default="models/models/final_scaler.joblib", help="Path to scaler")
    parser.add_argument("--attention", action="store_true", help="Enable temporal attention in model")
    parser.add_argument("--residual", action="store_true", help="Enable Residual Fallback with XGBoost baseline")
    parser.add_argument("--xgb-model", type=str, default="models/models/final_xgb_model.joblib", help="Path to XGBoost baseline model")
    parser.add_argument("--fps", type=int, default=4, help="Processing frame rate (model expects 4)")
    parser.add_argument("--calib-time", type=float, default=5.0, help="Calibration period in seconds")
    parser.add_argument("--show-mesh", type=bool, default=True, help="Draw face mesh on preview")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video file")
    parser.add_argument("--headless", action="store_true", help="Bypass GUI/window display for batch processing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Model Initialization
    try:
        model = FiLMGRUModel(
            num_classes=2, cnn_dim=64, geo_dim=11,
            geo_hidden=32, gru_hidden=64, gru_layers=1,
            dropout=0.3, use_film=True,
        ).to(device)
        model.use_attention = args.attention
        
        # Load weights
        model_path = Path(args.model)
        if not model_path.exists():
            # Try to auto-detect any film_gru checkpoint in models folders
            checkpoints = sorted(list(Path("models/models").glob("film_gru_fold*.pth"))) + \
                          sorted(list(Path("models").glob("film_gru_fold*.pth")))
            if checkpoints:
                model_path = checkpoints[0]
                print(f"Checkpoint {args.model} not found. Auto-loaded {model_path}")
            else:
                raise FileNotFoundError(f"Model checkpoint not found at {args.model}")
                
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"FiLM+GRU model loaded successfully from {model_path}.")
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        sys.exit(1)

    # 2. Scaler Initialization
    scaler = None
    scaler_path = Path(args.scaler)
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            print(f"MinMaxScaler loaded from {scaler_path}.")
        except Exception as e:
            print(f"Could not load scaler from {scaler_path}: {e}")
            
    if scaler is None:
        # Fallback 1: Try to fit scaler from behavioral_vectors.csv if it exists
        vectors_csv = Path("frame/csv/behavioral_vectors.csv")
        if vectors_csv.exists():
            try:
                print("Fitting a new MinMaxScaler using frame/csv/behavioral_vectors.csv...")
                df_vec = pd.read_csv(vectors_csv)
                scaler = MinMaxScaler()
                scaler.fit(df_vec[GEO_FEATURES].values)
                print("Fit MinMaxScaler successfully.")
            except Exception as e:
                print(f"Failed to fit scaler from dataset: {e}")
                
    if scaler is None:
        print("Warning: Running without scaling (identity scale).")
        class IdentityScaler:
            def transform(self, x): return x
        scaler = IdentityScaler()

    # 3. Residual Model Initialization (Optional Fallback)
    residual_model = None
    if args.residual:
        xgb_path = Path(args.xgb_model)
        if xgb_path.exists():
            try:
                residual_model = joblib.load(xgb_path)
                print(f"Residual Fallback Enabled. Loaded XGBoost baseline from {xgb_path}.")
            except Exception as e:
                print(f"Could not load XGBoost baseline model: {e}")
        else:
            print(f"Warning: Residual fallback enabled, but model not found at {args.xgb_model}. Bypassing.")

    # 4. MediaPipe Task Setup
    # Ensure model file is in Temp folder to prevent encoding/Unicode path problems
    landmarker_src = Path("face_landmarker.task")
    if not landmarker_src.exists():
        print("Error: face_landmarker.task not found in project root. Run pipeline or place task file there.")
        sys.exit(1)
        
    temp_landmarker_path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
    if not os.path.exists(temp_landmarker_path):
        shutil.copy(str(landmarker_src), temp_landmarker_path)
        
    base_options = mp_python.BaseOptions(model_asset_path=temp_landmarker_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # 5. Video Capture Initialization
    input_source = 0 if args.video is None else args.video
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {input_source}")
        sys.exit(1)
        
    # Get frame properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get FPS for frame-perfect sampling from video files
    is_video_file = (args.video is not None)
    if is_video_file:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or math.isnan(video_fps):
            video_fps = 30.0
        frame_sample_step = video_fps / args.fps
        print(f"Video Source Resolution: {w}x{h} | FPS: {video_fps:.2f} | Sampling Step: Every {frame_sample_step:.2f} frames")
    else:
        video_fps = 30.0
        frame_sample_step = None
        print(f"Webcam Source Resolution: {w}x{h} | Target Sampling: {args.fps} FPS")

    # Video Writer setup
    out_writer = None
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, video_fps, (w, h))
        print(f"Writing annotated output to: {args.output} (at {video_fps:.2f} FPS)")

    # Deques for history
    seq_len = 40
    frame_interval = 1.0 / args.fps
    
    # window_history stores features for the last 40 frames sampled at target FPS
    window_history = deque(maxlen=seq_len)
    
    # Store patch images for each frame: list of (3, 24, 24) np array
    patch_history = deque(maxlen=seq_len)
    
    # Variables for calibration
    is_calibrating = True
    calib_ear_vals = []
    calib_frames_target = int(args.calib_time * args.fps)
    alpha_threshold = 0.225 # default alpha threshold
    
    # Loop tracking variables
    last_sample_time = 0.0
    frames_read = 0
    sampled_count = 0
    
    fps_timer = time.time()
    fps_curr = 0.0
    
    # Model Outputs
    drowsy_prob = 0.0
    drowsy_state = "ALERT"

    print("\n" + "="*50)
    print("  DMS Stage E Demo Active")
    print("  Press 'q' in the window to quit.")
    print("  Press 'r' to force recalibration.")
    print("="*50 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_video_file and not args.headless:
                    # Loop video if running on file interactively
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    window_history.clear()
                    patch_history.clear()
                    frames_read = 0
                    sampled_count = 0
                    continue
                else:
                    break
                    
            frames_read += 1
            curr_time = time.time()
            
            # FPS tracking
            if curr_time - fps_timer >= 1.0:
                fps_curr = frames_read / (curr_time - fps_timer) if is_video_file else (frames_read - last_frame_count) / (curr_time - fps_timer)
                fps_timer = curr_time
                if not is_video_file:
                    last_frame_count = frames_read
            
            if not is_video_file and 'last_frame_count' not in locals():
                last_frame_count = 0

            # Decide whether to sample features and run inference
            should_sample = False
            if is_video_file:
                target_sample_idx = int(sampled_count * frame_sample_step)
                if frames_read - 1 >= target_sample_idx:
                    sampled_count += 1
                    should_sample = True
            else:
                if curr_time - last_sample_time >= frame_interval:
                    last_sample_time = curr_time
                    should_sample = True

            if should_sample:
                # Apply CLAHE pre-processing (GEMINI.md requirements)
                enhanced_frame = apply_clahe(frame)
                
                # Convert to MediaPipe image format
                img_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                
                # Run FaceMesh
                results = landmarker.detect(mp_image)
                
                # Frame statistics placeholder
                frame_stats = {
                    'ear': np.nan, 'mar': np.nan,
                    'pitch': np.nan, 'yaw': np.nan, 'roll': np.nan,
                    'eye_state': 0, 'valid': False
                }
                
                # Default patches
                patches_stacked = np.zeros((3, 24, 24), dtype=np.float32)
                
                if results.face_landmarks:
                    face_lms = results.face_landmarks[0]
                    frame_stats['valid'] = True
                    
                    # Extract Eye Aspect Ratio and Mouth Aspect Ratio
                    ear_val = eye_aspect_ratio(face_lms, LEFT_EYE_IDXS, w, h)
                    mar_val = mouth_aspect_ratio(face_lms, w, h)
                    pitch, yaw, roll = get_head_pose(face_lms, w, h)
                    
                    frame_stats['ear'] = ear_val
                    frame_stats['mar'] = mar_val
                    frame_stats['pitch'] = pitch
                    frame_stats['yaw'] = yaw
                    frame_stats['roll'] = roll
                    
                    # Crop eye and mouth patches (Isotropic padding rules)
                    l_eye_patch = crop_isotropic(enhanced_frame, face_lms, LEFT_EYE_IDXS, w, h)
                    r_eye_patch = crop_isotropic(enhanced_frame, face_lms, RIGHT_EYE_IDXS, w, h)
                    mouth_patch = crop_isotropic(enhanced_frame, face_lms, MOUTH_IDXS, w, h)
                    
                    # Convert to float and stack as 3 channels (Left eye, Right eye, Mouth)
                    patches_stacked = np.stack([
                        l_eye_patch.astype(np.float32) / 255.0,
                        r_eye_patch.astype(np.float32) / 255.0,
                        mouth_patch.astype(np.float32) / 255.0
                    ], axis=0)
                    
                    # Optional: Draw landmarks
                    if args.show_mesh:
                        draw_custom_mesh(frame, face_lms, w, h)
                        
                    # Calibration accumulation
                    if is_calibrating:
                        calib_ear_vals.append(ear_val)
                        if len(calib_ear_vals) >= calib_frames_target:
                            # 85th percentile represents baseline EAR_open
                            ear_open = np.percentile(calib_ear_vals, 85)
                            alpha_threshold = 0.75 * ear_open
                            is_calibrating = False
                            print(f"\nCalibration Completed!")
                            print(f"EAR_open: {ear_open:.4f} | Dynamic Alpha Threshold: {alpha_threshold:.4f}\n")
                            
                    # Set eye state based on threshold
                    frame_stats['eye_state'] = 1 if ear_val < alpha_threshold else 0
                
                # Append to sliding history deques
                window_history.append(frame_stats)
                patch_history.append(patches_stacked)
                
                # 6. Run Inference once we have enough window history
                if not is_calibrating and len(window_history) >= (seq_len // 2):
                    # Count valid frames in current sequence
                    valid_mask_seq = [float(f['valid']) for f in window_history]
                    
                    # Pad window sequence if it is not full yet
                    while len(window_history) < seq_len:
                        window_history.append(window_history[-1])
                        patch_history.append(patch_history[-1])
                        valid_mask_seq.append(0.0)
                        
                    # 6a. Compute confidence decay and forward fill patches (matching LateFusionDataset)
                    patch_seq_array = list(patch_history)
                    valid_indices = np.where(np.array(valid_mask_seq) > 0.0)[0]
                    confidence_seq = np.ones(seq_len, dtype=np.float32)
                    
                    if len(valid_indices) > 0:
                        for t in range(seq_len):
                            if valid_mask_seq[t] > 0.0:
                                continue
                            prev_v = valid_indices[valid_indices < t]
                            next_v = valid_indices[valid_indices > t]
                            if len(prev_v) and len(next_v):
                                dist = min(t - prev_v[-1], next_v[0] - t)
                                patch_seq_array[t] = patch_seq_array[prev_v[-1]]
                            elif len(prev_v):
                                dist = t - prev_v[-1]
                                patch_seq_array[t] = patch_seq_array[prev_v[-1]]
                            else:
                                dist = next_v[0] - t
                                patch_seq_array[t] = patch_seq_array[next_v[0]]
                              # Decay score
                            confidence_seq[t] = 0.85 ** dist
                    else:
                        confidence_seq[:] = 0.0
                        
                    # 6b. Aggregate geometry features
                    geo_features_vector = aggregate_window_features(list(window_history), args.fps)
                    
                    # 6c. Apply scaling
                    scaled_geo = scaler.transform(geo_features_vector.reshape(1, -1)).flatten().astype(np.float32)
                    
                    # 6d. Prepare tensors
                    patches_tensor = torch.from_numpy(np.stack(patch_seq_array)).unsqueeze(0).float().to(device) # (1, 40, 3, 24, 24)
                    valid_mask_tensor = torch.tensor(valid_mask_seq, dtype=torch.float32).unsqueeze(0).to(device) # (1, 40)
                    geo_tensor = torch.from_numpy(scaled_geo).unsqueeze(0).to(device) # (1, 11)
                    confidence_tensor = torch.from_numpy(confidence_seq).unsqueeze(0).to(device) # (1, 40)
                    
                    # 6e. FiLM+GRU Inference
                    with torch.no_grad():
                        logits = model(patches_tensor, valid_mask_tensor, geo_tensor, confidence_tensor)
                        
                        # Apply Residual Fallback formula if enabled
                        if args.residual and residual_model is not None:
                            # S_base probability of drowsy from XGBoost baseline
                            s_base = residual_model.predict_proba(scaled_geo.reshape(1, -1))[0, 1]
                            # Delta S = Tanh(DL score) * 0.15
                            delta_s = math.tanh(logits[0, 1] - logits[0, 0]) * 0.15
                            drowsy_prob = float(np.clip(s_base + delta_s, 0.0, 1.0))
                        else:
                            probs = torch.softmax(logits, dim=1)
                            drowsy_prob = float(probs[0, 1].item())
                            
                        drowsy_state = "DROWSY" if drowsy_prob > 0.5 else "ALERT"

            # 7. Render HUD on every frame for output/display
            metrics_display = {
                'ear': window_history[-1]['ear'] if len(window_history) > 0 else 0.0,
                'mar': window_history[-1]['mar'] if len(window_history) > 0 else 0.0,
                'pitch': window_history[-1]['pitch'] if len(window_history) > 0 else 0.0,
                'yaw': window_history[-1]['yaw'] if len(window_history) > 0 else 0.0,
                'roll': window_history[-1]['roll'] if len(window_history) > 0 else 0.0,
                'blink_rate': 0.0,
                'blink_duration': 0.0,
                'pose_jitter': 0.0,
                'perclos': 0.0
            }
            
            if len(window_history) > 0:
                sliding_features = aggregate_window_features(list(window_history), args.fps)
                metrics_display['perclos'] = float(sliding_features[0])
                metrics_display['blink_rate'] = float(sliding_features[1])
                metrics_display['blink_duration'] = float(sliding_features[2])
                metrics_display['pose_jitter'] = float(sliding_features[10])

            # Draw HUD
            calib_progress = len(calib_ear_vals) / calib_frames_target if is_calibrating else 1.0
            draw_hud(frame, metrics_display, drowsy_prob, drowsy_state, is_calibrating, calib_progress, alpha_threshold, fps_curr)
            
            # Save frame to output video if writer is active
            if out_writer is not None:
                out_writer.write(frame)
            
            # Interactive Display (skip if headless)
            if not args.headless:
                cv2.imshow("DMS Stage E Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("Force resetting calibration...")
                    is_calibrating = True
                    calib_ear_vals.clear()
                    window_history.clear()
                    patch_history.clear()
                    sampled_count = 0
            else:
                # Headless batch processing feedback
                if frames_read % 100 == 0:
                    print(f"Processed {frames_read} frames... State: {drowsy_state} ({drowsy_prob*100:.1f}%)")
                    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
            print(f"Saved completed output video.")
        cv2.destroyAllWindows()
        landmarker.close()
        print("Demo resources clean up completed successfully.")

if __name__ == "__main__":
    main()
