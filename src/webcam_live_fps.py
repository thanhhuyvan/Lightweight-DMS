"""
webcam_live_fps.py
------------------
Real-time webcam Driver Monitoring System (DMS) with advanced profiling HUD.
Specifically designed to measure and analyze latency, frame-rates, and performance
of each stage of the drowsiness detection pipeline.

Features:
  1. Real-time Laptop Webcam Feed
  2. Frame-accurate 4 FPS Sampling for Model History (40 frames = 10s sliding window)
  3. Real-time Face Mesh & Landmark detection (MediaPipe Tasks API)
  4. Isotropic Eye/Mouth Patch Cropping (padded 1:1, resized to 24x24)
  5. Geometric feature calculation (EAR, MAR, Head Pose, PERCLOS, Jitters)
  6. Hybrid FiLM+GRU Model Inference (with Residual Fallback to XGBoost)
  7. Advanced Live Profiling Panel (latency breakdown, loop FPS, model FPS)
  8. Interactive Keyboard Controls to toggle CLAHE, Inference, Face Mesh, Insets, etc.
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

# Import Config if available
try:
    from src.core_config import (
        CLAHE_CLIP_LIMIT,
        CLAHE_TILE_GRID,
        MIN_FACE_DETECTION_CONF,
        MIN_FACE_PRESENCE_CONF,
        MODEL_PATH as CONFIG_MODEL_PATH
    )
except ImportError:
    # Default Config values
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID = (8, 8)
    MIN_FACE_DETECTION_CONF = 0.4
    MIN_FACE_PRESENCE_CONF = 0.4
    CONFIG_MODEL_PATH = PROJECT_ROOT / 'face_landmarker.task'

# ---------------------------------------------------------------------------
# FiLM + GRU Model Architecture (Stage E SOTA)
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
# Global Indexes for Landmarks
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
# Mathematical Functions & Visual Helpers
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
    left   = (landmarks[61].x * img_w, landmarks[61].y * img_h)
    right  = (landmarks[291].x * img_w, landmarks[291].y * img_h)
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
    """Crops an image patch isotropically (1:1 aspect ratio) around landmarks."""
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
# Beautiful Rendering & HUD Drawer
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

def draw_hud(frame, metrics, prob, state, is_calibrating, calib_progress, alpha_thresh, fps_stats, show_profiler, show_insets, patches, show_help):
    h, w = frame.shape[:2]
    
    # ── Left DMS Panel (Width: 300px) ──
    left_overlay = frame.copy()
    cv2.rectangle(left_overlay, (0, 0), (320, h), (18, 18, 27), -1)
    cv2.addWeighted(left_overlay, 0.85, frame, 0.15, 0, frame)
    
    cv2.putText(frame, "DMS REALTIME FPS TEST", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 244), 2)
    cv2.line(frame, (15, 40), (300, 40), (49, 50, 68), 1)
    
    if is_calibrating:
        cv2.putText(frame, "STATUS: CALIBRATING", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (137, 180, 250), 2)
        cv2.rectangle(frame, (15, 80), (300, 95), (30, 30, 46), -1)
        progress_w = int(285 * calib_progress)
        cv2.rectangle(frame, (15, 80), (15 + progress_w, 95), (137, 180, 250), -1)
        cv2.putText(frame, f"{int(calib_progress*100)}%", (140, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (17, 17, 27), 2)
    else:
        alert_bg = (30, 58, 47) if state == "ALERT" else (30, 30, 58)
        alert_border = (46, 204, 113) if state == "ALERT" else (231, 76, 60)
        cv2.rectangle(frame, (15, 55), (300, 105), alert_bg, -1)
        cv2.rectangle(frame, (15, 55), (300, 105), alert_border, 2)
        cv2.putText(frame, state, (90 if state == "ALERT" else 75, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, alert_border, 3)

    # Real-Time Gauges
    y_offset = 135
    cv2.putText(frame, "REAL-TIME GAUGES", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (108, 112, 134), 1)
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
    ear_color = (46, 204, 113) if (not np.isnan(ear) and ear >= alpha_thresh) else (231, 76, 60)
    draw_gauge("Mean EAR", ear, 0.45, y_offset, ear_color)
    tick_pos = 15 + int(285 * min(alpha_thresh / 0.45, 1.0))
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
    cv2.putText(frame, f"Blink Avg Dur: {metrics.get('blink_duration', 0.0):.2f} s", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 214, 244), 1)
    y_offset += 20
    cv2.putText(frame, f"Pose Jitter: {metrics.get('pose_jitter', 0.0):.4f}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 214, 244), 1)
    
    y_offset += 25
    cv2.line(frame, (15, y_offset), (300, y_offset), (49, 50, 68), 1)
    y_offset += 15
    
    pitch = metrics.get('pitch', np.nan)
    yaw = metrics.get('yaw', np.nan)
    roll = metrics.get('roll', np.nan)
    pitch_str = "N/A" if np.isnan(pitch) else f"{pitch:.1f}"
    yaw_str = "N/A" if np.isnan(yaw) else f"{yaw:.1f}"
    roll_str = "N/A" if np.isnan(roll) else f"{roll:.1f}"
    cv2.putText(frame, f"Pitch:{pitch_str} Yaw:{yaw_str} Roll:{roll_str}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 214, 244), 1)
    
    if not is_calibrating:
        y_offset += 25
        cv2.putText(frame, "DROWSY PROBABILITY", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (108, 112, 134), 1)
        y_offset += 18
        cv2.rectangle(frame, (15, y_offset), (300, y_offset + 18), (30, 30, 46), -1)
        prob_color = (231, 76, 60) if prob > 0.5 else (46, 204, 113)
        prob_w = int(285 * prob)
        cv2.rectangle(frame, (15, y_offset), (15 + prob_w, y_offset + 18), prob_color, -1)
        cv2.putText(frame, f"{prob*100:.1f}%", (140, y_offset + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (205, 214, 244), 2)

    # Inset patches at the bottom of the left panel
    if show_insets and patches is not None:
        y_start = h - 110
        cv2.putText(frame, "INPUT PATCHES", (15, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (108, 112, 134), 1)
        # Left Eye (24x24 scale to 60x60)
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

    # ── Right Profile Panel (Width: 320px) ──
    if show_profiler:
        px_start = w - 320
        right_overlay = frame.copy()
        cv2.rectangle(right_overlay, (px_start, 0), (w, h), (30, 30, 46), -1)
        cv2.addWeighted(right_overlay, 0.85, frame, 0.15, 0, frame)
        
        cv2.putText(frame, "LATENCY & PERFORMANCE", (px_start + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 194, 231), 2)
        cv2.line(frame, (px_start + 15, 40), (w - 15, 40), (49, 50, 68), 1)
        
        # FPS values
        py_offset = 65
        cv2.putText(frame, f"Webcam Display FPS: {fps_stats['display']:.1f}", (px_start + 15, py_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (205, 214, 244), 2)
        py_offset += 25
        cv2.putText(frame, f"Model Inference FPS: {fps_stats['inference_fps']:.1f}", (px_start + 15, py_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (249, 226, 175), 1)
        py_offset += 25
        cv2.line(frame, (px_start + 15, py_offset), (w - 15, py_offset), (49, 50, 68), 1)
        py_offset += 20
        
        cv2.putText(frame, "LATENCY BREAKDOWN (ms)", (px_start + 15, py_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (108, 112, 134), 1)
        py_offset += 22
        
        latency_items = [
            ("Frame Acquisition", fps_stats['acq_ms'], (203, 166, 247)),
            ("Image Preprocess", fps_stats['prep_ms'], (137, 220, 235)),
            ("Landmark Detection", fps_stats['mesh_ms'], (166, 227, 161)),
            ("Patch Cropping", fps_stats['crop_ms'], (249, 226, 175)),
            ("Feature Extraction", fps_stats['feats_ms'], (250, 179, 135)),
            ("Model Inference", fps_stats['infer_ms'], (243, 139, 168)),
            ("HUD Render & Draw", fps_stats['render_ms'], (116, 199, 236))
        ]
        
        total_tracked_ms = 0.0
        for name, val, color in latency_items:
            cv2.putText(frame, f"{name}:", (px_start + 15, py_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (166, 173, 200), 1)
            cv2.putText(frame, f"{val:.1f} ms", (w - 100, py_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            total_tracked_ms += val
            py_offset += 16
            
        py_offset += 10
        cv2.line(frame, (px_start + 15, py_offset), (w - 15, py_offset), (49, 50, 68), 1)
        py_offset += 22
        cv2.putText(frame, f"Total Processing: {total_tracked_ms:.1f} ms", (px_start + 15, py_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (242, 205, 205), 2)
        py_offset += 25
        
        # Configuration indicators
        cv2.putText(frame, "PIPELINE SETTINGS", (px_start + 15, py_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (108, 112, 134), 1)
        py_offset += 20
        
        settings = [
            ("CLAHE Preprocess", fps_stats['cfg_clahe'], 'c'),
            ("PyTorch Model", fps_stats['cfg_pytorch'], 'p'),
            ("Residual Fallback", fps_stats['cfg_residual'], 'x'),
            ("Render FaceMesh", fps_stats['cfg_mesh'], 'm')
        ]
        for name, enabled, key in settings:
            status_text = "ENABLED" if enabled else "DISABLED"
            status_color = (166, 227, 161) if enabled else (108, 112, 134)
            cv2.putText(frame, f"[{key}] {name}:", (px_start + 15, py_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (205, 214, 244), 1)
            cv2.putText(frame, status_text, (w - 100, py_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
            py_offset += 16
            
        py_offset += 15
        cv2.putText(frame, "Press [h] to Toggle Key Help", (px_start + 15, py_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (137, 220, 235), 1)

    # ── Help Overlay Menu ──
    if show_help:
        # Render a translucent center box
        overlay = frame.copy()
        menu_w, menu_h = 420, 320
        mx, my = (w - menu_w) // 2, (h - menu_h) // 2
        cv2.rectangle(overlay, (mx, my), (mx + menu_w, my + menu_h), (17, 17, 27), -1)
        cv2.rectangle(overlay, (mx, my), (mx + menu_w, my + menu_h), (203, 166, 247), 2)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
        
        cv2.putText(frame, "DMS FPS TESTER - CONTROLS MENU", (mx + 30, my + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (203, 166, 247), 2)
        cv2.line(frame, (mx + 30, my + 45), (mx + menu_w - 30, my + 45), (49, 50, 68), 1)
        
        controls = [
            ("[q] Quit", "Close the app and release resources"),
            ("[r] Recalibrate", "Reset EAR calibration baseline"),
            ("[c] Toggle CLAHE", "Enable/disable contrast equalization"),
            ("[p] Toggle Inference", "Disable PyTorch for baseline speed test"),
            ("[x] Toggle Residual", "Toggle baseline XGBoost fallback"),
            ("[m] Toggle Mesh", "Show/hide face mesh landmark circles"),
            ("[i] Toggle Insets", "Show/hide the cropped input patches"),
            ("[f] Toggle Profiler", "Show/hide latency breakdown panel"),
            ("[h] Hide Menu", "Close this keyboard help overlay")
        ]
        
        cy_offset = my + 75
        for key, desc in controls:
            cv2.putText(frame, key, (mx + 30, cy_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (249, 226, 175), 1)
            cv2.putText(frame, desc, (mx + 170, cy_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (205, 214, 244), 1)
            cy_offset += 24

# ---------------------------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Real-Time Webcam DMS & FPS Benchmark")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--model", type=str, default="models/film_gru_fold3.pth", help="Path to PyTorch model weights")
    parser.add_argument("--scaler", type=str, default="models/final_scaler.joblib", help="Path to MinMaxScaler joblib")
    parser.add_argument("--xgb-model", type=str, default="models/final_xgb_model.joblib", help="Path to XGBoost baseline fallback model")
    parser.add_argument("--fps", type=int, default=4, help="Model evaluation frequency in Hz (default: 4)")
    parser.add_argument("--calib-time", type=float, default=5.0, help="Calibration duration in seconds")
    parser.add_argument("--width", type=int, default=1280, help="Desired webcam resolution width")
    parser.add_argument("--height", type=int, default=720, help="Desired webcam resolution height")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # 1. Load PyTorch FiLM+GRU Model
    model = None
    model_enabled = False
    try:
        model_path = Path(args.model)
        if model_path.exists():
            model = FiLMGRUModel(
                num_classes=2, cnn_dim=64, geo_dim=11,
                geo_hidden=32, gru_hidden=64, gru_layers=1,
                dropout=0.3, use_film=True
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model_enabled = True
            print(f"Loaded FiLM+GRU model: {model_path}")
        else:
            print(f"Warning: Model not found at {args.model}. Inference will run in fallback/dummy mode.")
    except Exception as e:
        print(f"Error loading PyTorch model: {e}. Model inference will be disabled.")

    # 2. Load MinMaxScaler Scaler
    scaler = None
    scaler_path = Path(args.scaler)
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            print(f"Loaded feature scaler: {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            
    if scaler is None:
        print("Using Identity Scaler fallback.")
        class IdentityScaler:
            def transform(self, x): return x
        scaler = IdentityScaler()

    # 3. Load XGBoost Baseline model
    xgb_model = None
    xgb_enabled = False
    xgb_path = Path(args.xgb_model)
    if xgb_path.exists():
        try:
            xgb_model = joblib.load(xgb_path)
            xgb_enabled = True
            print(f"Loaded Residual Fallback XGBoost model: {xgb_path}")
        except Exception as e:
            print(f"Error loading XGBoost model: {e}")

    # 4. MediaPipe FaceLandmarker Setup
    temp_landmarker_path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
    # Reference task file from config or check current directory
    landmarker_src = Path("face_landmarker.task")
    if not landmarker_src.exists():
        landmarker_src = CONFIG_MODEL_PATH
        
    if not landmarker_src.exists():
        # Let's try downloading face_landmarker if not found anywhere
        print("face_landmarker.task not found in root. Downloading...")
        import urllib.request
        MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        try:
            urllib.request.urlretrieve(MODEL_URL, "face_landmarker.task")
            landmarker_src = Path("face_landmarker.task")
            print("Downloaded successfully.")
        except Exception as e:
            print(f"Error downloading MediaPipe asset: {e}")
            sys.exit(1)

    if not os.path.exists(temp_landmarker_path):
        shutil.copy(str(landmarker_src), temp_landmarker_path)
        
    base_options = mp_python.BaseOptions(model_asset_path=temp_landmarker_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        min_face_detection_confidence=MIN_FACE_DETECTION_CONF,
        min_face_presence_confidence=MIN_FACE_PRESENCE_CONF
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # 5. Open Laptop Webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not access laptop webcam at index {args.camera}")
        sys.exit(1)
        
    # Attempt to set high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam initialized successfully: {w}x{h}")

    # 6. Initialize History & Calibration Deques
    seq_len = 40
    frame_interval = 1.0 / args.fps
    
    window_history = deque(maxlen=seq_len)
    patch_history = deque(maxlen=seq_len)
    
    # Latency tracking deques (size: 30) for smooth moving average display
    acq_deq = deque(maxlen=30)
    prep_deq = deque(maxlen=30)
    mesh_deq = deque(maxlen=30)
    crop_deq = deque(maxlen=30)
    feats_deq = deque(maxlen=30)
    infer_deq = deque(maxlen=30)
    render_deq = deque(maxlen=30)
    
    is_calibrating = True
    calib_ear_vals = []
    calib_frames_target = int(args.calib_time * args.fps)
    alpha_threshold = 0.225
    
    # UI Toggles
    show_profiler = True
    show_insets = True
    show_mesh = True
    show_help = False
    clahe_enabled = True
    
    # Loop metrics
    last_sample_time = 0.0
    frames_count = 0
    inference_count = 0
    drowsy_prob = 0.0
    drowsy_state = "ALERT"
    latest_patches = None
    
    # FPS Timers
    fps_display_timer = time.time()
    fps_display_frames = 0
    display_fps = 0.0
    
    fps_model_timer = time.time()
    model_fps = 0.0
    
    print("\n" + "="*60)
    print("  Real-time Webcam Driver Monitoring System (DMS) Active")
    print("  Controls Menu:")
    print("    [q] - Quit program           [r] - Reset Calibration")
    print("    [c] - Toggle CLAHE           [p] - Toggle PyTorch Model")
    print("    [x] - Toggle Residual        [m] - Toggle Face Mesh")
    print("    [i] - Toggle Crop Insets     [f] - Toggle Profiler Panel")
    print("    [h] - Toggle Help Menu")
    print("="*60 + "\n")

    try:
        while True:
            t_loop_start = time.time()
            
            # --- STAGE 1: Frame Acquisition ---
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab webcam frame.")
                break
                
            t_acq = time.time()
            acq_deq.append((t_acq - t_loop_start) * 1000.0)
            
            frames_count += 1
            curr_time = time.time()
            
            # Display FPS calculation
            fps_display_frames += 1
            if curr_time - fps_display_timer >= 1.0:
                display_fps = fps_display_frames / (curr_time - fps_display_timer)
                fps_display_timer = curr_time
                fps_display_frames = 0
                
            # Model FPS calculation
            if curr_time - fps_model_timer >= 1.0:
                model_fps = inference_count / (curr_time - fps_model_timer)
                fps_model_timer = curr_time
                inference_count = 0
                
            # Check model sampling condition
            should_sample = (curr_time - last_sample_time >= frame_interval)
            
            # --- STAGE 2: Image Preprocessing ---
            t_prep_start = time.time()
            if clahe_enabled:
                enhanced_frame = apply_clahe(frame, clip_limit=CLAHE_CLIP_LIMIT, tile_grid=CLAHE_TILE_GRID)
            else:
                enhanced_frame = frame.copy()
            
            img_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            prep_deq.append((time.time() - t_prep_start) * 1000.0)
            
            # --- STAGE 3: Landmark Detection (MediaPipe) ---
            t_mesh_start = time.time()
            results = landmarker.detect(mp_image)
            mesh_deq.append((time.time() - t_mesh_start) * 1000.0)
            
            # Frame statistics placeholder
            frame_stats = {
                'ear': np.nan, 'mar': np.nan,
                'pitch': np.nan, 'yaw': np.nan, 'roll': np.nan,
                'eye_state': 0, 'valid': False
            }
            patches_stacked = np.zeros((3, 24, 24), dtype=np.float32)
            
            # --- STAGE 4: Patch Cropping ---
            t_crop_start = time.time()
            face_lms = None
            if results.face_landmarks:
                face_lms = results.face_landmarks[0]
                frame_stats['valid'] = True
                
                # Crop left/right eye & mouth patches (Isotropic padding rule)
                l_eye_patch = crop_isotropic(enhanced_frame, face_lms, LEFT_EYE_IDXS, w, h)
                r_eye_patch = crop_isotropic(enhanced_frame, face_lms, RIGHT_EYE_IDXS, w, h)
                mouth_patch = crop_isotropic(enhanced_frame, face_lms, MOUTH_IDXS, w, h)
                
                patches_stacked = np.stack([
                    l_eye_patch.astype(np.float32) / 255.0,
                    r_eye_patch.astype(np.float32) / 255.0,
                    mouth_patch.astype(np.float32) / 255.0
                ], axis=0)
                
                latest_patches = patches_stacked
            crop_deq.append((time.time() - t_crop_start) * 1000.0)
            
            # --- STAGE 5: Feature Extraction ---
            t_feats_start = time.time()
            if face_lms is not None:
                # Geometric Calculations
                ear_val = eye_aspect_ratio(face_lms, LEFT_EYE_IDXS, w, h)
                mar_val = mouth_aspect_ratio(face_lms, w, h)
                pitch, yaw, roll = get_head_pose(face_lms, w, h)
                
                frame_stats['ear'] = ear_val
                frame_stats['mar'] = mar_val
                frame_stats['pitch'] = pitch
                frame_stats['yaw'] = yaw
                frame_stats['roll'] = roll
                
                # Calibration Phase
                if is_calibrating and should_sample:
                    calib_ear_vals.append(ear_val)
                    if len(calib_ear_vals) >= calib_frames_target:
                        ear_open = np.percentile(calib_ear_vals, 85)
                        alpha_threshold = 0.75 * ear_open
                        is_calibrating = False
                        print(f"Calibration Done! Baseline EAR_open: {ear_open:.4f} Threshold: {alpha_threshold:.4f}")
                        
                frame_stats['eye_state'] = 1 if ear_val < alpha_threshold else 0
                
            # Push into sliding window deques at target sampling rate
            if should_sample:
                last_sample_time = curr_time
                window_history.append(frame_stats)
                patch_history.append(patches_stacked)
            feats_deq.append((time.time() - t_feats_start) * 1000.0)
            
            # --- STAGE 6: Model Inference ---
            t_infer_start = time.time()
            if model_enabled and not is_calibrating and len(window_history) >= (seq_len // 2) and should_sample:
                inference_count += 1
                
                # Assemble temporal sequences with padding / interpolation
                valid_mask_seq = [float(f['valid']) for f in window_history]
                
                # Keep local copies of history lists
                win_hist_list = list(window_history)
                patch_hist_list = list(patch_history)
                
                # Pad out to seq_len
                while len(win_hist_list) < seq_len:
                    win_hist_list.append(win_hist_list[-1])
                    patch_hist_list.append(patch_hist_list[-1])
                    valid_mask_seq.append(0.0)
                    
                # Forward-fill patches for occlusions and decay confidence
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
                            patch_hist_list[t] = patch_hist_list[prev_v[-1]]
                        elif len(prev_v):
                            dist = t - prev_v[-1]
                            patch_hist_list[t] = patch_hist_list[prev_v[-1]]
                        else:
                            dist = next_v[0] - t
                            patch_hist_list[t] = patch_hist_list[next_v[0]]
                        
                        confidence_seq[t] = 0.85 ** dist
                else:
                    confidence_seq[:] = 0.0
                    
                # Aggregate sliding geometry features
                geo_features_vector = aggregate_window_features(win_hist_list, args.fps)
                
                # Scale geometry features using Min-Max Standard scaler
                scaled_geo = scaler.transform(geo_features_vector.reshape(1, -1)).flatten().astype(np.float32)
                
                # Tensors preparation
                patches_tensor = torch.from_numpy(np.stack(patch_hist_list)).unsqueeze(0).float().to(device)
                valid_mask_tensor = torch.tensor(valid_mask_seq, dtype=torch.float32).unsqueeze(0).to(device)
                geo_tensor = torch.from_numpy(scaled_geo).unsqueeze(0).to(device)
                confidence_tensor = torch.from_numpy(confidence_seq).unsqueeze(0).to(device)
                
                # Model inference
                with torch.no_grad():
                    logits = model(patches_tensor, valid_mask_tensor, geo_tensor, confidence_tensor)
                    
                    if xgb_enabled and xgb_model is not None:
                        # Residual Fallback Formula: S_final = S_base + Delta_S
                        s_base = xgb_model.predict_proba(scaled_geo.reshape(1, -1))[0, 1]
                        delta_s = math.tanh(logits[0, 1] - logits[0, 0]) * 0.15
                        drowsy_prob = float(np.clip(s_base + delta_s, 0.0, 1.0))
                    else:
                        probs = torch.softmax(logits, dim=1)
                        drowsy_prob = float(probs[0, 1].item())
                        
                    drowsy_state = "DROWSY" if drowsy_prob > 0.5 else "ALERT"
                    
            infer_deq.append((time.time() - t_infer_start) * 1000.0)
            
            # --- STAGE 7: HUD Rendering and UI Draw ---
            t_render_start = time.time()
            
            # Draw Face Mesh on source camera frame
            if show_mesh and face_lms is not None:
                draw_custom_mesh(frame, face_lms, w, h)
                
            # Prepare display metrics
            metrics_display = {
                'ear': window_history[-1]['ear'] if len(window_history) > 0 else np.nan,
                'mar': window_history[-1]['mar'] if len(window_history) > 0 else np.nan,
                'pitch': window_history[-1]['pitch'] if len(window_history) > 0 else np.nan,
                'yaw': window_history[-1]['yaw'] if len(window_history) > 0 else np.nan,
                'roll': window_history[-1]['roll'] if len(window_history) > 0 else np.nan,
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
                
            # Assemble FPS profiling statistics
            fps_stats = {
                'display': display_fps,
                'inference_fps': model_fps,
                'acq_ms': float(np.mean(acq_deq)) if acq_deq else 0.0,
                'prep_ms': float(np.mean(prep_deq)) if prep_deq else 0.0,
                'mesh_ms': float(np.mean(mesh_deq)) if mesh_deq else 0.0,
                'crop_ms': float(np.mean(crop_deq)) if crop_deq else 0.0,
                'feats_ms': float(np.mean(feats_deq)) if feats_deq else 0.0,
                'infer_ms': float(np.mean(infer_deq)) if infer_deq else 0.0,
                'render_ms': float(np.mean(render_deq)) if render_deq else 0.0,
                'cfg_clahe': clahe_enabled,
                'cfg_pytorch': model_enabled,
                'cfg_residual': xgb_enabled,
                'cfg_mesh': show_mesh
            }
            
            calib_progress = len(calib_ear_vals) / calib_frames_target if is_calibrating else 1.0
            draw_hud(frame, metrics_display, drowsy_prob, drowsy_state, is_calibrating, calib_progress, alpha_threshold, fps_stats, show_profiler, show_insets, latest_patches, show_help)
            
            cv2.imshow("DMS Realtime FPS Test", frame)
            
            render_deq.append((time.time() - t_render_start) * 1000.0)
            
            # --- KEYBOARD CONTROLS HANDLER ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Force resetting calibration...")
                is_calibrating = True
                calib_ear_vals.clear()
                window_history.clear()
                patch_history.clear()
            elif key == ord('c'):
                clahe_enabled = not clahe_enabled
                print(f"CLAHE preprocessing: {'ENABLED' if clahe_enabled else 'DISABLED'}")
            elif key == ord('p'):
                model_enabled = not model_enabled and model is not None
                print(f"PyTorch model inference: {'ENABLED' if model_enabled else 'DISABLED'}")
            elif key == ord('x'):
                xgb_enabled = not xgb_enabled and xgb_model is not None
                print(f"Residual Fallback (XGBoost): {'ENABLED' if xgb_enabled else 'DISABLED'}")
            elif key == ord('m'):
                show_mesh = not show_mesh
                print(f"Face Mesh rendering: {'SHOW' if show_mesh else 'HIDE'}")
            elif key == ord('i'):
                show_insets = not show_insets
                print(f"Patch insets rendering: {'SHOW' if show_insets else 'HIDE'}")
            elif key == ord('f'):
                show_profiler = not show_profiler
                print(f"Profiler panel: {'SHOW' if show_profiler else 'HIDE'}")
            elif key == ord('h'):
                show_help = not show_help
                
    except KeyboardInterrupt:
        print("\nReal-time webcam loop interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("Webcam resources released. Program completed.")

if __name__ == '__main__':
    main()
