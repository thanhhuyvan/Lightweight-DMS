"""
stats_aggregation.py — Stage 4: Statistical Aggregation (60s Sliding Window)

Goal:
    Aggregate frame-level signals into behavioral vectors that capture temporal
    drowsiness patterns over 60-second intervals.
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# ── Project path setup ──
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core_config import SUMMARY_FEATURES_CSV, CSV_DIR, TARGET_FPS

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════
WINDOW_SECONDS = 5                         # 10-second sliding window
WINDOW_SIZE    = WINDOW_SECONDS * TARGET_FPS  # 40 frames
STRIDE_SECONDS = 1                          # 1-second stride
STRIDE_FRAMES  = STRIDE_SECONDS * TARGET_FPS  # 4 frames
MIN_PERIODS    = WINDOW_SIZE // 2            # 20 frames (50% minimum)

# Output path
BEHAVIORAL_VECTORS_CSV = CSV_DIR / 'behavioral_vectors.csv'


def _resolve_col(df: pd.DataFrame, base_name: str) -> str:
    smooth = f'{base_name}_smooth'
    if smooth in df.columns:
        return smooth
    if base_name in df.columns:
        return base_name
    raise KeyError(f"Neither '{smooth}' nor '{base_name}' found in DataFrame.")


def _count_blinks(eye_state_array: np.ndarray) -> int:
    if len(eye_state_array) < 2:
        return 0
    transitions = np.diff(eye_state_array)
    return int(np.sum(transitions == 1))


def _mean_blink_duration(eye_state_array: np.ndarray, fps: int) -> float:
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


def aggregate_group(
    group: pd.DataFrame,
    ear_col: str,
    mar_col: str,
    p_col: str,
    y_col: str,
    r_col: str,
) -> pd.DataFrame:
    n_frames = len(group)
    if n_frames < MIN_PERIODS:
        return pd.DataFrame()

    # Pre-extract numpy arrays
    eye_states = group['eye_state'].values.astype(int)
    ear_values = group[ear_col].values.astype(float)
    mar_values = group[mar_col].values.astype(float)
    p_values   = group[p_col].values.astype(float)
    y_values   = group[y_col].values.astype(float)
    r_values   = group[r_col].values.astype(float)
    frame_files = group['frame_file'].values

    video_id = group['video_id'].iloc[0]
    participant_id = group['participant_id'].iloc[0]

    records = []

    for start in range(0, n_frames, STRIDE_FRAMES):
        end = start + WINDOW_SIZE
        if end > n_frames: end = n_frames
        win_len = end - start
        if win_len < MIN_PERIODS: break

        # Slice
        w_eye = eye_states[start:end]
        w_ear = ear_values[start:end]
        w_mar = mar_values[start:end]
        w_p   = p_values[start:end]
        w_y   = y_values[start:end]
        w_r   = r_values[start:end]

        perclos = (np.sum(w_eye == 1) / win_len) * 100.0
        n_blinks = _count_blinks(w_eye)
        blink_rate = (n_blinks / (win_len / TARGET_FPS)) * 60.0
        blink_avg_dur = _mean_blink_duration(w_eye, TARGET_FPS)

        # Robust stats (filtering NaNs)
        ear_valid = w_ear[~np.isnan(w_ear)]
        ear_mean = np.mean(ear_valid) if len(ear_valid) > 0 else np.nan
        ear_std  = np.std(ear_valid, ddof=1) if len(ear_valid) > 1 else 0.0

        mar_valid = w_mar[~np.isnan(w_mar)]
        mar_mean = np.mean(mar_valid) if len(mar_valid) > 0 else np.nan
        mar_max  = np.max(mar_valid) if len(mar_valid) > 0 else np.nan

        p_valid = w_p[~np.isnan(w_p)]
        y_valid = w_y[~np.isnan(w_y)]
        r_valid = w_r[~np.isnan(w_r)]
        
        jit_p = np.var(p_valid, ddof=1) if len(p_valid) > 1 else 0.0
        jit_y = np.var(y_valid, ddof=1) if len(y_valid) > 1 else 0.0
        jit_r = np.var(r_valid, ddof=1) if len(r_valid) > 1 else 0.0
        
        # Combined Pose Index (Emphasis on Pitch and Yaw for drowsiness)
        pose_jitter = jit_p + jit_y + (0.5 * jit_r)

        records.append({
            'video_id':           video_id,
            'participant_id':     participant_id,
            'window_start_frame': frame_files[start],
            'window_end_frame':   frame_files[end - 1],
            'window_start_idx':   start,
            'window_end_idx':     end - 1,
            'PERCLOS':            round(perclos, 4),
            'Blink_Rate':         round(blink_rate, 4),
            'Blink_Avg_Duration': round(blink_avg_dur, 4),
            'EAR_Mean':           round(float(ear_mean), 5) if not np.isnan(ear_mean) else np.nan,
            'EAR_Std':            round(float(ear_std), 5),
            'MAR_Mean':           round(float(mar_mean), 5) if not np.isnan(mar_mean) else np.nan,
            'MAR_Max':            round(float(mar_max), 5) if not np.isnan(mar_max) else np.nan,
            'Pitch_Jitter':       round(float(jit_p), 4),
            'Yaw_Jitter':         round(float(jit_y), 4),
            'Roll_Jitter':        round(float(jit_r), 4),
            'Pose_Jitter':        round(float(pose_jitter), 4),
        })

    return pd.DataFrame(records)


def compute_behavioral_vectors(df: pd.DataFrame) -> pd.DataFrame:
    ear_col = _resolve_col(df, 'mean_EAR')
    mar_col = _resolve_col(df, 'MAR')
    p_col   = _resolve_col(df, 'pitch')
    y_col   = _resolve_col(df, 'yaw')
    r_col   = _resolve_col(df, 'roll')

    logging.info(f"Aggregation using: EAR={ear_col}, MAR={mar_col}, Pose=[{p_col}, {y_col}, {r_col}]")

    grouped = df.groupby(['video_id', 'participant_id'])
    all_vectors = []
    for name, group in grouped:
        group = group.sort_values('frame_file').copy()
        vectors = aggregate_group(group, ear_col, mar_col, p_col, y_col, r_col)
        if not vectors.empty:
            all_vectors.append(vectors)

    if not all_vectors: return pd.DataFrame()
    return pd.concat(all_vectors, ignore_index=True)


def main():
    if not SUMMARY_FEATURES_CSV.exists():
        logging.error(f"Input file {SUMMARY_FEATURES_CSV} not found.")
        return

    logging.info(f"Reading frame-level features from {SUMMARY_FEATURES_CSV}...")
    df = pd.read_csv(SUMMARY_FEATURES_CSV)
    
    required_cols = ['video_id', 'participant_id', 'frame_file', 'eye_state']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logging.error(f"Missing required columns: {missing}.")
        return

    result = compute_behavioral_vectors(df)
    if result.empty:
        logging.error("No output generated.")
        return

    result.to_csv(BEHAVIORAL_VECTORS_CSV, index=False)
    logging.info(f"Aggregated {len(result)} behavioral vectors to {BEHAVIORAL_VECTORS_CSV}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    main()
