"""
stats_aggregation.py — Stage 4: Statistical Aggregation (60s Sliding Window)

Goal:
    Aggregate frame-level signals into behavioral vectors that capture temporal
    drowsiness patterns over 60-second intervals.

Architecture Position:
    Stage 3 (eye_state.py / features_perclos.py)
        → **Stage 4 (stats_aggregation.py)**
            → Stage 5 (ML Classification)

Input:
    SUMMARY_FEATURES_CSV (frame/csv/features_summary.csv)
    Required columns: video_id, participant_id, frame_file, face_detected,
                      mean_EAR_smooth (or mean_EAR), eye_state,
                      head_dx_smooth (or head_dx), head_dy_smooth (or head_dy),
                      MAR_smooth (or MAR)

Output:
    CSV_DIR / 'behavioral_vectors.csv'
    Each row represents one 60-second window with aggregated features:
        - PERCLOS          : % of time eyes are closed
        - Blink_Rate       : Number of blinks per minute
        - Blink_Avg_Duration: Average blink duration (seconds)
        - EAR_Mean         : Mean EAR in window
        - EAR_Std          : Std deviation of EAR
        - MAR_Mean         : Mean MAR (yawning indicator)
        - MAR_Max          : Max MAR in window (peak yawn)
        - Pose_Jitter_dx   : Variance of head dx (lateral sway)
        - Pose_Jitter_dy   : Variance of head dy (nodding)
        - Pose_Jitter      : Combined pose jitter (dx + dy variance)

Window Configuration:
    - Window Size : 240 frames (60 seconds at 4 FPS)
    - Stride      : 4 frames   (1 second)
    - Min Periods : 120 frames (50% of window — avoids noisy edge stats)

Usage:
    python -m src.stats_aggregation        (from PROJECT_ROOT)
    python src/stats_aggregation.py        (direct)

Dependencies:
    Requires eye_state.py to have been run first (needs 'eye_state' column).
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
WINDOW_SECONDS = 60                         # 60-second sliding window
WINDOW_SIZE    = WINDOW_SECONDS * TARGET_FPS  # 240 frames
STRIDE_SECONDS = 1                          # 1-second stride
STRIDE_FRAMES  = STRIDE_SECONDS * TARGET_FPS  # 4 frames
MIN_PERIODS    = WINDOW_SIZE // 2            # 120 frames (50% minimum)

# Output path
BEHAVIORAL_VECTORS_CSV = CSV_DIR / 'behavioral_vectors.csv'


# ══════════════════════════════════════════════════════════════
# Helper: Resolve column names (prefer _smooth, fallback to raw)
# ══════════════════════════════════════════════════════════════
def _resolve_col(df: pd.DataFrame, base_name: str) -> str:
    """
    Returns '{base_name}_smooth' if available, else '{base_name}'.
    Raises KeyError if neither exists.
    """
    smooth = f'{base_name}_smooth'
    if smooth in df.columns:
        return smooth
    if base_name in df.columns:
        return base_name
    raise KeyError(
        f"Neither '{smooth}' nor '{base_name}' found in DataFrame columns: "
        f"{list(df.columns)}"
    )


# ══════════════════════════════════════════════════════════════
# Core: Count blinks in a boolean series
# ══════════════════════════════════════════════════════════════
def _count_blinks(eye_state_array: np.ndarray) -> int:
    """
    Count the number of blink events in an eye_state array.

    A blink is defined as a rising edge (0 → 1 transition) in the
    eye_state signal. Each transition represents the START of one
    eye-closure event.

    Parameters
    ----------
    eye_state_array : np.ndarray
        1D array of binary values (0 = open, 1 = closed).

    Returns
    -------
    int
        Number of blink events (0→1 transitions).
    """
    if len(eye_state_array) < 2:
        return 0
    # Detect rising edges: current=1, previous=0
    transitions = np.diff(eye_state_array)
    return int(np.sum(transitions == 1))


def _mean_blink_duration(eye_state_array: np.ndarray, fps: int) -> float:
    """
    Calculate the average duration of blink/closure events in seconds.

    Parameters
    ----------
    eye_state_array : np.ndarray
        1D array of binary values (0 = open, 1 = closed).
    fps : int
        Frames per second.

    Returns
    -------
    float
        Average blink duration in seconds, or 0.0 if no blinks found.
    """
    if len(eye_state_array) < 2:
        return 0.0

    durations = []
    in_blink = False
    blink_start = 0

    for i, state in enumerate(eye_state_array):
        if state == 1 and not in_blink:
            # Rising edge — blink starts
            in_blink = True
            blink_start = i
        elif state == 0 and in_blink:
            # Falling edge — blink ends
            in_blink = False
            durations.append((i - blink_start) / fps)

    # Handle blink that extends to end of window
    if in_blink:
        durations.append((len(eye_state_array) - blink_start) / fps)

    return float(np.mean(durations)) if durations else 0.0


# ══════════════════════════════════════════════════════════════
# Core: Aggregate a single group (video_id × participant_id)
# ══════════════════════════════════════════════════════════════
def aggregate_group(
    group: pd.DataFrame,
    ear_col: str,
    mar_col: str,
    dx_col: str,
    dy_col: str,
) -> pd.DataFrame:
    """
    Apply the 60s sliding window to a single video×participant group.

    For each window position (stride = STRIDE_FRAMES):
        1. PERCLOS           — % frames closed
        2. Blink_Rate        — blinks per minute
        3. Blink_Avg_Duration— average closure duration (seconds)
        4. EAR_Mean          — mean EAR
        5. EAR_Std           — std of EAR
        6. MAR_Mean          — mean MAR (yawn indicator)
        7. MAR_Max           — max MAR in window (peak yawn)
        8. Pose_Jitter_dx    — variance of head_dx
        9. Pose_Jitter_dy    — variance of head_dy
       10. Pose_Jitter       — combined variance (dx + dy)

    Parameters
    ----------
    group : pd.DataFrame
        Frame-level data for one video×participant, sorted by frame_file.
    ear_col : str
        Column name for EAR values.
    mar_col : str
        Column name for MAR values.
    dx_col, dy_col : str
        Column names for head pose proxies.

    Returns
    -------
    pd.DataFrame
        Aggregated behavioral vectors, one row per window.
    """
    n_frames = len(group)
    if n_frames < MIN_PERIODS:
        logging.warning(
            f"Group ({group['video_id'].iloc[0]}, {group['participant_id'].iloc[0]}): "
            f"only {n_frames} frames — skipping (need ≥ {MIN_PERIODS})."
        )
        return pd.DataFrame()

    # Pre-extract numpy arrays for performance
    eye_states = group['eye_state'].values.astype(int)
    ear_values = group[ear_col].values.astype(float)
    mar_values = group[mar_col].values.astype(float)
    dx_values  = group[dx_col].values.astype(float)
    dy_values  = group[dy_col].values.astype(float)
    frame_files = group['frame_file'].values

    # Metadata (constant within group)
    video_id = group['video_id'].iloc[0]
    participant_id = group['participant_id'].iloc[0]

    records = []

    # Slide the window with stride
    for start in range(0, n_frames, STRIDE_FRAMES):
        end = start + WINDOW_SIZE
        if end > n_frames:
            end = n_frames

        # Actual window length
        win_len = end - start
        if win_len < MIN_PERIODS:
            # Window too small — skip remaining positions
            break

        # Slice arrays
        w_eye   = eye_states[start:end]
        w_ear   = ear_values[start:end]
        w_mar   = mar_values[start:end]
        w_dx    = dx_values[start:end]
        w_dy    = dy_values[start:end]

        # ── 1. PERCLOS (%) ──
        perclos = (np.sum(w_eye == 1) / win_len) * 100.0

        # ── 2. Blink Rate (blinks/minute) ──
        n_blinks = _count_blinks(w_eye)
        window_duration_sec = win_len / TARGET_FPS
        blink_rate = (n_blinks / window_duration_sec) * 60.0 if window_duration_sec > 0 else 0.0

        # ── 3. Average Blink Duration (seconds) ──
        blink_avg_dur = _mean_blink_duration(w_eye, TARGET_FPS)

        # ── 4 & 5. EAR statistics ──
        # Filter NaN for robust stats
        ear_valid = w_ear[~np.isnan(w_ear)]
        ear_mean = float(np.mean(ear_valid)) if len(ear_valid) > 0 else np.nan
        ear_std  = float(np.std(ear_valid, ddof=1)) if len(ear_valid) > 1 else 0.0

        # ── 6 & 7. MAR statistics ──
        mar_valid = w_mar[~np.isnan(w_mar)]
        mar_mean = float(np.mean(mar_valid)) if len(mar_valid) > 0 else np.nan
        mar_max  = float(np.max(mar_valid)) if len(mar_valid) > 0 else np.nan

        # ── 8, 9 & 10. Pose Jitter ──
        dx_valid = w_dx[~np.isnan(w_dx)]
        dy_valid = w_dy[~np.isnan(w_dy)]
        jitter_dx = float(np.var(dx_valid, ddof=1)) if len(dx_valid) > 1 else 0.0
        jitter_dy = float(np.var(dy_valid, ddof=1)) if len(dy_valid) > 1 else 0.0
        jitter_total = jitter_dx + jitter_dy

        records.append({
            'video_id':           video_id,
            'participant_id':     participant_id,
            'window_start_frame': frame_files[start],
            'window_end_frame':   frame_files[end - 1],
            'window_start_idx':   start,
            'window_end_idx':     end - 1,
            'window_length':      win_len,
            'PERCLOS':            round(perclos, 4),
            'Blink_Rate':         round(blink_rate, 4),
            'Blink_Avg_Duration': round(blink_avg_dur, 4),
            'EAR_Mean':           round(ear_mean, 5) if not np.isnan(ear_mean) else np.nan,
            'EAR_Std':            round(ear_std, 5),
            'MAR_Mean':           round(mar_mean, 5) if not np.isnan(mar_mean) else np.nan,
            'MAR_Max':            round(mar_max, 5) if not np.isnan(mar_max) else np.nan,
            'Pose_Jitter_dx':     round(jitter_dx, 6),
            'Pose_Jitter_dy':     round(jitter_dy, 6),
            'Pose_Jitter':        round(jitter_total, 6),
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════
def compute_behavioral_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main aggregation function — computes behavioral vectors for all
    video×participant groups.

    Parameters
    ----------
    df : pd.DataFrame
        Frame-level feature data (from features_summary.csv) with at least:
        video_id, participant_id, frame_file, eye_state,
        mean_EAR (or mean_EAR_smooth), MAR (or MAR_smooth),
        head_dx (or head_dx_smooth), head_dy (or head_dy_smooth).

    Returns
    -------
    pd.DataFrame
        Aggregated behavioral vectors (one row per window).
    """
    # Resolve column names (prefer _smooth variants)
    ear_col = _resolve_col(df, 'mean_EAR')
    mar_col = _resolve_col(df, 'MAR')
    dx_col  = _resolve_col(df, 'head_dx')
    dy_col  = _resolve_col(df, 'head_dy')

    logging.info(f"Column mapping: EAR={ear_col}, MAR={mar_col}, dx={dx_col}, dy={dy_col}")
    logging.info(
        f"Window config: size={WINDOW_SIZE} frames ({WINDOW_SECONDS}s), "
        f"stride={STRIDE_FRAMES} frames ({STRIDE_SECONDS}s), "
        f"min_periods={MIN_PERIODS} frames"
    )

    grouped = df.groupby(['video_id', 'participant_id'])
    logging.info(f"Processing {len(grouped)} video×participant groups...")

    all_vectors = []
    for name, group in grouped:
        group = group.sort_values('frame_file').copy()
        vectors = aggregate_group(group, ear_col, mar_col, dx_col, dy_col)
        if not vectors.empty:
            all_vectors.append(vectors)
            logging.info(
                f"  Group {name}: {len(group)} frames → {len(vectors)} windows"
            )

    if not all_vectors:
        logging.error("No behavioral vectors generated. Check input data.")
        return pd.DataFrame()

    result = pd.concat(all_vectors, ignore_index=True)
    logging.info(f"Total behavioral vectors: {len(result)}")

    # ── Summary statistics ──
    logging.info("=" * 50)
    logging.info("Aggregation Summary:")
    logging.info(f"  PERCLOS     — mean: {result['PERCLOS'].mean():.2f}%, "
                 f"max: {result['PERCLOS'].max():.2f}%")
    logging.info(f"  Blink_Rate  — mean: {result['Blink_Rate'].mean():.2f} blinks/min")
    logging.info(f"  EAR_Std     — mean: {result['EAR_Std'].mean():.5f}")
    logging.info(f"  Pose_Jitter — mean: {result['Pose_Jitter'].mean():.6f}")
    logging.info("=" * 50)

    return result


# ══════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════
def main():
    """Pipeline entry point — read features, aggregate, save."""

    # ── Validate input ──
    if not SUMMARY_FEATURES_CSV.exists():
        logging.error(
            f"Input file {SUMMARY_FEATURES_CSV} not found. "
            f"Run the upstream pipeline first (eye_state.py → features_perclos.py)."
        )
        return

    logging.info(f"Reading frame-level features from {SUMMARY_FEATURES_CSV}...")
    df = pd.read_csv(SUMMARY_FEATURES_CSV)
    logging.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    # ── Validate required columns ──
    required_cols = ['video_id', 'participant_id', 'frame_file', 'eye_state']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logging.error(
            f"Missing required columns: {missing}. "
            f"Ensure eye_state.py has been executed."
        )
        return

    # ── Run aggregation ──
    result = compute_behavioral_vectors(df)

    if result.empty:
        logging.error("No output generated.")
        return

    # ── Save output ──
    result.to_csv(BEHAVIORAL_VECTORS_CSV, index=False)
    logging.info(f"Behavioral vectors saved to {BEHAVIORAL_VECTORS_CSV}")
    logging.info(f"Shape: {result.shape[0]} rows × {result.shape[1]} columns")
    logging.info("Statistical aggregation complete.")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    main()
