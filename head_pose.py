"""
head_pose.py — Stage 1: Camera Matrix Configuration & cv2.solvePnP() Head Pose

Goal:
    Estimate 3D head orientation (Yaw, Pitch, Roll) from 2D facial landmarks
    using the Perspective-n-Point (PnP) algorithm.

Scientific Background:
    solvePnP finds the rotation (R) and translation (T) that map a known 3D
    object model to its observed 2D image projection. Given:
        - A set of 3D reference points (generic face model, in mm)
        - Their corresponding 2D image projections (from MediaPipe)
        - Camera intrinsic matrix K and distortion coefficients D
    OpenCV solves: s * [u, v, 1]^T = K * [R | T] * [X, Y, Z, 1]^T

Architecture Position:
    **Stage 1 (head_pose.py)** — runs alongside Mesh_apply.py
        → Stage 2 (to_csv.py — signal smoothing)
            → Stage 4 (stats_aggregation.py — Pose_Jitter from Euler angles)

Outputs:
    - rvec     : Rotation vector (Rodrigues, 3×1) — compact axis-angle form
    - tvec     : Translation vector (3×1) — camera-to-face displacement
    - yaw      : Left/Right head rotation (degrees)
    - pitch    : Up/Down head tilt (degrees)
    - roll     : Head lateral tilt (degrees)

Camera Matrix Setup:
    Since webcam intrinsic parameters are unknown, we use a standard pinhole
    model approximation:
        focal_length ≈ image_width   (common heuristic for standard webcams)
        cx = image_width / 2         (principal point at image center)
        cy = image_height / 2
    Distortion coefficients are set to zero (no lens distortion assumed).

3D Face Model:
    A canonical 6-point anthropometric model based on average adult face
    proportions. All coordinates are in millimeters with the nose tip as
    the origin. The model is gender-neutral and scale-invariant for pose
    estimation purposes.

Usage:
    # ── As a library (from Mesh_apply.py or other modules) ──
    from src.head_pose import HeadPoseEstimator

    estimator = HeadPoseEstimator(img_w=640, img_h=480)
    result = estimator.estimate(mediapipe_landmarks)
    print(result['yaw'], result['pitch'], result['roll'])

    # ── Standalone test with webcam ──
    python -m src.head_pose           (from PROJECT_ROOT)
    python src/head_pose.py           (direct execution)

    # ── Batch processing from landmarks CSV ──
    python src/head_pose.py --batch   (process landmarks_full.csv)

Dependencies:
    - opencv-python (cv2)
    - numpy
    - mediapipe (for standalone camera test only)
"""

import sys
import math
import logging
import argparse
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ── Project path setup ──
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core_config import RESIZE_DIM, CSV_DIR, RAW_LANDMARKS_CSV


# ══════════════════════════════════════════════════════════════
# 3D Face Model — Canonical Anthropometric Landmarks
# ══════════════════════════════════════════════════════════════
#
# 6-point model in world coordinates (millimeters).
# Nose tip is the origin (0, 0, 0).
# Y-axis points downward, Z-axis points backward (into the face).
#
# These values are derived from average adult anthropometric data
# (Farkas, 1994 — "Anthropometry of the Head and Face").
#
#   Landmark            X (mm)    Y (mm)    Z (mm)
#   ──────────────────────────────────────────────
#   Nose tip              0.0       0.0       0.0
#   Chin                  0.0    -330.0     -65.0
#   Left eye outer       -225.0   170.0    -135.0
#   Right eye outer       225.0   170.0    -135.0
#   Left mouth corner    -150.0  -150.0    -125.0
#   Right mouth corner    150.0  -150.0    -125.0
#
MODEL_POINTS_3D = np.array([
    (   0.0,    0.0,    0.0),    # Nose tip
    (   0.0, -330.0,  -65.0),    # Chin
    (-225.0,  170.0, -135.0),    # Left eye outer corner
    ( 225.0,  170.0, -135.0),    # Right eye outer corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    ( 150.0, -150.0, -125.0),    # Right mouth corner
], dtype=np.float64)

# ══════════════════════════════════════════════════════════════
# MediaPipe Face Mesh — Corresponding 2D Landmark Indices
# ══════════════════════════════════════════════════════════════
#
# These indices map to the same anatomical points as MODEL_POINTS_3D.
# MediaPipe uses a 468-point mesh (or 478 with iris).
#
#   Index   Anatomical Point
#   ─────   ─────────────────────
#     1     Nose tip
#   152     Chin (bottom of mandible)
#   263     Left eye outer corner
#    33     Right eye outer corner
#    61     Left mouth corner
#   291     Right mouth corner
#
LANDMARK_IDXS = [1, 152, 263, 33, 61, 291]


# ══════════════════════════════════════════════════════════════
# Camera Matrix Builder
# ══════════════════════════════════════════════════════════════
def build_camera_matrix(img_w: int, img_h: int) -> np.ndarray:
    """
    Construct the camera intrinsic matrix K for a pinhole camera model.

    The intrinsic matrix K encodes the internal parameters of the camera:

        K = | f_x   0    c_x |
            |  0   f_y   c_y |
            |  0    0     1  |

    Where:
        f_x, f_y = focal lengths in pixels (assumed equal for square pixels)
        c_x, c_y = principal point (assumed at image center)

    Approximation Rationale:
        For standard webcams with ~60° horizontal FOV:
            focal_length ≈ image_width
        This is derived from:
            f = (w/2) / tan(FOV/2) ≈ w for FOV ≈ 53°

    Parameters
    ----------
    img_w : int
        Image width in pixels.
    img_h : int
        Image height in pixels.

    Returns
    -------
    np.ndarray
        3×3 camera intrinsic matrix K (float64).
    """
    focal_length = float(img_w)  # Approximation for standard webcam
    cx = img_w / 2.0             # Principal point X (image center)
    cy = img_h / 2.0             # Principal point Y (image center)

    K = np.array([
        [focal_length,  0.0,          cx],
        [0.0,           focal_length, cy],
        [0.0,           0.0,          1.0],
    ], dtype=np.float64)

    return K


def build_distortion_coefficients() -> np.ndarray:
    """
    Construct the distortion coefficient vector D.

    For uncalibrated cameras, we assume zero distortion:
        D = [k1, k2, p1, p2, k3] = [0, 0, 0, 0, 0]

    Where:
        k1, k2, k3 = radial distortion coefficients
        p1, p2     = tangential distortion coefficients

    This is a reasonable assumption for:
        - Modern webcams with low barrel/pincushion distortion
        - Dashboard cameras at moderate FOV
        - Cases where intrinsic calibration is impractical

    Returns
    -------
    np.ndarray
        5×1 distortion coefficient vector D (all zeros).
    """
    D = np.zeros((5, 1), dtype=np.float64)
    return D


# ══════════════════════════════════════════════════════════════
# Rotation Vector → Euler Angles Conversion
# ══════════════════════════════════════════════════════════════
def rvec_to_euler_angles(rvec: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a Rodrigues rotation vector to Euler angles (Yaw, Pitch, Roll).

    Pipeline:
        1. rvec (3×1) → Rotation matrix R (3×3) via cv2.Rodrigues()
        2. R → Euler angles via decomposition (ZYX convention)

    Euler Angle Convention (ZYX — Tait-Bryan):
        - Yaw   (ψ) : Rotation around Y-axis (left ↔ right)
        - Pitch (θ) : Rotation around X-axis (up ↔ down)
        - Roll  (φ) : Rotation around Z-axis (tilt left ↔ right)

    Gimbal Lock Handling:
        When pitch ≈ ±90°, yaw and roll become degenerate.
        This function clamps the input to avoid NaN from arcsin.

    Parameters
    ----------
    rvec : np.ndarray
        3×1 Rodrigues rotation vector from cv2.solvePnP().

    Returns
    -------
    tuple of float
        (yaw, pitch, roll) in degrees.
        Yaw   : Negative = looking left, Positive = looking right
        Pitch : Negative = looking up,   Positive = looking down
        Roll  : Negative = tilt left,    Positive = tilt right
    """
    # Step 1: Rodrigues rotation vector → 3×3 rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Step 2: Decompose rotation matrix to Euler angles (ZYX convention)
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    #
    # From the rotation matrix elements:
    #   pitch = -arcsin(R[2,0])
    #   yaw   = atan2(R[2,1] / cos(pitch), R[2,2] / cos(pitch))
    #   roll  = atan2(R[1,0] / cos(pitch), R[0,0] / cos(pitch))

    # Clamp to avoid NaN from arcsin (gimbal lock region)
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    singular = sy < 1e-6  # Check for gimbal lock

    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw   = math.atan2(-rotation_matrix[2, 0], sy)
        roll  = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        # Gimbal lock — set roll = 0 by convention
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw   = math.atan2(-rotation_matrix[2, 0], sy)
        roll  = 0.0

    # Convert radians → degrees
    yaw_deg   = math.degrees(yaw)
    pitch_deg = math.degrees(pitch)
    roll_deg  = math.degrees(roll)

    return yaw_deg, pitch_deg, roll_deg


# ══════════════════════════════════════════════════════════════
# HeadPoseEstimator Class
# ══════════════════════════════════════════════════════════════
class HeadPoseEstimator:
    """
    Encapsulates camera parameters and provides head pose estimation
    via cv2.solvePnP().

    This class pre-computes and caches the camera matrix K and distortion
    coefficients D for a given image resolution, then provides methods
    to estimate head pose from MediaPipe landmarks.

    Attributes
    ----------
    img_w : int
        Image width in pixels.
    img_h : int
        Image height in pixels.
    K : np.ndarray
        3×3 camera intrinsic matrix.
    D : np.ndarray
        5×1 distortion coefficient vector.
    model_3d : np.ndarray
        6×3 canonical 3D face model points.
    landmark_idxs : list[int]
        MediaPipe indices corresponding to model_3d points.

    Example
    -------
    >>> estimator = HeadPoseEstimator(640, 480)
    >>> result = estimator.estimate(face_landmarks)
    >>> print(f"Yaw={result['yaw']:.1f}°, Pitch={result['pitch']:.1f}°")
    """

    def __init__(self, img_w: int = None, img_h: int = None):
        """
        Initialize the estimator with image dimensions.

        If no dimensions are provided, defaults to RESIZE_DIM from core_config
        (640×480).

        Parameters
        ----------
        img_w : int, optional
            Image width. Default: RESIZE_DIM[0] (640).
        img_h : int, optional
            Image height. Default: RESIZE_DIM[1] (480).
        """
        self.img_w = img_w if img_w is not None else RESIZE_DIM[0]
        self.img_h = img_h if img_h is not None else RESIZE_DIM[1]

        # Build and cache camera parameters
        self.K = build_camera_matrix(self.img_w, self.img_h)
        self.D = build_distortion_coefficients()

        # 3D model and landmark mapping
        self.model_3d = MODEL_POINTS_3D
        self.landmark_idxs = LANDMARK_IDXS

        logging.debug(
            f"HeadPoseEstimator initialized: "
            f"img={self.img_w}×{self.img_h}, "
            f"focal={self.K[0,0]:.1f}, "
            f"principal=({self.K[0,2]:.1f}, {self.K[1,2]:.1f})"
        )

    def _extract_2d_points(
        self, landmarks, img_w: int = None, img_h: int = None
    ) -> Optional[np.ndarray]:
        """
        Extract the 6 corresponding 2D image points from MediaPipe landmarks.

        Parameters
        ----------
        landmarks : list
            MediaPipe NormalizedLandmark list (468+ points).
            Each landmark has .x, .y attributes in [0, 1] range.
        img_w : int, optional
            Image width for denormalization. Default: self.img_w.
        img_h : int, optional
            Image height for denormalization. Default: self.img_h.

        Returns
        -------
        np.ndarray or None
            6×2 array of 2D pixel coordinates, or None if landmarks invalid.
        """
        w = img_w if img_w is not None else self.img_w
        h = img_h if img_h is not None else self.img_h

        if landmarks is None or len(landmarks) < max(self.landmark_idxs) + 1:
            return None

        try:
            points_2d = np.array([
                (landmarks[idx].x * w, landmarks[idx].y * h)
                for idx in self.landmark_idxs
            ], dtype=np.float64)
            return points_2d
        except (IndexError, AttributeError):
            return None

    def _extract_2d_points_from_arrays(
        self,
        lm_x_values: Dict[int, float],
        lm_y_values: Dict[int, float],
        img_w: int = None,
        img_h: int = None,
    ) -> Optional[np.ndarray]:
        """
        Extract 2D points from pre-extracted landmark coordinate dicts.

        Useful for batch processing from CSV data where landmarks are stored
        as individual columns (lm_{i}_x, lm_{i}_y).

        Parameters
        ----------
        lm_x_values : dict
            Mapping of landmark_index → normalized x coordinate.
        lm_y_values : dict
            Mapping of landmark_index → normalized y coordinate.
        img_w, img_h : int, optional
            Image dimensions for denormalization.

        Returns
        -------
        np.ndarray or None
            6×2 array of 2D pixel coordinates.
        """
        w = img_w if img_w is not None else self.img_w
        h = img_h if img_h is not None else self.img_h

        try:
            points_2d = np.array([
                (lm_x_values[idx] * w, lm_y_values[idx] * h)
                for idx in self.landmark_idxs
            ], dtype=np.float64)

            # Validate — no NaN allowed
            if np.any(np.isnan(points_2d)):
                return None

            return points_2d
        except (KeyError, TypeError):
            return None

    def solve_pnp(
        self, points_2d: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Run cv2.solvePnP() to find rotation and translation vectors.

        Uses the ITERATIVE solver (cv2.SOLVEPNP_ITERATIVE) which is
        the most robust for the 6-point configuration.

        The PnP problem finds [R|T] such that:
            s * [u, v, 1]^T = K * [R | T] * [X, Y, Z, 1]^T

        Parameters
        ----------
        points_2d : np.ndarray
            6×2 array of 2D image coordinates (pixels).

        Returns
        -------
        tuple(np.ndarray, np.ndarray) or None
            (rvec, tvec) — 3×1 rotation vector and 3×1 translation vector.
            Returns None if solvePnP fails.
        """
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=self.model_3d,
            imagePoints=points_2d,
            cameraMatrix=self.K,
            distCoeffs=self.D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        return rvec, tvec

    def estimate(
        self,
        landmarks,
        img_w: int = None,
        img_h: int = None,
    ) -> Optional[Dict[str, float]]:
        """
        Full head pose estimation from MediaPipe landmarks.

        This is the primary public API. It:
            1. Extracts 2D image points from landmarks
            2. Calls cv2.solvePnP() to get rvec and tvec
            3. Converts rvec to Euler angles (yaw, pitch, roll)

        Parameters
        ----------
        landmarks : list
            MediaPipe NormalizedLandmark list (468+ points).
        img_w, img_h : int, optional
            Override image dimensions if different from init.

        Returns
        -------
        dict or None
            {
                'rvec': np.ndarray (3×1),   # Rotation vector (Rodrigues)
                'tvec': np.ndarray (3×1),   # Translation vector
                'yaw': float,               # degrees (left ← → right)
                'pitch': float,             # degrees (up ← → down)
                'roll': float,              # degrees (tilt)
                'nose_2d': tuple,           # Nose tip 2D position (for viz)
            }
            Returns None if estimation fails.
        """
        w = img_w if img_w is not None else self.img_w
        h = img_h if img_h is not None else self.img_h

        # Step 1: Extract 2D points
        points_2d = self._extract_2d_points(landmarks, w, h)
        if points_2d is None:
            return None

        # Step 2: Solve PnP → rvec, tvec
        pnp_result = self.solve_pnp(points_2d)
        if pnp_result is None:
            return None

        rvec, tvec = pnp_result

        # Step 3: Convert rotation vector → Euler angles
        yaw, pitch, roll = rvec_to_euler_angles(rvec)

        # Nose tip 2D position (for visualization overlays)
        nose_2d = (
            float(landmarks[self.landmark_idxs[0]].x * w),
            float(landmarks[self.landmark_idxs[0]].y * h),
        )

        return {
            'rvec': rvec,
            'tvec': tvec,
            'yaw': round(yaw, 4),
            'pitch': round(pitch, 4),
            'roll': round(roll, 4),
            'nose_2d': nose_2d,
        }

    def estimate_from_csv_row(
        self,
        row: dict,
        img_w: int = None,
        img_h: int = None,
    ) -> Optional[Dict[str, float]]:
        """
        Estimate head pose from a single CSV row containing landmark columns.

        The CSV row must have columns: lm_{idx}_x, lm_{idx}_y for each
        index in LANDMARK_IDXS.

        Parameters
        ----------
        row : dict
            A dictionary (or pandas Series) with lm_{i}_x, lm_{i}_y columns.
        img_w, img_h : int, optional
            Override image dimensions.

        Returns
        -------
        dict or None
            Same as estimate() output, or None if data is missing/invalid.
        """
        w = img_w if img_w is not None else self.img_w
        h = img_h if img_h is not None else self.img_h

        try:
            lm_x = {idx: float(row[f'lm_{idx}_x']) for idx in self.landmark_idxs}
            lm_y = {idx: float(row[f'lm_{idx}_y']) for idx in self.landmark_idxs}
        except (KeyError, TypeError, ValueError):
            return None

        points_2d = self._extract_2d_points_from_arrays(lm_x, lm_y, w, h)
        if points_2d is None:
            return None

        pnp_result = self.solve_pnp(points_2d)
        if pnp_result is None:
            return None

        rvec, tvec = pnp_result
        yaw, pitch, roll = rvec_to_euler_angles(rvec)

        return {
            'rvec': rvec,
            'tvec': tvec,
            'yaw': round(yaw, 4),
            'pitch': round(pitch, 4),
            'roll': round(roll, 4),
        }

    def draw_pose_axes(
        self,
        frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        nose_2d: Tuple[float, float],
        axis_length: float = 100.0,
    ) -> np.ndarray:
        """
        Draw 3D coordinate axes on the frame for visualization.

        Draws three lines from the nose tip:
            - Red   (X-axis) : pointing right
            - Green (Y-axis) : pointing down
            - Blue  (Z-axis) : pointing forward (out of face)

        Parameters
        ----------
        frame : np.ndarray
            BGR image to draw on (will be modified in-place).
        rvec : np.ndarray
            Rotation vector from solvePnP.
        tvec : np.ndarray
            Translation vector from solvePnP.
        nose_2d : tuple
            (x, y) position of the nose tip in pixel coordinates.
        axis_length : float
            Length of the drawn axes in 3D units (mm).

        Returns
        -------
        np.ndarray
            Frame with axes drawn.
        """
        # Define 3D axis endpoints relative to nose tip
        axes_3d = np.array([
            [axis_length, 0, 0],     # X-axis (right)
            [0, axis_length, 0],     # Y-axis (down)
            [0, 0, axis_length],     # Z-axis (forward)
        ], dtype=np.float64)

        # Project 3D axis endpoints to 2D image plane
        axes_2d, _ = cv2.projectPoints(
            axes_3d,
            rvec,
            tvec,
            self.K,
            self.D,
        )

        origin = (int(nose_2d[0]), int(nose_2d[1]))

        # Draw axes: X=Red, Y=Green, Z=Blue
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        labels = ['X', 'Y', 'Z']
        for i, (color, label) in enumerate(zip(colors, labels)):
            end_point = (int(axes_2d[i][0][0]), int(axes_2d[i][0][1]))
            cv2.line(frame, origin, end_point, color, 2)
            cv2.putText(
                frame, label, end_point,
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        return frame

    def format_info(self) -> str:
        """Return a formatted string of the camera configuration."""
        return (
            f"Camera Matrix K:\n{self.K}\n"
            f"Distortion D: {self.D.flatten()}\n"
            f"Image: {self.img_w}×{self.img_h}\n"
            f"Focal Length: {self.K[0,0]:.1f} px\n"
            f"Principal Point: ({self.K[0,2]:.1f}, {self.K[1,2]:.1f})"
        )


# ══════════════════════════════════════════════════════════════
# Batch Processing (from landmarks_full.csv)
# ══════════════════════════════════════════════════════════════
def batch_process_landmarks(csv_path: Path = None, output_path: Path = None):
    """
    Process all frames in landmarks_full.csv and add Euler angle columns.

    Reads the raw landmark CSV, computes head pose for each frame where
    face was detected, and saves results to a new CSV file.

    Parameters
    ----------
    csv_path : Path, optional
        Input CSV with lm_{i}_x, lm_{i}_y columns. Default: RAW_LANDMARKS_CSV.
    output_path : Path, optional
        Output CSV path. Default: CSV_DIR / 'head_pose_euler.csv'.
    """
    import pandas as pd

    if csv_path is None:
        csv_path = RAW_LANDMARKS_CSV
    if output_path is None:
        output_path = CSV_DIR / 'head_pose_euler.csv'

    if not csv_path.exists():
        logging.error(f"Input file {csv_path} not found.")
        logging.error("Run the Mesh_apply.py pipeline first to generate landmarks.")
        return

    logging.info(f"Reading landmarks from {csv_path}...")

    # Only load needed columns (base + the 6 landmark pairs we need)
    base_cols = ['video_id', 'participant_id', 'frame_file', 'face_detected']
    lm_cols = []
    for idx in LANDMARK_IDXS:
        lm_cols.extend([f'lm_{idx}_x', f'lm_{idx}_y'])

    # Check which columns exist
    available = pd.read_csv(csv_path, nrows=0).columns.tolist()
    load_cols = [c for c in base_cols + lm_cols if c in available]
    missing_lm = [c for c in lm_cols if c not in available]

    if missing_lm:
        logging.error(f"Missing landmark columns in CSV: {missing_lm}")
        logging.error("The CSV must contain lm_{{idx}}_x and lm_{{idx}}_y columns.")
        return

    df = pd.read_csv(csv_path, usecols=load_cols)
    logging.info(f"Loaded {len(df)} rows.")

    # Initialize estimator
    estimator = HeadPoseEstimator()  # Uses RESIZE_DIM from config

    # Process each row
    yaw_list, pitch_list, roll_list = [], [], []
    success_count = 0

    for _, row in df.iterrows():
        if row.get('face_detected', False) is False:
            yaw_list.append(np.nan)
            pitch_list.append(np.nan)
            roll_list.append(np.nan)
            continue

        result = estimator.estimate_from_csv_row(row)
        if result is not None:
            yaw_list.append(result['yaw'])
            pitch_list.append(result['pitch'])
            roll_list.append(result['roll'])
            success_count += 1
        else:
            yaw_list.append(np.nan)
            pitch_list.append(np.nan)
            roll_list.append(np.nan)

    df['yaw'] = yaw_list
    df['pitch'] = pitch_list
    df['roll'] = roll_list

    # Save
    out_cols = base_cols + ['yaw', 'pitch', 'roll']
    df[out_cols].to_csv(output_path, index=False)

    logging.info(f"Head pose computed for {success_count}/{len(df)} frames.")
    logging.info(f"Results saved to {output_path}")

    # Summary stats
    valid = df.dropna(subset=['yaw'])
    if not valid.empty:
        logging.info("=" * 50)
        logging.info("Euler Angle Statistics:")
        logging.info(f"  Yaw   — mean: {valid['yaw'].mean():.2f}°, "
                     f"std: {valid['yaw'].std():.2f}°, "
                     f"range: [{valid['yaw'].min():.1f}°, {valid['yaw'].max():.1f}°]")
        logging.info(f"  Pitch — mean: {valid['pitch'].mean():.2f}°, "
                     f"std: {valid['pitch'].std():.2f}°, "
                     f"range: [{valid['pitch'].min():.1f}°, {valid['pitch'].max():.1f}°]")
        logging.info(f"  Roll  — mean: {valid['roll'].mean():.2f}°, "
                     f"std: {valid['roll'].std():.2f}°, "
                     f"range: [{valid['roll'].min():.1f}°, {valid['roll'].max():.1f}°]")
        logging.info("=" * 50)


# ══════════════════════════════════════════════════════════════
# Standalone Webcam Test
# ══════════════════════════════════════════════════════════════
def run_camera_test():
    """
    Open webcam and display real-time head pose estimation.
    Press 'q' to quit.

    Shows:
        - Yaw, Pitch, Roll angles as text overlay
        - 3D coordinate axes projected onto the face
        - Nose tip landmark highlighted
    """
    import os
    import tempfile
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    # ── Download model if needed ──
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    )
    model_path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
    if not os.path.exists(model_path):
        import urllib.request
        print("Downloading face_landmarker.task model...")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print("Download complete.")
        except Exception as e:
            print(f"ERROR: Cannot download model: {e}")
            return

    # ── Setup MediaPipe ──
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # ── Open camera ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        landmarker.close()
        return

    # Get actual camera resolution
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    estimator = HeadPoseEstimator(img_w=cam_w, img_h=cam_h)

    print("=" * 55)
    print("  Head Pose Live Test — Press 'q' to quit")
    print(f"  Camera: {cam_w}×{cam_h}")
    print(f"  Focal length: {estimator.K[0,0]:.1f} px")
    print(f"  Principal point: ({estimator.K[0,2]:.1f}, {estimator.K[1,2]:.1f})")
    print("=" * 55)

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = landmarker.detect(mp_image)

            display = frame_bgr.copy()

            if results.face_landmarks:
                face_lms = results.face_landmarks[0]
                result = estimator.estimate(face_lms, w, h)

                if result is not None:
                    yaw, pitch, roll = result['yaw'], result['pitch'], result['roll']

                    # Draw 3D axes on face
                    estimator.draw_pose_axes(
                        display, result['rvec'], result['tvec'],
                        result['nose_2d'], axis_length=80.0,
                    )

                    # Text overlay
                    color = (0, 255, 0)
                    cv2.putText(display, f"Yaw:   {yaw:+7.2f} deg",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(display, f"Pitch: {pitch:+7.2f} deg",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(display, f"Roll:  {roll:+7.2f} deg",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Highlight nose tip
                    nx, ny = int(result['nose_2d'][0]), int(result['nose_2d'][1])
                    cv2.circle(display, (nx, ny), 5, (0, 0, 255), -1)

                    # Console output
                    print(
                        f"Yaw={yaw:+7.2f}°  Pitch={pitch:+7.2f}°  "
                        f"Roll={roll:+7.2f}°"
                    )
                else:
                    cv2.putText(display, "PnP failed", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Head Pose Test (solvePnP)", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user.")
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Camera test ended.")


# ══════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Head Pose Estimation via cv2.solvePnP()",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/head_pose.py              # Live webcam test
  python src/head_pose.py --batch      # Process landmarks_full.csv
  python src/head_pose.py --info       # Show camera matrix config
        """,
    )
    parser.add_argument(
        '--batch', action='store_true',
        help='Batch process landmarks_full.csv and output Euler angles CSV.'
    )
    parser.add_argument(
        '--info', action='store_true',
        help='Display camera matrix K, distortion D, and 3D model info.'
    )

    args = parser.parse_args()

    if args.info:
        estimator = HeadPoseEstimator()
        print("\n" + "=" * 55)
        print("  Head Pose Estimator — Configuration")
        print("=" * 55)
        print(f"\n{estimator.format_info()}")
        print(f"\n3D Face Model (6 points, mm):\n{estimator.model_3d}")
        print(f"\nMediaPipe Landmark Indices: {estimator.landmark_idxs}")
        print(f"  [0] idx=1   → Nose tip")
        print(f"  [1] idx=152 → Chin")
        print(f"  [2] idx=263 → Left eye outer")
        print(f"  [3] idx=33  → Right eye outer")
        print(f"  [4] idx=61  → Left mouth corner")
        print(f"  [5] idx=291 → Right mouth corner")
        print("=" * 55)
        return

    if args.batch:
        batch_process_landmarks()
    else:
        run_camera_test()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    main()
