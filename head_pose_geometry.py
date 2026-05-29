"""
head_pose_geometry.py
---------------------
Defines the 3D canonical face model (World Coordinates) for 6 expression-invariant
keypoints and filters the corresponding 2D landmarks from a MediaPipe FaceLandmarker
result.

Keypoints used (invariant to facial expressions):
    ID   1  — Nose tip
    ID 152  — Chin
    ID  33  — Left eye outer corner
    ID 263  — Right eye outer corner
    ID  61  — Left mouth corner
    ID 291  — Right mouth corner

Usage (inside your mesh_apply.py processing loop):
    from head_pose_geometry import FACE_3D_MODEL, extract_pnp_points_2d
    img_pts = extract_pnp_points_2d(face_landmarks, frame_w, frame_h)
    if img_pts is not None:
        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL, img_pts,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. CANONICAL 3D FACE MODEL  (World Coordinates, unit = mm, origin = nose tip)
# ---------------------------------------------------------------------------
# Source: standard anthropometric face model widely used in gaze/pose literature.
# Z-axis points OUT of the face toward the camera.
# These values are expression-invariant (bony landmarks, not soft tissue).

FACE_3D_MODEL = np.array([
    [  0.0,    0.0,    0.0],   # ID   1 — Nose tip        (origin)
    [  0.0, -330.0,  -65.0],   # ID 152 — Chin
    [-225.0,  170.0, -135.0],  # ID  33 — Left  eye outer corner
    [ 225.0,  170.0, -135.0],  # ID 263 — Right eye outer corner
    [-150.0, -150.0, -125.0],  # ID  61 — Left  mouth corner
    [ 150.0, -150.0, -125.0],  # ID 291 — Right mouth corner
], dtype=np.float64)

# Landmark IDs that correspond row-by-row to FACE_3D_MODEL
PNP_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


# ---------------------------------------------------------------------------
# 2. APPROXIMATE CAMERA MATRIX  (used when no calibration file is available)
# ---------------------------------------------------------------------------

def build_camera_matrix(frame_w: int, frame_h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds an approximate camera intrinsic matrix K and zero distortion
    coefficients D, assuming:
        - focal length  fx = fy = frame_w  (common heuristic)
        - principal point cx = frame_w/2, cy = frame_h/2

    This is a reasonable approximation for standard webcams/dashcams.
    NOTE: For higher accuracy, replace with values from a checkerboard
          calibration (cv2.calibrateCamera).

    Args:
        frame_w: Frame width  in pixels.
        frame_h: Frame height in pixels.

    Returns:
        K : (3, 3) float64 camera matrix.
        D : (4, 1) float64 distortion coefficients (all zeros).
    """
    K = np.array([
        [frame_w,       0, frame_w / 2],
        [      0, frame_w, frame_h / 2],
        [      0,       0,           1],
    ], dtype=np.float64)

    D = np.zeros((4, 1), dtype=np.float64)

    return K, D


# ---------------------------------------------------------------------------
# 3. 2D LANDMARK EXTRACTION
# ---------------------------------------------------------------------------

def extract_pnp_points_2d(
    face_landmarks,
    frame_w: int,
    frame_h: int,
) -> np.ndarray | None:
    """
    Filters the 6 PnP keypoints from a MediaPipe FaceLandmarker result and
    converts normalised coordinates → pixel coordinates.

    Compatible with the NEW MediaPipe Tasks API used in this project
    (face_landmarker.task), where landmarks are accessed as:
        result.face_landmarks[face_idx][landmark_idx].x / .y

    Args:
        face_landmarks : List of NormalizedLandmark objects for ONE face
                         (i.e. result.face_landmarks[0]).
        frame_w        : Frame width  in pixels  (used to de-normalise x).
        frame_h        : Frame height in pixels  (used to de-normalise y).

    Returns:
        np.ndarray of shape (6, 1, 2) float64 — required format for
        cv2.solvePnP — or None if any landmark is missing / out of bounds.
    """
    if face_landmarks is None:
        logger.debug("extract_pnp_points_2d: face_landmarks is None.")
        return None

    total = len(face_landmarks)
    if total < max(PNP_LANDMARK_IDS) + 1:
        logger.warning(
            "extract_pnp_points_2d: only %d landmarks available, need >= %d.",
            total, max(PNP_LANDMARK_IDS) + 1,
        )
        return None

    img_pts = []
    for lm_id in PNP_LANDMARK_IDS:
        lm = face_landmarks[lm_id]

        # De-normalise: MediaPipe returns values in [0, 1]
        px = lm.x * frame_w
        py = lm.y * frame_h

        # Sanity check — landmark should be inside the frame
        if not (0 <= px <= frame_w and 0 <= py <= frame_h):
            logger.warning(
                "Landmark ID %d out of frame bounds (%.1f, %.1f). "
                "Frame: %dx%d. Skipping PnP for this frame.",
                lm_id, px, py, frame_w, frame_h,
            )
            return None

        img_pts.append([px, py])

    # solvePnP expects shape (N, 1, 2)
    return np.array(img_pts, dtype=np.float64).reshape(-1, 1, 2)


# ---------------------------------------------------------------------------
# 4. RODRIGUES → EULER ANGLES
# ---------------------------------------------------------------------------

def rvec_to_euler(rvec: np.ndarray) -> tuple[float, float, float]:
    """
    Converts a Rodrigues rotation vector (output of cv2.solvePnP) to
    Euler angles in degrees.

        pitch (θ) — nodding   (rotation around X-axis)
        yaw   (ψ) — turning   (rotation around Y-axis)
        roll  (φ) — tilting   (rotation around Z-axis)

    Convention: right-hand rule, camera coordinate system.

    Args:
        rvec: (3, 1) or (3,) rotation vector from cv2.solvePnP.

    Returns:
        (pitch_deg, yaw_deg, roll_deg) as Python floats.
    """
    R, _ = cv2.Rodrigues(rvec)

    # Tait-Bryan angles (ZYX convention)
    pitch_rad = np.arctan2( R[2, 1], R[2, 2])
    yaw_rad   = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll_rad  = np.arctan2( R[1, 0], R[0, 0])

    return (
        float(np.degrees(pitch_rad)),
        float(np.degrees(yaw_rad)),
        float(np.degrees(roll_rad)),
    )


# ---------------------------------------------------------------------------
# 5. CONVENIENCE WRAPPER  (full PnP solve in one call)
# ---------------------------------------------------------------------------

def solve_head_pose(
    face_landmarks,
    frame_w: int,
    frame_h: int,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs:   np.ndarray | None = None,
) -> dict | None:
    """
    Full head-pose estimation pipeline in a single call.

    Args:
        face_landmarks : result.face_landmarks[0] from MediaPipe Tasks.
        frame_w        : Frame width  in pixels.
        frame_h        : Frame height in pixels.
        camera_matrix  : Optional pre-calibrated K. If None, uses approximation.
        dist_coeffs    : Optional distortion D.  If None, uses zeros.

    Returns:
        dict with keys:
            'pitch'  (float, degrees)
            'yaw'    (float, degrees)
            'roll'   (float, degrees)
            'rvec'   (np.ndarray, (3,1))
            'tvec'   (np.ndarray, (3,1))
        or None if PnP failed.

    Example:
        result = landmarker.detect(mp_image)
        if result.face_landmarks:
            pose = solve_head_pose(result.face_landmarks[0], w, h)
            if pose:
                print(f"Pitch={pose['pitch']:.1f}° Yaw={pose['yaw']:.1f}°")
    """
    # Build approximate camera matrix if not provided
    if camera_matrix is None or dist_coeffs is None:
        camera_matrix, dist_coeffs = build_camera_matrix(frame_w, frame_h)

    # Extract 2D correspondences
    img_pts = extract_pnp_points_2d(face_landmarks, frame_w, frame_h)
    if img_pts is None:
        return None

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL,
        img_pts,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        logger.warning("cv2.solvePnP failed for this frame.")
        return None

    pitch, yaw, roll = rvec_to_euler(rvec)

    return {
        "pitch": pitch,
        "yaw":   yaw,
        "roll":  roll,
        "rvec":  rvec,
        "tvec":  tvec,
    }


# ---------------------------------------------------------------------------
# 6. QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=== FACE_3D_MODEL ===")
    labels = ["Nose tip", "Chin", "L eye corner", "R eye corner",
              "L mouth corner", "R mouth corner"]
    for label, pt in zip(labels, FACE_3D_MODEL):
        print(f"  {label:18s}  X={pt[0]:+7.1f}  Y={pt[1]:+7.1f}  Z={pt[2]:+7.1f}")

    print("\n=== Camera matrix (1920x1080 frame) ===")
    K, D = build_camera_matrix(1920, 1080)
    print(K)

    print("\n=== extract_pnp_points_2d (mock landmarks) ===")

    # Simulate 478 normalised landmarks (all at centre of a 640x480 frame)
    class MockLandmark:
        def __init__(self, x, y): self.x = x; self.y = y

    W, H = 640, 480
    mock_lms = [MockLandmark(0.5, 0.5)] * 478

    # Place the 6 PnP landmarks at realistic pixel positions
    expected_px = [
        (320, 260),  # nose tip
        (318, 420),  # chin
        (190, 210),  # left eye corner
        (450, 210),  # right eye corner
        (240, 370),  # left mouth
        (400, 370),  # right mouth
    ]
    for idx, (px, py) in zip(PNP_LANDMARK_IDS, expected_px):
        mock_lms[idx] = MockLandmark(px / W, py / H)

    img_pts = extract_pnp_points_2d(mock_lms, W, H)
    print(f"  img_pts shape : {img_pts.shape}")   # expect (6, 1, 2)
    print(f"  img_pts dtype : {img_pts.dtype}")
    for label, pt in zip(labels, img_pts[:, 0, :]):
        print(f"  {label:18s}  pixel=({pt[0]:.1f}, {pt[1]:.1f})")

    print("\n=== solve_head_pose (mock — frontal face) ===")
    pose = solve_head_pose(mock_lms, W, H)
    if pose:
        print(f"  Pitch = {pose['pitch']:+.2f}°")
        print(f"  Yaw   = {pose['yaw']:+.2f}°")
        print(f"  Roll  = {pose['roll']:+.2f}°")
    else:
        print("  PnP failed (expected with synthetic data).")
