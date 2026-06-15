import sys
import math
import time
import gc
import shutil
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core_config import *

# Landmark constants for features
LEFT_EYE_IDXS  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33,  160, 158, 133, 153, 144]
MOUTH_OUTER = [13, 14, 78, 308]
MOUTH_INNER = [312, 317]
NOSE_TIP_IDX = 1

# Iris indices
LEFT_IRIS_IDXS  = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDXS = [473, 474, 475, 476, 477]

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def eye_aspect_ratio(landmarks, eye_idxs, img_w, img_h):
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_idxs]
    v1 = euclidean(pts[1], pts[5])
    v2 = euclidean(pts[2], pts[4])
    h  = euclidean(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def mouth_aspect_ratio(landmarks, img_w, img_h):
    top    = (landmarks[13].x * img_w, landmarks[13].y * img_h)
    bottom = (landmarks[14].x * img_w, landmarks[14].y * img_h)
    left   = (landmarks[78].x * img_w, landmarks[78].y * img_h)
    right  = (landmarks[308].x * img_w, landmarks[308].y * img_h)
    vert  = euclidean(top, bottom)
    horiz = euclidean(left, right)
    return vert / (horiz + 1e-6)

def get_head_pose(landmarks, img_w, img_h):
    """
    Estimates Pitch, Yaw, and Roll using solvePnP.
    """
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
    if not success: return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rot_vec)
    pitch = math.asin(-rmat[1, 2])
    yaw = math.atan2(rmat[0, 2], rmat[2, 2])
    roll = math.atan2(rmat[1, 0], rmat[1, 1])

    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

def imread_unicode(path):
    try:
        img_array = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f'Error reading {path}: {e}')
        return None

def imwrite_unicode(path, img, quality=90):
    try:
        is_success, img_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if is_success:
            img_buf.tofile(str(path))
            return True
        return False
    except Exception as e:
        print(f'Error writing {path}: {e}')
        return False

def draw_custom_mesh(img, landmarks, w, h):
    annotated = img.copy()
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(annotated, (cx, cy), 1, (0, 180, 0), -1)
    if len(landmarks) >= 478:
        for idx in LEFT_IRIS_IDXS + RIGHT_IRIS_IDXS:
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 1, (255, 255, 255), -1)
    for idx in LEFT_EYE_IDXS + RIGHT_EYE_IDXS:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(annotated, (cx, cy), 2, (255, 100, 0), -1)
    for idx in MOUTH_OUTER + MOUTH_INNER:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(annotated, (cx, cy), 2, (0, 128, 255), -1)
    nose = landmarks[NOSE_TIP_IDX]
    cx, cy = int(nose.x * w), int(nose.y * h)
    cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
    return annotated

def create_landmarker(confidence=0.4):
    src_model_path = MODEL_PATH
    ascii_model_path = os.path.join(tempfile.gettempdir(), 'face_landmarker.task')
    if not os.path.exists(ascii_model_path):
        shutil.copy(str(src_model_path), ascii_model_path)
    
    base_options = python.BaseOptions(model_asset_path=ascii_model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        min_face_detection_confidence=confidence,
        min_face_presence_confidence=confidence
    )
    return vision.FaceLandmarker.create_from_options(options)

def main():
    t0 = time.time()
    clahe_frame_paths = sorted(list(FRAMES_CLAHE.rglob('*.jpg')))
    total = len(clahe_frame_paths)
    print(f'Found {total} CLAHE frames in {FRAMES_CLAHE}')

    records = []
    
    with create_landmarker(confidence=0.4) as landmarker_strict, \
         create_landmarker(confidence=0.15) as landmarker_loose:
         
        for idx, p in enumerate(clahe_frame_paths):
            rel_path = p.relative_to(FRAMES_CLAHE)
            out_path = FRAMES_MESH / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            parts = rel_path.parts
            video_id = parts[0]
            participant_id = parts[1]

            img_bgr = imread_unicode(p)
            if img_bgr is None: continue
            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            results = landmarker_strict.detect(mp_image)
            method = 'strict'
            
            if not results.face_landmarks:
                results = landmarker_loose.detect(mp_image)
                method = 'loose' if results.face_landmarks else 'failed'

            row = {
                'video_id': video_id, 
                'participant_id': participant_id, 
                'frame_file': p.name,
                'face_detected': False, 
                'detection_method': method,
                'EAR_left': np.nan,
                'EAR_right': np.nan,
                'mean_EAR': np.nan, 
                'MAR': np.nan,
                'pitch': np.nan,
                'yaw': np.nan,
                'roll': np.nan
            }

            if results.face_landmarks:
                face_lms = results.face_landmarks[0]
                
                left_ear  = eye_aspect_ratio(face_lms, LEFT_EYE_IDXS,  w, h)
                right_ear = eye_aspect_ratio(face_lms, RIGHT_EYE_IDXS, w, h)
                mar       = mouth_aspect_ratio(face_lms, w, h)
                pitch, yaw, roll = get_head_pose(face_lms, w, h)
                mean_ear  = (left_ear + right_ear) / 2

                row.update({
                    'face_detected': True,
                    'EAR_left': round(left_ear, 5),
                    'EAR_right': round(right_ear, 5),
                    'mean_EAR': round(mean_ear, 5),
                    'MAR': round(mar, 5),
                    'pitch': round(pitch, 2),
                    'yaw': round(yaw, 2),
                    'roll': round(roll, 2),
                })

                annotated = draw_custom_mesh(img_bgr, face_lms, w, h)
                color = (0, 255, 0) if method == 'strict' else (0, 255, 255)
                cv2.putText(annotated, f'P:{pitch:.1f} Y:{yaw:.1f} ({method})', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                imwrite_unicode(out_path, annotated, quality=90)
            else:
                fail_name = f"{video_id}_{participant_id}_{p.name}"
                shutil.copy(str(p), str(FAILED_DIR / fail_name))
                shutil.copy(str(p), str(out_path))

            records.append(row)
            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                elapsed = time.time() - t0
                fps = (idx + 1) / elapsed
                print(f'  Processed {idx+1}/{total} frames | {fps:.1f} FPS | Method: {method}')
                gc.collect()

    print('Finalizing DataFrame...')
    df_landmarks = pd.DataFrame(records)
    df_landmarks.to_csv(RAW_LANDMARKS_CSV, index=False)
    print(f'Mesh pipeline finished in {time.time()-t0:.1f}s | Saved to {RAW_LANDMARKS_CSV}')

if __name__ == '__main__':
    main()
