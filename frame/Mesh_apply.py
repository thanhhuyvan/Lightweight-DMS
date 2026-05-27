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

# Add Frame_exrtaction directory to sys.path to allow importing config
sys.path.append(str(Path(__file__).parent.parent / 'Frame_exrtaction'))
from config import *

# Landmark constants for features
LEFT_EYE_IDXS  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33,  160, 158, 133, 153, 144]
MOUTH_OUTER = [13, 14, 78, 308]
MOUTH_INNER = [312, 317]
NOSE_TIP_IDX = 1

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

def head_pose_proxy(landmarks, img_w, img_h):
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    cx, cy = np.mean(xs), np.mean(ys)
    face_w = max(xs) - min(xs)
    face_h = max(ys) - min(ys)
    nose   = landmarks[NOSE_TIP_IDX]
    nx, ny = nose.x * img_w, nose.y * img_h
    dx = (nx - cx) / (face_w + 1e-6)
    dy = (ny - cy) / (face_h + 1e-6)
    return dx, dy

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
    for lm in landmarks[:468]:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(annotated, (cx, cy), 1, (0, 180, 0), -1)
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

t0 = time.time()
clahe_frame_paths = sorted(list(FRAMES_CLAHE.rglob('*.jpg')))
total = len(clahe_frame_paths)
print(f'Found {total} CLAHE frames in {FRAMES_CLAHE}')

# Model setup
src_model_path = PROJECT_ROOT / 'face_landmarker.task'
ascii_model_path = os.path.join(tempfile.gettempdir(), 'face_landmarker.task')
shutil.copy(str(src_model_path), ascii_model_path)

base_options = python.BaseOptions(model_asset_path=ascii_model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

records = []

with vision.FaceLandmarker.create_from_options(options) as landmarker:
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
        results = landmarker.detect(mp_image)

        row = {
            'video_id': video_id, 
            'participant_id': participant_id, 
            'frame_file': p.name,
            'face_detected': False, 
            'EAR_left': np.nan,
            'EAR_right': np.nan,
            'mean_EAR': np.nan, 
            'MAR': np.nan,
            'head_dx': np.nan,
            'head_dy': np.nan
        }

        if results.face_landmarks:
            face_lms = results.face_landmarks[0]
            
            # Save ALL landmarks as flattened x,y,z coordinates
            # total 478 landmarks * 3 (x,y,z) = 1434 additional columns
            for i, lm in enumerate(face_lms):
                row[f'lm_{i}_x'] = round(lm.x, 6)
                row[f'lm_{i}_y'] = round(lm.y, 6)
                row[f'lm_{i}_z'] = round(lm.z, 6)

            left_ear  = eye_aspect_ratio(face_lms, LEFT_EYE_IDXS,  w, h)
            right_ear = eye_aspect_ratio(face_lms, RIGHT_EYE_IDXS, w, h)
            mar       = mouth_aspect_ratio(face_lms, w, h)
            dx, dy    = head_pose_proxy(face_lms, w, h)
            mean_ear  = (left_ear + right_ear) / 2

            row.update({
                'face_detected': True,
                'EAR_left': round(left_ear, 5),
                'EAR_right': round(right_ear, 5),
                'mean_EAR': round(mean_ear, 5),
                'MAR': round(mar, 5),
                'head_dx': round(dx, 5),
                'head_dy': round(dy, 5),
            })

            annotated = draw_custom_mesh(img_bgr, face_lms, w, h)
            cv2.putText(annotated, f'EAR:{mean_ear:.3f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            imwrite_unicode(out_path, annotated, quality=90)
        else:
            # If no face, we fill landmarks with NaN (already handled by row template if we adjust loop)
            # To ensure consistent columns, we should pre-fill or handle after loop
            pass
            shutil.copy(str(p), str(out_path))

        records.append(row)
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f'  Processed {idx+1}/{total} frames ({ (idx+1)/total*100:.1f}%) | {fps:.1f} FPS')
            gc.collect()

print('Finalizing DataFrame (this might take a moment due to many columns)...')
df_landmarks = pd.DataFrame(records)
csv_path = CSV_DIR / 'landmarks_full.csv'
df_landmarks.to_csv(csv_path, index=False)
print(f'Mesh pipeline finished in {time.time()-t0:.1f}s | Saved ALL landmarks to {csv_path}')
