import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import shutil
import tempfile
import time
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core_config import *

# Landmark indices for extraction
LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
MOUTH_IDXS = [61, 291, 13, 14] # Corners and top/bottom

# Output Directories
PATCH_ROOT = OUTPUT_ROOT / 'patches'
LEFT_EYE_DIR = PATCH_ROOT / 'left_eye'
RIGHT_EYE_DIR = PATCH_ROOT / 'right_eye'
MOUTH_DIR = PATCH_ROOT / 'mouth'

for d in [LEFT_EYE_DIR, RIGHT_EYE_DIR, MOUTH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def crop_isotropic(img, landmarks, idxs, img_w, img_h, target_size=(24, 24), padding_factor=1.2):
    """
    Extracts a region defined by landmarks, applies isotropic padding, and resizes.
    """
    pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in idxs])
    
    # Calculate Bounding Box
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    
    w = x_max - x_min
    h = y_max - y_min
    
    # Add a bit of padding around the features
    cx, cy = x_min + w/2, y_min + h/2
    side = max(w, h) * padding_factor
    
    # New BBox coordinates
    nx1 = int(cx - side/2)
    ny1 = int(cy - side/2)
    nx2 = int(cx + side/2)
    ny2 = int(cy + side/2)
    
    # Handle out-of-bounds with padding
    # 1. Create a larger canvas if needed or just clip and pad
    # Simplified: Clip and then pad to square
    
    # Crop with safety margin
    pad_val = int(side)
    img_padded = cv2.copyMakeBorder(img, pad_val, pad_val, pad_val, pad_val, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    roi = img_padded[ny1+pad_val:ny2+pad_val, nx1+pad_val:nx2+pad_val]
    
    if roi.size == 0:
        return np.zeros(target_size, dtype=np.uint8)
        
    # Resize to target
    resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to Grayscale if not already
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
    return resized

def create_landmarker():
    ascii_model_path = os.path.join(tempfile.gettempdir(), 'face_landmarker.task')
    if not os.path.exists(ascii_model_path):
        shutil.copy(str(MODEL_PATH), ascii_model_path)
    
    base_options = python.BaseOptions(model_asset_path=ascii_model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.15, # Loose detection for maximum patch recovery
        min_face_presence_confidence=0.15
    )
    return vision.FaceLandmarker.create_from_options(options)

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

def main():
    print("🚀 Starting Batch Patch Extraction (Isotropic Padding)")
    
    # Load all CLAHE frames
    frame_paths = sorted(list(FRAMES_CLAHE.rglob('*.jpg')))
    total = len(frame_paths)
    print(f"Found {total} frames to process.")
    
    t0 = time.time()
    extracted_count = 0
    failed_count = 0
    
    with create_landmarker() as landmarker:
        for idx, p in enumerate(tqdm(frame_paths, desc="Extracting Patches")):
            # Load Image (Unicode Safe)
            img = imread_unicode(p)
            if img is None:
                continue
            
            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            # Detect
            results = landmarker.detect(mp_image)
            
            if results.face_landmarks:
                face_lms = results.face_landmarks[0]
                
                # Extract 3 Patches
                l_eye = crop_isotropic(img, face_lms, LEFT_EYE_IDXS, w, h)
                r_eye = crop_isotropic(img, face_lms, RIGHT_EYE_IDXS, w, h)
                mouth = crop_isotropic(img, face_lms, MOUTH_IDXS, w, h)
                
                # Define Output Path (preserving structure: video_id_participant_id_frame.jpg)
                rel_parts = p.relative_to(FRAMES_CLAHE).parts
                prefix = "_".join(rel_parts[:-1])
                filename = f"{prefix}_{p.name}"
                
                imwrite_unicode(LEFT_EYE_DIR / filename, l_eye)
                imwrite_unicode(RIGHT_EYE_DIR / filename, r_eye)
                imwrite_unicode(MOUTH_DIR / filename, mouth)
                
                extracted_count += 1
            else:
                failed_count += 1
            
            # Intermediate stats
            if (idx + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  Progress: {idx+1}/{total} | Success: {extracted_count} | Failed: {failed_count} | Speed: {(idx+1)/elapsed:.1f} fps")

    duration = time.time() - t0
    print(f"\n✅ Extraction Finished!")
    print(f"Total Duration: {duration/60:.2f} minutes")
    print(f"Successfully Extracted: {extracted_count} frames")
    print(f"Failed Detections: {failed_count}")
    print(f"Output Directory: {PATCH_ROOT}")

if __name__ == "__main__":
    main()
