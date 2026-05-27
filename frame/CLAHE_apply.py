import sys
import time
from pathlib import Path
import numpy as np
import cv2

# Add Frame_exrtaction directory to sys.path to allow importing config
sys.path.append(str(Path(__file__).parent.parent / 'Frame_exrtaction'))
from config import *

def imread_unicode(path):
    try:
        img_array = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f'Error reading {path}: {e}')
        return None

def imwrite_unicode(path, img, quality=95):
    try:
        is_success, img_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if is_success:
            img_buf.tofile(str(path))
            return True
        return False
    except Exception as e:
        print(f'Error writing {path}: {e}')
        return False

def apply_clahe(frame_bgr: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    if frame_bgr is None:
        return None
    lab   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq  = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

t0 = time.time()
# Recursively find all raw frames
raw_frame_paths = sorted(list(FRAMES_RAW.rglob('*.jpg')))
total = len(raw_frame_paths)
print(f'Found {total} raw frames in {FRAMES_RAW}')

for idx, p in enumerate(raw_frame_paths):
    # Determine relative path from FRAMES_RAW
    rel_path = p.relative_to(FRAMES_RAW)
    out_path = FRAMES_CLAHE / rel_path
    
    # Skip if already exists (optional optimization)
    # if out_path.exists(): continue
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    frame = imread_unicode(p)
    if frame is None: continue
    
    enhanced = apply_clahe(frame, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID)
    imwrite_unicode(out_path, enhanced, quality=95)
    
    if (idx + 1) % 500 == 0 or (idx + 1) == total:
        print(f'  CLAHE Preprocessing: {idx + 1}/{total} completed...')

print(f'CLAHE completed in {time.time()-t0:.1f}s')
