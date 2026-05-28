import sys
import time
from pathlib import Path
import numpy as np
import cv2
import shutil
import multiprocessing

# Add project root to sys.path to allow importing src.core_config
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core_config import *

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

def process_video_worker(args):
    video_path, p_folder, target_fps, resize_dim, clip_limit, tile_grid, raw_base, clahe_base = args
    video_name = video_path.stem
    
    # Target directories
    raw_target_dir = raw_base / video_name / p_folder
    clahe_target_dir = clahe_base / video_name / p_folder
    
    # Delete existing folders if they exist to start fresh
    try:
        if raw_target_dir.exists():
            shutil.rmtree(raw_target_dir)
        raw_target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not clean raw directory {raw_target_dir}: {e}")
        
    try:
        if clahe_target_dir.exists():
            shutil.rmtree(clahe_target_dir)
        clahe_target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not clean clahe directory {clahe_target_dir}: {e}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return video_path.name, False, 0
        
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(src_fps / target_fps))

    saved_idx = 0
    frame_idx = 0

    print(f"Starting: {video_path.name} for {p_folder} (Source FPS={src_fps:.1f})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            if resize_dim:
                frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
            
            # Save Raw frame
            out_raw = raw_target_dir / f"frame_{saved_idx:05d}.jpg"
            imwrite_unicode(out_raw, frame, quality=95)
            
            # Apply CLAHE and save CLAHE frame
            clahe_frame = apply_clahe(frame, clip_limit, tile_grid)
            out_clahe = clahe_target_dir / f"frame_{saved_idx:05d}.jpg"
            imwrite_unicode(out_clahe, clahe_frame, quality=95)
            
            saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"Completed: {video_path.name} for {p_folder} ({saved_idx} frames extracted)")
    return video_path.name, True, saved_idx

def main():
    t_start = time.time()
    
    # Gather tasks
    tasks = []
    for p_id, p_folder in PARTICIPANT_MAP.items():
        p_video_dir = VIDEO_ROOT / p_id
        if not p_video_dir.exists():
            continue
            
        for video_file in p_video_dir.glob('*.*'):
            if video_file.suffix.lower() not in ['.mp4', '.mov', '.avi']:
                continue
            
            tasks.append((
                video_file,
                p_folder,
                TARGET_FPS,
                RESIZE_DIM,
                CLAHE_CLIP_LIMIT,
                CLAHE_TILE_GRID,
                FRAMES_RAW,
                FRAMES_CLAHE
            ))
            
    print(f"Found {len(tasks)} videos to process.")
    
    if not tasks:
        print("No videos found.")
        return

    num_workers = min(18, len(tasks))
    print(f"Starting multiprocessing pool with {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_video_worker, tasks)
        
    success_count = sum(1 for r in results if r[1])
    total_frames = sum(r[2] for r in results)
    
    print(f"\n--- Processing Summary ---")
    for name, success, count in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  Video: {name:<12} | Status: {status:<8} | Frames: {count}")
    
    print(f"\nSuccessfully processed {success_count}/{len(tasks)} videos.")
    print(f"Total frames extracted: {total_frames}")
    print(f"Elapsed time: {time.time() - t_start:.1f}s")

if __name__ == '__main__':
    main()
