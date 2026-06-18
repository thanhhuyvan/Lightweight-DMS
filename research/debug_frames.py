import sys
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core_config import FRAMES_CLAHE

frame_paths = sorted(list(FRAMES_CLAHE.rglob('*.jpg')))
print(f"Total frames: {len(frame_paths)}")

if len(frame_paths) > 0:
    p = frame_paths[0]
    print(f"Checking first frame: {p}")
    img = cv2.imread(str(p))
    if img is None:
        print("FAIL: cv2.imread returned None")
    else:
        print(f"SUCCESS: Image shape {img.shape}")
else:
    print("FAIL: No frames found")
