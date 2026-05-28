import sys
from pathlib import Path
import warnings
import numpy as np

# -- Global Settings --
warnings.filterwarnings('ignore')
np.random.seed(42)

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# -- Project Root & Directory Structure --
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
VIDEO_ROOT   = PROJECT_ROOT / 'Video_container'
OUTPUT_ROOT  = PROJECT_ROOT / 'frame'
LOG_DIR      = PROJECT_ROOT / 'logs'

# Asset Directories
FRAMES_RAW    = OUTPUT_ROOT / 'frames_raw'
FRAMES_CLAHE  = OUTPUT_ROOT / 'frames_clahe'
FRAMES_MESH   = OUTPUT_ROOT / 'frames_mesh'
FAILED_DIR    = OUTPUT_ROOT / 'failed_detections'
CSV_DIR       = OUTPUT_ROOT / 'csv'

# Ensure directories exist
for d in [FRAMES_RAW, FRAMES_CLAHE, FRAMES_MESH, FAILED_DIR, CSV_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -- Pipeline Configuration --
TARGET_FPS         = 4
CLAHE_CLIP_LIMIT   = 2.0
CLAHE_TILE_GRID    = (8, 8)
RESIZE_DIM         = (640, 480)

# MediaPipe Thresholds
FACE_MESH_CONF     = 0.4  # Default detection confidence
FACE_MESH_TRACKING = 0.4  # Default tracking confidence
MIN_FACE_DETECTION_CONF = 0.4
MIN_FACE_PRESENCE_CONF  = 0.4

# -- File Paths --
MODEL_PATH = PROJECT_ROOT / 'face_landmarker.task'
RAW_LANDMARKS_CSV = CSV_DIR / 'landmarks_full.csv'
SUMMARY_FEATURES_CSV = CSV_DIR / 'features_summary.csv'

# -- Metadata --
PARTICIPANT_MAP = {
    '01': 'participant1',
    '02': 'partcipant2',
    '03': 'participant3',
    '04': 'partcipant4',
    '05': 'participant5',
    '06': 'participant6'
}

def get_venv_python():
    """Returns path to venv python if it exists, else sys.executable."""
    venv_python = PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'
    return str(venv_python) if venv_python.exists() else sys.executable

print(f"Config Loaded -- Root: {PROJECT_ROOT}")
