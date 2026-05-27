import os, gc, math, time, warnings, sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mediapipe as mp
from pathlib import Path
from IPython.display import display

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# -- Reproducibility --
np.random.seed(42)

# -- Root paths --
PROJECT_ROOT  = Path(r'E:\Buồn ngủ')
VIDEO_ROOT    = PROJECT_ROOT / 'Video_container'
OUTPUT_ROOT   = PROJECT_ROOT / 'frame'

FRAMES_RAW    = OUTPUT_ROOT / 'frames_raw'
FRAMES_CLAHE  = OUTPUT_ROOT / 'frames_clahe'
FRAMES_MESH   = OUTPUT_ROOT / 'frames_mesh'
CSV_DIR       = OUTPUT_ROOT / 'csv'

for d in [FRAMES_RAW, FRAMES_CLAHE, FRAMES_MESH, CSV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -- Preprocessing config --
TARGET_FPS         = 4
CLAHE_CLIP_LIMIT   = 2.0
CLAHE_TILE_GRID    = (8, 8)
RESIZE_DIM         = (640, 480)
FACE_MESH_CONF     = 0.5
FACE_MESH_TRACKING = 0.5

# Helper to map participant IDs to folder names
PARTICIPANT_MAP = {
    '01': 'participant1',
    '02': 'partcipant2',
    '03': 'participant3',
    '04': 'partcipant4',
    '05': 'participant5',
    '06': 'participant6'
}

print('Config OK -- project root:', PROJECT_ROOT)
