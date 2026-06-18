import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core_config import SUMMARY_FEATURES_CSV, TARGET_FPS

def calculate_perclos(df, window_seconds=60):
    """
    Calculates PERCLOS (Percentage of Eye Closure) over a sliding window.
    PERCLOS = (Frames with eye_state == 1) / (Total frames in window) * 100
    """
    window_size = window_seconds * TARGET_FPS
    logging.info(f"Calculating PERCLOS with window size of {window_seconds}s ({window_size} frames)...")
    
    # We must group by video_id to avoid bleeding across different videos/participants
    grouped = df.groupby(['video_id', 'participant_id'])
    
    processed_list = []
    for name, group in grouped:
        group = group.sort_values('frame_file').copy()
        
        # Calculate rolling PERCLOS
        # center=False because in real-time we only have past data
        group['PERCLOS'] = group['eye_state'].rolling(
            window=window_size, 
            min_periods=1
        ).mean() * 100
        
        processed_list.append(group)
        
    return pd.concat(processed_list)

def main():
    if not SUMMARY_FEATURES_CSV.exists():
        logging.error(f"Input file {SUMMARY_FEATURES_CSV} not found. Run eye_state.py first.")
        return

    logging.info(f"Loading features from {SUMMARY_FEATURES_CSV}...")
    df = pd.read_csv(SUMMARY_FEATURES_CSV)
    
    if 'eye_state' not in df.columns:
        logging.error("Column 'eye_state' not found. Run eye_state.py first.")
        return

    df_final = calculate_perclos(df)
    
    logging.info(f"Updating {SUMMARY_FEATURES_CSV} with PERCLOS feature...")
    df_final.to_csv(SUMMARY_FEATURES_CSV, index=False)
    logging.info("PERCLOS calculation complete.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
