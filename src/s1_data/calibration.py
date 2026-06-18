import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core_config import SUMMARY_FEATURES_CSV, CSV_DIR

# Calibration settings
CALIBRATION_WINDOW_SEC = 5  # First 5 seconds for calibration
TARGET_FPS = 4
T_CALIB_FRAMES = CALIBRATION_WINDOW_SEC * TARGET_FPS

def calculate_calibration(df):
    """
    Calculates per-participant baseline EAR_open and dynamic alpha threshold.
    EAR_open is the 85th percentile of EAR during the first T_CALIB_FRAMES.
    alpha = 0.75 * EAR_open.
    """
    logging.info(f"Starting calibration (T_calib = {CALIBRATION_WINDOW_SEC}s, {T_CALIB_FRAMES} frames)...")
    
    calibration_results = []
    
    # Check if we have the required columns
    ear_col = 'mean_EAR_smooth' if 'mean_EAR_smooth' in df.columns else 'mean_EAR'
    
    if ear_col not in df.columns:
        logging.error(f"Neither 'mean_EAR_smooth' nor 'mean_EAR' found in columns: {df.columns}")
        return pd.DataFrame()

    logging.info(f"Using column '{ear_col}' for calibration.")

    # Group by participant
    for p_id, p_group in df.groupby('participant_id'):
        # Get the first T_CALIB_FRAMES for this participant
        # Ensure temporal order by frame_file (e.g., frame_00001.jpg)
        p_group = p_group.sort_values('frame_file')
        calib_data = p_group.head(T_CALIB_FRAMES)
        
        # Filter for frames where face was detected to get valid EAR
        valid_ear = calib_data[calib_data['face_detected'] == True][ear_col]
        
        if len(valid_ear) < (T_CALIB_FRAMES / 2):
            logging.warning(f"Participant {p_id}: Only {len(valid_ear)}/{T_CALIB_FRAMES} frames have valid EAR. Calibration might be inaccurate.")
            
        if not valid_ear.empty:
            # 85th percentile represents 'natural open eye' baseline (ignoring occasional blinks)
            ear_open = np.percentile(valid_ear, 85)
            # Dynamic threshold alpha = 0.75 * EAR_open as per Task #3
            alpha = 0.75 * ear_open
        else:
            logging.error(f"Participant {p_id}: No valid EAR data found for calibration. Using defaults.")
            ear_open = 0.3 # Default estimate
            alpha = 0.225
            
        calibration_results.append({
            'participant_id': p_id,
            'EAR_open': round(ear_open, 5),
            'alpha': round(alpha, 5)
        })
        logging.info(f"Participant {p_id}: EAR_open={ear_open:.4f}, Alpha={alpha:.4f}")
        
    return pd.DataFrame(calibration_results)

def main():
    if not SUMMARY_FEATURES_CSV.exists():
        logging.error(f"Input file {SUMMARY_FEATURES_CSV} not found. Run to_csv.py first.")
        return

    logging.info(f"Reading {SUMMARY_FEATURES_CSV} for calibration...")
    df = pd.read_csv(SUMMARY_FEATURES_CSV)
    
    calib_df = calculate_calibration(df)
    
    if not calib_df.empty:
        calib_output_path = CSV_DIR / 'calibration_params.csv'
        calib_df.to_csv(calib_output_path, index=False)
        logging.info(f"Calibration parameters saved to {calib_output_path}")

if __name__ == '__main__':
    # Setup simple logging if run directly
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
