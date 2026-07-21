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

def apply_eye_state(df, calib_df):
    """
    Applies the dynamic alpha threshold to calculate the binary eye state B(t).
    B(t) = 1 if EAR < alpha (closed), else 0 (open).
    """
    logging.info("Calculating binary eye state B(t) using dynamic thresholds...")
    
    # Merge calibration parameters
    df = df.merge(calib_df[['participant_id', 'alpha']], on='participant_id', how='left')
    
    # Default alpha if missing (should not happen if calibration ran)
    df['alpha'] = df['alpha'].fillna(0.225)
    
    # Choose column
    ear_col = 'mean_EAR_smooth' if 'mean_EAR_smooth' in df.columns else 'mean_EAR'
    logging.info(f"Using column '{ear_col}' for eye state calculation.")
    
    # B(t) calculation: 1 for closed (drowsy/blink), 0 for open
    df['eye_state'] = (df[ear_col] < df['alpha']).astype(int)
    
    closed_count = df['eye_state'].sum()
    total_count = len(df)
    logging.info(f"Eye state calculated: {closed_count}/{total_count} frames marked as 'closed' ({closed_count/total_count*100:.2f}%)")
    
    return df

def main():
    calib_params_path = CSV_DIR / 'calibration_params.csv'
    
    if not SUMMARY_FEATURES_CSV.exists():
        logging.error(f"Input file {SUMMARY_FEATURES_CSV} not found.")
        return
    if not calib_params_path.exists():
        logging.error(f"Calibration file {calib_params_path} not found. Run calibration.py first.")
        return

    logging.info(f"Loading features and calibration parameters...")
    df = pd.read_csv(SUMMARY_FEATURES_CSV)
    calib_df = pd.read_csv(calib_params_path)
    
    df_final = apply_eye_state(df, calib_df)
    
    # Save back to SUMMARY_FEATURES_CSV or a new one? 
    # Let's keep it in SUMMARY_FEATURES_CSV to maintain a single feature source
    logging.info(f"Updating {SUMMARY_FEATURES_CSV} with B(t) eye state...")
    df_final.to_csv(SUMMARY_FEATURES_CSV, index=False)
    logging.info("Eye state calculation complete.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
