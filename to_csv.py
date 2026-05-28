import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

# Add project root to sys.path to allow importing src.core_config
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core_config import RAW_LANDMARKS_CSV, SUMMARY_FEATURES_CSV

def process_features(df):
    """
    Cleans and smooths the features from the raw landmark data.
    """
    print("Processing features (Advanced Interpolation & Smoothing)...")
    
    # 1. Group by video and participant
    grouped = df.groupby(['video_id', 'participant_id'])
    
    processed_list = []
    
    for name, group in grouped:
        group = group.copy().sort_values('frame_file')
        
        # 2. Refined Interpolation (FD-05)
        # Using polynomial order 2 for smoother transitions than linear
        cols_to_fix = ['mean_EAR', 'MAR', 'head_dx', 'head_dy']
        for col in cols_to_fix:
            if col in group.columns:
                # Polynomial interpolation (order 2) is smoother for biological signals
                # We still limit to 4 frames to avoid hallucinating long gaps
                try:
                    group[col] = group[col].interpolate(method='polynomial', order=2, limit=4)
                except Exception:
                    # Fallback to linear if polynomial fails (e.g., not enough points)
                    group[col] = group[col].interpolate(method='linear', limit=4)
                
                # Fill remaining NaNs at the edges
                group[col] = group[col].ffill().bfill()
        
        # 3. Advanced Smoothing (Savitzky-Golay Filter)
        # Savitzky-Golay preserves peaks/valleys better than a simple rolling mean.
        # Window length must be odd. 5 frames at 4 FPS is 1.25s.
        for col in cols_to_fix:
            if col in group.columns:
                # We only apply filter if we have enough data points
                if len(group) > 5:
                    try:
                        # Ensure no NaNs before filtering
                        clean_vals = group[col].fillna(method='ffill').fillna(method='bfill').values
                        group[f'{col}_smooth'] = savgol_filter(clean_vals, window_length=5, polyorder=2)
                    except Exception:
                        group[f'{col}_smooth'] = group[col].rolling(window=3, min_periods=1, center=True).mean()
                else:
                    group[f'{col}_smooth'] = group[col].rolling(window=3, min_periods=1, center=True).mean()
            
        processed_list.append(group)
        
    if not processed_list:
        return pd.DataFrame()
        
    return pd.concat(processed_list)

def main():
    if not RAW_LANDMARKS_CSV.exists():
        print(f'Error: Input file {RAW_LANDMARKS_CSV} not found.')
        return

    base_cols = ['video_id', 'participant_id', 'frame_file', 'face_detected']
    feature_cols = ['mean_EAR', 'MAR', 'head_dx', 'head_dy']
    
    # detection_method was added in FD-03/06
    optional_cols = ['detection_method']
    available_cols = pd.read_csv(RAW_LANDMARKS_CSV, nrows=0).columns.tolist()
    load_cols = base_cols + feature_cols + [c for c in optional_cols if c in available_cols]

    print(f'Reading {RAW_LANDMARKS_CSV}...')
    try:
        df = pd.read_csv(RAW_LANDMARKS_CSV, usecols=load_cols)
        
        df_final = process_features(df)
        
        if df_final.empty:
            print("No data to process.")
            return

        smooth_cols = [f'{c}_smooth' for c in feature_cols]
        save_cols = load_cols + smooth_cols
        
        print(f"Saving summarized features to {SUMMARY_FEATURES_CSV}...")
        df_final[save_cols].to_csv(SUMMARY_FEATURES_CSV, index=False)
        
        print(f'Successfully saved summarized features.')
        print(f'Total rows processed: {len(df_final)}')
        
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == '__main__':
    main()
