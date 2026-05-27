import sys
import pandas as pd
from pathlib import Path

# Thêm thÆ° má»¥c Frame_exrtaction vÃ o sys.path Ä‘á»ƒ cho phÃ©p import config
sys.path.append(str(Path(__file__).parent / 'Frame_exrtaction'))
from config import *

# ÄÆ°á»ng dáº«n file input (file nÃ y Ä‘Æ°á»£c táº¡o ra sau khi cháº¡y mesh_apply.py)
input_csv = CSV_DIR / 'landmarks.csv'
output_csv = CSV_DIR / 'features_summary.csv'

if input_csv.exists():
    print(f'Reading {input_csv}...')
    df = pd.read_csv(input_csv)
    
    # Chá»n cÃ¡c cá»™t Ä‘áº·c trÆ°ng cáº§n thiáº¿t
    feature_cols = [
        'video_id', 
        'participant_id', 
        'frame_file', 
        'face_detected', 
        'EAR_left', 
        'EAR_right', 
        'mean_EAR', 
        'MAR', 
        'head_dx', 
        'head_dy'
    ]
    
    # Kiá»ƒm tra xem cÃ¡c cá»™t cÃ³ tá»“n táº¡i khÃ´ng (phÃ²ng trÆ°á»ng há»£p tÃªn cá»™t trong mesh_apply.py thay Ä‘á»•i)
    existing_cols = [c for c in feature_cols if c in df.columns]
    df_features = df[existing_cols].copy()
    
    # LÆ°u file summary
    df_features.to_csv(output_csv, index=False)
    print(f'Saved features summary to: {output_csv}')
    print(df_features.head(10))
else:
    print(f'Error: Input file {input_csv} not found. Please ensure mesh_apply.py has finished.')
