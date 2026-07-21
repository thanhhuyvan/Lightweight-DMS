import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core_config import RAW_LANDMARKS_CSV, FAILED_DIR, OUTPUT_ROOT

def main():
    if not RAW_LANDMARKS_CSV.exists():
        print(f"Error: {RAW_LANDMARKS_CSV} not found.")
        return

    print(f"Analyzing failures from {RAW_LANDMARKS_CSV}...")
    
    # Load only necessary columns
    cols = ['video_id', 'participant_id', 'frame_file', 'face_detected']
    df = pd.read_csv(RAW_LANDMARKS_CSV, usecols=cols)
    
    total_frames = len(df)
    failed_df = df[df['face_detected'] == False]
    failed_count = len(failed_df)
    
    print(f"\n--- Global Stats ---")
    print(f"Total Frames: {total_frames}")
    print(f"Failed Frames: {failed_count} ({failed_count/total_frames*100:.2f}%)")
    
    # Analysis by Participant
    print(f"\n--- Failures by Participant ---")
    p_stats = df.groupby('participant_id')['face_detected'].value_counts(normalize=True).unstack()
    if False in p_stats.columns:
        p_fail_pct = p_stats[False] * 100
        print(p_fail_pct.sort_values(ascending=False))
    else:
        print("No failures found per participant.")
        
    # Analysis by Video
    print(f"\n--- Top 10 Problematic Videos (Failure Count) ---")
    v_stats = failed_df.groupby('video_id').size().sort_values(ascending=False).head(10)
    print(v_stats)
    
    # Check physical folder
    physical_files = list(FAILED_DIR.glob('*.jpg'))
    print(f"\n--- Physical Audit ---")
    print(f"Images in {FAILED_DIR.name}: {len(physical_files)}")
    
    # Generate Plot
    plt.figure(figsize=(10, 6))
    v_stats.plot(kind='bar', color='salmon')
    plt.title('Top 10 Problematic Videos (Failed Detections)')
    plt.ylabel('Number of Failed Frames')
    plt.xlabel('Video ID')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    report_path = OUTPUT_ROOT / 'failure_analysis.png'
    plt.savefig(report_path)
    print(f"\nReport saved to: {report_path}")

if __name__ == '__main__':
    main()
