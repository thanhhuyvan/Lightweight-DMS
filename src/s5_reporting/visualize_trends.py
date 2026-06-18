"""
visualize_trends.py — Diagnosis & Visualization Tool

Generates deep-dive visualizations to understand participant variance
and temporal prediction stability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import logging

# --- Setup Paths ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CSV_PATH = PROJECT_ROOT / 'frame' / 'csv' / 'behavioral_vectors.csv'
MODEL_PATH = PROJECT_ROOT / 'models' / 'baseline_rf_model.joblib'
REPORT_DIR = PROJECT_ROOT / 'report'
REPORT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_data():
    if not CSV_PATH.exists():
        logging.error(f"Data not found: {CSV_PATH}")
        return None
    return pd.read_csv(CSV_PATH)

def plot_temporal_comparison(df, model):
    """
    Plots the actual vs predicted state over time for a sample participant.
    """
    logging.info("Generating temporal trend visualization...")
    
    # Pick a participant and a video sequence
    # Let's take 'participant1' and visualize their transition if possible
    sample_p = 'participant1'
    p_data = df[df['participant_id'] == sample_p].copy().sort_values(['video_id', 'window_start_idx'])
    
    if p_data.empty:
        logging.warning("No data for participant1 found.")
        return

    # Features used in baseline
    features = ['PERCLOS', 'Blink_Rate', 'EAR_Std', 'Pose_Jitter']
    # Calculate EAR_Variance as in train_baseline
    p_data['EAR_Variance'] = p_data['EAR_Std'] ** 2
    X = p_data[['PERCLOS', 'Blink_Rate', 'EAR_Variance', 'Pose_Jitter']]
    
    # Predict
    p_data['predicted_id'] = model.predict(X)
    
    plt.figure(figsize=(15, 7))
    
    # Plot Actual
    plt.step(range(len(p_data)), p_data['video_id'], label='Ground Truth (Actual)', 
             where='post', color='gray', alpha=0.5, linewidth=4)
    
    # Plot Predicted
    plt.step(range(len(p_data)), p_data['predicted_id'], label='Model Prediction (Baseline)', 
             where='post', color='red', linestyle='--')
    
    # Plot PERCLOS trend (scaled to fit 0-10 range for visibility)
    plt.plot(range(len(p_data)), p_data['PERCLOS'] / 10, label='PERCLOS (Scaled 0.1x)', 
             color='blue', alpha=0.3)

    plt.title(f'Temporal Prediction Stability: {sample_p}')
    plt.xlabel('Sliding Window Index (Stride: 1s)')
    plt.ylabel('Drowsiness Level (0=Alert, 5=Low, 10=Drowsy)')
    plt.yticks([0, 5, 10], ['Alert', 'Low', 'Drowsy'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = REPORT_DIR / 'temporal_trend_diagnosis.png'
    plt.savefig(save_path)
    logging.info(f"Saved: {save_path}")

def plot_participant_variance(df):
    """
    Visualizes how core features vary across participants for the SAME state.
    This explains why GroupKFold is hard.
    """
    logging.info("Generating participant variance visualization...")
    
    # Focus on ALERT state (video_id == 0) to see baseline differences
    alert_df = df[df['video_id'] == 0]
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='participant_id', y='PERCLOS', data=alert_df, palette='Set2')
    plt.title('Variance of PERCLOS across Participants (All in ALERT state)')
    plt.ylabel('PERCLOS (%)')
    plt.xlabel('Participant ID')
    plt.grid(axis='y', alpha=0.3)
    
    save_path = REPORT_DIR / 'participant_variance_perclos.png'
    plt.savefig(save_path)
    logging.info(f"Saved: {save_path}")

    # Also check EAR_Mean variance
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='participant_id', y='EAR_Mean', data=alert_df, palette='Set3')
    plt.title('Anatomical Variance: Mean EAR across Participants (ALERT state)')
    plt.ylabel('Mean EAR')
    plt.xlabel('Participant ID')
    plt.grid(axis='y', alpha=0.3)
    
    save_path = REPORT_DIR / 'anatomical_variance_ear.png'
    plt.savefig(save_path)
    logging.info(f"Saved: {save_path}")

def main():
    df = load_data()
    if df is None: return
    
    if not MODEL_PATH.exists():
        logging.error("Baseline model not found. Run train_baseline.py first.")
        return
        
    model = joblib.load(MODEL_PATH)
    
    plot_temporal_comparison(df, model)
    plot_participant_variance(df)
    
    logging.info("\nVisualization complete. Check the 'report/' folder.")

if __name__ == '__main__':
    main()
