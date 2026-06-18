"""
diagnostic_viz.py — Deep Diagnostic Tool

Aims to find out WHY the model fails on specific participants by:
1. Analyzing feature 'alignment' after normalization.
2. Plotting per-participant confusion matrices.
3. Checking correlation between data quality (face detection) and error rate.
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
VECTORS_CSV = PROJECT_ROOT / 'frame' / 'csv' / 'behavioral_vectors.csv'
FEATURES_CSV = PROJECT_ROOT / 'frame' / 'csv' / 'features_summary.csv'
MODEL_PATH = PROJECT_ROOT / 'models' / 'improved_model.joblib'
SCALER_PATH = PROJECT_ROOT / 'models' / 'improved_scaler.joblib'
REPORT_DIR = PROJECT_ROOT / 'report' / 'diagnostics'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_and_prepare():
    if not VECTORS_CSV.exists() or not MODEL_PATH.exists():
        logging.error("Required files missing.")
        return None, None, None
    
    df = pd.read_csv(VECTORS_CSV)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Re-apply the same normalization as in train_improved.py
    baselines = df[df['video_id'] == 0].groupby('participant_id')[['EAR_Mean', 'MAR_Mean']].mean().reset_index()
    baselines.columns = ['participant_id', 'EAR_base', 'MAR_base']
    df = df.merge(baselines, on='participant_id', how='left')
    df['EAR_base'] = df['EAR_base'].fillna(df['EAR_Mean'].mean())
    df['MAR_base'] = df['MAR_base'].fillna(df['MAR_Mean'].mean())
    df['EAR_Relative'] = df['EAR_Mean'] / df['EAR_base']
    df['MAR_Relative'] = df['MAR_Mean'] / df['MAR_base']
    
    return df, model, scaler

def diagnostic_plots(df, model, scaler):
    exclude_cols = ['video_id', 'participant_id', 'window_start_frame', 'window_end_frame', 
                    'window_start_idx', 'window_end_idx', 'EAR_base', 'MAR_base']
    features = [c for c in df.columns if c not in exclude_cols]
    
    X = df[features]
    y_true = df['video_id'].map({0: 0, 5: 1, 10: 2})
    
    X_scaled = scaler.transform(X)
    df['pred'] = model.predict(X_scaled)
    df['correct'] = (df['pred'] == y_true)
    
    # 1. Feature Drift Analysis: Are 'Relative' features actually aligned?
    # In Alert state (video_id 0), EAR_Relative SHOULD be 1.0 for everyone.
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='participant_id', y='EAR_Relative', data=df[df['video_id'] == 0])
    plt.axhline(1.0, color='red', linestyle='--')
    plt.title('Feature Drift: EAR_Relative at ALERT State (Goal is 1.0)')
    plt.savefig(REPORT_DIR / 'drift_ear_relative.png')
    
    # 2. Error Heatmap: Which Participant vs Which State
    error_rates = df.groupby(['participant_id', 'video_id'])['correct'].mean().unstack()
    plt.figure(figsize=(10, 8))
    sns.heatmap(error_rates, annot=True, cmap='RdYlGn', vmin=0, vmax=1)
    plt.title('Accuracy Heatmap (Participant vs State)')
    plt.savefig(REPORT_DIR / 'accuracy_heatmap.png')
    
    # 3. Why did Fold 2 fail? Let's look at participant2 specifically
    # Find the most problematic participant
    worst_p = error_rates.mean(axis=1).idxmin()
    best_p = error_rates.mean(axis=1).idxmax()
    logging.info(f"Worst Participant: {worst_p} | Best Participant: {best_p}")
    
    # Compare feature distribution of Worst vs Best
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.kdeplot(data=df[df['participant_id'] == worst_p], x='EAR_Relative', hue='video_id', ax=axes[0], fill=True)
    axes[0].set_title(f'Feature Separation (EAR_Rel) - WORST: {worst_p}')
    sns.kdeplot(data=df[df['participant_id'] == best_p], x='EAR_Relative', hue='video_id', ax=axes[1], fill=True)
    axes[1].set_title(f'Feature Separation (EAR_Rel) - BEST: {best_p}')
    plt.savefig(REPORT_DIR / 'best_vs_worst_separation.png')

    # 4. Correlation with Data Quality
    # We need to know if 'face_detected' rate affects accuracy
    if FEATURES_CSV.exists():
        raw_df = pd.read_csv(FEATURES_CSV)
        quality = raw_df.groupby(['participant_id', 'video_id'])['face_detected'].mean().reset_index()
        quality.columns = ['participant_id', 'video_id', 'detection_rate']
        
        # Merge with accuracy
        acc_flat = error_rates.stack().reset_index()
        acc_flat.columns = ['participant_id', 'video_id', 'accuracy']
        quality_acc = quality.merge(acc_flat, on=['participant_id', 'video_id'])
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='detection_rate', y='accuracy', hue='participant_id', data=quality_acc, s=100)
        plt.title('Data Quality (Face Detection Rate) vs Accuracy')
        plt.savefig(REPORT_DIR / 'quality_vs_accuracy.png')

def main():
    df, model, scaler = load_and_prepare()
    if df is not None:
        diagnostic_plots(df, model, scaler)
        logging.info(f"Diagnostics saved to {REPORT_DIR}")

if __name__ == '__main__':
    main()
