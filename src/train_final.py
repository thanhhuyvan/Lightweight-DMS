"""
train_final.py — Stage 5: Final Optimized Training (Baseline Rescue Edition)

Goal: Reach F1 > 0.70 for Geometry-only baseline using:
1. Min-Max scaling per participant (Removes anatomical bias).
2. Binary Classification (Alert vs. Extreme Drowsy) to establish ceiling.
3. 10s Window dataset (Higher sample density).
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# --- Setup Paths ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
VECTORS_CSV = PROJECT_ROOT / 'frame' / 'csv' / 'behavioral_vectors.csv'
FEATURES_CSV = PROJECT_ROOT / 'frame' / 'csv' / 'features_summary.csv'
MODEL_DIR = PROJECT_ROOT / 'models'
REPORT_DIR = PROJECT_ROOT / 'report' / 'final'
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

def min_max_scale_group(group):
    norm_cols = ['EAR_Mean', 'MAR_Mean', 'Blink_Rate', 'Pose_Jitter']
    for col in norm_cols:
        if col in group.columns:
            mi = group[col].min()
            ma = group[col].max()
            if ma > mi:
                group[f'{col}_Norm'] = (group[col] - mi) / (ma - mi)
            else:
                group[f'{col}_Norm'] = 0.5
    return group

def load_clean_and_engineer():
    if not VECTORS_CSV.exists():
        logging.error("Behavioral vectors CSV missing.")
        return None
    
    df = pd.read_csv(VECTORS_CSV)
    
    # 1. Binary Focus: Filter for video_id 0 and 10 only
    logging.info("Step 1: Filtering for Binary Classification (0 vs 10)...")
    df_binary = df[df['video_id'].isin([0, 10])].copy()
    
    if df_binary.empty:
        logging.error("No binary data found.")
        return None

    # 2. Per-Participant Min-Max Normalization (Manual Loop for robustness)
    logging.info("Step 2: Applying Per-Participant Min-Max Normalization...")
    processed_groups = []
    for p_id, group in df_binary.groupby('participant_id'):
        processed_groups.append(min_max_scale_group(group.copy()))
    
    df_binary = pd.concat(processed_groups, ignore_index=True)
    
    # 3. Delta Features
    logging.info("Step 3: Engineering Delta Features...")
    df_binary = df_binary.sort_values(['participant_id', 'video_id', 'window_start_idx'])
    
    # Pre-calculate deltas to avoid groupby index issues
    df_binary['EAR_Mean_Norm_Delta'] = df_binary.groupby(['participant_id', 'video_id'])['EAR_Mean_Norm'].diff(periods=3).fillna(0)
    df_binary['PERCLOS_Delta'] = df_binary.groupby(['participant_id', 'video_id'])['PERCLOS'].diff(periods=3).fillna(0)
    df_binary['Pose_Jitter_Norm_Delta'] = df_binary.groupby(['participant_id', 'video_id'])['Pose_Jitter_Norm'].diff(periods=3).fillna(0)
        
    return df_binary

def run_final_training(df):
    features = [
        'PERCLOS', 'Blink_Rate_Norm', 'EAR_Mean_Norm', 'EAR_Std', 
        'Pitch_Jitter', 'Yaw_Jitter', 'Pose_Jitter_Norm',
        'EAR_Mean_Norm_Delta', 'PERCLOS_Delta', 'Pose_Jitter_Norm_Delta'
    ]
    
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    y = df['video_id'].map({0: 0, 10: 1})
    groups = df['participant_id']
    
    logging.info(f"Training Baseline Rescue Model (XGBoost Binary) with {len(available_features)} features.")
    
    gkf = GroupKFold(n_splits=5)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fold_results.append(f1)
        logging.info(f"Fold {fold} | Test: {df.iloc[test_idx]['participant_id'].unique()} | F1: {f1:.4f} | Acc: {acc:.4f}")
        
        if fold == 1:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                        xticklabels=['Alert', 'Drowsy'], yticklabels=['Alert', 'Drowsy'])
            plt.title('Binary Baseline Confusion Matrix')
            plt.savefig(REPORT_DIR / 'binary_confusion_matrix.png')

    logging.info("-" * 30)
    logging.info(f"RESCUE BASELINE MEAN F1-SCORE: {np.mean(fold_results):.4f}")
    logging.info("-" * 30)
    
    importances = pd.Series(clf.feature_importances_, index=available_features).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    importances.plot(kind='barh', color='darkred')
    plt.title('Feature Importance (Binary Baseline)')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'binary_feature_importance.png')

def main():
    df = load_clean_and_engineer()
    if df is not None:
        run_final_training(df)

if __name__ == '__main__':
    main()
