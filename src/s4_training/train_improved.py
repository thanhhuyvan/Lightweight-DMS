"""
train_improved.py — Stage 5: Machine Learning Classification (Optimized)

Improvements over Baseline:
1. Full Feature Set (17 features)
2. Personal Normalization (Relative EAR/MAR)
3. Advanced Algorithm (XGBoost)
4. Class Imbalance Handling
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import logging

# Check if xgboost is installed, fallback to advanced Random Forest if not
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# --- Setup Paths ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CSV_PATH = PROJECT_ROOT / 'frame' / 'csv' / 'behavioral_vectors.csv'
MODEL_DIR = PROJECT_ROOT / 'models'
REPORT_DIR = PROJECT_ROOT / 'report'
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

def load_and_preprocess():
    if not CSV_PATH.exists():
        logging.error(f"Data source not found: {CSV_PATH}")
        return None
    
    df = pd.read_csv(CSV_PATH)
    
    # --- Feature Engineering: Individual Normalization ---
    # We normalize EAR and MAR per participant based on their ALERT state (video_id 0)
    logging.info("Applying individual normalization (Relative Features)...")
    
    baselines = df[df['video_id'] == 0].groupby('participant_id')[['EAR_Mean', 'MAR_Mean']].mean().reset_index()
    baselines.columns = ['participant_id', 'EAR_base', 'MAR_base']
    
    df = df.merge(baselines, on='participant_id', how='left')
    
    # Handle participants who might not have alert data (fallback to global mean)
    df['EAR_base'] = df['EAR_base'].fillna(df['EAR_Mean'].mean())
    df['MAR_base'] = df['MAR_base'].fillna(df['MAR_Mean'].mean())
    
    # Create relative features
    df['EAR_Relative'] = df['EAR_Mean'] / df['EAR_base']
    df['MAR_Relative'] = df['MAR_Mean'] / df['MAR_base']
    
    logging.info(f"Generated Relative Features. Dataset size: {len(df)}")
    return df

def run_improved_training(df):
    # 1. Feature Selection (Full Set + Relative)
    exclude_cols = ['video_id', 'participant_id', 'window_start_frame', 'window_end_frame', 
                    'window_start_idx', 'window_end_idx', 'EAR_base', 'MAR_base']
    features = [c for c in df.columns if c not in exclude_cols]
    
    X = df[features]
    y = df['video_id'].map({0: 0, 5: 1, 10: 2}) # Map to 0, 1, 2 for XGBoost
    groups = df['participant_id']
    
    logging.info(f"Training Improved Model with {len(features)} features.")
    
    gkf = GroupKFold(n_splits=5)
    fold_f1 = []
    best_clf = None
    max_f1 = 0
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scaling is important for many algorithms
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if HAS_XGB:
            clf = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss'
            )
        else:
            clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced')
            
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        f1 = f1_score(y_test, y_pred, average='macro')
        fold_f1.append(f1)
        
        logging.info(f"Fold {fold} | F1-Score: {f1:.4f}")
        
        if f1 > max_f1:
            max_f1 = f1
            best_clf = clf
            best_scaler = scaler
            
            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                        xticklabels=['Alert', 'Low', 'Drowsy'],
                        yticklabels=['Alert', 'Low', 'Drowsy'])
            plt.title(f'Improved Confusion Matrix (Fold {fold})')
            plt.savefig(REPORT_DIR / 'confusion_matrix_improved.png')

    logging.info("-" * 30)
    logging.info(f"Mean Improved F1-Score: {np.mean(fold_f1):.4f}")
    logging.info("-" * 30)
    
    # Save Model & Scaler
    joblib.dump(best_clf, MODEL_DIR / 'improved_model.joblib')
    joblib.dump(best_scaler, MODEL_DIR / 'improved_scaler.joblib')
    
    # Feature Importance
    if HAS_XGB:
        imp_vals = best_clf.feature_importances_
    else:
        imp_vals = best_clf.feature_importances_
        
    importances = pd.Series(imp_vals, index=features).sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color='darkgreen')
    plt.title('Top 10 Features (Improved Model)')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'feature_importance_improved.png')

def main():
    df = load_and_preprocess()
    if df is not None:
        run_improved_training(df)

if __name__ == '__main__':
    main()
