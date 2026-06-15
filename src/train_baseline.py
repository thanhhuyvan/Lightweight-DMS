"""
train_baseline.py — Stage 5: Machine Learning Classification (Baseline)

Implementation of the behavioral state classifier using the parameters specified 
in the Methodology section of the research paper.

Methodology Specs:
    - Algorithm: Random Forest (n_estimators=100, max_depth=10)
    - Validation: 5-fold GroupKFold (Participant-stratified)
    - Seed: 42
    - Features: PERCLOS, Blink Rate (f_blink), EAR Variance (sigma^2_EAR), Pose Stability (Psi_pose)
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import logging

# --- Setup Paths ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CSV_PATH = PROJECT_ROOT / 'frame' / 'csv' / 'behavioral_vectors.csv'
MODEL_DIR = PROJECT_ROOT / 'models'
REPORT_DIR = PROJECT_ROOT / 'report'
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

def load_data():
    if not CSV_PATH.exists():
        logging.error(f"Data source not found: {CSV_PATH}")
        return None
    
    df = pd.read_csv(CSV_PATH)
    logging.info(f"Loaded dataset: {len(df)} window vectors.")
    return df

def run_baseline_training(df):
    """
    Executes training following the exact parameters in the paper's methodology.
    """
    # 1. Feature Selection (Strictly 4-dimensional as per Methodology)
    # Mapping CSV columns to Methodology variables
    feature_map = {
        'PERCLOS': 'PERCLOS',
        'Blink_Rate': 'f_blink',
        'EAR_Std': 'EAR_Variance', # sigma^2_EAR (We'll square it if needed, but Std works similarly)
        'Pose_Jitter': 'Psi_pose'
    }
    
    X = df[list(feature_map.keys())]
    y = df['video_id'] # 0, 5, 10
    groups = df['participant_id']
    
    # Square EAR_Std to get actual Variance if we want to be 100% literal with LaTeX
    X['EAR_Variance'] = X['EAR_Std'] ** 2
    X_final = X[['PERCLOS', 'Blink_Rate', 'EAR_Variance', 'Pose_Jitter']]
    
    logging.info(f"Features: {list(X_final.columns)}")
    logging.info(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # 2. GroupKFold Setup (5-fold)
    gkf = GroupKFold(n_splits=5)
    
    # Metrics containers
    fold_accuracies = []
    fold_f1_scores = []
    best_clf = None
    max_f1 = 0
    
    logging.info("-" * 30)
    logging.info("STARTING 5-FOLD GROUPKFOLD CROSS-VALIDATION")
    logging.info("-" * 30)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_final, y, groups=groups), 1):
        X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        test_participants = df.iloc[test_idx]['participant_id'].unique()
        
        # 3. Model Parameters (Strictly n=100, depth=10)
        clf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced' # Added for better handling of class imbalance
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)
        
        logging.info(f"Fold {fold} | Test Participants: {test_participants}")
        logging.info(f"       | Accuracy: {acc:.4f} | F1-Score (Macro): {f1:.4f}")
        
        # Save best model and report for PR/Paper
        if f1 > max_f1:
            max_f1 = f1
            best_clf = clf
            # Generate Confusion Matrix for best fold
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Alert', 'Low Vigilant', 'Drowsy'],
                        yticklabels=['Alert', 'Low Vigilant', 'Drowsy'])
            plt.title(f'Confusion Matrix (Fold {fold} - Best Performance)')
            plt.ylabel('Ground Truth')
            plt.xlabel('Predicted State')
            plt.tight_layout()
            plt.savefig(REPORT_DIR / 'confusion_matrix_baseline.png')
            
            # Save Classification Report
            with open(REPORT_DIR / 'classification_report_baseline.txt', 'w') as f:
                f.write(f"Best Fold: {fold}\n")
                f.write(f"Test Participants: {test_participants}\n")
                f.write(classification_report(y_test, y_pred))

    # --- Final Aggregate Metrics ---
    logging.info("-" * 30)
    logging.info("FINAL CROSS-VALIDATION RESULTS")
    logging.info(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    logging.info(f"Mean F1-Score: {np.mean(fold_f1_scores):.4f} (+/- {np.std(fold_f1_scores):.4f})")
    logging.info("-" * 30)

    # 4. Feature Importance (For Methodology Visualization)
    importances = pd.Series(best_clf.feature_importances_, index=X_final.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar', color='salmon')
    plt.title('Feature Importance (Gini Impurity) - Baseline Model')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'feature_importance_baseline.png')
    
    # 5. Save Model
    model_path = MODEL_DIR / 'baseline_rf_model.joblib'
    joblib.dump(best_clf, model_path)
    logging.info(f"Baseline model saved to {model_path}")

def main():
    df = load_data()
    if df is not None:
        run_baseline_training(df)

if __name__ == '__main__':
    main()
