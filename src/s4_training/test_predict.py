import joblib
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
VECTORS_CSV = PROJECT_ROOT / 'frame' / 'csv' / 'behavioral_vectors.csv'

def main():
    df = pd.read_csv(VECTORS_CSV)
    
    # Preprocess as in train_improved
    df_imp = df.copy()
    baselines = df_imp[df_imp['video_id'] == 0].groupby('participant_id')[['EAR_Mean', 'MAR_Mean']].mean().reset_index()
    baselines.columns = ['participant_id', 'EAR_base', 'MAR_base']
    df_imp = df_imp.merge(baselines, on='participant_id', how='left')
    df_imp['EAR_base'] = df_imp['EAR_base'].fillna(df_imp['EAR_Mean'].mean())
    df_imp['MAR_base'] = df_imp['MAR_base'].fillna(df_imp['MAR_Mean'].mean())
    df_imp['EAR_Relative'] = df_imp['EAR_Mean'] / df_imp['EAR_base']
    df_imp['MAR_Relative'] = df_imp['MAR_Mean'] / df_imp['MAR_base']
    
    # Preprocess as in train_final
    df_final = df.copy()
    processed_groups = []
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
    for p_id, group in df_final.groupby('participant_id'):
        processed_groups.append(min_max_scale_group(group.copy()))
    df_final = pd.concat(processed_groups, ignore_index=True)
    df_final = df_final.sort_values(['participant_id', 'video_id', 'window_start_idx'])
    df_final['EAR_Mean_Norm_Delta'] = df_final.groupby(['participant_id', 'video_id'])['EAR_Mean_Norm'].diff(periods=3).fillna(0)
    df_final['PERCLOS_Delta'] = df_final.groupby(['participant_id', 'video_id'])['PERCLOS'].diff(periods=3).fillna(0)
    df_final['Pose_Jitter_Norm_Delta'] = df_final.groupby(['participant_id', 'video_id'])['Pose_Jitter_Norm'].diff(periods=3).fillna(0)
    
    models_dir = PROJECT_ROOT / 'src' / 'models'
    
    # Define possible feature configurations
    # 1. 11 features: standard features
    feat_11_std = ['PERCLOS', 'Blink_Rate', 'Blink_Avg_Duration', 'EAR_Mean', 'EAR_Std', 
                   'MAR_Mean', 'MAR_Max', 'Pitch_Jitter', 'Yaw_Jitter', 'Roll_Jitter', 'Pose_Jitter']
    
    # 2. 13 features: improved features (std + relative)
    feat_13 = feat_11_std + ['EAR_Relative', 'MAR_Relative']
    
    # 3. 11 features: final features (from final script, maybe including MAR_Mean_Norm?)
    # Let's try 11 features from final
    feat_11_final = ['PERCLOS', 'Blink_Rate_Norm', 'EAR_Mean_Norm', 'MAR_Mean_Norm', 'EAR_Std', 
                     'Pitch_Jitter', 'Yaw_Jitter', 'Pose_Jitter_Norm',
                     'EAR_Mean_Norm_Delta', 'PERCLOS_Delta', 'Pose_Jitter_Norm_Delta']
    
    # 4. 12 features: final pose features (12 features)
    # Could be final features + Roll_Jitter or MAR_Mean_Norm?
    feat_12 = ['PERCLOS', 'Blink_Rate_Norm', 'EAR_Mean_Norm', 'MAR_Mean_Norm', 'EAR_Std', 
               'Pitch_Jitter', 'Yaw_Jitter', 'Roll_Jitter', 'Pose_Jitter_Norm',
               'EAR_Mean_Norm_Delta', 'PERCLOS_Delta', 'Pose_Jitter_Norm_Delta']
    
    # Test models
    models = {
        'baseline_rf_model': (models_dir / 'baseline_rf_model.joblib', df, feat_11_std, None),
        'improved_model': (models_dir / 'improved_model.joblib', df_imp, feat_13, models_dir / 'improved_scaler.joblib'),
        'final_xgb_model': (models_dir / 'final_xgb_model.joblib', df_final, feat_11_final, None),
        'final_xgb_pose_model': (models_dir / 'final_xgb_pose_model.joblib', df_final, feat_12, None)
    }
    
    for name, (path, data, feats, scaler_path) in models.items():
        if path.exists():
            model = joblib.load(path)
            X = data[feats].values
            if scaler_path and scaler_path.exists():
                scaler = joblib.load(scaler_path)
                X = scaler.transform(X)
            
            try:
                preds = model.predict(X)
                print(f"Model {name} predict success. Unique predictions: {np.unique(preds)}")
            except Exception as e:
                print(f"Model {name} predict failed: {e}")
                
if __name__ == '__main__':
    main()
