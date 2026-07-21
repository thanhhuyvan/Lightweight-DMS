import joblib
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
VECTORS_CSV = PROJECT_ROOT / 'frame' / 'csv' / 'behavioral_vectors.csv'

def main():
    df = pd.read_csv(VECTORS_CSV)
    print("CSV columns:", df.columns.tolist())
    
    # Let's load the models
    models_dir = PROJECT_ROOT / 'src' / 'models'
    
    models = {
        'baseline_rf_model': models_dir / 'baseline_rf_model.joblib',
        'improved_model': models_dir / 'improved_model.joblib',
        'final_xgb_model': models_dir / 'final_xgb_model.joblib',
        'final_xgb_pose_model': models_dir / 'final_xgb_pose_model.joblib'
    }
    
    for name, path in models.items():
        if path.exists():
            model = joblib.load(path)
            print(f"\nModel: {name}")
            print(f"Type: {type(model)}")
            print(f"n_features_in_: {getattr(model, 'n_features_in_', 'N/A')}")
            if hasattr(model, 'classes_'):
                print(f"classes_: {model.classes_}")
        else:
            print(f"\nModel: {name} NOT FOUND at {path}")

if __name__ == '__main__':
    main()
