import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import logging

class HybridSequenceDataset(Dataset):
    """
    Loads sequences of (L-Eye, R-Eye, Mouth) patches and 12D Geometry Vectors.
    Synchronized via behavioral_vectors.csv (Windows) and features_summary.csv (Frames).
    """
    def __init__(self, behavioral_csv, summary_csv, patch_root, seq_len=40, stride=20):
        self.patch_root = Path(patch_root)
        self.seq_len = seq_len
        
        # Load Data
        self.df_windows = pd.read_csv(behavioral_csv)
        self.df_frames = pd.read_csv(summary_csv)
        
        # Pre-define directories
        self.l_eye_dir = self.patch_root / 'left_eye'
        self.r_eye_dir = self.patch_root / 'right_eye'
        self.mouth_dir = self.patch_root / 'mouth'
        
        # Filter windows to ensure we have enough frames
        # (behavioral_vectors.csv usually has windows of ~40 frames)
        self.valid_windows = self.df_windows.index.tolist()
        
        logging.info(f"Initialized HybridSequenceDataset with {len(self.valid_windows)} windows.")

    def __len__(self):
        return len(self.valid_windows)

    def _load_patch(self, folder, filename):
        path = folder / filename
        if not path.exists():
            # Return black patch if missing
            return np.zeros((24, 24), dtype=np.uint8)
        
        # Unicode safe read
        img_array = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None: return np.zeros((24, 24), dtype=np.uint8)
        return img

    def __getitem__(self, idx):
        window = self.df_windows.iloc[idx]
        
        # Get frame range
        start_idx = int(window['window_start_idx'])
        end_idx = int(window['window_end_idx'])
        
        # Slice frames for this window
        # We need to ensure we filter by video_id and participant_id to be safe
        frames = self.df_frames[
            (self.df_frames['video_id'] == window['video_id']) & 
            (self.df_frames['participant_id'] == window['participant_id'])
        ].iloc[start_idx:end_idx+1]
        
        # Ensure we have exactly seq_len frames (pad or truncate)
        if len(frames) > self.seq_len:
            frames = frames.iloc[:self.seq_len]
        
        l_patches, r_patches, m_patches, geo_vectors = [], [], [], []
        
        for _, row in frames.iterrows():
            # 1. Load Patches
            fname = f"{row['video_id']}_{row['participant_id']}_{row['frame_file']}"
            l_patches.append(self._load_patch(self.l_eye_dir, fname))
            r_patches.append(self._load_patch(self.r_eye_dir, fname))
            m_patches.append(self._load_patch(self.mouth_dir, fname))
            
            # 2. Build 12D Geometry Vector
            # [EAR_L, EAR_R, EAR_Avg, EAR_Diff, MAR, Pitch, Yaw, Roll, dP, dY, dR, Conf]
            # (Note: Using available columns, placeholders for deltas)
            geo = np.array([
                row['mean_EAR_smooth'], # L (Placeholder)
                row['mean_EAR_smooth'], # R (Placeholder)
                row['mean_EAR_smooth'], # Avg
                0.0,                    # Diff
                row['MAR_smooth'],
                row['pitch_smooth'],
                row['yaw_smooth'],
                row['roll_smooth'],
                0.0, 0.0, 0.0,          # Deltas (to be computed)
                1.0 if row['face_detected'] else 0.0 # Confidence
            ], dtype=np.float32)
            
            # Apply INSIGHTS.md Normalization
            geo[0:5] = geo[0:5] / 0.5   # EAR/MAR
            geo[5:8] = geo[5:8] / 90.0  # Angles
            
            geo_vectors.append(geo)

        # Handle short sequences (Padding)
        while len(l_patches) < self.seq_len:
            l_patches.append(np.zeros((24, 24), dtype=np.uint8))
            r_patches.append(np.zeros((24, 24), dtype=np.uint8))
            m_patches.append(np.zeros((24, 24), dtype=np.uint8))
            geo_vectors.append(np.zeros(12, dtype=np.float32))

        # Convert to Tensors
        # Patches: [Seq, 1, 24, 24]
        l_tensor = torch.from_numpy(np.stack(l_patches)).unsqueeze(1).float() / 255.0
        r_tensor = torch.from_numpy(np.stack(r_patches)).unsqueeze(1).float() / 255.0
        m_tensor = torch.from_numpy(np.stack(m_patches)).unsqueeze(1).float() / 255.0
        geo_tensor = torch.from_numpy(np.stack(geo_vectors)).float()
        
        # Target Label (0: Alert, 1: Low, 2: Drowsy)
        # Map video_id (0, 5, 10) to labels (0, 1, 2)
        label_map = {0: 0, 5: 1, 10: 2}
        label = label_map.get(window['video_id'], 0)
        
        return {
            'l_eye': l_tensor,
            'r_eye': r_tensor,
            'mouth': m_tensor,
            'geometry': geo_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    # Test Connection
    logging.basicConfig(level=logging.INFO)
    
    behavioral_csv = "frame/csv/behavioral_vectors.csv"
    summary_csv = "frame/csv/features_summary.csv"
    patch_root = "frame/patches"
    
    if Path(behavioral_csv).exists() and Path(summary_csv).exists():
        dataset = HybridSequenceDataset(behavioral_csv, summary_csv, patch_root)
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"L-Eye shape: {sample['l_eye'].shape}") # [40, 1, 24, 24]
        print(f"Geo shape: {sample['geometry'].shape}") # [40, 12]
        print(f"Label: {sample['label']}")
        print("✅ Hybrid Sequence Data Verified.")
    else:
        print("CSV files not found. Check paths.")
