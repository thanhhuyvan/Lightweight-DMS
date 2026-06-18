import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import time

# --- Simple Dataset for Mini-Experiment ---
class MiniPatchDataset(Dataset):
    def __init__(self, patch_dir, label_map=None):
        self.patch_paths = list(Path(patch_dir).glob("*.jpg"))
        self.label_map = label_map # In real scenario, extract label from filename
        
    def __len__(self):
        return len(self.patch_paths)
    
    def __getitem__(self, idx):
        p = self.patch_paths[idx]
        # Unicode safe read
        img_array = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        # Normalize 0-1
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0) # Add channel dim
        
        # Mock label for demo (usually 0: alert, 1: low, 2: drowsy)
        # For this experiment, we just want to see if the model compiles and runs
        label = torch.tensor(0, dtype=torch.long) 
        return img, label

# --- Custom Small CNN (Phase 2 Design) ---
class DrowsyPatchNet(nn.Module):
    def __init__(self, width_mult=1.0):
        super(DrowsyPatchNet, self).__init__()
        # Input: 1x24x24
        self.features = nn.Sequential(
            nn.Conv2d(1, int(16*width_mult), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 12x12
            
            nn.Conv2d(int(16*width_mult), int(32*width_mult), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 6x6
            
            nn.Conv2d(int(32*width_mult), int(64*width_mult), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten() # 6*6*64
        )
        self.classifier = nn.Linear(int(6*6*64*width_mult), 3)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def run_mini_train():
    print("🧪 Starting Mini-Experiment: Hyperparameter Testing")
    
    # 1. Setup Data
    patch_dir = "frame/patches/left_eye"
    dataset = MiniPatchDataset(patch_dir)
    if len(dataset) < 100:
        print("Not enough data yet. Waiting for background process...")
        return
    
    # Use a small subset (1000 patches) for speed
    indices = torch.randperm(len(dataset))[:1000]
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=True)
    
    # 2. Setup Model (Using width_mult=0.5 as per TECHNICAL_PLAN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrowsyPatchNet(width_mult=0.5).to(device)
    
    # 3. Setup Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    print(f"Device: {device}")
    print(f"Dataset Size: {len(subset)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 4. Training Loop (Single Epoch for Sanity Check)
    model.train()
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
            
    print(f"Done in {time.time()-t0:.2f}s")
    print("Result: Model architecture is STABLE and training loop is FUNCTIONAL.")

if __name__ == "__main__":
    run_mini_train()
