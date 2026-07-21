"""
train_hybrid.py — Phase 2: Hybrid Evolution Training

Goal: Train the FiLM-CNN-GRU network on synchronized Multimodal Data.
Features: 
- Geometric-Steered Vision (FiLM)
- Temporal Modeling (GRU)
- Zero Participant Leakage (GroupKFold)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.s3_models.models_hybrid import HybridNet
from src.s3_models.hybrid_dataset import HybridSequenceDataset
from src.core_config import MODEL_SAVE_DIR, PROJECT_ROOT

# Settings
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        l, r = batch['l_eye'].to(DEVICE), batch['r_eye'].to(DEVICE)
        m, g = batch['mouth'].to(DEVICE), batch['geometry'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(l, r, m, g)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(loader), correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            l, r = batch['l_eye'].to(DEVICE), batch['r_eye'].to(DEVICE)
            m, g = batch['mouth'].to(DEVICE), batch['geometry'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(l, r, m, g)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), correct / total

def run_hybrid_training():
    logging.info(f"🚀 Starting Hybrid Evolution Training on {DEVICE}")
    
    # 1. Prepare Dataset
    behavioral_csv = "frame/csv/behavioral_vectors.csv"
    summary_csv = "frame/csv/features_summary.csv"
    patch_root = "frame/patches"
    
    full_dataset = HybridSequenceDataset(behavioral_csv, summary_csv, patch_root)
    
    # 2. GroupKFold (Ensure zero participant leakage)
    # We use the participant_id from the behavioral_vectors.csv
    df_windows = pd.read_csv(behavioral_csv)
    groups = df_windows['participant_id']
    gkf = GroupKFold(n_splits=5)
    
    # For this official "startup", we'll run Fold 1
    train_idx, val_idx = next(gkf.split(np.arange(len(groups)), groups=groups))
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model, Optimizer, Criterion
    model = HybridNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc = validate(model, val_loader, criterion)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        
        logging.info(f"Epoch {epoch}/{EPOCHS} | T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | V-Acc: {v_acc:.4f}")
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), MODEL_SAVE_DIR / 'hybrid_model_best.pth')
            logging.info(f"✨ New Best Model Saved (V-Acc: {best_acc:.4f})")

    # 5. Plot Results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss Trend')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy Trend')
    plt.legend()
    
    REPORT_DIR = PROJECT_ROOT / 'report' / 'hybrid'
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORT_DIR / 'training_curves.png')
    logging.info(f"📈 Training curves saved to {REPORT_DIR}")

if __name__ == "__main__":
    run_hybrid_training()
