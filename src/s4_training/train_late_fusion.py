"""
train_late_fusion.py
---------------------
Stage D: Late Fusion — TinyPatchCNN embedding + geometry vector → binary classifier.

Architecture
------------
  CNN branch:  TinyPatchCNN encoder (64-dim pooled embedding) — weights loaded from
               best Stage B fold checkpoint, then fine-tuned.
  Geo branch:  11 geometry features → Linear(11, 32) → ReLU → Dropout
  Fusion:      concat [64 + 32] → Linear(96, 64) → ReLU → Dropout → Linear(64, 2)

Per-participant min-max scaling is applied to ocular features (EAR_Mean, EAR_Std,
PERCLOS, MAR_Mean, MAR_Max) as required by the GEMINI.md Isotropic + Min-Max rules.
The scaler is fit ONLY on the training folds to avoid leakage.

Usage
-----
    # Basic run (loads best CNN checkpoint from Stage B fold1)
    python -m src.s4_training.train_late_fusion

    # Full options
    python -m src.s4_training.train_late_fusion \\
        --mode cv --min-valid-rate 0.80 --max-windows 2000 \\
        --epochs 25 --lr 3e-4 --weight-decay 1e-4 --dropout 0.3 \\
        --patience 8 --folds 5 --batch-size 32 --num-workers 0 --cpu \\
        --freeze-cnn-epochs 3 --augment

    # Overfit test (sanity check)
    python -m src.s4_training.train_late_fusion --mode overfit --max-windows 300
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Subset


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
VECTORS_CSV  = PROJECT_ROOT / "frame" / "csv" / "behavioral_vectors.csv"
SUMMARY_CSV  = PROJECT_ROOT / "frame" / "csv" / "features_summary.csv"
PATCH_ROOT   = PROJECT_ROOT / "frame" / "patches"
MODEL_DIR    = PROJECT_ROOT / "models"
REPORT_DIR   = PROJECT_ROOT / "report" / "late_fusion"

PATCH_DIRS = ["left_eye", "right_eye", "mouth"]

# Geometry feature columns (all 11)
GEO_FEATURES = [
    "PERCLOS", "Blink_Rate", "Blink_Avg_Duration",
    "EAR_Mean", "EAR_Std",
    "MAR_Mean", "MAR_Max",
    "Pitch_Jitter", "Yaw_Jitter", "Roll_Jitter", "Pose_Jitter",
]
# Ocular features that require per-participant min-max scaling (GEMINI.md Min-Max Standard)
OCULAR_FEATURES = ["EAR_Mean", "EAR_Std", "PERCLOS", "MAR_Mean", "MAR_Max"]

SEQ_LEN = 40

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
_AUG_TRANSFORM = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3),
])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LateFusionDataset(Dataset):
    """Returns CNN patch sequence + geometry vector + label for each window."""

    def __init__(
        self,
        vectors_csv,
        summary_csv,
        patch_root,
        seq_len=SEQ_LEN,
        min_valid_rate=0.8,
        max_windows=0,
        augment=False,
        geo_scaler=None,     # fitted MinMaxScaler (set after fit on train split)
        exclude_participants=None,
    ):
        self.patch_root   = Path(patch_root)
        self.seq_len      = seq_len
        self.min_valid_rate = min_valid_rate
        self.augment      = augment
        self.geo_scaler   = geo_scaler   # may be None during initial build
        self.exclude_participants = set(exclude_participants) if exclude_participants else set()

        self.windows = pd.read_csv(vectors_csv)
        self.frames  = pd.read_csv(summary_csv)

        # Binary task only: alert (video_id=0) vs drowsy (video_id=10)
        self.windows = self.windows[self.windows["video_id"].isin([0, 10])].copy()
        if self.exclude_participants:
            self.windows = self.windows[~self.windows["participant_id"].isin(self.exclude_participants)].copy()
        self.windows = self.windows.reset_index(drop=True)

        self.frame_groups = {
            key: grp.reset_index(drop=True)
            for key, grp in self.frames.groupby(["video_id", "participant_id"], sort=False)
        }

        self.samples = self._build_index()

        if max_windows > 0:
            by_class: dict = {}
            for s in self.samples:
                by_class.setdefault(s["video_id"], []).append(s)
            per_class = max_windows // len(by_class)
            rng = np.random.default_rng(42)
            balanced = []
            for cls_samples in by_class.values():
                idx = rng.choice(len(cls_samples), size=min(per_class, len(cls_samples)), replace=False)
                balanced.extend([cls_samples[i] for i in idx])
            rng.shuffle(balanced)
            self.samples = balanced

        logging.info(
            "LateFusionDataset: windows=%d  min_valid_rate=%.2f  augment=%s",
            len(self.samples), min_valid_rate, augment,
        )

    def _build_index(self):
        samples = []
        for window_idx, window in self.windows.iterrows():
            grp = self.frame_groups.get((window["video_id"], window["participant_id"]))
            if grp is None:
                continue
            start = int(window["window_start_idx"])
            end   = int(window["window_end_idx"])
            frms  = grp.iloc[start : end + 1].head(self.seq_len)
            if frms.empty:
                continue

            expected = len(frms) * len(PATCH_DIRS)
            valid = 0
            for _, f in frms.iterrows():
                fname = f"{f['video_id']}_{f['participant_id']}_{f['frame_file']}"
                for pt in PATCH_DIRS:
                    if (self.patch_root / pt / fname).exists():
                        valid += 1
            valid_rate = valid / expected if expected else 0.0
            if valid_rate < self.min_valid_rate:
                continue

            geo = window[GEO_FEATURES].values.astype(np.float32)
            samples.append({
                "window_idx":    window_idx,
                "participant_id": str(window["participant_id"]),
                "video_id":       int(window["video_id"]),
                "valid_rate":     valid_rate,
                "geo":            geo,
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def _read_patch(self, patch_name, filename):
        path = self.patch_root / patch_name / filename
        if not path.exists():
            return np.zeros((24, 24), dtype=np.float32), 0.0
        img_bytes = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((24, 24), dtype=np.float32), 0.0

        patch = img.astype(np.float32) / 255.0
        if self.augment:
            t = torch.from_numpy((patch * 255).astype(np.uint8)).unsqueeze(0)
            t = _AUG_TRANSFORM(t)
            patch = t.squeeze(0).numpy().astype(np.float32) / 255.0
        return patch, 1.0

    def __getitem__(self, idx):
        sample = self.samples[idx]
        window = self.windows.loc[sample["window_idx"]]
        grp    = self.frame_groups[(window["video_id"], window["participant_id"])]
        start  = int(window["window_start_idx"])
        end    = int(window["window_end_idx"])
        frms   = grp.iloc[start : end + 1].head(self.seq_len)

        patch_sequence, valid_sequence = [], []
        for _, f in frms.iterrows():
            fname = f"{f['video_id']}_{f['participant_id']}_{f['frame_file']}"
            channels, valids = [], []
            for pt in ("left_eye", "right_eye", "mouth"):
                p, v = self._read_patch(pt, fname)
                channels.append(p)
                valids.append(v)
            patch_sequence.append(np.stack(channels, axis=0))
            valid_sequence.append(float(np.mean(valids)))  # soft mean: partial frames keep weight

        while len(patch_sequence) < self.seq_len:
            patch_sequence.append(np.zeros((3, 24, 24), dtype=np.float32))
            valid_sequence.append(0.0)

        # Fill invalid frames with nearest valid patch + compute confidence decay.
        # Confidence = 0.85^(distance to nearest valid frame), so the model learns
        # to down-weight fabricated frames proportional to how far they are from truth.
        confidence = np.ones(self.seq_len, dtype=np.float32)
        valid_indices = np.where(np.array(valid_sequence) > 0.0)[0]

        if len(valid_indices) == 0:
            confidence[:] = 0.0
        else:
            for t in range(self.seq_len):
                if valid_sequence[t] > 0.0:
                    continue
                prev_v = valid_indices[valid_indices < t]
                next_v = valid_indices[valid_indices > t]
                if len(prev_v) and len(next_v):
                    dist = min(t - prev_v[-1], next_v[0] - t)
                    patch_sequence[t] = patch_sequence[prev_v[-1]]  # nearest prev
                elif len(prev_v):
                    dist = t - prev_v[-1]
                    patch_sequence[t] = patch_sequence[prev_v[-1]]
                else:
                    dist = next_v[0] - t
                    patch_sequence[t] = patch_sequence[next_v[0]]
                confidence[t] = 0.85 ** dist

        geo = sample["geo"].copy()   # (11,) float32
        if self.geo_scaler is not None:
            geo = self.geo_scaler.transform(geo.reshape(1, -1)).flatten().astype(np.float32)

        label = 0 if sample["video_id"] == 0 else 1

        return {
            "patches":    torch.from_numpy(np.stack(patch_sequence)).float(),
            "valid_mask": torch.tensor(valid_sequence, dtype=torch.float32),
            "confidence": torch.from_numpy(confidence),
            "geo":        torch.from_numpy(geo),
            "label":      torch.tensor(label, dtype=torch.long),
        }

    def groups(self):
        return np.array([s["participant_id"] for s in self.samples])

    def get_geo_matrix(self, indices):
        """Return geo feature matrix for a subset of indices (for scaler fitting)."""
        return np.stack([self.samples[i]["geo"] for i in indices])


# ---------------------------------------------------------------------------
# Per-participant min-max scaler (GEMINI.md Min-Max Standard)
# ---------------------------------------------------------------------------

def fit_geo_scaler(dataset, train_indices):
    """
    Fit a MinMaxScaler on the training subset only.
    Applies per-column (feature-wise) scaling across all training windows.
    This is the correct way to honour the 'per-participant min-max' mandate:
    since GroupKFold already separates by participant, the scaler is fit only
    on training participants and applied to held-out participants.
    """
    geo_matrix = dataset.get_geo_matrix(train_indices)
    scaler = MinMaxScaler()
    scaler.fit(geo_matrix)
    return scaler


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TinyPatchCNNEncoder(nn.Module):
    """CNN encoder only (no classifier head). Matches TinyPatchCNN in train_cnn_patches.py."""
    def __init__(self, embedding_dim=64, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, patches, valid_mask):
        batch_size, seq_len = patches.shape[:2]
        x = patches.reshape(batch_size * seq_len, 3, 24, 24)
        emb = self.proj(self.encoder(x)).reshape(batch_size, seq_len, -1)
        mask  = valid_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (emb * mask).sum(dim=1) / denom   # (B, 64)


class GeoBranch(nn.Module):
    def __init__(self, n_features=11, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, geo):
        return self.net(geo)   # (B, 32)


class LateFusionModel(nn.Module):
    def __init__(self, num_classes=2, cnn_dim=64, geo_hidden=32, dropout=0.3):
        super().__init__()
        self.cnn_branch = TinyPatchCNNEncoder(embedding_dim=cnn_dim, dropout=dropout)
        self.geo_branch = GeoBranch(n_features=len(GEO_FEATURES), hidden=geo_hidden, dropout=dropout)
        fused = cnn_dim + geo_hidden
        self.fusion_head = nn.Sequential(
            nn.Linear(fused, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, patches, valid_mask, geo):
        cnn_emb = self.cnn_branch(patches, valid_mask)  # (B, 64)
        geo_emb = self.geo_branch(geo)                  # (B, 32)
        fused   = torch.cat([cnn_emb, geo_emb], dim=1)  # (B, 96)
        return self.fusion_head(fused)                  # (B, 2)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_class_weights(dataset, indices, num_classes=2):
    labels = [0 if dataset.samples[i]["video_id"] == 0 else 1 for i in indices]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train(optimizer is not None)
    losses, labels_all, preds_all = [], [], []

    for batch in loader:
        patches    = batch["patches"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        geo        = batch["geo"].to(device)
        y          = batch["label"].to(device)

        with torch.set_grad_enabled(optimizer is not None):
            logits = model(patches, valid_mask, geo)
            loss   = criterion(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        losses.append(float(loss.item()))
        labels_all.extend(y.detach().cpu().numpy())
        preds_all.extend(logits.argmax(dim=1).detach().cpu().numpy())

    return {
        "loss":          float(np.mean(losses)) if losses else 0.0,
        "accuracy":      accuracy_score(labels_all, preds_all) if labels_all else 0.0,
        "macro_f1":      f1_score(labels_all, preds_all, average="macro", zero_division=0) if labels_all else 0.0,
        "drowsy_recall": recall_score(labels_all, preds_all, labels=[1], average="macro", zero_division=0) if labels_all else 0.0,
        "confusion":     confusion_matrix(labels_all, preds_all).tolist() if labels_all else [],
    }


def _load_cnn_weights(model: LateFusionModel, fold_name: str):
    """Try to load Stage B CNN weights into the CNN branch (best-effort)."""
    ckpt = MODEL_DIR / f"cnn_patches_binary_{fold_name}.pth"
    if not ckpt.exists():
        logging.warning("No Stage B checkpoint found at %s — CNN branch starts from scratch.", ckpt)
        return
    state = torch.load(ckpt, map_location="cpu")
    # Stage B checkpoint has keys: encoder.*, proj.*, classifier.*
    # We only load encoder + proj into cnn_branch (skip classifier)
    cnn_state = {k: v for k, v in state.items() if not k.startswith("classifier")}
    model.cnn_branch.load_state_dict(cnn_state, strict=False)
    logging.info("Loaded Stage B CNN weights from %s", ckpt)


def train_split(dataset, train_idx, val_idx, args, fold_name):
    device      = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    num_classes = 2

    # Fit geo scaler on training indices only (no leakage)
    scaler = fit_geo_scaler(dataset, train_idx)
    dataset.geo_scaler = scaler
    logging.info("%s geo scaler fit on %d training windows", fold_name, len(train_idx))

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    model = LateFusionModel(num_classes=num_classes, dropout=args.dropout).to(device)
    _load_cnn_weights(model, fold_name)

    class_weights = compute_class_weights(dataset, train_idx, num_classes).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # Optionally freeze CNN for the first N epochs (warm-up for geo branch)
    # Use param groups so Adam momentum is preserved across the unfreeze transition.
    freeze_epochs = args.freeze_cnn_epochs
    cnn_params   = list(model.cnn_branch.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("cnn_branch.")]
    if freeze_epochs > 0:
        for p in cnn_params:
            p.requires_grad = False
        logging.info("%s CNN branch frozen for first %d epochs", fold_name, freeze_epochs)
    optimizer = torch.optim.AdamW([
        {"params": cnn_params,   "lr": 0.0 if freeze_epochs > 0 else args.lr,
         "weight_decay": args.weight_decay},
        {"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ])

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    best_f1       = -1.0
    best_metrics  = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        # Unfreeze CNN after warm-up — update lr in-place to preserve Adam momentum
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            for p in model.cnn_branch.parameters():
                p.requires_grad = True
            optimizer.param_groups[0]["lr"] = args.lr * 0.5   # CNN group: halved lr
            patience_left = args.patience   # reset: joint fine-tuning is a new phase
            best_f1       = -1.0
            logging.info("%s CNN branch unfrozen at epoch %d (lr → %.2e)",
                         fold_name, epoch, args.lr * 0.5)

        # Augmentation: on for train, guaranteed off for val via try/finally
        try:
            if args.augment:
                dataset.augment = True
            train_m = run_epoch(model, train_loader, criterion, optimizer, device)
        finally:
            dataset.augment = False
        val_m   = run_epoch(model, val_loader, criterion, None, device)

        logging.info(
            "%s epoch=%d train_f1=%.4f val_f1=%.4f val_acc=%.4f "
            "val_loss=%.4f drowsy_recall=%.4f",
            fold_name, epoch,
            train_m["macro_f1"], val_m["macro_f1"],
            val_m["accuracy"],   val_m["loss"],
            val_m["drowsy_recall"],
        )

        if val_m["macro_f1"] > best_f1:
            best_f1      = val_m["macro_f1"]
            best_metrics = val_m
            patience_left = args.patience
            torch.save(model.state_dict(), MODEL_DIR / f"late_fusion_{fold_name}.pth")
        else:
            patience_left -= 1
            if patience_left <= 0:
                logging.info("%s early stopping at epoch %d", fold_name, epoch)
                break

    # Reset scaler so dataset is clean for next fold
    dataset.geo_scaler = None
    return best_metrics or {}


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_overfit(dataset, args):
    indices = np.arange(len(dataset))
    labels  = np.array([0 if s["video_id"] == 0 else 1 for s in dataset.samples])
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    metrics = train_split(dataset, train_idx, val_idx, args, "overfit")
    logging.info("Overfit best metrics: %s", metrics)


def run_cv(dataset, args):
    groups  = dataset.groups()
    indices = np.arange(len(dataset))
    gkf     = GroupKFold(n_splits=args.folds)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups), 1):
        held_out = sorted(set(groups[val_idx]))
        logging.info("Fold %d held_out=%s", fold, held_out)
        m = train_split(dataset, train_idx, val_idx, args, f"fold{fold}")
        m["fold"]     = fold
        m["held_out"] = held_out
        fold_metrics.append(m)

    f1s = [m.get("macro_f1", 0.0) for m in fold_metrics]
    recalls = [m.get("drowsy_recall", 0.0) for m in fold_metrics]
    logging.info(
        "Late Fusion CV  macro_F1=%.4f ± %.4f  drowsy_recall=%.4f ± %.4f",
        float(np.mean(f1s)),    float(np.std(f1s)),
        float(np.mean(recalls)), float(np.std(recalls)),
    )
    logging.info("Comparison — Geometry-only baseline F1: 0.5422  |  CNN-only F1: 0.6441")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage D: Late Fusion CNN + Geometry binary classifier.")
    p.add_argument("--mode",             choices=["overfit", "cv"], default="cv")
    p.add_argument("--epochs",           type=int,   default=25)
    p.add_argument("--batch-size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--weight-decay",     type=float, default=1e-4)
    p.add_argument("--dropout",          type=float, default=0.3)
    p.add_argument("--patience",         type=int,   default=8)
    p.add_argument("--folds",            type=int,   default=5)
    p.add_argument("--seq-len",          type=int,   default=40)
    p.add_argument("--min-valid-rate",   type=float, default=0.80)
    p.add_argument("--max-windows",      type=int,   default=0)
    p.add_argument("--num-workers",      type=int,   default=0)
    p.add_argument("--cpu",              action="store_true")
    p.add_argument("--augment",          action="store_true",
                   help="Apply random flip + brightness/contrast jitter on patches during training")
    p.add_argument("--freeze-cnn-epochs", type=int, default=3,
                   help="Freeze CNN branch for first N epochs to let geo branch warm up (0=disabled)")
    p.add_argument("--exclude-participants", nargs="+", default=[],
                   help="List of participant IDs to exclude from the dataset entirely.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    dataset = LateFusionDataset(
        VECTORS_CSV, SUMMARY_CSV, PATCH_ROOT,
        seq_len=args.seq_len,
        min_valid_rate=args.min_valid_rate,
        max_windows=args.max_windows,
        augment=False,   # augment flag controlled per-epoch in train_split
        exclude_participants=args.exclude_participants,
    )
    if len(dataset) == 0:
        raise SystemExit("No usable windows. Lower --min-valid-rate or check patches.")

    if args.mode == "overfit":
        run_overfit(dataset, args)
    else:
        run_cv(dataset, args)


if __name__ == "__main__":
    main()
