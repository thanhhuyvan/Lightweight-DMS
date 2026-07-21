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
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
VECTORS_CSV = PROJECT_ROOT / "frame" / "csv" / "behavioral_vectors.csv"
SUMMARY_CSV = PROJECT_ROOT / "frame" / "csv" / "features_summary.csv"
PATCH_ROOT = PROJECT_ROOT / "frame" / "patches"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report" / "cnn_patches"

PATCH_DIRS = {
    "left_eye": PATCH_ROOT / "left_eye",
    "right_eye": PATCH_ROOT / "right_eye",
    "mouth": PATCH_ROOT / "mouth",
}


# ---------------------------------------------------------------------------
# Augmentation transforms (applied per-frame patch, grayscale float [0,1])
# ---------------------------------------------------------------------------
_AUG_TRANSFORM = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3),
])


class PatchWindowDataset(Dataset):
    def __init__(
        self,
        vectors_csv,
        summary_csv,
        patch_root,
        task="binary",
        seq_len=40,
        min_valid_rate=0.8,
        max_windows=0,
        augment=False,
        exclude_participants=None,
    ):
        self.patch_root = Path(patch_root)
        self.seq_len = seq_len
        self.task = task
        self.min_valid_rate = min_valid_rate
        self.augment = augment
        self.exclude_participants = set(exclude_participants) if exclude_participants else set()

        self.windows = pd.read_csv(vectors_csv)
        self.frames = pd.read_csv(summary_csv)
        if task == "binary":
            self.windows = self.windows[self.windows["video_id"].isin([0, 10])].copy()
        if self.exclude_participants:
            self.windows = self.windows[~self.windows["participant_id"].isin(self.exclude_participants)].copy()
        self.windows = self.windows.reset_index(drop=True)

        self.frame_groups = {
            key: group.reset_index(drop=True)
            for key, group in self.frames.groupby(["video_id", "participant_id"], sort=False)
        }

        self.samples = self._build_index()
        if max_windows > 0:
            # Balanced class sampling: equal windows per class to avoid ordering bias.
            # CSV is sorted by video_id so naive [:N] would grab one class only.
            by_class: dict = {}
            for s in self.samples:
                vid = s["video_id"]
                by_class.setdefault(vid, []).append(s)
            per_class = max_windows // len(by_class)
            balanced = []
            rng = np.random.default_rng(42)
            for cls_samples in by_class.values():
                idx = rng.choice(len(cls_samples), size=min(per_class, len(cls_samples)), replace=False)
                balanced.extend([cls_samples[i] for i in idx])
            rng.shuffle(balanced)
            self.samples = balanced

        logging.info(
            "PatchWindowDataset: task=%s windows=%d min_valid_rate=%.2f augment=%s",
            task,
            len(self.samples),
            min_valid_rate,
            augment,
        )

    def _build_index(self):
        samples = []
        for window_idx, window in self.windows.iterrows():
            group = self.frame_groups.get((window["video_id"], window["participant_id"]))
            if group is None:
                continue

            start_idx = int(window["window_start_idx"])
            end_idx = int(window["window_end_idx"])
            frames = group.iloc[start_idx : end_idx + 1].head(self.seq_len)
            if frames.empty:
                continue

            expected = len(frames) * len(PATCH_DIRS)
            valid = 0
            for _, frame in frames.iterrows():
                filename = f"{frame['video_id']}_{frame['participant_id']}_{frame['frame_file']}"
                for patch_name in PATCH_DIRS:
                    if (self.patch_root / patch_name / filename).exists():
                        valid += 1

            valid_rate = valid / expected if expected else 0.0
            if valid_rate < self.min_valid_rate:
                continue

            samples.append(
                {
                    "window_idx": window_idx,
                    "participant_id": str(window["participant_id"]),
                    "video_id": int(window["video_id"]),
                    "valid_rate": valid_rate,
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def _read_patch(self, patch_name, filename):
        path = self.patch_root / patch_name / filename
        if not path.exists():
            return np.zeros((24, 24), dtype=np.float32), 0.0

        img_array = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((24, 24), dtype=np.float32), 0.0

        patch = img.astype(np.float32) / 255.0

        if self.augment:
            # Convert to (1, H, W) uint8 tensor for torchvision transforms
            t = torch.from_numpy((patch * 255).astype(np.uint8)).unsqueeze(0)
            t = _AUG_TRANSFORM(t)
            patch = t.squeeze(0).numpy().astype(np.float32) / 255.0

        return patch, 1.0

    def __getitem__(self, idx):
        sample = self.samples[idx]
        window = self.windows.loc[sample["window_idx"]]
        group = self.frame_groups[(window["video_id"], window["participant_id"])]

        start_idx = int(window["window_start_idx"])
        end_idx = int(window["window_end_idx"])
        frames = group.iloc[start_idx : end_idx + 1].head(self.seq_len)

        patch_sequence = []
        valid_sequence = []
        for _, frame in frames.iterrows():
            filename = f"{frame['video_id']}_{frame['participant_id']}_{frame['frame_file']}"
            channels = []
            valids = []
            for patch_name in ("left_eye", "right_eye", "mouth"):
                patch, valid = self._read_patch(patch_name, filename)
                channels.append(patch)
                valids.append(valid)
            patch_sequence.append(np.stack(channels, axis=0))
            valid_sequence.append(float(np.mean(valids)))  # soft mean: partial frames keep weight

        while len(patch_sequence) < self.seq_len:
            patch_sequence.append(np.zeros((3, 24, 24), dtype=np.float32))
            valid_sequence.append(0.0)

        video_id = int(window["video_id"])
        if self.task == "binary":
            label = 0 if video_id == 0 else 1
        else:
            label = {0: 0, 5: 1, 10: 2}[video_id]

        return {
            "patches": torch.from_numpy(np.stack(patch_sequence)).float(),
            "valid_mask": torch.tensor(valid_sequence, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }

    def groups(self):
        return np.array([sample["participant_id"] for sample in self.samples])


class TinyPatchCNN(nn.Module):
    def __init__(self, num_classes, embedding_dim=64, dropout=0.2):
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
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, patches, valid_mask):
        batch_size, seq_len = patches.shape[:2]
        x = patches.reshape(batch_size * seq_len, 3, 24, 24)
        embeddings = self.proj(self.encoder(x)).reshape(batch_size, seq_len, -1)

        mask = valid_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (embeddings * mask).sum(dim=1) / denom
        return self.classifier(pooled)


def compute_class_weights(dataset, indices, num_classes):
    labels = []
    for idx in indices:
        video_id = dataset.samples[idx]["video_id"]
        if dataset.task == "binary":
            labels.append(0 if video_id == 0 else 1)
        else:
            labels.append({0: 0, 5: 1, 10: 2}[video_id])

    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train(optimizer is not None)
    losses, labels, preds = [], [], []

    for batch in loader:
        patches = batch["patches"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        y = batch["label"].to(device)

        with torch.set_grad_enabled(optimizer is not None):
            logits = model(patches, valid_mask)
            loss = criterion(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        losses.append(float(loss.item()))
        labels.extend(y.detach().cpu().numpy())
        preds.extend(logits.argmax(dim=1).detach().cpu().numpy())

    drowsy_label = 1 if criterion.weight.numel() == 2 else 2
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy_score(labels, preds) if labels else 0.0,
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0) if labels else 0.0,
        "drowsy_recall": recall_score(labels, preds, labels=[drowsy_label], average="macro", zero_division=0) if labels else 0.0,
        "confusion": confusion_matrix(labels, preds).tolist() if labels else [],
    }


def train_split(dataset, train_idx, val_idx, args, fold_name):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    num_classes = 2 if args.task == "binary" else 3

    # Augmentation: only the training split sees transforms.
    # We temporarily flip the flag on the dataset because Subset shares the
    # same underlying Dataset object. We restore it before the val loader runs.
    train_subset = Subset(dataset, train_idx)
    val_subset   = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = TinyPatchCNN(num_classes=num_classes, dropout=args.dropout).to(device)
    class_weights = compute_class_weights(dataset, train_idx, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_metrics = None
    patience_left = args.patience
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Augmentation: on for train, guaranteed off for val via try/finally
        try:
            if args.augment:
                dataset.augment = True
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        finally:
            dataset.augment = False
        val_metrics = run_epoch(model, val_loader, criterion, None, device)
        logging.info(
            "%s epoch=%d train_f1=%.4f val_f1=%.4f val_acc=%.4f val_loss=%.4f",
            fold_name,
            epoch,
            train_metrics["macro_f1"],
            val_metrics["macro_f1"],
            val_metrics["accuracy"],
            val_metrics["loss"],
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            patience_left = args.patience
            torch.save(model.state_dict(), MODEL_DIR / f"cnn_patches_{args.task}_{fold_name}.pth")
        else:
            patience_left -= 1
            if patience_left <= 0:
                logging.info("%s early stopping at epoch %d", fold_name, epoch)
                break

    return best_metrics or {}


def run_overfit(dataset, args):
    indices = np.arange(len(dataset))
    if args.task == "binary":
        labels = np.array([0 if s["video_id"] == 0 else 1 for s in dataset.samples])
    else:
        labels = np.array([{0: 0, 5: 1, 10: 2}[s["video_id"]] for s in dataset.samples])
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    metrics = train_split(dataset, train_idx, val_idx, args, "overfit")
    logging.info("Overfit best metrics: %s", metrics)


def run_cv(dataset, args):
    groups = dataset.groups()
    indices = np.arange(len(dataset))
    gkf = GroupKFold(n_splits=args.folds)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups), 1):
        held_out = sorted(set(groups[val_idx]))
        logging.info("Fold %d held_out=%s", fold, held_out)
        metrics = train_split(dataset, train_idx, val_idx, args, f"fold{fold}")
        metrics["fold"] = fold
        metrics["held_out"] = held_out
        fold_metrics.append(metrics)

    f1s = [m.get("macro_f1", 0.0) for m in fold_metrics]
    logging.info("CV macro F1 mean=%.4f std=%.4f", float(np.mean(f1s)), float(np.std(f1s)))


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight CNN-only patch baseline.")
    parser.add_argument("--mode", choices=["overfit", "cv"], default="overfit")
    parser.add_argument("--task", choices=["binary", "three-class"], default="binary")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--min-valid-rate", type=float, default=0.8)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--exclude-participants", nargs="+", default=[],
                        help="List of participant IDs to exclude from the dataset entirely.")
    parser.add_argument("--augment", action="store_true",
                        help="Apply random flip + brightness/contrast jitter during training")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    task = "binary" if args.task == "binary" else "three-class"

    dataset = PatchWindowDataset(
        VECTORS_CSV,
        SUMMARY_CSV,
        PATCH_ROOT,
        task=task,
        seq_len=args.seq_len,
        min_valid_rate=args.min_valid_rate,
        max_windows=args.max_windows,
        augment=args.augment,
        exclude_participants=args.exclude_participants,
    )
    if len(dataset) == 0:
        raise SystemExit("No usable windows after filtering. Lower --min-valid-rate or inspect patches.")

    if args.mode == "overfit":
        run_overfit(dataset, args)
    else:
        run_cv(dataset, args)


if __name__ == "__main__":
    main()
