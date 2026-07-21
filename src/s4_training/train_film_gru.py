"""
train_film_gru.py
------------------
Stage E: FiLM + GRU — temporal drowsiness classifier with geometry-conditioned CNN.

Architecture
------------
  CNN encoder : TinyPatchCNN encoder → per-frame 64-dim embeddings (seq_len × 64)
  Geo encoder : 11 geo features → Linear(11→32) → ReLU → Dropout  (conditioning signal)
  FiLM layer  : geo_cond (32-dim) → γ, β (64-dim each) → γ * frame_emb + β
                Initialized as identity (γ=1, β=0) so training starts stable.
  GRU         : (B, seq_len, 64) → (B, seq_len, gru_hidden)
                Uses last *valid* frame's hidden state for classification.
  Head        : Dropout → Linear(gru_hidden → 2)

Why FiLM + GRU beats Late Fusion
---------------------------------
  Late Fusion (Stage D) pooled patches to a single vector — no temporal order.
  GRU processes 40 frames sequentially → captures how eye/mouth change over time.
  FiLM conditions each frame's visual embedding on the geometry signal, so the
  GRU sees geometry-adapted visual features rather than raw CNN activations.
  This addresses the cross-participant domain shift: same γ/β shift per window
  re-centres participant4/2's patches toward the training distribution.

Per-participant min-max scaling (GEMINI.md Min-Max Standard)
-------------------------------------------------------------
  Scaler is fit on training fold indices only. Applied to geo at __getitem__ time.

Dataset
-------
  Reuses LateFusionDataset from train_late_fusion.py (patches + geo + label).

Usage
-----
    # Sanity check overfit (should reach F1 ~1.0 quickly)
    python -m src.s4_training.train_film_gru --mode overfit --max-windows 300 --epochs 15

    # Full CV run (recommended for laptop)
    python -m src.s4_training.train_film_gru \\
        --mode cv --min-valid-rate 0.80 --max-windows 2000 \\
        --epochs 30 --lr 3e-4 --weight-decay 1e-4 --dropout 0.3 \\
        --gru-hidden 64 --gru-layers 1 \\
        --patience 8 --folds 5 --batch-size 16 --num-workers 0 --cpu \\
        --freeze-cnn-epochs 3 --augment

    # Larger GRU (if you have time / GPU)
    python -m src.s4_training.train_film_gru \\
        --mode cv --max-windows 2000 --epochs 30 \\
        --gru-hidden 128 --gru-layers 2 --batch-size 16 --cpu
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold, train_test_split
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Reuse dataset + scaler utilities from Stage D (no code duplication)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.s4_training.train_late_fusion import (  # noqa: E402
    GEO_FEATURES,
    LateFusionDataset,
    fit_geo_scaler,
    VECTORS_CSV,
    SUMMARY_CSV,
    PATCH_ROOT,
)

MODEL_DIR  = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report" / "film_gru"


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
_AUG_TRANSFORM = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3),
])


# ---------------------------------------------------------------------------
# Cached Dataset (RAM Cache for massive speedup on CPU/disk-bound training)
# ---------------------------------------------------------------------------

class CachedLateFusionDataset(LateFusionDataset):
    """
    Subclass of LateFusionDataset that caches all decoded patch sequences,
    valid masks, and labels in RAM to avoid expensive disk I/O and PNG decoding
    in the training loop. Augmentation and geometry scaling are applied on-the-fly.
    """
    def __init__(self, *args, cache=True, **kwargs):
        self.original_augment = kwargs.get("augment", False)
        kwargs["augment"] = False
        super().__init__(*args, **kwargs)
        self.cache = cache
        self.cached_samples = []

        if self.cache:
            logging.info("RAM CACHE: Pre-decoding and caching all patch sequences in memory...")
            n_samples = len(self.samples)
            for i in range(n_samples):
                if (i + 1) % 500 == 0 or i == 0 or (i + 1) == n_samples:
                    logging.info("  Cached %d/%d samples...", i + 1, n_samples)
                item = super().__getitem__(i)
                self.cached_samples.append({
                    "patches": item["patches"],
                    "valid_mask": item["valid_mask"],
                    "confidence": item["confidence"],
                    "label": item["label"],
                })
            logging.info("RAM CACHE: Decoded and loaded %d samples in RAM successfully.", n_samples)

    def __getitem__(self, idx):
        if not self.cache:
            self.augment = self.original_augment
            return super().__getitem__(idx)

        cached = self.cached_samples[idx]
        patches = cached["patches"].clone()
        valid_mask = cached["valid_mask"]
        confidence = cached["confidence"]
        label = cached["label"]

        if self.augment:
            seq_len = patches.shape[0]
            for t in range(seq_len):
                if valid_mask[t] > 0.0:
                    frame_uint8 = (patches[t] * 255.0).byte()
                    frame_aug = _AUG_TRANSFORM(frame_uint8)
                    patches[t] = frame_aug.float() / 255.0

        sample = self.samples[idx]
        geo = sample["geo"].copy()
        if self.geo_scaler is not None:
            geo = self.geo_scaler.transform(geo.reshape(1, -1)).flatten().astype(np.float32)

        return {
            "patches": patches,
            "valid_mask": valid_mask,
            "confidence": confidence,
            "geo": torch.from_numpy(geo),
            "label": label,
        }



# ---------------------------------------------------------------------------
# FiLM Layer
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Scales and shifts feature maps using a conditioning signal.
    Initialized as identity (γ=1, β=0) for training stability.
    """
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, feature_dim)
        self.beta_net  = nn.Linear(cond_dim, feature_dim)
        # Identity init: start as a no-op so early epochs aren't destabilised
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.ones_(self.gamma_net.bias)    # γ → 1
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)    # β → 0

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, seq_len, feature_dim)
            cond: (B, cond_dim)
        Returns:
            (B, seq_len, feature_dim)  — FiLM-modulated sequence
        """
        gamma = self.gamma_net(cond).unsqueeze(1)  # (B, 1, feature_dim)
        beta  = self.beta_net(cond).unsqueeze(1)   # (B, 1, feature_dim)
        return gamma * x + beta


# ---------------------------------------------------------------------------
# CNN Frame Encoder (shared weights across all 40 frames)
# ---------------------------------------------------------------------------

class FrameCNNEncoder(nn.Module):
    """
    Encodes each frame patch independently.
    Identical to TinyPatchCNNEncoder but returns per-frame embeddings
    (B, seq_len, embedding_dim) instead of a pooled vector.
    """
    def __init__(self, embedding_dim: int = 64, dropout: float = 0.2):
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

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, seq_len, 3, 24, 24)
        Returns:
            (B, seq_len, embedding_dim)
        """
        B, seq_len = patches.shape[:2]
        x = patches.reshape(B * seq_len, 3, 24, 24)
        emb = self.proj(self.encoder(x))            # (B*seq_len, embedding_dim)
        return emb.reshape(B, seq_len, -1)          # (B, seq_len, embedding_dim)


# ---------------------------------------------------------------------------
# FiLM + GRU Model
# ---------------------------------------------------------------------------

class FiLMGRUModel(nn.Module):
    """
    Full Stage E model.
    Pipeline per window:
      patches → FrameCNNEncoder → frame_emb (B, T, 64)
      geo     → GeoEncoder      → geo_cond  (B, 32)
      FiLM(frame_emb, geo_cond) → modulated_emb (B, T, 64)
      zero-mask invalid frames
      GRU(modulated_emb) → hidden states (B, T, gru_hidden)
      last valid hidden → Dropout → Linear → logits (B, 2)
    """
    def __init__(
        self,
        num_classes:  int = 2,
        cnn_dim:      int = 64,
        geo_dim:      int = 11,
        geo_hidden:   int = 32,
        gru_hidden:   int = 64,
        gru_layers:   int = 1,
        dropout:      float = 0.3,
        use_film:     bool = True,
    ):
        super().__init__()
        self.use_film = use_film
        self.cnn_encoder = FrameCNNEncoder(embedding_dim=cnn_dim, dropout=dropout)
        self.geo_encoder  = nn.Sequential(
            nn.Linear(geo_dim, geo_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if self.use_film:
            self.film = FiLMLayer(cond_dim=geo_hidden, feature_dim=cnn_dim)
        # Both paths feed (cnn_dim + geo_hidden) into GRU:
        # FiLM path:    FiLM-modulated cnn_emb || geo_cond
        # no-FiLM path: raw cnn_emb            || geo_cond
        gru_input_size = cnn_dim + geo_hidden

        self.gru  = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # Temporal attention: learns which frames in the window matter most
        # instead of blindly using the last valid hidden state.
        self.use_attention = False   # toggled externally via flag
        self.attn = nn.Linear(gru_hidden, 1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, num_classes),
        )

    def forward(
        self,
        patches:    torch.Tensor,   # (B, seq_len, 3, 24, 24)
        valid_mask: torch.Tensor,   # (B, seq_len)  float 0/1
        geo:        torch.Tensor,   # (B, geo_dim)
        confidence: torch.Tensor = None,  # (B, seq_len) decay weights, None = no decay
    ) -> torch.Tensor:
        B, seq_len = patches.shape[:2]

        # 1. Per-frame CNN embeddings
        frame_emb = self.cnn_encoder(patches)          # (B, seq_len, 64)

        # 2. Geo conditioning vector
        geo_cond = self.geo_encoder(geo)               # (B, 32)

        # 3. Conditioning: FiLM modulation vs standard concatenation
        mask = valid_mask.unsqueeze(-1)                # (B, seq_len, 1)
        if self.use_film:
            # Fix: zero invalid frames BEFORE FiLM so β bias doesn't leak into
            # invalid positions, then re-zero after to kill any residual β.
            frame_emb = frame_emb * mask
            frame_emb = self.film(frame_emb, geo_cond)
            frame_emb = frame_emb * mask               # re-zero to kill β leakage
            # Fix: also inject geo into GRU input so temporal dynamics are
            # conditioned on geometry, not just the CNN embeddings.
            geo_replicated = geo_cond.unsqueeze(1).expand(-1, seq_len, -1)
            gru_input = torch.cat([frame_emb, geo_replicated], dim=-1)  # (B, seq_len, 96)
        else:
            frame_emb = frame_emb * mask
            geo_replicated = geo_cond.unsqueeze(1).expand(-1, seq_len, -1)
            gru_input = torch.cat([frame_emb, geo_replicated], dim=-1)  # (B, seq_len, 96)

        # Apply confidence decay as soft input gate: uncertain frames contribute less
        if confidence is not None:
            gru_input = gru_input * confidence.unsqueeze(-1)  # (B, seq_len, 96)

        # 5. GRU over the frame sequence
        gru_out, _ = self.gru(gru_input)              # (B, seq_len, gru_hidden)

        # 6. Pick last *valid* frame's hidden state OR attention-weighted sum
        if self.use_attention:
            # Mask padding positions with -inf before softmax so they get ~0 weight
            scores = self.attn(gru_out).squeeze(-1)           # (B, seq_len)
            scores = scores.masked_fill(valid_mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, seq_len, 1)
            last_h = (gru_out * weights).sum(dim=1)               # (B, gru_hidden)
        else:
            lengths  = valid_mask.sum(dim=1).long().clamp(min=1)
            last_idx = (lengths - 1).clamp(0, seq_len - 1)
            last_h   = gru_out[torch.arange(B, device=gru_out.device), last_idx]

        return self.head(last_h)                       # (B, num_classes)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_cnn_weights(model: FiLMGRUModel, fold_name: str):
    """
    Load CNN encoder weights from Stage D (preferred) or Stage B checkpoint.
    Only encoder + proj weights are transferred (classifier head is not shared).
    """
    for prefix, fname in [
        ("late_fusion", f"late_fusion_{fold_name}.pth"),
        ("cnn_patches", f"cnn_patches_binary_{fold_name}.pth"),
    ]:
        ckpt_path = MODEL_DIR / fname
        if not ckpt_path.exists():
            continue
        state = torch.load(ckpt_path, map_location="cpu")
        # Map checkpoint keys → FrameCNNEncoder keys
        cnn_state = {}
        for k, v in state.items():
            if k.startswith("cnn_branch.encoder."):
                new_k = k.replace("cnn_branch.encoder.", "encoder.")
                cnn_state[new_k] = v
            elif k.startswith("cnn_branch.proj."):
                new_k = k.replace("cnn_branch.proj.", "proj.")
                cnn_state[new_k] = v
            elif k.startswith("encoder.") or k.startswith("proj."):
                cnn_state[k] = v
        missing, unexpected = model.cnn_encoder.load_state_dict(cnn_state, strict=False)
        logging.info(
            "Loaded CNN weights from %s (source=%s) | missing=%d unexpected=%d",
            ckpt_path.name, prefix, len(missing), len(unexpected),
        )
        return
    logging.warning("No CNN checkpoint found for %s — starting from scratch.", fold_name)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_class_weights(dataset, indices, num_classes=2):
    labels = [0 if dataset.samples[i]["video_id"] == 0 else 1 for i in indices]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, residual_model=None, geo_scaler=None, scheduler=None):
    model.train(optimizer is not None)
    losses, labels_all, preds_all = [], [], []

    for batch in loader:
        patches    = batch["patches"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        geo        = batch["geo"].to(device)
        y          = batch["label"].to(device)
        confidence = batch["confidence"].to(device) if "confidence" in batch else None

        with torch.set_grad_enabled(optimizer is not None):
            logits = model(patches, valid_mask, geo, confidence)

            # Residual fallback: S_final = S_base + Tanh(ΔS) * 0.15
            if residual_model is not None:
                geo_np = geo.detach().cpu().numpy()
                s_base = torch.tensor(
                    residual_model.predict_proba(geo_np)[:, 1],
                    dtype=torch.float32, device=device,
                )  # (B,) — prob of drowsy from XGBoost
                delta = torch.tanh(logits[:, 1] - logits[:, 0]) * 0.15
                logits_drowsy = s_base + delta          # (B,)
                logits = torch.stack([1 - logits_drowsy, logits_drowsy], dim=1)

            loss   = criterion(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:   # OneCycleLR steps per batch
                    scheduler.step()

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


# ---------------------------------------------------------------------------
# Train one fold / split
# ---------------------------------------------------------------------------

def train_split(dataset, train_idx, val_idx, args, fold_name):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Geo scaler: fit on train only
    scaler = fit_geo_scaler(dataset, train_idx)
    dataset.geo_scaler = scaler

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    model = FiLMGRUModel(
        num_classes=2,
        cnn_dim=64,
        geo_dim=len(GEO_FEATURES),
        geo_hidden=32,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        use_film=not args.no_film,
    ).to(device)
    model.use_attention = args.attention

    # Residual fallback: load XGBoost geometry baseline for S_base
    residual_model = None
    if args.residual:
        xgb_path = MODEL_DIR.parent / "models" / "baseline_rf_model.joblib"
        if xgb_path.exists():
            import joblib as _joblib
            residual_model = _joblib.load(xgb_path)
            logging.info("Residual fallback: loaded XGBoost baseline from %s", xgb_path)

    _load_cnn_weights(model, fold_name)

    class_weights = compute_class_weights(dataset, train_idx).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # CNN freeze warm-up — use param groups so Adam momentum is preserved on unfreeze
    freeze_epochs = args.freeze_cnn_epochs
    cnn_params   = list(model.cnn_encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("cnn_encoder.")]
    if freeze_epochs > 0:
        for p in cnn_params:
            p.requires_grad = False
        logging.info("%s CNN frozen for first %d epochs", fold_name, freeze_epochs)
    optimizer = torch.optim.AdamW([
        {"params": cnn_params,   "lr": 0.0 if freeze_epochs > 0 else args.lr,
         "weight_decay": args.weight_decay},
        {"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ])

    # OneCycleLR: created after CNN unfreeze (inside the loop at epoch freeze+1).
    # Cannot be created upfront because frozen epochs must not step it.
    scheduler = None
    onecycle_ready = False  # flag: scheduler created and ready to step

    # If no freeze, create OneCycleLR immediately (no frozen phase to skip)
    if args.onecycle and freeze_epochs == 0:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[args.lr * 0.5, args.lr],
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=100.0,
        )
        onecycle_ready = True
        logging.info("%s OneCycleLR created: max_lr=[%.2e, %.2e] over %d epochs",
                     fold_name, args.lr * 0.5, args.lr, args.epochs)

    # SWA: average weights over epochs >= swa_start for a flatter, more generalizable
    # minimum. swa_start=12 is data-driven: one epoch after the latest observed
    # convergence across all LOPO folds (participant5 converged at epoch 11).
    swa_model     = None
    swa_scheduler = None
    if args.swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model     = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
        logging.info("%s SWA enabled: averaging starts at epoch %d, swa_lr=%.2e",
                     fold_name, args.swa_start, args.swa_lr)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    best_f1, best_metrics, patience_left = -1.0, None, args.patience
    in_swa_phase = False

    for epoch in range(1, args.epochs + 1):

        # Unfreeze CNN after warm-up — update lr in-place to preserve Adam momentum
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            for p in model.cnn_encoder.parameters():
                p.requires_grad = True
            if not args.onecycle:
                optimizer.param_groups[0]["lr"] = args.lr * 0.5   # CNN group: halved lr
            patience_left = args.patience   # reset: joint fine-tuning is a new phase
            best_f1       = -1.0
            logging.info("%s CNN branch unfrozen at epoch %d (lr → %.2e)",
                         fold_name, epoch, args.lr * 0.5)
            # Create OneCycleLR NOW after unfreeze — avoids graph reuse during frozen epochs
            if args.onecycle and not onecycle_ready:
                joint_epochs = args.epochs - freeze_epochs
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=[args.lr * 0.5, args.lr],
                    steps_per_epoch=len(train_loader),
                    epochs=joint_epochs,
                    pct_start=0.3,
                    div_factor=10.0,
                    final_div_factor=100.0,
                )
                onecycle_ready = True
                logging.info("%s OneCycleLR created: max_lr=[%.2e, %.2e] over %d joint epochs",
                             fold_name, args.lr * 0.5, args.lr, joint_epochs)

        # Enter SWA phase: switch to swa_lr, disable early stopping
        if args.swa and epoch == args.swa_start and not in_swa_phase:
            in_swa_phase  = True
            patience_left = 999   # early stopping disabled during SWA averaging
            logging.info("%s entering SWA averaging phase at epoch %d", fold_name, epoch)

        # Augmentation: on for train, guaranteed off for val via try/finally
        try:
            if args.augment:
                dataset.augment = True
            active_scheduler = scheduler if (not in_swa_phase and epoch > freeze_epochs) else None
            train_m = run_epoch(model, train_loader, criterion, optimizer, device, residual_model, scheduler=active_scheduler)
        finally:
            dataset.augment = False

        # Step SWA
        if in_swa_phase and swa_model is not None:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        val_m = run_epoch(model, val_loader, criterion, None, device, residual_model)

        logging.info(
            "%s epoch=%d  train_f1=%.4f  val_f1=%.4f  val_acc=%.4f  "
            "val_loss=%.4f  drowsy_recall=%.4f%s",
            fold_name, epoch,
            train_m["macro_f1"], val_m["macro_f1"],
            val_m["accuracy"],   val_m["loss"],
            val_m["drowsy_recall"],
            "  [SWA]" if in_swa_phase else "",
        )

        if not in_swa_phase:
            if val_m["macro_f1"] > best_f1:
                best_f1, best_metrics, patience_left = val_m["macro_f1"], val_m, args.patience
                torch.save(model.state_dict(), MODEL_DIR / f"film_gru_{fold_name}.pth")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logging.info("%s early stopping at epoch %d", fold_name, epoch)
                    break
        else:
            if val_m["macro_f1"] > best_f1:
                best_f1, best_metrics = val_m["macro_f1"], val_m

    # After SWA: update BatchNorm stats, save averaged model
    if args.swa and swa_model is not None and in_swa_phase:
        logging.info("%s updating SWA BatchNorm statistics...", fold_name)
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.module.state_dict(), MODEL_DIR / f"film_gru_{fold_name}.pth")
        logging.info("%s SWA model saved → film_gru_%s.pth", fold_name, fold_name)

    dataset.geo_scaler = None   # reset for next fold
    return best_metrics or {}


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_overfit(dataset, args):
    indices = np.arange(len(dataset))
    labels  = np.array([0 if s["video_id"] == 0 else 1 for s in dataset.samples])
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
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

    f1s     = [m.get("macro_f1", 0.0) for m in fold_metrics]
    recalls = [m.get("drowsy_recall", 0.0) for m in fold_metrics]
    logging.info("=" * 60)
    logging.info("FiLM+GRU CV Results")
    logging.info("  macro_F1      = %.4f ± %.4f", float(np.mean(f1s)),    float(np.std(f1s)))
    logging.info("  drowsy_recall = %.4f ± %.4f", float(np.mean(recalls)), float(np.std(recalls)))
    logging.info("Progression:")
    logging.info("  Geometry-only    F1 = 0.5422  (Baseline)")
    logging.info("  CNN-only         F1 = 0.6441  (Stage B)")
    logging.info("  Late Fusion      F1 = 0.7497  (Stage D)")
    logging.info("  FiLM + GRU       F1 = %.4f  (Stage E)  SOTA goal: >0.80", float(np.mean(f1s)))
    logging.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage E: FiLM + GRU drowsiness classifier.")
    p.add_argument("--mode",              choices=["overfit", "cv"], default="cv")
    p.add_argument("--epochs",            type=int,   default=30)
    p.add_argument("--batch-size",        type=int,   default=16,
                   help="Lower than Stage D (16) due to GRU memory overhead")
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--dropout",           type=float, default=0.3)
    p.add_argument("--patience",          type=int,   default=8)
    p.add_argument("--folds",             type=int,   default=5)
    p.add_argument("--seq-len",           type=int,   default=40)
    p.add_argument("--min-valid-rate",    type=float, default=0.80)
    p.add_argument("--max-windows",       type=int,   default=0)
    p.add_argument("--num-workers",       type=int,   default=0)
    p.add_argument("--cpu",               action="store_true")
    p.add_argument("--augment",           action="store_true",
                   help="Random flip + brightness/contrast jitter on patches (train only)")
    p.add_argument("--freeze-cnn-epochs", type=int,   default=3,
                   help="Freeze CNN encoder for first N epochs (0=disabled)")
    # GRU hyperparameters
    p.add_argument("--gru-hidden",        type=int,   default=64,
                   help="GRU hidden size. 64=fast (laptop), 128=richer (GPU)")
    p.add_argument("--gru-layers",        type=int,   default=1,
                   help="Number of GRU layers. 1=fast, 2=deeper")
    p.add_argument("--no-cache",          action="store_true",
                   help="Disable RAM caching of patches (saves RAM but runs slower)")
    p.add_argument("--no-film",           action="store_true",
                   help="Disable FiLM modulation and use standard feature concatenation inside GRU instead.")
    p.add_argument("--attention",         action="store_true",
                   help="Use attention-weighted sum over GRU outputs instead of last valid hidden state.")
    p.add_argument("--residual",          action="store_true",
                   help="Enable residual fallback: S_final = S_base(XGBoost) + Tanh(ΔS)*0.15.")
    p.add_argument("--exclude-participants", nargs="+", default=[],
                   help="List of participant IDs to exclude from the dataset entirely.")
    # OneCycleLR
    p.add_argument("--onecycle",          action="store_true",
                   help="Use OneCycleLR scheduler instead of fixed lr.")
    # SWA
    p.add_argument("--swa",               action="store_true",
                   help="Enable SWA weight averaging after --swa-start epoch.")
    p.add_argument("--swa-start",         type=int,   default=12,
                   help="Epoch to begin SWA averaging (default=12).")
    p.add_argument("--swa-lr",            type=float, default=1e-4,
                   help="Constant LR during SWA averaging phase (default=1e-4).")
    return p.parse_args()


def main():
    import sys as _sys
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/train_film_gru.log", encoding="utf-8"),
            logging.StreamHandler(),
        ]
    )
    args = parse_args()

    # Log exactly what command was run so results are always traceable
    logging.info("=" * 60)
    logging.info("RUN COMMAND : python %s", " ".join(_sys.argv))
    logging.info("ARGS        : %s", vars(args))
    logging.info("=" * 60)

    dataset = CachedLateFusionDataset(
        VECTORS_CSV, SUMMARY_CSV, PATCH_ROOT,
        seq_len=args.seq_len,
        min_valid_rate=args.min_valid_rate,
        max_windows=args.max_windows,
        augment=False,   # controlled per-epoch in train_split
        cache=not args.no_cache,
        exclude_participants=args.exclude_participants,
    )
    if len(dataset) == 0:
        raise SystemExit("No usable windows. Lower --min-valid-rate or check patches.")

    logging.info(
        "Stage E — FiLM+GRU | gru_hidden=%d gru_layers=%d freeze_cnn=%d augment=%s cache=%s",
        args.gru_hidden, args.gru_layers, args.freeze_cnn_epochs, args.augment, not args.no_cache,
    )

    if args.mode == "overfit":
        run_overfit(dataset, args)
    else:
        run_cv(dataset, args)


if __name__ == "__main__":
    main()
