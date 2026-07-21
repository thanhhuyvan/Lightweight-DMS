# Lightweight Training Findings and Hyperparameter Plan

## Purpose

This note defines the practical training setup for the current laptop-limited stage of the project. The goal is not to fully optimize the final CNN--FiLM--GRU architecture yet. The goal is to produce reliable, reportable findings with low compute cost.

## Current Key Findings

### 1. Geometry-only is a limited but useful baseline

The verified geometry baseline is:

```text
Binary alert vs. drowsy F1: 0.5422
```

This should be treated as the current honest reference result. Geometry features are useful for safety and interpretability, but they are unlikely to provide a high final F1 by themselves under participant-separated validation.

### 2. Previous high XGBoost result is not safe to report as final

The previously observed result around:

```text
Macro F1 ~= 0.895
```

should be treated as suspicious because model directories and evaluation paths are inconsistent:

```text
training saves to: models/
evaluation loads from: src/models/
```

This can accidentally evaluate stale or externally trained models. It may also involve leakage from scaler fitting or saved-model reuse.

### 3. CNN branch has a data-quality bottleneck

Patch audits show that image data is not equally reliable across participants:

```text
participant1 missing patch rate: ~23.96%
participant3 missing patch rate: ~9.00%
participant1 video_id=10 face_detected=True: 63.9%
```

This is important because the current dataset silently replaces missing patches with black images. The CNN may therefore learn detector failure patterns instead of drowsiness appearance.

### 4. Full hybrid training is not the right next step

FiLM + GRU should not be tuned yet. The current priority is to verify whether a simple CNN can learn useful visual signal from clean patches.

## Low-Compute Experiment Order

Use the following order. Stop when a stage fails and diagnose before adding complexity.

| Stage | Model | Goal | Compute Cost |
| :--- | :--- | :--- | :--- |
| 1 | Geometry-only | Confirm baseline | Low |
| 2 | CNN-only small overfit | Verify patch-label signal | Low |
| 3 | CNN-only binary CV | Test visual generalization | Medium-low |
| 4 | Late fusion binary | Test whether CNN helps geometry | Medium |
| 5 | GRU / FiLM | Final architecture only after earlier stages pass | High |

## Recommended Lightweight Hyperparameters

### A. CNN-Only Small Overfit Test

Purpose:

```text
Check whether the CNN can memorize a tiny clean subset.
```

If this fails, the patch pipeline or label alignment is broken.

Recommended setup:

```text
task: binary, video_id 0 vs 10
subset size: 200-300 windows
participant split: not required for this overfit test
missing patch policy: drop windows with valid_patch_rate < 0.90
epochs: 30
batch_size: 32
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-4
loss: CrossEntropyLoss
dropout: 0.1
augmentation: none first
checkpoint metric: training accuracy / training F1
expected result: training F1 should approach >0.90
```

Interpretation:

```text
Pass: image patches and labels are probably aligned.
Fail: patch loading, labels, missing-data handling, or architecture has a bug.
```

### B. CNN-Only Binary GroupKFold

Purpose:

```text
Test whether visual patches generalize to unseen participants.
```

Recommended setup:

```text
task: binary, video_id 0 vs 10
validation: GroupKFold by participant
missing patch policy: drop windows with valid_patch_rate < 0.80
epochs: 20-30
batch_size: 32
optimizer: AdamW
learning_rate: 5e-4
weight_decay: 1e-4
loss: weighted CrossEntropyLoss
dropout: 0.2
scheduler: ReduceLROnPlateau
scheduler_patience: 3
early_stopping_patience: 6
checkpoint metric: validation macro F1
```

Light augmentation:

```text
brightness jitter: +/- 10%
contrast jitter: +/- 10%
translation: <= 2 px
gaussian noise: very light
horizontal flip: avoid initially
```

Report:

```text
fold participant
macro F1
drowsy recall
confusion matrix
mean +/- std
```

### C. CNN-Only 3-Class

Purpose:

```text
Measure difficulty after binary CNN works.
```

Recommended setup:

```text
task: 3-class, video_id 0 vs 5 vs 10
validation: GroupKFold by participant
missing patch policy: drop windows with valid_patch_rate < 0.80
epochs: 30-40
batch_size: 32
optimizer: AdamW
learning_rate: 3e-4
weight_decay: 1e-4
loss: weighted CrossEntropyLoss or FocalLoss
focal_gamma: 1.5
dropout: 0.3
checkpoint metric: validation macro F1
```

Interpretation:

The low-vigilance class is expected to be hardest because it lies between alert and drowsy. A lower 3-class F1 is acceptable if binary separation improves.

### D. Late Fusion Binary

Purpose:

```text
Check whether CNN features improve over geometry.
```

Recommended setup:

```text
input: CNN embedding + geometry vector
cnn_embedding_dim: 64
fusion_hidden_dim: 64
dropout: 0.3
optimizer: AdamW
learning_rate: 3e-4
weight_decay: 1e-4
epochs: 30
loss: weighted CrossEntropyLoss
checkpoint metric: validation macro F1
```

Expected reportable comparison:

```text
Geometry-only binary F1: 0.5422
CNN-only binary F1: TBD
Late-fusion binary F1: TBD
```

### E. GRU / FiLM Deferred Setup

Do not tune FiLM or GRU until CNN-only and late fusion are stable.

When ready:

```text
seq_len: 40
gru_hidden_dim: 64
gru_layers: 1
batch_size: 16
learning_rate: 1e-4 to 3e-4
gradient_clip_norm: 1.0
dropout: 0.3
missing_frame_mask: required
checkpoint metric: validation macro F1
```

## Suggested Report Table

Use this table format even before all experiments are complete:

| Model | Task | Validation | Missing Patch Policy | F1 / Status | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Geometry XGBoost | Binary 0 vs 10 | GroupKFold | N/A | 0.5422 | Verified baseline |
| CNN-only | Binary 0 vs 10 | Small overfit | Drop < 0.90 valid | Pending | Patch sanity check |
| CNN-only | Binary 0 vs 10 | GroupKFold | Drop < 0.80 valid | Pending | Visual baseline |
| Late fusion | Binary 0 vs 10 | GroupKFold | Drop < 0.80 valid | Pending | First hybrid |
| FiLM + GRU | 3-class | GroupKFold | Mask required | Deferred | Final target |

## What To Say In The Report

Recommended wording:

```text
Due to limited local compute, the current experimental phase prioritizes leakage-safe baselines and data-quality diagnostics over full hybrid hyperparameter optimization. The verified geometry baseline reaches 0.5422 F1 on binary alert-vs-drowsy classification. Patch audits reveal participant-dependent missing visual data, especially for participant1 in the drowsy sequence, motivating a staged training protocol: CNN-only validation, late fusion, and only then temporal FiLM-GRU optimization.
```

## Practical Laptop Settings

Use these defaults for local experiments:

```text
num_workers: 0 or 2
pin_memory: false unless using CUDA
image_size: 24x24
batch_size: 32
mixed_precision: not needed on CPU
save_every_epoch: false
save_best_only: true
log_every_n_batches: 20
```

If training is too slow:

```text
reduce epochs to 10
train binary only
use one fold first
use smaller subset
freeze CNN encoder if using late fusion
```

## Decision Rules

Use these rules to avoid wasting compute:

```text
If CNN cannot overfit 200-300 clean windows:
    fix patch loading / labels / missing handling.

If CNN overfits but fails GroupKFold:
    add augmentation and participant-specific diagnostics.

If CNN-only works but late fusion fails:
    fix geometry scaling and fusion architecture.

If late fusion beats baselines:
    then try GRU.

If GRU works:
    then try FiLM.
```
