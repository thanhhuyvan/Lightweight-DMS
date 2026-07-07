# Pretraining Setup Checklist

Use this checklist before running any new training job. The goal is to avoid stale models, leakage, and silent patch failures.

## Environment

- Use the project root as the working directory.
- Install the dependencies from `requirements.txt`.
- Confirm that `torch`, `sklearn`, `xgboost`, `cv2`, and `pandas` import successfully.

Suggested command:

```bash
python -c "import torch, sklearn, xgboost, cv2, pandas; print('training deps ok')"
```

## Data

Required files:

```text
frame/csv/behavioral_vectors.csv
frame/csv/features_summary.csv
frame/patches/left_eye/
frame/patches/right_eye/
frame/patches/mouth/
```

Run the patch audit before training:

```bash
python src/s4_training/audit_hybrid_data.py --per-cell 20 --sample-step 5
```

Known data-quality findings:

```text
participant1 missing patch rate: ~23.96%
participant3 missing patch rate: ~9.00%
participant1 video_id=10 face_detected=True: 63.9%
```

## Model Directory

Use only:

```text
models/
```

Do not evaluate stale models from:

```text
src/models/
```

## First Training Target

Start with the CNN-only binary setup:

```bash
python src/s4_training/train_cnn_patches.py --mode overfit --task binary --max-windows 300
```

Only after the overfit test passes, run a lightweight fold:

```bash
python src/s4_training/train_cnn_patches.py --mode cv --task binary --folds 5 --epochs 20
```

## Pass / Fail Rules

If CNN cannot overfit 200-300 clean windows:

```text
fix patch loading, missing-patch handling, or label alignment
```

If CNN overfits but fails GroupKFold:

```text
add augmentation and inspect participant-specific domain shift
```

If CNN-only works:

```text
move to late fusion before trying GRU or FiLM
```
