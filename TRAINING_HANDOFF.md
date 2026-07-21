# Training Handoff — Drowsiness Detection Project

**Date:** 2026-06-29  
**Prepared by:** Ablation study session (laptop)  
**For:** Training machine (GPU recommended)

---

## What This Project Is

Drowsiness detection from facial video using a hybrid architecture:
- **Visual branch:** TinyPatchCNN encodes eye + mouth patches per frame
- **Geometry branch:** 11 behavioral features (EAR, PERCLOS, MAR, head pose jitter)
- **FiLM:** geometry conditions CNN embeddings (γ·x + β per frame)
- **GRU:** processes 40-frame sliding window temporally
- **Output:** binary alert (0) vs drowsy (1)

Primary goal: **macro F1 > 0.80** — already achieved on clean folds (0.9988).

---

## Files That Matter

### Train
| File | Purpose |
|------|---------|
| `src/s4_training/train_film_gru.py` | **Main training script** — FiLM+GRU, all fixes applied |
| `src/s4_training/train_late_fusion.py` | Stage D baseline, also contains `LateFusionDataset` used by train_film_gru |
| `src/s4_training/train_cnn_patches.py` | Stage B CNN-only baseline |

### Evaluate
| File | Purpose |
|------|---------|
| `src/s4_training/evaluate_all.py` | **Run this after training** — generates all plots + comparison table |

### Data
| Path | What |
|------|------|
| `frame/csv/behavioral_vectors.csv` | Sliding window geometry features |
| `frame/csv/features_summary.csv` | Per-frame landmark summary |
| `frame/patches/{left_eye,right_eye,mouth}/` | 24×24 grayscale patch images |
| `models/` | Saved fold checkpoints go here |

### Ignore
- `src/s3_models/` — old hybrid architecture, superseded
- `src/models/` — stale model dir from old pipeline, do not evaluate from here
- `research/` — exploratory demos only

---

## Bugs Fixed (Already in Code)

1. **Failed frame interpolation** (`train_late_fusion.py` `__getitem__`)  
   Zero-patches from missed mesh detection now forward/backward filled from nearest valid neighbor. GRU no longer sees zero discontinuities mid-sequence.

2. **FiLM β-leakage** (`train_film_gru.py` `FiLMGRUModel.forward`)  
   Invalid frames are masked before AND after FiLM. Previously `γ·0 + β = β` was injecting the bias into invalid frames.

3. **GRU capacity mismatch** (`train_film_gru.py`)  
   Both FiLM and no-FiLM paths now feed 96-dim input to GRU (64 CNN + 32 geo). Previously FiLM used 64-dim — unfair ablation.

---

## Critical Dataset Insight

| Participant | Mesh Failure | Notes |
|-------------|-------------|-------|
| `participant1` | ~46% windows fail | **Exclude from all runs** — severe outlier |
| `partcipant2` | 0% fail (audit) but GRU collapses | Clustered burst failures within windows pass the 0.80 rate but corrupt GRU. Interpolation fix should help. |
| `participant6` | 0% fail (audit) but weak | Same issue as partcipant2. |
| `partcipant4`, `participant3`, `participant5` | Clean | These drive the good results. |

> ⚠️ Note the typo: `partcipant2` and `partcipant4` (missing 'i') — this is in the actual filenames and CSV. Use exactly these strings in `--exclude-participants`.

---

## Run Commands

### Full ablation (recommended order):

```bash
# 1. FiLM+GRU — proposed SOTA
python -m src.s4_training.train_film_gru --mode cv --min-valid-rate 0.80 --max-windows 2000 --epochs 15 --lr 3e-4 --weight-decay 1e-4 --dropout 0.3 --gru-hidden 64 --gru-layers 1 --patience 4 --folds 5 --batch-size 16 --num-workers 4 --freeze-cnn-epochs 3 --augment --exclude-participants participant1

# 2. no-FiLM ablation (same capacity, fair comparison)
python -m src.s4_training.train_film_gru --mode cv --min-valid-rate 0.80 --max-windows 2000 --epochs 15 --lr 3e-4 --weight-decay 1e-4 --dropout 0.3 --gru-hidden 64 --gru-layers 1 --patience 4 --folds 5 --batch-size 16 --num-workers 4 --freeze-cnn-epochs 3 --augment --no-film --exclude-participants participant1

# 3. FiLM+GRU + Attention
python -m src.s4_training.train_film_gru --mode cv --min-valid-rate 0.80 --max-windows 2000 --epochs 15 --lr 3e-4 --weight-decay 1e-4 --dropout 0.3 --gru-hidden 64 --gru-layers 1 --patience 4 --folds 5 --batch-size 16 --num-workers 4 --freeze-cnn-epochs 3 --augment --attention --exclude-participants participant1

# 4. Generate all reports
python -m src.s4_training.evaluate_all --exclude-participants participant1
```

> Remove `--cpu` if running on GPU (it's not in the commands above intentionally).  
> Increase `--max-windows` beyond 2000 if GPU memory allows — more data = better generalization for partcipant2/6.

### Optional — residual fallback experiment:
```bash
python -m src.s4_training.train_film_gru --mode cv ... --residual --exclude-participants participant1
```
Requires `models/baseline_rf_model.joblib` to exist (it does).

---

## Expected Results

| Model | Expected macro F1 | Notes |
|-------|------------------|-------|
| XGBoost baseline | ~0.54 | Geometry only |
| CNN-only | ~0.64 | Visual only |
| Late Fusion | ~0.75 | Static fusion |
| FiLM+GRU (fixed) | **>0.90** | Achieved 0.9988 on 3 clean folds |
| FiLM+GRU + Attention | TBD | Should match or exceed above |

If full 5-fold (including partcipant2 + participant6) drops below 0.85, the interpolation fix didn't fully solve the burst-failure problem — consider raising `--min-valid-rate` to 0.90.

---

## Output

After training, `report/evaluation/` will contain:
- `cm_*.png` — confusion matrix per model
- `pr_*.png` — precision-recall curve per model  
- `comparison_bar.png` — macro F1 bar chart across all models
- `comparison_table.csv` — full results table
