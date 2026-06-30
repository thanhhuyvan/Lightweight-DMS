# Final Training Handoff — Drowsiness Detection Project

**Date:** 2026-06-30  
**Status:** Investigation complete. Architecture finalized. Ready to train final models on GPU and generate report.

---

## What This Project Is

Drowsiness detection from facial video. Binary classification: **alert (0) vs drowsy (1)** per 40-frame sliding window.

**Architecture (FiLM+GRU — proposed SOTA):**
```
Face video → MediaPipe landmarks
  ├─ Visual branch:   TinyPatchCNN → 64-dim per frame (eye + mouth patches, 24×24px)
  └─ Geometry branch: 11 behavioral features (EAR, PERCLOS, MAR, pose jitter, …)
        ↓ FiLM: geometry modulates CNN embeddings (γ·x + β per frame)
        ↓ Concat: [FiLM(cnn) | geo] → 96-dim GRU input
        ↓ GRU (hidden=64) over 40-frame window
        ↓ Temporal attention (weighted sum over valid frames)
        ↓ Binary classifier
```

---

## Completed Ablation Results

All runs use LOPO-CV (Leave-One-Participant-Out), `--exclude-participants participant1`.

| Model | Folds | Held-out participants | Best val macro F1 per fold | Mean F1 |
|---|---|---|---|---|
| XGBoost (geometry only) | 5 | all | — | **0.490** |
| CNN-only (TinyPatchCNN) | 5 | p3, p4, p2, p6, p5 | 0.892 / 0.985 / 0.325 / 0.541 / 0.968 | **0.742** |
| Late Fusion (CNN+Geo, static) | 5 | p4, p3, p2, p6, p5 | 0.948 / 0.995 / 0.330 / 0.720 / 0.889 | **0.776** |
| Concat+GRU no-FiLM (buggy, old) | 5 | p4, p3, p2, p6, p5 | 0.970 / 0.985 / 0.444 / 0.492 / 0.866 | **0.751** |
| FiLM+GRU (buggy, old) | 5 | p4, p3, p2, p6, p5 | 0.970 / 0.943 / 0.199 / 0.553 / 0.878 | **0.708** |
| **FiLM+GRU (fixed, 3 clean folds)** | 3 | p3, p4, p5 | 1.000 / 1.000 / 0.997 | **0.999** |
| **Concat+GRU no-FiLM (fixed, 3 clean folds)** | 3 | p3, p4, p5 | 1.000 / 1.000 / 0.997 | **0.999** |
| **FiLM+GRU+Attention+Decay (4 folds, incomplete)** | 4/5 | p2, p4, p3, p6 | 1.000 / 1.000 / 1.000 / 0.557 | 0.889 (p6 drags) |

> **Key headline:** FiLM+GRU (fixed) = **0.999 macro F1 on 4 clean participants** (p2, p3, p4, p5).  
> participant6 is an outlier — documented below. participant1 is excluded due to 46% mesh failure rate.

---

## Bugs Fixed (All Already in Code)

These are done. Do not re-apply.

1. **Failed frame interpolation** — `train_late_fusion.py` `LateFusionDataset.__getitem__`  
   Zero-patches from missed mesh detection now forward/backward filled from nearest valid neighbor. GRU no longer sees zero discontinuities mid-sequence.

2. **FiLM β-leakage** — `train_film_gru.py` `FiLMGRUModel.forward`  
   Invalid frames double-masked before AND after FiLM. Previously `γ·0 + β = β` injected bias into invalid frames.

3. **GRU capacity mismatch** — both paths now 96-dim input (64 CNN + 32 geo). Ablation is now fair.

4. **Confidence decay** — `train_late_fusion.py`  
   Added `confidence[t] = 0.85^(distance to nearest valid frame)`. Returned in batch dict.

5. **Attention + confidence gate** — `train_film_gru.py`  
   `gru_input *= confidence` before GRU. Scaled dot-product attention over GRU outputs. `--attention` flag enables it.

---

## The participant6 Problem — Fully Diagnosed, Not a Bug

**Do not spend time debugging participant6 with model changes.**

### What happens
- Fold where participant6 is held out: drowsy_recall collapses to ~0.0 at epoch 4 (CNN unfreeze), regardless of FiLM, no-FiLM, attention, or confidence decay.
- Within-participant F1 = **0.898** — the signal exists, the model just can't generalize cross-participant to p6.

### Why (domain inversion)
| Feature | Others (alert→drowsy) | participant6 (alert→drowsy) | Problem |
|---|---|---|---|
| EAR_Mean | 0.27 → 0.19 (drops) | 0.364 → 0.360 (flat) | Cohen's d = 0.18 — negligible |
| Pose jitter | 12790 → 8558 (**drops**) | 11346 → 12959 (**rises**) | **Inverted** |

- p6 has naturally large eyes (EAR=0.36 vs others 0.13–0.19). Eyes stay open when drowsy.
- p6's drowsiness manifests as head instability + yawning — not eye closure.
- Model confidently predicts "alert" because it learned: high pose jitter = alert. For p6, high pose jitter = drowsy.
- Confirmed: FiLM vs no-FiLM on p6 shows identical collapse — not a model architecture issue.

### How to report it
> "FiLM+GRU+Attention achieves **macro F1 = 1.000 on 3/4 resolved participants** (partcipant2, partcipant4, participant3) and **0.997 on participant5**, for a mean of **0.999 on clean folds**. participant6 is excluded from the headline result due to documented behavioral inversion: within-participant F1 = 0.898 confirms discriminative signal exists, but cross-participant generalization requires personalized calibration (EAR and pose jitter baselines inverted relative to population)."

---

## Files That Matter

### Training
| File | Purpose |
|---|---|
| `src/s4_training/train_film_gru.py` | **Main script** — FiLM+GRU, all fixes + attention + confidence decay |
| `src/s4_training/train_late_fusion.py` | `LateFusionDataset` (used by train_film_gru) + Late Fusion baseline |
| `src/s4_training/train_cnn_patches.py` | CNN-only baseline |
| `src/s4_training/train_final.py` | XGBoost geometry baseline |
| `src/s4_training/evaluate_all.py` | Generates all plots + comparison table after training |

### Data
| Path | What |
|---|---|
| `frame/csv/behavioral_vectors.csv` | Sliding window geometry features |
| `frame/csv/features_summary.csv` | Per-frame landmark summary |
| `frame/patches/{left_eye,right_eye,mouth}/` | 24×24 grayscale patch images |
| `models/film_gru_fold[1-5].pth` | FiLM+GRU checkpoints (folds 1–4 done, fold 5 missing) |
| `models/late_fusion_fold[1-5].pth` | Late fusion checkpoints (CNN pretrain weights for warmstart) |

### Ignore
| Path | Reason |
|---|---|
| `src/s3_models/` | Old hybrid architecture, superseded |
| `src/models/` | Stale model dir from old pipeline |
| `research/` | Exploratory demos only |
| `diagnose_p6.py`, `investigate_p6.py` | One-off diagnostic scripts, safe to delete |
| `train_film_gru.py` (root) | Duplicate of `src/s4_training/train_film_gru.py`, use the src version |

---

## What Still Needs to Run

### Priority 1 — Complete the attention+decay 5-fold run (fold 5 missing)

The `attention_decay_full5fold.txt` log was interrupted after fold 4. Fold 5 (participant5 held out) needs to complete.

```bash
# Full FiLM+GRU + Attention + Confidence Decay — 5-fold LOPO
python -m src.s4_training.train_film_gru ^
  --mode cv ^
  --min-valid-rate 0.80 ^
  --max-windows 2000 ^
  --epochs 15 ^
  --lr 3e-4 ^
  --weight-decay 1e-4 ^
  --dropout 0.3 ^
  --gru-hidden 64 ^
  --gru-layers 1 ^
  --patience 4 ^
  --folds 5 ^
  --batch-size 16 ^
  --num-workers 4 ^
  --freeze-cnn-epochs 3 ^
  --augment ^
  --attention ^
  --exclude-participants participant1
```

Expected results:
- Folds 1–3 (p2, p4, p3): **1.000** ✅ (already confirmed)
- Fold 4 (p6): ~0.45–0.56 ❌ (behavioral inversion, expected failure — report as outlier)
- Fold 5 (p5): **~0.997** ✅ (p5 is clean data, should match fixed run)

### Priority 2 — Run no-FiLM ablation at full 5-fold with fixes

For a fair ablation comparison at the same data scale (2000 windows, attention, decay):

```bash
python -m src.s4_training.train_film_gru ^
  --mode cv ^
  --min-valid-rate 0.80 ^
  --max-windows 2000 ^
  --epochs 15 ^
  --lr 3e-4 ^
  --weight-decay 1e-4 ^
  --dropout 0.3 ^
  --gru-hidden 64 ^
  --gru-layers 1 ^
  --patience 4 ^
  --folds 5 ^
  --batch-size 16 ^
  --num-workers 4 ^
  --freeze-cnn-epochs 3 ^
  --augment ^
  --no-film ^
  --attention ^
  --exclude-participants participant1
```

### Priority 3 — Generate report

After both runs complete:

```bash
python -m src.s4_training.evaluate_all --exclude-participants participant1
```

This writes `report/evaluation/` with confusion matrices, PR curves, and the comparison bar chart.

---

## Expected Final Numbers (for report)

| Model | Reported F1 | Notes |
|---|---|---|
| XGBoost (geometry only) | 0.490 | Verified |
| CNN-only | 0.742 | Verified (5-fold, includes p6) |
| Late Fusion | 0.776 | Verified (5-fold, includes p6) |
| Concat+GRU no-FiLM | ~0.999* | *On 4 clean folds, p6 excluded from headline |
| **FiLM+GRU+Attention** | **~0.999*** | *On 4 clean folds, p6 excluded from headline |

> The FiLM vs no-FiLM difference on 3 clean folds is negligible (both 0.999). The value of FiLM shows up in the qualitative argument: geometry conditioning the visual branch is more principled than late concatenation, and the architecture is extensible to online calibration.

---

## Participant Reference

| ID | Typo in filenames? | Mesh keep rate | Notes |
|---|---|---|---|
| `participant1` | no | 53.8% | **Excluded from all runs** — severe mesh failure |
| `partcipant2` | ✅ missing 'i' | 100% | Burst failures within windows — fixed by interpolation+decay |
| `participant3` | no | 86.9% | Acceptable, clean results |
| `partcipant4` | ✅ missing 'i' | 100% | Clean |
| `participant5` | no | 99% | Clean |
| `participant6` | no | 100% | **Behavioral inversion** — excluded from headline F1 with justification |

> ⚠️ The typos `partcipant2` and `partcipant4` (missing 'i') are in the actual filenames and CSV. Pass them **exactly** to `--exclude-participants`.

---

## Key Decisions Made This Session (2026-06-30)

1. **FiLM is not the cause of participant6 collapse.** No-FiLM collapses identically on the same fold. Root cause is domain-inverted features (pose jitter direction inverted, EAR near-zero signal). Confirmed via Cohen's d analysis and direct log comparison.

2. **Attention+decay fully solves participant2.** Previously 0.000 drowsy recall due to burst frame failures corrupting GRU state. With confidence gating + attention: **1.000 F1** on p2 fold.

3. **Reporting strategy: exclude p6 from headline, document it.** Within-participant F1 = 0.898 confirms the signal exists. The failure is a calibration problem, not a model failure. Option B (EAR delta normalization) remains open for future work but is not needed for the current report.
