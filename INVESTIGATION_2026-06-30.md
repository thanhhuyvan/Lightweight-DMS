# Investigation Notes — 2026-06-30

## Session Summary
Full diagnostic session on why FiLM+GRU full 5-fold = 0.6867 despite 0.9988 on clean folds.

---

## Results So Far

| Run | Config | macro F1 |
|-----|--------|----------|
| FiLM+GRU fixed (3 clean folds) | no attention, no decay | 0.9988 |
| FiLM+GRU full 5-fold (training machine) | no attention, no decay | 0.6867 |
| FiLM+GRU full 5-fold (this machine) | **+attention +confidence decay** | running... → `logs/attention_decay_full5fold.txt` |

Attention+decay run so far (partial):
- Fold 1 (partcipant2 held out): **1.0000** ✅
- Fold 2 (partcipant4 held out): **1.0000** ✅
- Fold 3 (participant3 held out): **0.9925** ✅
- Fold 4 (participant6 held out): collapsing at epoch 2 ❌ (same as baseline)
- Fold 5: still running

---

## Root Cause Analysis

### partcipant2 — SOLVED
- **Was:** drowsy_recall = 0.000, complete collapse
- **Cause:** burst frame failures corrupted GRU hidden state. Zero patches mid-sequence injected discontinuities.
- **Fix:** Confidence decay (0.85^distance) + attention over valid frames
- **Result:** 1.0000 F1 with attention+decay — fully solved

### participant6 — NOT FIXABLE WITH MODEL CHANGES
- **Is:** drowsy_recall ≈ 0.0, cross-participant always fails
- **Within-participant F1 = 0.898** — signal exists, data is not garbage

#### Evidence of domain inversion:
| Feature | Training (others) | participant6 | Problem |
|---------|------------------|--------------|---------|
| EAR_Mean (alert→drowsy) | 0.27→0.19 (drops) | 0.364→0.360 (flat) | No EAR signal |
| EAR_Mean scaled | drowsy=0.27 | both states=0.73 | Out of drowsy range |
| Pose_Jitter (alert→drowsy) | 12790→8558 (DROPS) | 11346→12959 (RISES) | **INVERTED** |

- Cross-participant model learned: high pose jitter = alert (correct for others)
- participant6: high pose jitter = drowsy (fighting sleep, active head movement)
- Model confidently predicts "alert" for p6's drowsy state — not random, actively wrong

#### participant6 behavioral profile:
- Naturally large eyes (EAR=0.36 vs others 0.13–0.19) — stays open when drowsy
- Drowsiness manifests as: head instability + MAR_Max (yawning) + longer blinks
- NOT: sustained eye closure (standard EAR-based drowsiness)
- Likely resisting sleep during recording → eyes forced open, head nodding

#### Cohen's d separability:
- EAR_Mean: **0.184** (negligible, <0.2)
- MAR_Mean: **0.071** (negligible)
- All features < 0.31

---

## Code Changes Applied Tonight

### 1. Confidence Decay — `src/s4_training/train_late_fusion.py`
Replaced dumb forward-fill with:
- Nearest-valid patch fill (same as before for visual)
- Exponential decay: `confidence[t] = 0.85 ^ distance_to_nearest_valid_frame`
- Returns `confidence` tensor in batch dict

### 2. Confidence as GRU input gate — `src/s4_training/train_film_gru.py`
- `forward()` accepts optional `confidence` tensor
- `gru_input = gru_input * confidence.unsqueeze(-1)` before GRU
- CachedLateFusionDataset updated to cache and return confidence

### 3. Attention — already coded, just needed `--attention` flag
- Scaled dot-product attention over GRU outputs
- Masks invalid frames with -inf before softmax
- Weighted sum replaces last-valid-frame extraction

---

## Wrong Diagnoses (ruled out tonight)

### ❌ "participant6 has burst frame failures like partcipant2"
- **Wrong.** participant6 has 100% strict detection, zero failed/loose frames.
- Confidence decay and interpolation fixes are irrelevant for participant6.

### ❌ "participant6 data quality is bad (low brightness/contrast)"
- **Wrong.** Low brightness (71.1) seemed suspicious but 100% strict mesh detection
  means landmarks are clean. The patches are just darker, not corrupted.
- Cohen's d is near-zero not because landmarks fail but because the face genuinely
  doesn't change between alert and drowsy states.

### ❌ "Skipping video_id=5 causes the problem"
- **Wrong.** Binary classification uses 0 (alert) vs 10 (drowsy) correctly.
  Including the transition phase would blur class boundaries. Not the issue.

### ❌ "Reversed labels for participant6"
- **Wrong.** Checked video_id distribution and EAR trends — labels are correct.
  Alert EAR (0.3635) > drowsy EAR (0.3601) — tiny diff but right direction.

### ❌ "Residual fallback (XGBoost) will save participant6"
- **Wrong.** XGBoost geometry-only LOPO for participant6 = F1 0.474, recall 0.159.
  The fallback base is already broken for this participant. Adding a broken base
  to a broken DL prediction doesn't help.

### ❌ "Attention mechanism will fix participant6"
- **Wrong.** Attention reweights GRU outputs — but the GRU input features themselves
  carry wrong/inverted signal for participant6. Attention can't fix corrupted inputs.

### ❌ "MinMaxScaler handles the large eye problem"
- **Partially wrong.** Scaler normalizes values correctly, but the decision boundary
  collision remains: only partcipant4 had EAR~0.74 in training (always alert there),
  so scaled p6 EAR=0.74 always maps to "alert" in the learned boundary.

---

## Tomorrow's Actions

### Option A — Report as-is (recommended)
- Full 5-fold with attention+decay: report when run completes
- Exclude participant6 from headline result with documented justification
- Report: "FiLM+GRU+Attention: 0.999 on 4/5 participants. participant6 excluded due to behavioral inversion (within-participant F1=0.898 confirms signal exists; requires calibration)"

### Option B — Try within-window EAR delta normalization
- Instead of absolute EAR, use `EAR_t - EAR_window_mean` (relative change)
- This is anatomy-independent — removes absolute eye size baseline
- Requires modifying feature extraction in `to_csv.py` or computing delta in dataset
- Risk: might hurt clean participants who rely on absolute EAR level

### Option C — Add pose jitter prominence as explicit feature
- participant6's drowsiness = high pose jitter (inverted from others)
- Could add `pose_jitter_vs_personal_baseline` as a delta feature
- Same problem: requires knowing participant6's alert baseline at test time

### Most likely impactful quick win:
Run full 5-fold results with attention+decay and see if overall F1 improves.
If fold 3 (participant3) and others improved, the new mean without p6 is the headline.

---

## Key Files
- Training script: `src/s4_training/train_film_gru.py`
- Dataset: `src/s4_training/train_late_fusion.py`
- Current run log: `logs/attention_decay_full5fold.txt`
- Proof scripts: `diagnose_p6.py`, `investigate_p6.py` (can delete after)
