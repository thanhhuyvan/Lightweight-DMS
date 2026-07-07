# Hybrid Stabilization Plan

## Current Diagnosis

The current hybrid/CNN path is not yet trustworthy. Geometry-only is limited, but the CNN branch is also not receiving clean enough supervision to outperform it.

Fresh reruns showed:

| Path | Task | Result |
| :--- | :--- | :--- |
| `train_improved.py` | 3-class: `0`, `5`, `10` | Macro F1 around `0.3145` |
| `train_final.py` | Binary: `0` vs `10` | F1 around `0.5422` |

The previously reported `0.895` F1 is likely affected by leakage, stale saved models, or evaluation mismatch. The repo has two model folders:

```text
models/
src/models/
```

Training scripts save to `models/`, while `evaluate_all.py` loads from `src/models/`. This makes old/friend-generated models easy to evaluate accidentally.

## Key Insight

CNN should eventually carry more useful signal than geometry, but only if the patch data is reliable. Right now, the image branch is weakened by missing/detector-failed frames.

Audit result from `src/s4_training/audit_hybrid_data.py`:

```text
participant1 missing patch rate: ~23.96%
participant3 missing patch rate: ~9.00%
participant1 video_id=10 face_detected=True: only 63.9%
```

This means the model sees many black replacement patches, especially for a drowsy video. Since `HybridSequenceDataset` silently replaces missing patches with black images, the CNN/GRU learns a mixed signal: real eye/mouth appearance plus detector failure artifacts.

## Main Problems To Fix

1. **Missing patches are treated as real images**
   - Current behavior: missing patch -> black `24x24` image.
   - Problem: the model cannot tell missing data from real dark visual content.

2. **Hybrid model is too complex too early**
   - Current path jumps to FiLM + GRU.
   - Problem: if performance is poor, we cannot tell whether the failure is from patches, fusion, temporal modeling, or training.

3. **Evaluation is inconsistent**
   - Some scripts use `models/`.
   - `evaluate_all.py` uses `src/models/`.
   - This can compare stale or externally generated models against current data.

4. **Geometry vector contains placeholders**
   - Left/right EAR are both set to average EAR.
   - EAR difference is `0.0`.
   - Pose deltas are `0.0`.
   - This weakens FiLM because the conditioning signal is incomplete.

5. **Hybrid training only runs one fold**
   - `train_hybrid.py` trains only the first GroupKFold split.
   - Evaluation expects five fold models.

## Stable Experiment Strategy

### Stage 0: Lock Clean Evaluation

Goal: make sure every metric is honest.

Rules:

- Use one canonical model directory: `models/`.
- For every fold:
  - split by participant with `GroupKFold`
  - fit scalers only on the train fold
  - train only on the train fold
  - evaluate only on held-out participant(s)
- Report:
  - per-fold F1
  - mean F1
  - held-out participant names
  - confusion matrix

Acceptance:

```text
No model should be evaluated on data it was trained or scaled on.
```

### Stage 1: Patch Data Audit

Goal: prove the image input is usable.

Use:

```bash
python src/s4_training/audit_hybrid_data.py --per-cell 20 --sample-step 5
```

Check:

- missing patch rate by participant
- missing patch rate by class
- near-black patch rate
- frame slice length
- pixel mean/std by class

Acceptance:

```text
Missing patch rate should be explicitly handled before CNN training.
No silent black replacement should be treated as normal input.
```

### Stage 2: CNN-Only Baseline

Goal: prove visual patches contain predictive signal.

Model:

```text
left_eye patch + right_eye patch + mouth patch -> CNN -> class
```

No geometry. No FiLM. No GRU.

Recommended first task:

```text
Binary: video_id 0 vs 10
```

Then:

```text
3-class: video_id 0 vs 5 vs 10
```

Missing patch policy:

- Option A: drop windows with too many missing frames.
- Option B: add valid masks.
- Option C: copy nearest valid patch within the same sequence.

Acceptance:

```text
CNN-only binary F1 should beat the geometry binary baseline or clearly overfit a small train subset.
```

If CNN cannot overfit a small subset, the image pipeline is broken.

### Stage 3: Late Fusion

Goal: combine CNN signal with geometry without letting geometry dominate too early.

Model:

```text
CNN embedding + geometry window vector -> MLP classifier
```

Avoid FiLM at this stage.

Reason:

Late fusion makes it easier to measure whether CNN contributes useful information.

Acceptance:

```text
Late fusion should outperform geometry-only and CNN-only on the same clean folds.
```

### Stage 4: Temporal CNN + GRU

Goal: add sequence dynamics only after image classification works.

Model:

```text
per-frame CNN embedding -> GRU -> classifier
```

Use patch masks so missing frames do not become fake visual evidence.

Acceptance:

```text
GRU should improve recall for drowsy or low-vigilance states without reducing macro F1.
```

### Stage 5: FiLM / Geometry-Steered CNN

Goal: let geometry guide visual interpretation after the visual branch is proven.

Only add FiLM after:

- CNN-only works
- late fusion works
- missing patches are masked or filtered
- geometry vector is real, not placeholder-filled

Fix geometry vector first:

- real left EAR
- real right EAR
- real EAR asymmetry
- real pose deltas
- face detection confidence
- patch valid mask

Acceptance:

```text
FiLM should beat late fusion. If it does not, keep late fusion.
```

## Participant1 Stress Test

Participant1 is a known weak point:

```text
participant1 video_id=10 face_detected=True: 63.9%
```

Always report two results:

```text
All participants
All participants except participant1
```

Interpretation:

- If performance improves sharply without participant1, the issue is data quality/domain shift.
- If performance remains poor, the issue is likely model/data alignment.

## Immediate Next Scripts

### 1. `train_cnn_patches.py`

Purpose:

```text
CNN-only baseline for eye/mouth patches.
```

Requirements:

- GroupKFold by participant
- binary mode first: `0` vs `10`
- optional 3-class mode
- missing patch filtering
- per-fold F1 reporting
- small-subset overfit mode

### 2. `train_late_fusion.py`

Purpose:

```text
CNN embedding + geometry vector baseline.
```

Requirements:

- reuse trained CNN encoder or train jointly
- compare against geometry-only and CNN-only
- report macro F1 and drowsy recall

### 3. `evaluate_clean_cv.py`

Purpose:

```text
One canonical evaluator for all non-neural baselines.
```

Requirements:

- no stale `src/models/` loading
- fold-local scaler fitting
- fold-local model training
- participant names in output

## Success Criteria

Short-term:

```text
Clean binary CNN-only model beats 0.5422 F1 or proves patch pipeline failure.
```

Mid-term:

```text
Late fusion beats both geometry-only and CNN-only.
```

Final:

```text
FiLM + GRU beats late fusion under clean GroupKFold validation.
```

If FiLM + GRU does not beat late fusion, the stable final model should be late fusion, not the more complex architecture.
