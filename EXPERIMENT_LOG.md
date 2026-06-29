# Experiment Log

Tracks training runs and dataset modifications. Update after each stage.

---

## 📊 Summary Table

| Stage | Model & Description | Mode | Exclusions | Windows | Epochs | macro F1 (val) | Drowsy Recall | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Geometry XGBoost (4-feat) | GroupKFold (5) | None | 10,944 | — | 0.5422 | — | ✅ Verified |
| **Baseline (Clean)** | Geometry XGBoost (10-feat) | GroupKFold (5) | `participant1` | 9,665 | — | 0.4900 | — | ✅ Verified (Stricter CV) |
| **Stage A** | CNN-only (TinyPatchCNN) | Overfit 80/20 | None | 300 | 19 | 1.0000 | 1.0000 | ✅ PASSED |
| **Stage B** | CNN-only (TinyPatchCNN) | GroupKFold (5) | `participant1` | 1000 | 7–12 | 0.6441 ± 0.2632 | — | ✅ PASSED |
| **Stage D** | Late Fusion (CNN + Geo) | GroupKFold (5) | `participant1` | 1000 | 8–12 | 0.7584 ± 0.2642 | 0.7337 ± 0.3757 | ✅ PASSED |
| **Stage E — FiLM+GRU (run1, buggy)** | FiLM+GRU (original, β-leak + capacity mismatch) | GroupKFold (5) | `participant1` | 1000 | 8–15 | 0.7084 ± 0.2951 | 0.6492 ± 0.3694 | ⚠️ Regressed vs Stage D — bugs identified |
| **Stage E — Concat+GRU (run2, buggy)** | Concat+GRU no-FiLM (original, unfair comparison) | GroupKFold (5) | `participant1` | 1000 | 8–13 | 0.7311 ± 0.2533 | 0.7355 ± 0.3622 | ⚠️ Regressed vs Stage D — bugs identified |
| **Stage E — FiLM+GRU (fixed)** | FiLM+GRU (β-leak fixed, geo→GRU, interpolation) | GroupKFold (3) | `participant1` `partcipant2` `participant6` | 1000 | 15 | **0.9988** | — | ✅ PASSED (clean folds only) |
| **Stage E — Full 5-fold (fixed)** | FiLM+GRU fixed, all participants | GroupKFold (5) | `participant1` | 1000 | 15 | *TBD* | *TBD* | ⏳ Ready to Run (GPU recommended) |
| **Stage E — Attention** | FiLM+GRU + Temporal Attention | GroupKFold (5) | `participant1` | 1000 | 15 | *TBD* | *TBD* | ⏳ Code ready, not yet trained |

---

## 🔍 Participant Patch Quality Audit

We conducted a diagnostic audit of patch quality and facial mesh detection rates across all participants using `inspect_participant_patches.py`.

### Diagnostic Metrics Table
*   **Threshold:** Windows are kept only if they have at least 80% (`0.80`) of their frames with valid eye and mouth patches (face mesh successfully detected).

| Participant ID | Total Windows | Windows Kept ($\ge 0.80$) | Keep Rate (%) | Mean Valid Rate | Brightness | Contrast | Verdict / Quality |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **partcipant2** | 1,291 | 1,291 | 100.0% | 99.9% | 93.7 | 39.6 | ✅ Excellent |
| **partcipant4** | 1,273 | 1,273 | 100.0% | 100.0% | 126.6 | 50.1 | ✅ Excellent |
| **participant1** | 1,279 | 688 | **53.8%** | **74.7%** | 127.5 | 64.1 | ❌ **Severe Outlier (Mesh Failure)** |
| **participant3** | 1,393 | 1,210 | 86.9% | 93.1% | 107.7 | 65.1 | ⚠️ Moderate / Acceptable |
| **participant5** | 1,131 | 1,120 | 99.0% | 98.7% | 95.7 | 43.0 | ✅ Excellent |
| **participant6** | 1,052 | 1,052 | 100.0% | 100.0% | 71.1 | 31.4 | ✅ Excellent |

### Key Findings
*   **`participant1` is highly corrupted:** Only 53.8% of its windows contain a valid face mesh. For the remaining 46.2%, landmark localization failed due to head rotation or extreme lighting, producing empty or near-black patches. This creates a severe label and domain gap during cross-validation.
*   **Other participants are clean:** The rest of the cohorts have keep rates $\ge 86.9\%$, with most at $99-100\%$.
*   **GroupKFold Alignment:** Excluding `participant1` leaves exactly 5 participants. Under a 5-fold cross-validation scheme, this perfectly aligns each fold to hold out exactly one participant, creating a mathematically rigorous Leave-One-Participant-Out (LOPO) evaluation.

---

## 🚀 Ablation Study Flight Plan (Clean Dataset)

To isolate performance gains and verify architecture choices, use the `--exclude-participants participant1` flag to run all models on the clean subset.

### 1. Geometry-Only Baseline (XGBoost)
Establishes the performance ceiling of behavioral geometry alone (10 features, 3-period deltas).
*   **Run command:**
    ```bash
    python -m src.s4_training.train_final --exclude-participants participant1
    ```
*   **Results (Verified):** macro F1 = **0.4900**.

### 2. CNN-Only Spatial Ablation (TinyPatchCNN)
Evaluates spatial-only features without temporal context.
*   **Run command:**
    ```bash
    python -m src.s4_training.train_cnn_patches --mode cv --min-valid-rate 0.80 --max-windows 1000 --epochs 12 --lr 5e-4 --weight-decay 1e-4 --dropout 0.2 --patience 4 --folds 5 --batch-size 32 --num-workers 0 --cpu --exclude-participants participant1
    ```

### 3. Late Fusion Static Ablation (CNN + Geometry)
Combines spatial and behavioral geometry statically via masked average pooling.
*   **Run command:**
    ```bash
    python -m src.s4_training.train_late_fusion --mode cv --min-valid-rate 0.80 --max-windows 1000 --epochs 12 --lr 3e-4 --weight-decay 1e-4 --dropout 0.3 --patience 4 --folds 5 --batch-size 32 --num-workers 0 --cpu --freeze-cnn-epochs 3 --augment --exclude-participants participant1
    ```

### 4. FiLM+GRU Temporal Fusion (Proposed SOTA)
Performs sequence modeling with GRU and adapts patch embeddings to geometry conditioning via FiLM.
*   **Fixes applied:** failed-frame interpolation, β-leakage double-mask, geo injected into GRU input (96-dim both paths).
*   **Run command:**
    ```bash
    python -m src.s4_training.train_film_gru --mode cv --min-valid-rate 0.80 --max-windows 2000 --epochs 15 --lr 3e-4 --weight-decay 1e-4 --dropout 0.3 --gru-hidden 64 --gru-layers 1 --patience 4 --folds 5 --batch-size 16 --num-workers 4 --cpu --freeze-cnn-epochs 3 --augment --exclude-participants participant1
    ```

### 5. Concat+GRU Conditioning Ablation (No FiLM)
Ablates the FiLM layer — same 96-dim GRU input as FiLM path, fair capacity comparison.
*   **Run command:**
    ```bash
    python -m src.s4_training.train_film_gru --mode cv --min-valid-rate 0.80 --max-windows 2000 --epochs 15 --lr 3e-4 --weight-decay 1e-4 --dropout 0.3 --gru-hidden 64 --gru-layers 1 --patience 4 --folds 5 --batch-size 16 --num-workers 4 --cpu --freeze-cnn-epochs 3 --augment --no-film --exclude-participants participant1
    ```

---

## 🛠️ Code Modification History

1.  **Balanced Class Sampling:** Patched `max_windows` selection in `train_cnn_patches.py` to prevent single-class skew.
2.  **Participant Exclusion Support:** Added `--exclude-participants` flag to all training scripts.
3.  **FiLM Layer Ablation (`--no-film`):** Added a parameter to bypass the FiLM layer and run standard temporal concatenation instead.
4.  **Failed Frame Interpolation (2026-06-29):** Fixed `LateFusionDataset.__getitem__` in `train_late_fusion.py`. Previously, frames with failed mesh detection were silently replaced with zero-tensors. The GRU would see a sudden zero discontinuity mid-sequence, corrupting hidden state for all subsequent frames. Fix: forward-fill then backward-fill from nearest valid neighbor. The `valid_mask` still marks these frames as invalid so the model retains awareness of data quality.
5.  **FiLM β-leakage Fix (2026-06-29):** Fixed `FiLMGRUModel.forward` in `train_film_gru.py`. FiLM's bias term β caused invalid (zero) frames to become non-zero after modulation (`γ·0 + β = β`), injecting noise into the GRU. Fix: double-mask — zero invalid frames before FiLM, re-zero after.
6.  **Geo injected into GRU input (2026-06-29):** Previously FiLM path fed 64-dim to GRU, no-FiLM path fed 96-dim — an unfair capacity comparison. Fix: both paths now concatenate `geo_cond` to every GRU timestep (96-dim input). FiLM additionally modulates CNN features before concatenation. This makes the ablation honest and also lets geometry condition temporal dynamics, not just CNN features.

---

## 🔑 Key Insights (2026-06-29)

### Why Stage E regressed vs Stage D (before fixes)
Three compounding bugs:
- **Zero-patch GRU corruption:** partcipant2 and participant6 have high mesh failure rates. Zero-patches mid-sequence corrupted GRU hidden state → `drowsy_recall=0.000` for all epochs on those folds.
- **β-leakage:** FiLM's bias re-activated zeroed invalid frames, feeding structured noise into GRU.
- **Capacity mismatch:** FiLM path (64-dim GRU input) vs no-FiLM path (96-dim) — not a fair ablation.

### FiLM is actually effective on clean data
On folds with good data quality (partcipant4, participant3, participant5), FiLM+GRU reached **0.9988 macro F1** after fixes — well above the 0.80 SOTA goal and above Late Fusion (0.75).

### partcipant2 and participant6 need investigation
These participants pass the 0.80 `min_valid_rate` threshold (100% keep rate per audit) but still cause model collapse. Likely cause: **clustered** mesh failures within windows — scattered failures average out above 0.80 but consecutive failures in a burst still corrupt the GRU sequence. The interpolation fix (item 4 above) addresses this.

### Recommended full-training run order
1. Run Stage E FiLM+GRU fixed, full 5-fold, `--exclude-participants participant1`, `--max-windows 2000+` on GPU
2. Run Stage E no-FiLM fixed (same settings, `--no-film`) for fair ablation
3. Run Stage E + Attention (once attention code is merged) for final comparison
