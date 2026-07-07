"""
inspect_participant_patches.py
-------------------------------
Diagnose patch quality per participant.

Outputs
-------
- report/diagnostics/patch_grid_<participant>.png   — 6x8 grid of sample patches
- report/diagnostics/patch_stats.csv                — per-participant stats table
- Console summary table (brightness, contrast, valid_rate)

Usage
-----
    python -m src.s4_training.inspect_participant_patches
    python -m src.s4_training.inspect_participant_patches --participants partcipant2 partcipant4
    python -m src.s4_training.inspect_participant_patches --patch-type left_eye --n-samples 48
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
VECTORS_CSV  = PROJECT_ROOT / "frame" / "csv" / "behavioral_vectors.csv"
SUMMARY_CSV  = PROJECT_ROOT / "frame" / "csv" / "features_summary.csv"
PATCH_ROOT   = PROJECT_ROOT / "frame" / "patches"
PATCH_DIRS   = ["left_eye", "right_eye", "mouth"]
DIAG_DIR     = PROJECT_ROOT / "report" / "diagnostics"

SEQ_LEN = 40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_patches_for_participant(participant_id: str, patch_type: str,
                                  vectors_df, frames_df, n_samples: int):
    """Yield up to n_samples (image_array, valid) for one participant."""
    patch_dir = PATCH_ROOT / patch_type
    windows   = vectors_df[
        (vectors_df["participant_id"] == participant_id) &
        (vectors_df["video_id"].isin([0, 10]))
    ]
    frame_groups = {
        key: grp.reset_index(drop=True)
        for key, grp in frames_df.groupby(["video_id", "participant_id"], sort=False)
    }

    collected = []
    for _, w in windows.iterrows():
        grp = frame_groups.get((w["video_id"], w["participant_id"]))
        if grp is None:
            continue
        frm = grp.iloc[int(w["window_start_idx"]):int(w["window_end_idx"]) + 1].head(SEQ_LEN)
        for _, f in frm.iterrows():
            fname = f"{f['video_id']}_{f['participant_id']}_{f['frame_file']}"
            p = patch_dir / fname
            if p.exists():
                img_bytes = np.fromfile(str(p), dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    collected.append(img)
                    if len(collected) >= n_samples:
                        return collected
    return collected


def make_grid(images, rows: int, cols: int, cell: int = 48) -> np.ndarray:
    """Stack images into an (rows × cols) grid, each cell resized to (cell, cell)."""
    canvas = np.zeros((rows * cell, cols * cell), dtype=np.uint8)
    for k, img in enumerate(images[: rows * cols]):
        r, c = divmod(k, cols)
        resized = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        canvas[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = resized
    return canvas


def patch_stats(images) -> dict:
    if not images:
        return {"n": 0, "mean_brightness": float("nan"),
                "std_brightness": float("nan"), "mean_contrast": float("nan")}
    arr = np.stack([i.astype(np.float32) for i in images])
    return {
        "n": len(images),
        "mean_brightness": float(arr.mean()),
        "std_brightness":  float(arr.std()),
        "mean_contrast":   float(np.mean([i.std() for i in images])),
    }


# ---------------------------------------------------------------------------
# Per-participant valid-rate stats
# ---------------------------------------------------------------------------

def valid_rate_stats(participant_id: str, vectors_df, frames_df, threshold=0.80):
    """Return (n_windows_kept, n_windows_total, mean_valid_rate) for a participant."""
    windows = vectors_df[
        (vectors_df["participant_id"] == participant_id) &
        (vectors_df["video_id"].isin([0, 10]))
    ]
    frame_groups = {
        key: grp.reset_index(drop=True)
        for key, grp in frames_df.groupby(["video_id", "participant_id"], sort=False)
    }

    rates = []
    for _, w in windows.iterrows():
        grp = frame_groups.get((w["video_id"], w["participant_id"]))
        if grp is None:
            rates.append(0.0)
            continue
        frm = grp.iloc[int(w["window_start_idx"]):int(w["window_end_idx"]) + 1].head(SEQ_LEN)
        expected = len(frm) * len(PATCH_DIRS)
        valid = 0
        for _, f in frm.iterrows():
            fname = f"{f['video_id']}_{f['participant_id']}_{f['frame_file']}"
            for pt in PATCH_DIRS:
                if (PATCH_ROOT / pt / fname).exists():
                    valid += 1
        rates.append(valid / expected if expected else 0.0)

    rates_arr = np.array(rates)
    return {
        "total_windows": len(rates_arr),
        "kept_at_0.80":  int((rates_arr >= threshold).sum()),
        "mean_valid_rate": float(rates_arr.mean()) if len(rates_arr) else float("nan"),
        "min_valid_rate":  float(rates_arr.min()) if len(rates_arr) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Inspect patch quality per participant.")
    parser.add_argument("--participants", nargs="+", default=None,
                        help="Participant IDs to inspect (default: all)")
    parser.add_argument("--patch-type", default="left_eye",
                        choices=["left_eye", "right_eye", "mouth"],
                        help="Which patch channel to visualise (default: left_eye)")
    parser.add_argument("--n-samples", type=int, default=48,
                        help="Max patches to sample per participant for the grid")
    parser.add_argument("--grid-rows", type=int, default=6)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--cell-size", type=int, default=48,
                        help="Pixel size of each cell in the output grid")
    args = parser.parse_args()

    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    vectors_df = pd.read_csv(VECTORS_CSV)
    frames_df  = pd.read_csv(SUMMARY_CSV)

    all_participants = sorted(vectors_df["participant_id"].unique())
    targets = args.participants if args.participants else all_participants
    logging.info("Inspecting participants: %s", targets)
    logging.info("Patch type: %s | samples/participant: %d", args.patch_type, args.n_samples)

    rows_out = []
    for pid in targets:
        logging.info("--- %s ---", pid)

        # 1. Collect patches and build visual grid
        images = iter_patches_for_participant(
            pid, args.patch_type, vectors_df, frames_df, args.n_samples
        )
        stats = patch_stats(images)
        logging.info("  Collected %d patches | brightness=%.1f ± %.1f | contrast=%.1f",
                     stats["n"], stats["mean_brightness"],
                     stats["std_brightness"], stats["mean_contrast"])

        if images:
            grid = make_grid(images, args.grid_rows, args.grid_cols, args.cell_size)
            out_path = DIAG_DIR / f"patch_grid_{pid}_{args.patch_type}.png"
            cv2.imwrite(str(out_path), grid)
            logging.info("  Grid saved → %s", out_path)
        else:
            logging.warning("  No patches found for %s!", pid)

        # 2. Compute valid-rate stats
        vr = valid_rate_stats(pid, vectors_df, frames_df)
        logging.info(
            "  Windows: total=%d  kept@0.80=%d  mean_valid_rate=%.3f  min_valid_rate=%.3f",
            vr["total_windows"], vr["kept_at_0.80"],
            vr["mean_valid_rate"], vr["min_valid_rate"],
        )

        rows_out.append({
            "participant_id": pid,
            "patch_type":     args.patch_type,
            **stats,
            **vr,
        })

    # 3. Save summary CSV
    df_out = pd.DataFrame(rows_out)
    csv_path = DIAG_DIR / "patch_stats.csv"
    df_out.to_csv(csv_path, index=False)
    logging.info("\nSummary saved → %s", csv_path)

    # 4. Print console table
    print("\n" + "=" * 75)
    print(f"{'Participant':<18} {'#Patches':>8} {'Brightness':>11} {'Contrast':>10} "
          f"{'mean_vr':>8} {'kept@.80':>9} {'total':>7}")
    print("=" * 75)
    for r in rows_out:
        print(f"{r['participant_id']:<18} {r['n']:>8} {r['mean_brightness']:>11.1f} "
              f"{r['mean_contrast']:>10.1f} {r['mean_valid_rate']:>8.3f} "
              f"{r['kept_at_0.80']:>9} {r['total_windows']:>7}")
    print("=" * 75)
    print("\n⚠️  RED FLAGS to look for:")
    print("   mean_brightness < 40  → patches are too dark")
    print("   mean_contrast   < 15  → patches are blurry / flat")
    print("   mean_valid_rate < 0.7 → many missing patches for this participant")


if __name__ == "__main__":
    main()
