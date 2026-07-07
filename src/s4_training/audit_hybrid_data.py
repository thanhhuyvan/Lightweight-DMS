import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
VECTORS_CSV = PROJECT_ROOT / "frame" / "csv" / "behavioral_vectors.csv"
SUMMARY_CSV = PROJECT_ROOT / "frame" / "csv" / "features_summary.csv"
PATCH_ROOT = PROJECT_ROOT / "frame" / "patches"


PATCH_KINDS = {
    "left_eye": PATCH_ROOT / "left_eye",
    "right_eye": PATCH_ROOT / "right_eye",
    "mouth": PATCH_ROOT / "mouth",
}


def read_gray(path):
    img_array = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    return img


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size} mean={arr.mean():.3f} std={arr.std():.3f} "
        f"p05={np.percentile(arr, 5):.3f} p50={np.percentile(arr, 50):.3f} "
        f"p95={np.percentile(arr, 95):.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-windows", type=int, default=0, help="0 means audit all windows.")
    parser.add_argument("--per-class", type=int, default=0, help="Audit up to N windows from each video_id.")
    parser.add_argument(
        "--per-cell",
        type=int,
        default=0,
        help="Audit up to N windows from each (video_id, participant_id) cell.",
    )
    parser.add_argument("--sample-step", type=int, default=1, help="Audit every Nth window.")
    args = parser.parse_args()

    windows = pd.read_csv(VECTORS_CSV)
    frames = pd.read_csv(SUMMARY_CSV)

    required_window_cols = {
        "video_id",
        "participant_id",
        "window_start_idx",
        "window_end_idx",
    }
    required_frame_cols = {"video_id", "participant_id", "frame_file"}
    missing_window_cols = required_window_cols - set(windows.columns)
    missing_frame_cols = required_frame_cols - set(frames.columns)
    if missing_window_cols or missing_frame_cols:
        raise SystemExit(
            f"Missing columns. windows={sorted(missing_window_cols)} "
            f"frames={sorted(missing_frame_cols)}"
        )

    frame_groups = {
        key: group.reset_index(drop=True)
        for key, group in frames.groupby(["video_id", "participant_id"], sort=False)
    }

    if args.per_cell > 0:
        selected_indices = []
        for _, group in windows.groupby(["video_id", "participant_id"], sort=True):
            stepped = list(group.index[:: max(args.sample_step, 1)])
            selected_indices.extend(stepped[: args.per_cell])
    elif args.per_class > 0:
        selected_indices = []
        for _, group in windows.groupby("video_id", sort=True):
            stepped = list(group.index[:: max(args.sample_step, 1)])
            selected_indices.extend(stepped[: args.per_class])
    else:
        selected_indices = list(range(0, len(windows), max(args.sample_step, 1)))
    if args.max_windows > 0 and args.per_class <= 0:
        selected_indices = selected_indices[: args.max_windows]

    totals = defaultdict(int)
    by_class = defaultdict(lambda: defaultdict(int))
    by_participant = defaultdict(lambda: defaultdict(int))
    pixel_means = defaultdict(list)
    pixel_stds = defaultdict(list)
    frame_count_by_class = defaultdict(list)
    bad_examples = []

    for idx in selected_indices:
        window = windows.iloc[idx]
        label = int(window["video_id"])
        participant = str(window["participant_id"])
        key = (window["video_id"], window["participant_id"])
        group = frame_groups.get(key)

        totals["windows"] += 1
        by_class[label]["windows"] += 1
        by_participant[participant]["windows"] += 1

        if group is None:
            totals["missing_frame_group"] += 1
            by_class[label]["missing_frame_group"] += 1
            if len(bad_examples) < 10:
                bad_examples.append((idx, "missing_frame_group", key))
            continue

        start_idx = int(window["window_start_idx"])
        end_idx = int(window["window_end_idx"])
        window_frames = group.iloc[start_idx : end_idx + 1]
        frame_count_by_class[label].append(len(window_frames))

        if len(window_frames) == 0:
            totals["empty_frame_slice"] += 1
            by_class[label]["empty_frame_slice"] += 1
            if len(bad_examples) < 10:
                bad_examples.append((idx, "empty_frame_slice", key, start_idx, end_idx, len(group)))
            continue

        for _, frame in window_frames.iterrows():
            filename = f"{frame['video_id']}_{frame['participant_id']}_{frame['frame_file']}"
            for kind, folder in PATCH_KINDS.items():
                path = folder / filename
                totals[f"{kind}_expected"] += 1
                by_class[label][f"{kind}_expected"] += 1
                by_participant[participant][f"{kind}_expected"] += 1

                if not path.exists():
                    totals[f"{kind}_missing"] += 1
                    by_class[label][f"{kind}_missing"] += 1
                    by_participant[participant][f"{kind}_missing"] += 1
                    if len(bad_examples) < 10:
                        bad_examples.append((idx, f"{kind}_missing", filename))
                    continue

                img = read_gray(path)
                if img is None:
                    totals[f"{kind}_decode_failed"] += 1
                    by_class[label][f"{kind}_decode_failed"] += 1
                    by_participant[participant][f"{kind}_decode_failed"] += 1
                    continue

                mean = float(img.mean())
                std = float(img.std())
                pixel_means[(label, kind)].append(mean)
                pixel_stds[(label, kind)].append(std)
                if mean < 2.0 and std < 2.0:
                    totals[f"{kind}_near_black"] += 1
                    by_class[label][f"{kind}_near_black"] += 1
                    by_participant[participant][f"{kind}_near_black"] += 1

    print("=== Hybrid Data Audit ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Windows audited: {totals['windows']} / {len(windows)}")
    print("\nClass window counts:")
    for label in sorted(by_class):
        print(f"  video_id={label}: {by_class[label]['windows']}")

    print("\nPatch missing / near-black rates by class:")
    for label in sorted(by_class):
        print(f"  video_id={label}")
        for kind in PATCH_KINDS:
            expected = by_class[label][f"{kind}_expected"]
            missing = by_class[label][f"{kind}_missing"]
            decode_failed = by_class[label][f"{kind}_decode_failed"]
            near_black = by_class[label][f"{kind}_near_black"]
            miss_rate = missing / expected if expected else 0.0
            black_rate = near_black / expected if expected else 0.0
            print(
                f"    {kind}: expected={expected} missing={missing} "
                f"({miss_rate:.2%}) decode_failed={decode_failed} "
                f"near_black={near_black} ({black_rate:.2%})"
            )

    print("\nFrame slice lengths by class:")
    for label in sorted(frame_count_by_class):
        print(f"  video_id={label}: {summarize(frame_count_by_class[label])}")

    print("\nPatch pixel statistics by class:")
    for label in sorted(by_class):
        print(f"  video_id={label}")
        for kind in PATCH_KINDS:
            print(f"    {kind} mean: {summarize(pixel_means[(label, kind)])}")
            print(f"    {kind} std:  {summarize(pixel_stds[(label, kind)])}")

    print("\nParticipant missing patch rates:")
    for participant in sorted(by_participant):
        line = [f"  {participant}:"]
        for kind in PATCH_KINDS:
            expected = by_participant[participant][f"{kind}_expected"]
            missing = by_participant[participant][f"{kind}_missing"]
            miss_rate = missing / expected if expected else 0.0
            line.append(f"{kind}={miss_rate:.2%}")
        print(" ".join(line))

    if bad_examples:
        print("\nFirst bad examples:")
        for item in bad_examples:
            print(f"  {item}")


if __name__ == "__main__":
    main()
