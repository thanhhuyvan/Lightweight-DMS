"""Quick patch dataset diagnostic — count alert vs drowsy windows per threshold."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
VECTORS_CSV  = PROJECT_ROOT / "frame" / "csv" / "behavioral_vectors.csv"
SUMMARY_CSV  = PROJECT_ROOT / "frame" / "csv" / "features_summary.csv"
PATCH_ROOT   = PROJECT_ROOT / "frame" / "patches"
PATCH_DIRS   = ["left_eye", "right_eye", "mouth"]

windows = pd.read_csv(VECTORS_CSV)
frames  = pd.read_csv(SUMMARY_CSV)

binary = windows[windows["video_id"].isin([0, 10])].copy()
print(f"Total binary windows (before patch filter): {len(binary)}")
print(binary["video_id"].value_counts().to_string())

frame_groups = {
    key: grp.reset_index(drop=True)
    for key, grp in frames.groupby(["video_id", "participant_id"], sort=False)
}

SEQ_LEN = 40
rates = []
for _, w in binary.iterrows():
    grp = frame_groups.get((w["video_id"], w["participant_id"]))
    if grp is None:
        rates.append((w["video_id"], w["participant_id"], 0.0))
        continue
    frm = grp.iloc[int(w["window_start_idx"]):int(w["window_end_idx"]) + 1].head(SEQ_LEN)
    expected = len(frm) * len(PATCH_DIRS)
    valid = 0
    for _, f in frm.iterrows():
        fname = f"{f['video_id']}_{f['participant_id']}_{f['frame_file']}"
        for p in PATCH_DIRS:
            if (PATCH_ROOT / p / fname).exists():
                valid += 1
    rates.append((w["video_id"], w["participant_id"], valid / expected if expected else 0.0))

import numpy as np
rates_arr = pd.DataFrame(rates, columns=["video_id", "participant_id", "valid_rate"])

print("\n--- Valid rate stats by video_id ---")
print(rates_arr.groupby("video_id")["valid_rate"].describe().round(3).to_string())

for thresh in [0.90, 0.80, 0.70, 0.60, 0.50, 0.30, 0.0]:
    kept = rates_arr[rates_arr["valid_rate"] >= thresh]
    alert  = (kept["video_id"] == 0).sum()
    drowsy = (kept["video_id"] == 10).sum()
    print(f"  thresh={thresh:.2f} → alert={alert:4d}  drowsy={drowsy:4d}  total={len(kept):4d}")
