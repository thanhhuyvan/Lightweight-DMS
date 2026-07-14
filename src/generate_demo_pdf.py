"""
generate_demo_pdf.py
--------------------
Generates a 2-page demo PDF for the drowsiness detection project.

Page 1: Pipeline overview
  Raw frame → CLAHE → Mesh overlay → Eye/Mouth patches → Sliding window concept

Page 2: Signal + Predictions
  EAR / PERCLOS over time with alert/drowsy zones + sample window thumbnails

Usage:
    python generate_demo_pdf.py

Output:
    report/demo.pdf
"""

import os
import sys
# Force UTF-8 output on Windows to handle path with Vietnamese characters
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent
MESH_DIR    = ROOT / "frame" / "frames_mesh"
CLAHE_DIR   = ROOT / "frame" / "frames_clahe"
RAW_DIR     = ROOT / "frame" / "frames_raw"
PATCH_DIR   = ROOT / "frame" / "patches"
SUMMARY_CSV = ROOT / "frame" / "csv" / "features_summary.csv"
OUT_PDF     = ROOT / "report" / "demo.pdf"

PARTICIPANT = "participant3"
ALERT_VID   = 0    # video_id = 0  → alert
DROWSY_VID  = 10   # video_id = 10 → drowsy

# Colour scheme
C_ALERT  = "#2ECC71"   # green
C_DROWSY = "#E74C3C"   # red
C_BG     = "#F8F9FA"
C_DARK   = "#2C3E50"
C_ACCENT = "#3498DB"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_img(path, rgb=True):
    """Load image — handles paths with Unicode characters (Vietnamese folder name)."""
    try:
        with open(str(path), "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if rgb else img
    except Exception:
        return None


def pick_frame(video_id, participant, n=500):
    """Pick a representative frame from the middle of a video condition."""
    d = MESH_DIR / str(video_id) / participant
    frames = sorted(d.glob("*.jpg"))
    if not frames:
        return None
    idx = min(n, len(frames) - 1)
    return frames[idx]


def pick_patch(video_id, participant, frame_file, patch_type):
    """Return path to a patch file. patch_type: left_eye / right_eye / mouth"""
    stem = Path(frame_file).stem          # e.g. frame_00500
    name = f"{video_id}_{participant}_{stem}.jpg"
    p = PATCH_DIR / patch_type / name
    return p if p.exists() else None


def find_raw_frame(video_id, participant, frame_file):
    """Try to find the raw (non-mesh) frame."""
    for d in [RAW_DIR / str(video_id) / participant,
              CLAHE_DIR / str(video_id) / participant]:
        p = d / Path(frame_file).name
        if p.exists():
            return p
    return None


def add_label(ax, text, color, fontsize=11, loc="lower center"):
    ax.text(0.5, -0.08, text, transform=ax.transAxes,
            ha="center", va="top", fontsize=fontsize,
            color=color, fontweight="bold")


def hide_ax(ax):
    ax.set_visible(False)


def frame_ax(ax, color=C_ACCENT, lw=2):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Page 1 — Pipeline overview
# ---------------------------------------------------------------------------

def page1(pdf):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor=C_BG)   # A4 landscape
    fig.suptitle(
        "Drowsiness Detection Pipeline",
        fontsize=18, fontweight="bold", color=C_DARK, y=0.97
    )

    # Layout: 3 rows
    #   Row 0: title row (spacer)
    #   Row 1: raw | arrow | clahe | arrow | mesh | arrow | patches (left, right, mouth)
    #   Row 2: sliding window diagram

    gs = gridspec.GridSpec(
        3, 9,
        figure=fig,
        top=0.90, bottom=0.32,
        left=0.03, right=0.97,
        hspace=0.4, wspace=0.15,
        height_ratios=[0.05, 1, 0.05]
    )

    # ── Row 1: pipeline frames ──────────────────────────────────────────────
    alert_mesh_path  = pick_frame(ALERT_VID,  PARTICIPANT, n=400)
    drowsy_mesh_path = pick_frame(DROWSY_VID, PARTICIPANT, n=400)

    # We show alert condition for the pipeline strip
    frame_file = alert_mesh_path.name if alert_mesh_path else "frame_00400.jpg"

    raw_path   = find_raw_frame(ALERT_VID, PARTICIPANT, frame_file)
    clahe_path = CLAHE_DIR / str(ALERT_VID) / PARTICIPANT / frame_file if (
        CLAHE_DIR / str(ALERT_VID) / PARTICIPANT / frame_file).exists() else None
    mesh_path  = alert_mesh_path

    le_path = pick_patch(ALERT_VID, PARTICIPANT, frame_file, "left_eye")
    re_path = pick_patch(ALERT_VID, PARTICIPANT, frame_file, "right_eye")
    mo_path = pick_patch(ALERT_VID, PARTICIPANT, frame_file, "mouth")

    def show_frame(gs_pos, img_path, label, border_color=C_ACCENT):
        ax = fig.add_subplot(gs_pos)
        img = load_img(img_path) if img_path and Path(img_path).exists() else None
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor("#CCCCCC")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
        frame_ax(ax, color=border_color)
        add_label(ax, label, border_color, fontsize=9)
        return ax

    def show_arrow(gs_pos):
        ax = fig.add_subplot(gs_pos)
        ax.annotate("", xy=(0.85, 0.5), xytext=(0.15, 0.5),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        return ax

    show_frame(gs[1, 0], raw_path,   "① Raw Frame",   C_DARK)
    show_arrow(gs[1, 1])
    show_frame(gs[1, 2], clahe_path, "② CLAHE",       C_ACCENT)
    show_arrow(gs[1, 3])
    show_frame(gs[1, 4], mesh_path,  "③ Mesh Overlay", C_ACCENT)
    show_arrow(gs[1, 5])
    show_frame(gs[1, 6], le_path,    "Left Eye",       C_ALERT)
    show_frame(gs[1, 7], re_path,    "Right Eye",      C_ALERT)
    show_frame(gs[1, 8], mo_path,    "Mouth",          C_ALERT)

    # Patch group label
    ax_patch_label = fig.add_axes([0.72, 0.285, 0.25, 0.02])
    ax_patch_label.text(0.5, 0.5, "④ Extracted Patches (24×24 px)",
                        ha="center", va="center", fontsize=9,
                        color=C_ALERT, fontweight="bold")
    ax_patch_label.axis("off")

    # ── Row 2: sliding window diagram ───────────────────────────────────────
    gs2 = gridspec.GridSpec(
        1, 1,
        figure=fig,
        top=0.28, bottom=0.05,
        left=0.05, right=0.95,
    )
    ax_win = fig.add_subplot(gs2[0, 0])
    ax_win.set_facecolor(C_BG)
    ax_win.set_xlim(0, 100)
    ax_win.set_ylim(0, 1)
    ax_win.axis("off")

    # Draw frame timeline
    n_frames = 100
    frame_h = 0.35
    frame_y = 0.55
    for i in range(n_frames):
        color = "#AAAAAA" if i < 20 or i > 65 else (C_ALERT if i < 50 else C_DROWSY)
        rect = mpatches.FancyBboxPatch(
            (i + 0.1, frame_y), 0.8, frame_h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white", lw=0.3, alpha=0.7
        )
        ax_win.add_patch(rect)

    # Sliding window highlight
    win_start = 30
    win_end   = win_start + 40
    win_rect = mpatches.FancyBboxPatch(
        (win_start, frame_y - 0.08), 40, frame_h + 0.16,
        boxstyle="round,pad=0.05",
        facecolor="none", edgecolor=C_ACCENT, lw=2.5
    )
    ax_win.add_patch(win_rect)
    ax_win.text(win_start + 20, frame_y + frame_h + 0.15,
                "40-frame sliding window  (~10 sec at 4fps)",
                ha="center", va="bottom", fontsize=10,
                color=C_ACCENT, fontweight="bold")

    # Arrow down to prediction
    ax_win.annotate("", xy=(win_start + 20, 0.18),
                    xytext=(win_start + 20, frame_y - 0.1),
                    arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.5))

    # Prediction badge
    pred_box = mpatches.FancyBboxPatch(
        (win_start + 14, 0.04), 12, 0.16,
        boxstyle="round,pad=0.05",
        facecolor=C_ALERT, edgecolor="none", alpha=0.9
    )
    ax_win.add_patch(pred_box)
    ax_win.text(win_start + 20, 0.12, "ALERT",
                ha="center", va="center", fontsize=12,
                color="white", fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_ALERT,   label="Alert (label 0)"),
        mpatches.Patch(facecolor=C_DROWSY,  label="Drowsy (label 10)"),
        mpatches.Patch(facecolor="#AAAAAA", label="No label / transition"),
    ]
    ax_win.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.7)

    ax_win.set_title(
        "⑤ Sliding Window → One Prediction per Window (Alert / Drowsy)",
        fontsize=10, color=C_DARK, pad=4
    )

    plt.savefig(pdf, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("✓ Page 1 done")


# ---------------------------------------------------------------------------
# Page 2 — Signal over time + sample predictions
# ---------------------------------------------------------------------------

def page2(pdf):
    df = pd.read_csv(SUMMARY_CSV)
    p3 = df[df["participant_id"] == PARTICIPANT].copy()

    # Build continuous timeline: alert (vid=0) then drowsy (vid=10)
    alert_df  = p3[p3["video_id"] == ALERT_VID].reset_index(drop=True)
    drowsy_df = p3[p3["video_id"] == DROWSY_VID].reset_index(drop=True)

    # Subsample for speed — take every 5th frame
    step = 5
    alert_df  = alert_df.iloc[::step].reset_index(drop=True)
    drowsy_df = drowsy_df.iloc[::step].reset_index(drop=True)

    n_alert  = len(alert_df)
    n_drowsy = len(drowsy_df)
    n_total  = n_alert + n_drowsy

    combined = pd.concat([alert_df, drowsy_df], ignore_index=True)
    x = np.arange(n_total)

    fig = plt.figure(figsize=(11.69, 8.27), facecolor=C_BG)
    fig.suptitle(
        f"Drowsiness Signal & Model Predictions — {PARTICIPANT}",
        fontsize=18, fontweight="bold", color=C_DARK, y=0.97
    )

    gs = gridspec.GridSpec(
        3, 1,
        figure=fig,
        top=0.91, bottom=0.25,
        left=0.08, right=0.97,
        hspace=0.55
    )

    def shade_zones(ax):
        ax.axvspan(0,       n_alert,  alpha=0.10, color=C_ALERT,  label="Alert zone")
        ax.axvspan(n_alert, n_total,  alpha=0.10, color=C_DROWSY, label="Drowsy zone")
        ax.axvline(n_alert, color=C_DARK, lw=1.5, linestyle="--", alpha=0.6)
        ax.text(n_alert / 2,      ax.get_ylim()[1] * 0.92,
                "ALERT",  ha="center", fontsize=9,  color=C_ALERT,  fontweight="bold")
        ax.text(n_alert + n_drowsy / 2, ax.get_ylim()[1] * 0.92,
                "DROWSY", ha="center", fontsize=9, color=C_DROWSY, fontweight="bold")

    # ── EAR plot ─────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, combined["mean_EAR"], color=C_ACCENT, lw=0.8, alpha=0.5, label="EAR (raw)")
    ax1.plot(x, combined["mean_EAR_smooth"], color=C_ACCENT, lw=2.0, label="EAR (smoothed)")
    ax1.set_ylabel("EAR", fontsize=10, color=C_DARK)
    ax1.set_title("Eye Aspect Ratio (EAR) — lower = more closed eyes", fontsize=10, color=C_DARK)
    ax1.set_facecolor(C_BG)
    ax1.set_xlim(0, n_total)
    shade_zones(ax1)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xticklabels([])

    # ── PERCLOS plot ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(x, combined["PERCLOS"], color=C_DROWSY, lw=1.5, label="PERCLOS")
    ax2.set_ylabel("PERCLOS", fontsize=10, color=C_DARK)
    ax2.set_title("PERCLOS — percentage of time eyes are closed (higher = more drowsy)", fontsize=10, color=C_DARK)
    ax2.set_facecolor(C_BG)
    ax2.set_xlim(0, n_total)
    shade_zones(ax2)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xticklabels([])

    # ── MAR (yawning) plot ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(x, combined["MAR"], color="#9B59B6", lw=0.8, alpha=0.5, label="MAR (raw)")
    ax3.plot(x, combined["MAR_smooth"], color="#9B59B6", lw=2.0, label="MAR (smoothed)")
    ax3.set_ylabel("MAR", fontsize=10, color=C_DARK)
    ax3.set_title("Mouth Aspect Ratio (MAR) — spikes indicate yawning", fontsize=10, color=C_DARK)
    ax3.set_xlabel("Frame index (subsampled)", fontsize=9, color=C_DARK)
    ax3.set_facecolor(C_BG)
    ax3.set_xlim(0, n_total)
    shade_zones(ax3)
    ax3.legend(loc="upper right", fontsize=8)

    # ── Sample window thumbnails (bottom strip) ───────────────────────────────
    # Pick 3 alert windows and 3 drowsy windows
    sample_windows = [
        (ALERT_VID,  200,  "ALERT",  C_ALERT),
        (ALERT_VID,  600,  "ALERT",  C_ALERT),
        (ALERT_VID,  1000, "ALERT",  C_ALERT),
        (DROWSY_VID, 200,  "DROWSY", C_DROWSY),
        (DROWSY_VID, 600,  "DROWSY", C_DROWSY),
        (DROWSY_VID, 1000, "DROWSY", C_DROWSY),
    ]

    thumb_gs = gridspec.GridSpec(
        1, 6,
        figure=fig,
        top=0.20, bottom=0.03,
        left=0.05, right=0.97,
        wspace=0.08
    )

    fig.text(0.5, 0.22,
             "Sample Windows — Center Frame + Ground Truth Label",
             ha="center", fontsize=11, color=C_DARK, fontweight="bold")

    for col, (vid, frame_n, label, color) in enumerate(sample_windows):
        ax = fig.add_subplot(thumb_gs[0, col])
        mesh_dir = MESH_DIR / str(vid) / PARTICIPANT
        frames   = sorted(mesh_dir.glob("*.jpg"))
        idx = min(frame_n, len(frames) - 1)
        img = load_img(frames[idx]) if frames else None
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor("#CCCCCC")
        frame_ax(ax, color=color, lw=3)
        ax.set_xticks([]); ax.set_yticks([])

        # Label badge below image
        ax.text(0.5, -0.12, label,
                transform=ax.transAxes,
                ha="center", va="top", fontsize=10,
                color="white", fontweight="bold",
                bbox=dict(facecolor=color, edgecolor="none",
                          boxstyle="round,pad=0.3"))

        # Frame timestamp
        t_sec = round(idx / 4, 1)
        ax.set_title(f"t = {t_sec}s", fontsize=8, color=C_DARK, pad=3)

    plt.savefig(pdf, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("✓ Page 2 done")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    # Check required dirs exist
    missing = []
    for d in [MESH_DIR, PATCH_DIR, SUMMARY_CSV.parent]:
        if not d.exists():
            missing.append(str(d))
    if missing:
        print("ERROR: missing paths:", missing)
        sys.exit(1)

    print(f"Generating demo PDF -> {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        page1(pdf)
        page2(pdf)

        # PDF metadata
        meta = pdf.infodict()
        meta["Title"]   = "Drowsiness Detection — Demo"
        meta["Author"]  = PARTICIPANT
        meta["Subject"] = "FiLM+GRU pipeline visualization"

    print(f"\n✓ Saved: {OUT_PDF}")
    print("  Open it to review — 2 pages, A4 landscape.")
