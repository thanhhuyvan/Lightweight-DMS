"""
generate_ablation_chart_v2.py
------------------------------
Vertical bar chart — F1 Score on Y axis, models on X axis.
Output: report/ablation_comparison_v2.png / .pdf

Run: python generate_ablation_chart_v2.py
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path(__file__).parent / "report"
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
models = [
    "XGBoost\n(Geometry Only)",
    "CNN Only\n(TinyPatchCNN)",
    "Late Fusion\n(CNN + Geometry)",
    "Concat+GRU\n(No FiLM)",
    "FiLM+GRU\n+ Attention\n(Ours)",
]

f1_scores  = [0.490, 0.742, 0.776, 0.810, 0.827]
std_scores = [0.031, 0.263, 0.264, 0.180, 0.144]   # from logs

COLORS = [
    "#4A4E69",   # XGBoost     — muted purple-grey
    "#5E81AC",   # CNN only    — slate blue
    "#81A1C1",   # Late Fusion — lighter blue
    "#88C0D0",   # Concat+GRU  — cyan
    "#A3BE8C",   # FiLM+GRU    — green (best)
]

EDGE_COLORS = [
    "#6C7086",
    "#7AA2C8",
    "#A3C3D9",
    "#A8D8E8",
    "#C3D9A8",
]

SOTA_GOAL  = 0.80
BAR_WIDTH  = 0.55

# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#11111B")
ax.set_facecolor("#1A1A2E")

x = np.arange(len(models))

bars = ax.bar(
    x, f1_scores,
    width=BAR_WIDTH,
    color=COLORS,
    edgecolor=EDGE_COLORS,
    linewidth=1.2,
    zorder=3,
    yerr=std_scores,
    error_kw=dict(ecolor="#CDD6F4", elinewidth=1.2, capsize=5, capthick=1.2, alpha=0.7),
)

# ── Gradient-style shading on best bar ───────────────────────────────────
best_idx = f1_scores.index(max(f1_scores))
bars[best_idx].set_edgecolor("#2ECC71")
bars[best_idx].set_linewidth(2.5)

# ── Value labels on top of bars ───────────────────────────────────────────
for i, (bar, score, std) in enumerate(zip(bars, f1_scores, std_scores)):
    color = "#2ECC71" if i == best_idx else "#CDD6F4"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        score + std + 0.018,
        f"{score:.3f}",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold", color=color,
        zorder=5,
    )

# ── SOTA goal line ────────────────────────────────────────────────────────
ax.axhline(SOTA_GOAL, color="#E74C3C", linestyle="--",
           linewidth=1.8, zorder=4, alpha=0.9)
ax.text(len(models) - 0.5, SOTA_GOAL + 0.012,
        f"SOTA Target  (F1 = {SOTA_GOAL})",
        color="#E74C3C", fontsize=9, ha="right", fontweight="bold")

# ── Improvement bracket: XGBoost → FiLM+GRU ──────────────────────────────
bracket_y = 0.92
ax.annotate(
    "", xy=(x[best_idx], bracket_y), xytext=(x[0], bracket_y),
    arrowprops=dict(arrowstyle="<->", color="#F39C12", lw=1.8),
    annotation_clip=False,
)
ax.text((x[0] + x[best_idx]) / 2, bracket_y + 0.022,
        "+33.7 pp improvement",
        ha="center", fontsize=9.5, color="#F39C12",
        fontweight="bold")

# ── "Ours" callout ────────────────────────────────────────────────────────
ax.annotate(
    "Proposed\nArchitecture",
    xy=(x[best_idx], f1_scores[best_idx] + std_scores[best_idx] + 0.07),
    xytext=(x[best_idx] + 0.6, f1_scores[best_idx] + std_scores[best_idx] + 0.13),
    fontsize=9, color="#2ECC71", fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color="#2ECC71", lw=1.3),
    annotation_clip=False,
)

# ── Axes styling ──────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10, color="#CDD6F4", linespacing=1.4)
ax.set_ylabel("Macro F1 Score", fontsize=12, color="#CDD6F4", labelpad=10)
ax.set_ylim(0.0, 1.05)
ax.set_xlim(-0.5, len(models) - 0.5)

ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.yaxis.set_tick_params(colors="#6C7086")
ax.xaxis.set_tick_params(colors="#CDD6F4", length=0)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_edgecolor("#313244")

ax.yaxis.grid(True, color="#2D2D44", linestyle="--", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

# ── Title ─────────────────────────────────────────────────────────────────
ax.set_title(
    "Ablation Study — Macro F1 Score by Architecture\n"
    "(5-fold LOPO-CV, participant1 excluded, error bars = ±1 std)",
    fontsize=13, fontweight="bold", color="#CDD6F4", pad=14,
)

# ── Legend ────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor="#4A4E69", edgecolor="#6C7086", label="Geometry baseline"),
    mpatches.Patch(facecolor="#5E81AC", edgecolor="#7AA2C8", label="Visual (CNN) baseline"),
    mpatches.Patch(facecolor="#81A1C1", edgecolor="#A3C3D9", label="Static fusion"),
    mpatches.Patch(facecolor="#88C0D0", edgecolor="#A8D8E8", label="Temporal (no FiLM)"),
    mpatches.Patch(facecolor="#A3BE8C", edgecolor="#C3D9A8", label="FiLM+GRU+Attention (proposed)"),
]
ax.legend(
    handles=legend_elements, loc="upper left",
    fontsize=8.5, framealpha=0.15,
    facecolor="#1A1A2E", edgecolor="#313244",
    labelcolor="#CDD6F4",
)

fig.text(
    0.99, 0.01,
    "participant6 excluded from headline F1 (behavioral inversion documented)",
    ha="right", fontsize=7.5, color="#6C7086", style="italic",
)

plt.tight_layout(rect=[0, 0.02, 1, 1])

# ── Save ──────────────────────────────────────────────────────────────────
png_path = OUT_DIR / "ablation_comparison_v2.png"
pdf_path = OUT_DIR / "ablation_comparison_v2.pdf"

plt.savefig(png_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
