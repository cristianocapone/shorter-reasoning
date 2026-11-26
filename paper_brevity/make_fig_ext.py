#!/usr/bin/env python3
"""Fig. 3: published length-reduction methods against their starting models (second pre-registered test).
Accuracy on MATH-500 at budgets 1k-16k, with cut attempts scored as wrong (dashed) and with forced answers (solid).
    python3 paper_brevity/make_fig_ext.py  ->  paper_brevity/figures/fig_ext.{pdf,png}   (reads paper_brevity/data/natcomm_ext.json)
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "data", "natcomm_ext.json")))
INK, INK2, MUTED, AXIS, GRID = "#1d1d1b", "#4f4e4a", "#6f6e69", "#a3a29b", "#ebeae6"
C_BASE = "#2f3e5c"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "pdf.fonttype": 42, "figure.facecolor": "white"})
B = [1024, 2048, 4096, 8192, 16384]
PANELS = [("arora_a02", "r1base", "Length penalty (\u03b1 = 0.2)", "R1-Distill-Qwen-1.5B", "#1b7f6e"),
          ("thinkprune", "r1base", "ThinkPrune", "R1-Distill-Qwen-1.5B", "#c98a2b"),
          ("dler", "r1base", "DLER", "R1-Distill-Qwen-1.5B", "#6a4c93"),
          ("l1max4k", "dsr", "L1-Max (told 4k tokens)", "DeepScaleR-1.5B", "#c4573a")]

fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.35), sharey=True)
fig.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.27, wspace=0.12)
for k, (ax, (m, b, title, bname, col)) in enumerate(zip(axs, PANELS)):
    for arm, c, lw in ((b, C_BASE, 1.3), (m, col, 1.6)):
        cut = [R["arms"][arm][str(t)]["cut_as_wrong"] for t in B]
        frc = [R["arms"][arm][str(t)]["forced"] for t in B]
        ax.plot(range(5), cut, color=c, lw=lw, ls=(0, (3, 2)), marker="o", ms=2.6, mfc="white", mew=0.9)
        ax.plot(range(5), frc, color=c, lw=lw, marker="o", ms=2.6, mew=0)
    tok = {c["budget"]: c["tokens_rel_pct"] for c in R["contrasts"] if c["a"] == m and c["b"] == b}
    ax.text(0.97, 0.05, f"tokens with forced\nanswers at 8k: {tok[8192]:+.0f}%".replace("-", "−"),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4, color=INK2)
    ax.set_title(title, fontsize=7.4, color=col, pad=11, fontweight="bold")
    ax.text(0.5, 1.01, f"vs {bname}", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3, color=MUTED)
    ax.set_xticks(range(5)); ax.set_xticklabels(["1k", "2k", "4k", "8k", "16k"])
    ax.axvspan(1.75, 3.25, color=GRID, zorder=0, lw=0)
    ax.set_ylim(0, 92); ax.set_xlim(-0.3, 4.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.text(-0.1, 1.17, "abcd"[k], transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom")
    ax.set_xlabel("token budget", fontsize=7)
axs[0].set_ylabel("accuracy (%)", fontsize=7)
# legend
from matplotlib.lines import Line2D
h = [Line2D([], [], color=INK2, lw=1.4, marker="o", ms=2.6, mew=0, label="forced answer"),
     Line2D([], [], color=INK2, lw=1.4, ls=(0, (3, 2)), marker="o", ms=2.6, mfc="white", mew=0.9, label="cut attempts scored wrong"),
     Line2D([], [], color=C_BASE, lw=1.4, label="starting model")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=6.6, bbox_to_anchor=(0.53, -0.02))
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_ext.{ext}"), dpi=220)
print("wrote figures/fig_ext.png")
