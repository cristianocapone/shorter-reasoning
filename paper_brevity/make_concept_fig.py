#!/usr/bin/env python3
"""Fig. 1 (schematic, idealised curves, no data): what a token limit does to reasoning attempts (a) and the three
things a length-reducing method can change, read from accuracy with forced answers against tokens (b).
    python3 paper_brevity/make_concept_fig.py  ->  paper_brevity/figures/fig_concept.{pdf,png}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures")
INK, INK2, MUTED = "#1d1d1b", "#4f4e4a", "#6f6e69"
AXIS, GRID = "#a3a29b", "#ebeae6"
C_BASE, C_STOP, C_FAST, C_DEEP, C_CUT, C_OK = "#2f3e5c", "#1b7f6e", "#c98a2b", "#6a4c93", "#c4573a", "#1b7f6e"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7.5,
    "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "pdf.fonttype": 42, "figure.facecolor": "white"})

fig = plt.figure(figsize=(7.2, 2.9))
axa = fig.add_axes([0.035, 0.12, 0.44, 0.74])
axb = fig.add_axes([0.575, 0.17, 0.4, 0.69])

# ---------- a: three attempts against the token limit ----------
LIM = 8.0
axa.set_xlim(0, 12.6); axa.set_ylim(-0.2, 3.55); axa.axis("off")
rows = [  # (y, length, answer found at, label)
    (2.75, 4.2, 3.4, "finishes"),
    (1.75, 5.4, 4.5, "finishes"),
    (0.75, 11.4, 5.9, "cut")]
for y, L, found, kind in rows:
    end = min(L, LIM)
    axa.add_patch(FancyBboxPatch((0, y - 0.17), end, 0.34, boxstyle="round,pad=0,rounding_size=0.1",
                                 fc="#d9dde6", ec="none"))
    if kind == "cut":  # the part the limit throws away
        axa.add_patch(FancyBboxPatch((LIM, y - 0.17), L - LIM, 0.34, boxstyle="round,pad=0,rounding_size=0.1",
                                     fc="none", ec="#b8bcc6", ls=(0, (2, 2)), lw=0.7))
    axa.plot([found], [y], "o", ms=4.2, color=C_OK, zorder=5)
    if kind == "finishes":
        axa.text(end + 0.2, y, "answer ✓", va="center", color=C_OK, fontsize=7.5, fontweight="bold", fontfamily="DejaVu Sans")
axa.annotate("right answer already\nreached here", xy=(5.9, 0.6), xytext=(3.0, -0.12), fontsize=6.6, color=INK2,
             ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
axa.axvline(LIM, ymin=0.02, ymax=0.93, color=INK, lw=1.0, ls=(0, (4, 2)))
axa.text(LIM, 3.42, "token limit", ha="center", va="bottom", fontsize=7.2, color=INK)
axa.text(LIM + 0.25, 1.06, "✂", fontsize=10, color=C_CUT, va="center", fontfamily="DejaVu Sans")
axa.text(LIM + 0.25, 0.43, "scored wrong", fontsize=6.8, color=C_CUT, va="center")
axa.text(LIM + 0.25, 0.08, "forced answer ✓", fontsize=6.8, color=C_OK, va="center", fontweight="bold", fontfamily="DejaVu Sans")
axa.text(0, 3.42, "reasoning attempts", fontsize=7.2, color=INK2, va="bottom")
axa.text(-0.25, 3.62, "a", fontsize=10, fontweight="bold", color=INK, va="bottom")

# ---------- b: accuracy with a forced answer, against tokens ----------
t = np.linspace(0, 16, 400)
acc = lambda A, tau: A * (1 - np.exp(-t / tau))
axb.plot(t, acc(0.80, 3.2), color=C_BASE, lw=1.8, zorder=3)
axb.plot(t, acc(0.80, 1.9), color=C_FAST, lw=1.5, zorder=2)
axb.plot(t, acc(0.89, 3.2), color=C_DEEP, lw=1.5, zorder=2)
axb.text(15.9, 0.80 - 0.035, "starting model", color=C_BASE, ha="right", va="top", fontsize=7)
axb.text(15.9, 0.89 + 0.012, "deeper: reasons better", color=C_DEEP, ha="right", va="bottom", fontsize=7)
axb.annotate("faster: same answers\nwith fewer tokens", xy=(2.2, acc(0.80, 1.9)[55]), xytext=(4.6, 0.30),
             color=C_FAST, fontsize=7, arrowprops=dict(arrowstyle="-", color=C_FAST, lw=0.6))
# stops earlier: same curve, typical length moves left
yb = 0.05
axb.plot([9.0], [yb], "|", ms=8, mew=1.4, color=C_BASE)
axb.plot([6.4], [yb], "|", ms=8, mew=1.4, color=C_STOP)
axb.annotate("", xy=(6.55, yb), xytext=(8.85, yb), arrowprops=dict(arrowstyle="->", color=C_STOP, lw=1.0))
axb.text(6.2, yb + 0.045, "stops earlier: same curve,\nends sooner", color=C_STOP, fontsize=7, ha="left", va="bottom")
axb.text(9.2, yb, "typical\nlength", color=MUTED, fontsize=6.2, va="center")
axb.set_xlim(0, 16); axb.set_ylim(0, 1.0)
axb.set_xticks([]); axb.set_yticks([])
axb.set_xlabel("tokens of reasoning before the answer is forced", fontsize=7.2)
axb.set_ylabel("accuracy with a forced answer", fontsize=7.2)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)
axb.text(-1.6, 1.06, "b", fontsize=10, fontweight="bold", color=INK, va="bottom")

os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_concept.{ext}"), dpi=220)
print("wrote", os.path.join(OUT, "fig_concept.png"))
