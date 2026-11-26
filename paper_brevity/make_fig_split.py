#!/usr/bin/env python3
"""Fig. 1: what a token limit does to reasoning (a, schematic), how an accuracy gain splits into a part from finishing
and a part that survives a forced answer (b, schematic), and that split measured against the token budget for four
published methods and shortest replay (c-g, data).
    python3 paper_brevity/make_fig_split.py -> paper_brevity/figures/fig_split.{pdf,png}"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(os.path.join(ROOT, "data", "natcomm_ext.json")))
Q = json.load(open(os.path.join(ROOT, "data", "natcomm_32k.json")))
INK, INK2, MUTED, AXIS, GRID = "#1d1d1b", "#4f4e4a", "#6f6e69", "#a3a29b", "#ebeae6"
C_FIN, C_ANS, C_APP, C_OK, C_CUT = "#c98a2b", "#2f3e5c", "#a3a29b", "#1b7f6e", "#c4573a"
SYM = dict(fontfamily="DejaVu Sans")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
                     "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": AXIS,
                     "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "pdf.fonttype": 42})
fig = plt.figure(figsize=(7.2, 4.6))

# ---- a: attempts against the limit ----
axa = fig.add_axes([0.03, 0.58, 0.50, 0.36]); LIM = 8.0
axa.set_xlim(0, 12.6); axa.set_ylim(-0.25, 3.55); axa.axis("off")
for y, L, found, kind in ((2.75, 4.2, 3.4, "fin"), (1.75, 5.4, 4.5, "fin"), (0.75, 11.4, 5.9, "cut")):
    end = min(L, LIM)
    axa.add_patch(FancyBboxPatch((0, y - 0.17), end, 0.34, boxstyle="round,pad=0,rounding_size=0.1", fc="#d9dde6", ec="none"))
    if kind == "cut":
        axa.add_patch(FancyBboxPatch((LIM, y - 0.17), L - LIM, 0.34, boxstyle="round,pad=0,rounding_size=0.1",
                                     fc="none", ec="#b8bcc6", ls=(0, (2, 2)), lw=0.7))
    axa.plot([found], [y], "o", ms=4.2, color=C_OK, zorder=5)
    if kind == "fin":
        axa.text(end + 0.2, y, "answer ✓", va="center", color=C_OK, fontsize=7.2, fontweight="bold", **SYM)
axa.annotate("right answer already\nreached here", xy=(5.9, 0.6), xytext=(3.0, -0.2), fontsize=6.4, color=INK2,
             ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
axa.axvline(LIM, ymin=0.02, ymax=0.93, color=INK, lw=1.0, ls=(0, (4, 2)))
axa.text(LIM, 3.42, "token limit", ha="center", va="bottom", fontsize=7, color=INK)
axa.text(LIM + 0.25, 1.06, "✂", fontsize=10, color=C_CUT, va="center", **SYM)
axa.text(LIM + 0.25, 0.43, "scored wrong", fontsize=6.6, color=C_CUT, va="center")
axa.text(LIM + 0.25, 0.06, "forced answer ✓", fontsize=6.6, color=C_OK, va="center", fontweight="bold", **SYM)
axa.text(0, 3.42, "reasoning attempts", fontsize=7, color=INK2, va="bottom")
axa.text(-0.3, 3.62, "a", fontsize=10, fontweight="bold", color=INK, va="bottom")

# ---- b: the split of a gain (schematic) ----
axb = fig.add_axes([0.62, 0.60, 0.36, 0.32])
for x, fin, ans, lab in ((0, 7.0, 1.5, "tight limit"), (1, 0.8, 0.6, "loose limit")):
    axb.bar(x, ans, width=0.5, color=C_ANS)
    axb.bar(x, fin, bottom=ans, width=0.5, color=C_FIN)
    axb.text(x, -0.5, lab, ha="center", va="top", fontsize=6.6, color=INK2)
axb.text(0.33, 1.5 + 3.5, "from finishing\nin time", color=C_FIN, fontsize=6.6, va="center")
axb.text(0.33, 0.75, "with forced answers:\nfaster progress\nor better answers", color=C_ANS, fontsize=6.4, va="center")
axb.annotate("", xy=(-0.32, 8.5), xytext=(-0.32, 0), arrowprops=dict(arrowstyle="<->", color=C_APP, lw=0.8))
axb.text(-0.38, 4.25, "apparent gain", rotation=90, ha="right", va="center", fontsize=6.4, color=MUTED)
axb.set_xlim(-0.7, 1.6); axb.set_ylim(0, 9); axb.axis("off")
axb.text(-0.75, 9.4, "b", fontsize=10, fontweight="bold", color=INK, va="bottom")
axb.text(0.45, 9.4, "an accuracy gain under a limit (schematic)", fontsize=6.8, color=INK2, va="bottom", ha="center")

# ---- c-g: the split measured against the budget ----
panels = []
for m, b, name in (("arora_a02", "r1base", "Length penalty"), ("thinkprune", "r1base", "ThinkPrune"),
                   ("dler", "r1base", "DLER"), ("l1max4k", "dsr", "L1-Max (told 4k)")):
    T = [2048, 4096, 8192, 16384]
    fo = [E["arms"][m][str(t)]["forced"] - E["arms"][b][str(t)]["forced"] for t in T]
    ap = [E["arms"][m][str(t)]["cut_as_wrong"] - E["arms"][b][str(t)]["cut_as_wrong"] for t in T]
    tb = 4096 if m == "l1max4k" else 8192
    tok = [c["tokens_rel_pct"] for c in E["contrasts"] if c["a"] == m and c["b"] == b and c["budget"] == tb][0]
    panels.append((name, T, fo, ap, tok, tb))
F = Q["forced"]["arms"]; T = [4096, 8192, 16384, 32768]
fo = [F["S"][str(t)]["forced"] - F["base"][str(t)]["forced"] for t in T]
ap = [F["S"][str(t)]["convention"] - F["base"][str(t)]["convention"] for t in T]
panels.append(("Shortest replay", T, fo, ap, Q["m32"]["contrasts"][0]["tokens_rel_pct"]["est"], 32768))
W = 0.17; X0 = 0.055
for k, (name, T, fo, ap, tok, tb) in enumerate(panels):
    ax = fig.add_axes([X0 + k * (W + 0.023), 0.11, W, 0.33])
    fin = [a - f for a, f in zip(ap, fo)]
    ax.axhline(0, color=AXIS, lw=0.5)
    ax.fill_between(range(len(T)), 0, fin, color=C_FIN, alpha=0.18, lw=0)
    ax.plot(range(len(T)), ap, color=C_APP, lw=1.0, ls=(0, (3, 2)))
    ax.plot(range(len(T)), fin, color=C_FIN, lw=1.6, marker="o", ms=2.6)
    ax.plot(range(len(T)), fo, color=C_ANS, lw=1.6, marker="o", ms=2.6)
    if name.startswith("L1-Max"):
        ax.axvspan(1.5, len(T) - 0.6, color="#f1f0ec", zorder=0, lw=0)
        ax.text(2.45, 0.96, "beyond its\n4k instruction", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=5.6, color=MUTED)
    ax.set_xticks(range(len(T))); ax.set_xticklabels([f"{t // 1024}k" for t in T])
    ax.set_title(name, fontsize=7, pad=9, color=INK)
    ax.text(0.5, 1.0, f"tokens at {tb // 1024}k: {tok:+.0f}%".replace("-", "−"), transform=ax.transAxes, ha="center",
            va="bottom", fontsize=5.8, color=MUTED)
    ax.set_xlabel("token budget", fontsize=6.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=6.2)
    ax.text(-0.22, 1.13, "cdefg"[k], transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom")
    if k == 0:
        ax.set_ylabel("gain over starting model (points)", fontsize=6.6)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=C_APP, lw=1.0, ls=(0, (3, 2)), label="apparent gain (cut attempts scored wrong)"),
     Line2D([], [], color=C_FIN, lw=1.6, marker="o", ms=2.6, label="from finishing in time"),
     Line2D([], [], color=C_ANS, lw=1.6, marker="o", ms=2.6, label="with forced answers (faster progress or better answers)")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=6.4, bbox_to_anchor=(0.5, -0.005))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_split.{ext}"), dpi=240)
print("wrote figures/fig_split.png")
