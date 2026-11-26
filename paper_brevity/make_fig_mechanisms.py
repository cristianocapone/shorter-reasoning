#!/usr/bin/env python3
"""Fig. 2: three ways of shortening reasoning (exploratory, existing runs).
 a-c  reasoning-length distributions of a method and its starting model: rescaling (length penalty), tail cut (DLER),
      path choice (shortest replay)
 d    scale x shape map of every model, colour = gain with forced answers at 8k
 e    path choice per problem: share of long attempts on problems that the untouched model always solves short,
      sometimes long, or always long (Qwen3-4B, 32k generations, 8 attempts per problem)
 f    rescaling test: share of attempts finished against the starting model's lengths multiplied by the median ratio
    python3 paper_brevity/make_fig_mechanisms.py -> paper_brevity/figures/fig_mechanisms.{pdf,png}"""
import glob, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(ROOT)
E = json.load(open(os.path.join(ROOT, "data", "natcomm_ext.json")))
Q = json.load(open(os.path.join(ROOT, "data", "natcomm_32k.json")))
INK, INK2, MUTED, AXIS, GRID = "#1d1d1b", "#4f4e4a", "#6f6e69", "#a3a29b", "#ebeae6"
COL = {"arora_a02": "#1b7f6e", "dler": "#6a4c93", "S": "#2f3e5c", "R": "#90959e", "base": "#c9ccd3"}
NAME = {"arora_a0": "RL, no penalty", "arora_a02": "length penalty", "thinkprune": "ThinkPrune", "dler": "DLER",
        "dsr": "DeepScaleR", "l1max4k": "L1-Max", "S": "shortest replay", "R": "random replay", "L": "longest replay",
        "KA": "keep-all"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
                     "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": AXIS,
                     "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "pdf.fonttype": 42})


def ext(arm, fins=False):
    x = []
    for f in glob.glob(f"{REPO}/runs_synth_ext/runs_synth/agent_ncx_{arm}_s0/shard_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            x += [v if (fi or not fins) else 1e9 for v, fi in zip(r["lens"], r["fins"])]
    return np.sort(np.array(x, float))


def qw(arm, by_task=False):
    d = defaultdict(list)
    for s in (0, 1):
        for l in open(f"{ROOT}/data/nc32_records/agent_nc32_q4_{arm}_s{s}/forced.jsonl"):
            r = json.loads(l)
            if r["budget"] == 32768:
                d[r["task_id"]].append(r["n_tokens"])
    return d if by_task else np.sort(np.concatenate([np.array(v, float) for v in d.values()]))


fig = plt.figure(figsize=(7.2, 4.7))
# ---- a-c: distributions ----
P = [("rescaling: length penalty", ext("r1base"), ext("arora_a02"), "R1-Distill-Qwen-1.5B", "length penalty", COL["arora_a02"], 16384),
     ("tail cut: DLER", ext("r1base"), ext("dler"), "R1-Distill-Qwen-1.5B", "DLER", COL["dler"], 16384),
     ("path choice: shortest replay", qw("base"), qw("S"), "Qwen3-4B, untouched", "shortest replay", COL["S"], 32768)]
for k, (title, b, m, bn, mn, c, cap) in enumerate(P):
    ax = fig.add_axes([0.04 + k * 0.33, 0.60, 0.29, 0.27])
    bins = np.logspace(np.log10(150), np.log10(cap * 1.05), 40)
    ax.hist(b, bins=bins, weights=np.ones(len(b)) / len(b), histtype="stepfilled", color="#d3d6dc", label=bn)
    ax.hist(m, bins=bins, weights=np.ones(len(m)) / len(m), histtype="step", color=c, lw=1.5, label=mn)
    ax.set_xscale("log"); ax.set_yticks([])
    ax.set_title(title, fontsize=7.2, pad=3, color=INK, loc="left")
    ax.set_xlabel("reasoning length (tokens)", fontsize=6.6)
    ax.legend(frameon=False, fontsize=5.6, loc="upper right", bbox_to_anchor=(1.0, 1.32), ncol=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(-0.05, 1.08, "abc"[k], transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom")

# ---- d: scale x shape ----
axd = fig.add_axes([0.06, 0.09, 0.27, 0.38])
pairs = [("arora_a0", "r1base", "e"), ("arora_a02", "r1base", "e"), ("thinkprune", "r1base", "e"), ("dler", "r1base", "e"),
         ("dsr", "r1base", "e"), ("l1max4k", "dsr", "e"), ("S", "base", "q"), ("R", "base", "q"), ("L", "base", "q"), ("KA", "base", "q")]
norm = TwoSlopeNorm(vmin=-7, vcenter=0, vmax=7); cmap = plt.get_cmap("RdBu")
axd.axhspan(0.85, 1.15, color=GRID, lw=0, zorder=0)
for txt, y in (("rescaling", 1.0), ("tail cut", 0.5), ("path choice", 1.42)):
    axd.text(0.21, y, txt, fontsize=6.0, color=MUTED, va="center")
off = {"arora_a0": (0, 0.08, "center"), "R": (0, 0.08, "center"), "KA": (0, -0.08, "center"), "L": (0, -0.08, "center")}
for m, b, src in pairs:
    Lm, Lb = (ext(m), ext(b)) if src == "e" else (qw(m), qw(b))
    sc = np.quantile(Lm, 0.5) / np.quantile(Lb, 0.5)
    sh = (np.quantile(Lm, 0.9) / np.quantile(Lb, 0.9)) / sc
    bud = "4096" if m == "l1max4k" else "8192"
    g = (E["arms"][m][bud]["forced"] - E["arms"][b][bud]["forced"]) if src == "e" else \
        (Q["forced"]["arms"][m]["8192"]["forced"] - Q["forced"]["arms"]["base"]["8192"]["forced"])
    axd.scatter([sc], [sh], s=30, c=[g], cmap=cmap, norm=norm, edgecolors=INK2, linewidths=0.5, zorder=3)
    dx, dy, ha = off.get(m, (0.03, 0, "left"))
    axd.text(sc + dx, sh + dy, NAME[m], fontsize=5.6, color=INK2, va="center", ha=ha)
axd.set_xlim(0.2, 1.35); axd.set_ylim(0.3, 1.62)
axd.set_xlabel("scale: median length ratio", fontsize=6.6)
axd.set_ylabel("shape: tail ratio / median ratio", fontsize=6.6)
cax = axd.inset_axes([0.58, 0.93, 0.4, 0.035])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[-6, 0, 6])
cax.set_title("gain, forced answers, 8k", fontsize=5.4, color=INK2, pad=2); cb.ax.tick_params(labelsize=5.4); cb.outline.set_linewidth(0.4)

# ---- e: path choice per problem ----
axe = fig.add_axes([0.42, 0.09, 0.24, 0.38])
B, S, R = qw("base", True), qw("S", True), qw("R", True)
ids = sorted(set(B) & set(S) & set(R)); thr = 3217
pb = np.array([np.mean(np.array(B[t]) > thr) for t in ids])
groups = [("always short", pb == 0), ("sometimes long", (pb > 0) & (pb < 1)), ("always long", pb == 1)]
for gi, (gname, mask) in enumerate(groups):
    for ai, (arm, D, c) in enumerate((("untouched", B, COL["base"]), ("random replay", R, COL["R"]), ("shortest replay", S, COL["S"]))):
        v = np.mean([np.mean(np.array(D[t]) > thr) for t, mm in zip(ids, mask) if mm])
        axe.bar(gi + (ai - 1) * 0.26, v, width=0.24, color=c, label=arm if gi == 0 else None)
    axe.text(gi, -0.12, f"{gname}\n(n = {mask.sum()})", ha="center", va="top", fontsize=5.9, color=INK2)
axe.set_xticks([]); axe.set_ylim(0, 1.05); axe.set_ylabel(f"share of long attempts (> {thr:,} tokens)", fontsize=6.6)
axe.legend(frameon=False, fontsize=5.8, loc="upper left")
axe.text(0.5, -0.33, "problems, by the untouched model's 8 attempts", transform=axe.transAxes, ha="center", fontsize=6.2, color=INK2)
for s in ("top", "right"):
    axe.spines[s].set_visible(False)

# ---- f: rescaling test ----
axf = fig.add_axes([0.75, 0.09, 0.23, 0.38])
t = np.logspace(np.log10(500), np.log10(16384), 120)
Lb = ext("r1base", True); Fb = lambda x: np.searchsorted(Lb, x, side="right") / len(Lb)
axf.plot(t, Fb(t), color="#6f7480", lw=1.3)
for m in ("arora_a02", "dler"):
    Lm = ext(m, True); lam = np.median(ext(m)) / np.median(ext("r1base"))
    axf.plot(t, np.searchsorted(Lm, t, side="right") / len(Lm), color=COL[m], lw=1.5)
    axf.plot(t, Fb(t / lam), color=COL[m], lw=1.0, ls=(0, (3, 2)))
axf.text(520, 0.74, "length penalty", color=COL["arora_a02"], fontsize=6.0)
axf.text(1700, 0.96, "DLER", color=COL["dler"], fontsize=6.0, va="bottom")
axf.text(3600, 0.33, "starting\nmodel", color="#6f7480", fontsize=6.0)
axf.set_xscale("log", base=2); axf.set_xticks([512, 2048, 8192]); axf.set_xticklabels(["0.5k", "2k", "8k"])
axf.set_ylim(0, 1.04); axf.set_xlabel("token budget", fontsize=6.6); axf.set_ylabel("share of attempts finished", fontsize=6.6)
axf.plot([], [], color=INK2, lw=1.4, label="observed"); axf.plot([], [], color=INK2, lw=1.0, ls=(0, (3, 2)), label="lengths × λ")
axf.legend(frameon=False, fontsize=5.8, loc="lower right")
for ax, k in ((axd, "d"), (axe, "e"), (axf, "f")):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(-0.2, 1.04, k, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom")
for ext_ in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_mechanisms.{ext_}"), dpi=240)
print("wrote figures/fig_mechanisms.png")
