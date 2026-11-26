"""Figure 2 of the Nature Communications revision (natcomm/main_v2.tex): the advantage of shortest replay exists only
while the cap binds, and a forced answer recovers it. Data: data/natcomm_32k.json (synth/natcomm_32k_analysis.py on the
32k MATH-500 generations of every Qwen3-4B arm, two sampling seeds for untouched, S, R, L).

    env -u PYTHONPATH .venv_mps/bin/python paper_brevity/make_fig_cap.py  -> figures/fig_cap.{pdf,png}

a  accuracy at thinking budgets 4k-32k, a sample not finished by t scored as wrong ("convention" in the JSON)
b  the same with a forced answer at t ("forced")
c  shortest minus random replay under both scorings, paired-bootstrap 95% CIs
d  tokens per sample at the 32k cap, relative change from the untouched model, with 95% CIs
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(ROOT, "data", "natcomm_32k.json")))
OUT = os.path.join(ROOT, "figures", "fig_cap")

C = {"S": "#1b7f6e", "R": "#90959e", "L": "#c4573a", "base": "#2f3e5c", "KA": "#7a6a9a"}
NAME = {"S": "shortest", "R": "random", "L": "longest", "base": "untouched", "KA": "keep-all"}
INK, INK2, AXIS, GRID = "#1d1d1b", "#4f4e4a", "#a3a29b", "#ebeae6"
FS, FSS = 6.6, 6.0
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": FS,
    "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.labelsize": FS,
    "xtick.labelsize": FSS, "ytick.labelsize": FSS, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white",
})
BUD = ["4096", "8192", "16384", "32768"]
XL = ["4k", "8k", "16k", "32k"]
x = np.arange(4)
ARMS = ["base", "L", "R", "S"]                       # S drawn last, on top

fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.25), dpi=300, gridspec_kw=dict(width_ratios=[1, 1, 1, 0.9]))
fig.subplots_adjust(left=0.06, right=0.985, bottom=0.2, top=0.82, wspace=0.5)


def letter(ax, l, title):
    ax.text(-0.22, 1.16, l, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")
    ax.text(-0.10, 1.16, title, transform=ax.transAxes, fontsize=7.0, fontweight="bold", color=INK, va="bottom")


for ax, key, l, title in ((axs[0], "convention", "a", "Cut sample scored wrong"),
                          (axs[1], "forced", "b", "Forced answer at the budget")):
    for a in ARMS:
        y = [D["forced"]["arms"][a][b][key] for b in BUD]
        ax.plot(x, y, "-o", color=C[a], lw=1.3 if a == "S" else 1.0, ms=3.2, zorder=3 if a == "S" else 2,
                label=NAME[a])
    ax.set_xticks(x)
    ax.set_xticklabels(XL)
    ax.set_ylim(35, 95)
    ax.set_yticks([40, 50, 60, 70, 80, 90])
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_xlabel("thinking budget (tokens)")
    letter(ax, l, title)
axs[0].set_ylabel("accuracy per sample (%)")
axs[0].legend(frameon=False, fontsize=FSS, loc="lower right", handlelength=1.4, borderaxespad=0.2)

# c: S - R under both scorings
ax = axs[2]
rows = {k: [] for k in ("convention", "forced")}
for c in D["forced"]["contrasts"]:
    if c["a"] == "S" and c["b"] == "R":
        for k in rows:
            rows[k].append((str(c["budget"]), c[k]["est"], c[k]["ci95"]))
for k, off, col, lab in (("convention", -0.1, C["S"], "cut scored wrong"), ("forced", 0.1, INK2, "forced answer")):
    r = sorted(rows[k], key=lambda t: int(t[0]))
    xs = x + off
    est = [t[1] for t in r]
    ax.plot(xs, est, "o", ms=3.6, color=col, mfc=col if k == "convention" else "white", mew=1.0, zorder=3, label=lab)
    for xi, t in zip(xs, r):
        ax.plot([xi, xi], t[2], color=col, lw=1.2, zorder=2)
ax.axhline(0, color=AXIS, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(XL)
ax.set_xlabel("thinking budget (tokens)")
ax.set_ylabel("shortest minus random (points)")
ax.grid(axis="y", color=GRID, lw=0.5)
ax.legend(frameon=False, fontsize=FSS, loc="upper right", handlelength=1.0, borderaxespad=0.2)
letter(ax, "c", "Shortest minus random")

# d: tokens at 32k relative to untouched
ax = axs[3]
tok = {c["a"]: c["tokens_rel_pct"] for c in D["m32"]["contrasts"] if c["b"] == "base"}
tok["S"] = next(c["tokens_rel_pct"] for c in D["m32"]["contrasts"] if c["a"] == "S" and c["b"] == "base")
order = ["S", "R", "KA", "L"]
for i, a in enumerate(order):
    t = tok[a]
    ax.barh(i, t["est"], color=C[a], height=0.62)
    ax.plot(t["ci95"], [i, i], color=INK, lw=0.9)
    ax.text(t["ci95"][1] + 0.4 if t["est"] > 0 else 0.5, i, f"{t['est']:+.1f}%".replace("-", "−"),
            va="center", ha="left", fontsize=FSS, color=INK)
ax.axvline(0, color=AXIS, lw=0.8)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([NAME[a] for a in order])
ax.invert_yaxis()
ax.set_xlim(-9, 10.5)
ax.set_xticks([-5, 0, 5])
ax.set_xlabel("tokens vs untouched (%)")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
letter(ax, "d", "Tokens at the 32k cap")

fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
print("wrote", OUT)
for k in rows:
    print(k, sorted(rows[k], key=lambda t: int(t[0])))
print("tokens", {a: tok[a] for a in order})
