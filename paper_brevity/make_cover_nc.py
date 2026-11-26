"""Figure 1 of the Nature Communications version (natcomm/main.tex): the cover strip of make_cover.py with panel b
redrawn around the selection contrast. The ARC version keeps figures/cover.pdf from make_cover.py, unchanged.

    env -u PYTHONPATH .venv_mps/bin/python paper_brevity/make_cover_nc.py
    -> paper_brevity/figures/cover_nc.pdf      vector, Arial embedded as TrueType (pdf.fonttype 42)
       paper_brevity/figures/cover_nc.png      300 dpi
       paper_brevity/figures/cover_nc_caption.md   caption draft; its numbers come from the same data

Differences from make_cover.py (editor review, natcomm): panel b plots shortest minus random and shortest minus longest
replay in every setting (the selection contrast, which is the claim) instead of each rule minus the untouched model
(that comparison is Supplementary Fig. 1 and Table 1), and replaces the two ARC-AGI-2 headline numbers against the
untouched model by a test pooled over the four settings (Stouffer's method on the one-sided bootstrap tails stored in
the JSON; a tail of 0 is floored at 1/10,000, the bootstrap's resolution, which can only make the pooled P larger).
Panel a says that only the selection rule differs (the replayed solutions differ in content, not only in length).

a  The selection rule (a schematic, not data): one problem's verified solutions in sampling order; keep the shortest,
   a random or the longest one; fine-tune identical copies; the untouched model is the control.
b  Shortest minus random and shortest minus longest replay in all four settings (forest plot; paired-bootstrap 95%
   intervals; ARC converted from puzzles of 72 to percentage points), with the pooled test above the plot.
c  Mechanism. Accuracy = P(finish within the cap) x P(correct | finished), so a change in accuracy splits exactly into
   a finishing term and a quality term (midpoint form, below), shown for shortest and longest replay on both MATH-500
   model families with 95% intervals. Under it, the ARC agent's share of messages cut at the cap and its restart cost
   (tokens per code-carrying message) for S, R and L, read from the message logs.

       dAcc = dF * (Q_arm + Q_base) / 200  +  dQ * (F_arm + F_base) / 200        (F, Q in %, dAcc in points)
              '--- finishing term ---'        '--- quality term ---'

   dF and dQ are the paired-bootstrap contrasts stored in the JSON. The two terms add up to the stored accuracy
   contrast up to rounding (checked at run time), and their intervals are the contrasts' intervals times the same
   positive factors, so a term is "significant" exactly when its contrast is.

Every number is read at run time from paper_brevity/data/paper_stats.json and paper_brevity/data/arc_msgs.json and
nothing else, so re-running after the JSON changes (e.g. the third untouched R1-Llama-8B seed) refreshes values, axis
limits, label positions, the footnote and the caption. The only hard-coded numbers are method constants that the data
files do not hold: LoRA r32, one epoch (panel a), the 8k-token cap per MATH/APPS sample (caption) and the 10,000
bootstrap resamples behind the stored tails (the floor of the pooled test).
The script prints every number it put on the figure, warns when a claim made by a heading stops holding, and checks
the layout (text overlaps, text on data marks, clipping, text below 6 pt) and the fonts embedded in the PDF.
"""
import json
import os
import re
import shutil
import subprocess
from statistics import NormalDist

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import Bbox, blended_transform_factory  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))                 # .../paper_brevity
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
OUT = os.path.join(FIGS, "cover_nc")
with open(os.path.join(DATA, "paper_stats.json")) as fh:
    STATS = json.load(fh)
with open(os.path.join(DATA, "arc_msgs.json")) as fh:
    MSGS = json.load(fh)

# method constants (not in the data files)
LORA_RANK, EPOCHS, MATH_CAP_K = 32, 1, 8
N_BOOT = 10000                                                      # resamples behind every stored one-sided tail

# ================================================================ style
# Paper palette (make_figs.py) with one change: R #8a8f98 -> #90959e. The dataviz validator (validate_palette.js,
# light mode, white surface) hard-fails S vs #8a8f98 on the normal-vision floor (OKLab dE 14.4 < 15); #90959e is one
# lightness step up, same hue, dE 16.0, still 3.0:1 on white. Accepted flags: S vs L is 7.4 under protan simulation
# (warn band), legal because every S/R/L mark carries its letter; navy `base` (the reference, drawn as the zero line)
# and grey R (the neutral control) sit outside the categorical lightness/chroma bands by role; S2 is an ordinal step
# of S (passes the --ordinal checks) and is always labelled "S2" / "round 2".
C = {"S": "#1b7f6e", "R": "#90959e", "L": "#c4573a", "base": "#2f3e5c", "S2": "#0b4f43"}
NAME = {"S": "shortest", "R": "random", "L": "longest", "base": "untouched", "S2": "round 2"}
INK, INK2, MUTED = "#1d1d1b", "#4f4e4a", "#6f6e69"                # 16.9:1, 8.3:1, 5.1:1 on white
AXIS, GRID, PALE, BAND = "#a3a29b", "#ebeae6", "#dcdde1", "#f1f0ec"
FS, FS_SMALL, FS_HEAD, FS_LETTER, FS_HERO = 6.6, 6.0, 7.4, 9.0, 14.0

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS, "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.labelsize": FS, "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

W, H = 7.2, 3.0                                                     # inches: 183 mm wide
HEAD_Y = H - 0.15                                                   # heading baseline
fig = plt.figure(figsize=(W, H), dpi=300)                           # measure text at output resolution
SHOWN = []                                                          # every number printed on the figure
WARNINGS = []


def warn(msg):
    WARNINGS.append(msg)
    print("WARNING:", msg)


def num(what, text):
    SHOWN.append((what, text))
    return text


def signed(v, nd=1):
    s = f"{v:+.{nd}f}"
    if float(s) == 0.0:
        s = f"{0.0:.{nd}f}"
    return s.replace("-", "−")


def tick_txt(v):
    return "0" if abs(v) < 1e-9 else signed(v, 0)


def two_sided(p_one):
    """paper_stats p is the one-sided bootstrap P(diff <= 0); two-sided = 2 min(p, 1 - p)."""
    return 2.0 * min(p_one, 1.0 - p_one)


def fmt_p(p):
    return "P < 0.001" if p < 0.001 else (f"P = {p:.3f}" if p < 0.01 else f"P = {p:.2f}")


def ax_at(x, y, w, h):
    """Axes placed in inches from the figure's lower-left corner."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def ftext(x, y, s, **kw):
    """Figure text placed in inches."""
    return fig.text(x / W, y / H, s, **kw)


def heading(x_in, letter, text):
    ftext(x_in, HEAD_Y, letter, fontsize=FS_LETTER, fontweight="bold", color=INK, va="baseline")
    ftext(x_in + 0.16, HEAD_Y, text, fontsize=FS_HEAD, fontweight="bold", color=INK, va="baseline")


def width_in(t):
    return t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi


def measure(s, **kw):
    """Width (in) of a string set with these text properties, without leaving it on the figure."""
    t = fig.text(0, 0, s, **kw)
    w = width_in(t)
    t.remove()
    return w


def bg(*artists):
    """Background artists (grid, bands): the text-on-mark check ignores them."""
    for a in artists:
        for b in (a if isinstance(a, (list, tuple)) else [a]):
            b.set_gid("bg")
    return artists[0] if len(artists) == 1 else artists


# ================================================================ data checks shared by panels b and c
arc = STATS["arc"]
A = arc["arms"]
N_TASKS = arc["n_tasks"]
CAP = MSGS["cap"]
# arc_msgs.json and paper_stats.json describe the same agent logs: the cut share and the restart cost must agree
msg_tok = {}
for a in ("S", "R", "L", "base"):
    X = np.asarray(MSGS["arms"][a], dtype=float)
    tok, cut, code = X[:, 0], X[:, 3], X[:, 4]
    msg_tok[a] = tok.sum()
    assert abs(100 * cut.mean() - A[a]["cut_msgs"]) < 0.05, f"{a}: cut share differs between the two files"
    assert abs(np.minimum(tok, CAP).mean() / code.mean() / 1000 - A[a]["restart_cost_k"]) < 0.05, \
        f"{a}: restart cost differs between the two files"
# which arms' logs are complete: tokens / all seeds' solves reproduces tokens_per_solved_k exactly for those
per_seed = {}
for a in ("S", "R", "L", "base"):
    full = abs(msg_tok[a] / sum(A[a]["seeds"]) / 1000 / A[a]["tokens_per_solved_k"] - 1) < 0.005
    per_seed[a] = len(MSGS["arms"][a]) / len(A[a]["seeds"]) if full else None
msgs_per_seed = np.mean([v for v in per_seed.values() if v])
LOGGED = {a: (len(A[a]["seeds"]) if per_seed[a] else int(round(len(MSGS["arms"][a]) / msgs_per_seed)))
          for a in per_seed}
PARTIAL = [a for a in LOGGED if LOGGED[a] < len(A[a]["seeds"])]
# tokens_per_solved_k comes from the per-task token totals, which every seed has; only the per-message logs (panel c's
# ARC table, which shows S/R/L) are missing for some seeds of the untouched agent
print("ARC message logs, seeds covered:", LOGGED)

# ================================================================ a: the selection rule (schematic, not data)
AW, AH = 2.20, HEAD_Y - 0.10
axa = ax_at(0.0, 0.0, AW, AH)
axa.set_xlim(0, AW)
axa.set_ylim(0, AH)
axa.axis("off")
heading(0.04, "a", "Replay one verified solution per problem")
SEGMENTS = []                                                       # arrows drawn with annotate, for the checks


def arrow(ax, xy_from, xy_to, color, lw, head, zorder=3):
    ax.annotate("", xy=xy_to, xytext=xy_from, zorder=zorder,
                arrowprops=dict(arrowstyle=f"-|>,head_length={head[0]},head_width={head[1]}", color=color, lw=lw,
                                shrinkA=0, shrinkB=0))
    SEGMENTS.append((ax, xy_from, xy_to))


def check(ax, x, y, s=0.03, color=INK2, lw=0.9):
    ax.plot([x - s, x - 0.35 * s, x + s], [y + 0.05 * s, y - 0.6 * s, y + 0.75 * s], color=color, lw=lw,
            solid_capstyle="round", solid_joinstyle="round")


def cross(ax, x, y, s=0.024, color=MUTED, lw=0.9):
    ax.plot([x - s, x + s], [y - s, y + s], color=color, lw=lw, solid_capstyle="round")
    ax.plot([x - s, x + s], [y + s, y - s], color=color, lw=lw, solid_capstyle="round")


def chip(ax, x, y, size, color, letter=None, fs=7.4):
    ax.add_patch(FancyBboxPatch((x, y - size / 2), size, size, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=color, ec="none"))
    if letter:
        ax.text(x + size / 2, y - 0.004, letter, fontsize=fs, fontweight="bold", color="white", ha="center",
                va="center", gid="inmark")


BAR_X0, BAR_H = 0.08, 0.085
ROWS_A = [(0.60, None), (0.27, "S"), (0.86, None), (0.48, "R"), (0.72, None), (1.00, "L")]   # sampling order
Y_INTRO = AH - 0.25
Y0_A, DY_A = Y_INTRO - 0.31, 0.205
X_RULE, X_CHIP, CHIP = 1.585, 1.95, 0.225
X_CAP = BAR_X0 + 1.14                                               # the token cap on the schematic length axis
axa.text(BAR_X0, Y_INTRO, "one training problem: the model's own\nsolutions (failed ones are discarded)",
         fontsize=FS_SMALL, color=INK2, va="center", ha="left", linespacing=1.15)
axa.text(X_CHIP + CHIP / 2, Y_INTRO, "fine-tuned\ncopy", fontsize=FS_SMALL, color=INK2, ha="center", va="center",
         linespacing=1.15)
for i, (length, arm) in enumerate(ROWS_A):
    y = Y0_A - i * DY_A
    axa.add_patch(FancyBboxPatch((BAR_X0, y - BAR_H / 2), length, BAR_H,
                                 boxstyle="round,pad=0,rounding_size=0.03", fc=C[arm] if arm else PALE, ec="none"))
    check(axa, BAR_X0 + length + 0.07, y)
    if arm:
        arrow(axa, (BAR_X0 + length + 0.14, y), (X_CHIP - 0.025, y), C[arm], 1.2 if arm == "S" else 0.9,
              (0.32, 0.16))
        axa.text(X_RULE, y + 0.03, f"keep {NAME[arm]}", fontsize=FS, color=INK, ha="center", va="bottom",
                 fontweight="bold" if arm == "S" else "normal")
        chip(axa, X_CHIP, y, CHIP, C[arm], arm)
# an attempt that is never replayed: cut at the token cap before it answered
y_cut = Y0_A - len(ROWS_A) * DY_A
axa.add_patch(FancyBboxPatch((BAR_X0, y_cut - BAR_H / 2), X_CAP - BAR_X0, BAR_H,
                             boxstyle="round,pad=0,rounding_size=0.03", fc="white", ec=PALE, lw=0.8))
cross(axa, X_CAP + 0.07, y_cut)
axa.text(X_CAP + 0.14, y_cut, "cut at the cap", fontsize=FS_SMALL, color=MUTED, va="center")
y_len = y_cut - 0.17
axa.plot([X_CAP, X_CAP], [y_len - 0.03, y_cut + BAR_H / 2 + 0.02], color=INK2, lw=0.8, solid_capstyle="butt")
arrow(axa, (BAR_X0, y_len), (X_CAP + 0.12, y_len), MUTED, 0.6, (0.3, 0.14))
axa.text(BAR_X0, y_len - 0.06, "solution length (tokens)", fontsize=FS_SMALL, color=MUTED, va="top")
axa.text(X_CAP, y_len - 0.06, "token cap", fontsize=FS_SMALL, color=INK2, ha="center", va="top")
y_base = y_len - 0.03
chip(axa, X_CHIP, y_base, CHIP, C["base"])
axa.text(X_CHIP - 0.06, y_base, "untouched\nmodel", fontsize=FS_SMALL, color=INK2, ha="right", va="center",
         linespacing=1.1)
HAS_S2 = any("S2-base" in STATS[k]["contrasts"] for k in ("arc", "math_qwen4b", "math_llama8b", "apps_qwen4b"))
axa.text(BAR_X0, 0.27, f"Same problems, answers and recipe (LoRA r{LORA_RANK}, {EPOCHS} epoch);\n"
         "only the selection rule differs.", fontsize=FS_SMALL, color=INK2,
         va="center", linespacing=1.25, style="italic")

# ================================================================ b: the selection contrast in every setting
GROUPS = [  # data key, benchmark, model, accuracy metric
    ("arc", "ARC-AGI-2", "Qwen3.8-27B\nagent", "solved"),
    ("math_qwen4b", "MATH-500", "Qwen3-4B", "acc"),
    ("math_llama8b", "MATH-500", "R1-Llama-8B", "acc"),
    ("apps_qwen4b", "APPS", "Qwen3-4B", "acc"),
]
OTHERS_B = ["R", "L"]                                               # shortest minus each other replay rule
rows_b, groups_b = [], []
yy = 0.0
for key, bench, model, metric in GROUPS:
    blk = STATS[key]
    scale = 100.0 / blk["n_tasks"] if key == "arc" else 1.0        # ARC: puzzles of 72 -> percentage points
    ys = []
    for o in OTHERS_B:
        c = blk["contrasts"][f"S-{o}"][metric]
        rows_b.append(dict(key=key, other=o, y=yy, d=c["diff"] * scale, lo=c["lo"] * scale, hi=c["hi"] * scale,
                           raw=c, sig=(c["lo"] > 0) or (c["hi"] < 0)))
        ys.append(yy)
        yy -= 1.0
    groups_b.append(dict(key=key, bench=bench, model=model, y0=ys[0], y1=ys[-1]))
    yy -= 0.55

# claim made by the caption: shortest has the highest accuracy of the three replay rules in every setting (point
# estimates), i.e. every row of the forest lies right of zero
RANK_OK = all(r["d"] > 0 for r in rows_b)
if not RANK_OK:
    warn("shortest replay no longer ranks first among the replay rules in every setting: "
         + ", ".join(f"{r['key']} S-{r['other']} {r['d']:+.2f}" for r in rows_b if r["d"] <= 0))


# --- pooled test over the four settings: Stouffer's z from the one-sided bootstrap tails P(diff <= 0)
def stouffer(other):
    nd = NormalDist()
    tails = [max(STATS[k]["contrasts"][f"S-{other}"][m]["p"], 1.0 / N_BOOT) for k, _, _, m in GROUPS]
    z = sum(nd.inv_cdf(1.0 - p) for p in tails) / len(tails) ** 0.5
    return z, two_sided(1.0 - nd.cdf(z))


POOLED = {o: stouffer(o) for o in OTHERS_B}
if not all(p < 0.05 for _, p in POOLED.values()):
    warn(f"pooled selection test no longer significant: {POOLED}")
XB0 = 2.38                                                          # panel b left edge (in)
heading(XB0, "b", "Shortest minus random or longest replay")
ftext(XB0 + 0.16, HEAD_Y - 0.19, "pooled over the four settings (Stouffer):", fontsize=FS_SMALL, color=INK2,
      va="baseline")
ftext(XB0 + 0.16, HEAD_Y - 0.31,
      "vs random " + num("b pooled S-R, two-sided P", fmt_p(POOLED["R"][1]))
      + ",  vs longest " + num("b pooled S-L, two-sided P", fmt_p(POOLED["L"][1])),
      fontsize=FS_SMALL, color=INK, va="baseline")

# --- forest
BX, BW, BY = 3.06, 1.74, 0.50
BTOP = HEAD_Y - 0.46
axb = ax_at(BX, BY, BW, BTOP - BY)
x_lo, x_hi = min(r["lo"] for r in rows_b), max(r["hi"] for r in rows_b)
xmin, xmax = x_lo - 0.45, x_hi + 0.45
step = 2.0 if (x_hi - x_lo) > 6 else 1.0
axb.set_xlim(xmin, xmax)
axb.set_ylim(min(r["y"] for r in rows_b) - 0.65, rows_b[0]["y"] + 0.75)
xt = np.arange(np.ceil(xmin / step) * step, xmax + 1e-9, step)
for v in xt:
    if abs(v) > 1e-9:
        bg(axb.axvline(v, color=GRID, lw=0.5, zorder=0))
axb.axvline(0, color=AXIS, lw=0.8, zorder=1)
col = C["S"]
for r in rows_b:
    axb.plot([r["lo"], r["hi"]], [r["y"], r["y"]], color=col, lw=1.4, solid_capstyle="round", zorder=2)
    axb.plot([r["d"]], [r["y"]], "o", ms=4.8, mfc=col if r["sig"] else "white", mec=col, mew=1.0, zorder=3)
axb.set_yticks([r["y"] for r in rows_b])
axb.set_yticklabels(["vs " + r["other"] for r in rows_b])
for lab in axb.get_yticklabels():
    lab.set_color(INK)
    lab.set_fontsize(FS)
axb.tick_params(axis="y", length=0, pad=3)
axb.spines["left"].set_visible(False)
axb.set_xticks(xt)
axb.set_xticklabels([num("b x tick (pp)", tick_txt(v)) for v in xt])
axb.set_xlabel(f"shortest minus the other rule (percentage points)\nARC: % of its {num('b ARC puzzles', str(N_TASKS))}"
               " puzzles", labelpad=2, linespacing=1.2)
GL_X = -0.20                                                        # group labels, axes fraction
ytr = axb.get_yaxis_transform()
for g in groups_b:
    yc = (g["y0"] + g["y1"]) / 2
    axb.text(GL_X, yc + 0.28, g["bench"], transform=ytr, fontsize=FS, fontweight="bold", color=INK, ha="right",
             va="bottom")
    axb.text(GL_X, yc + 0.18, g["model"], transform=ytr, fontsize=FS_SMALL, color=INK2, ha="right", va="top",
             linespacing=1.1)
# significance key in the empty space right of the last group's intervals
last = groups_b[-1]
key_rows = [r for r in rows_b if r["key"] == last["key"]]
kx = max(xmin + 0.47 * (xmax - xmin), max(r["hi"] for r in key_rows) + 0.08 * (xmax - xmin))
for i, (filled, txt) in enumerate(((True, "95% CI excludes 0"), (False, "95% CI includes 0"))):
    yk = key_rows[0]["y"] - 0.05 - i * 0.95
    axb.plot([kx], [yk], "o", ms=4.2, mfc=INK2 if filled else "white", mec=INK2, mew=0.9)
    axb.annotate(num("b key", txt), xy=(kx, yk), xytext=(5, 0), textcoords="offset points", ha="left",
                 va="center", fontsize=FS_SMALL, color=INK2)

# ================================================================ c: mechanism
# shortest replay against the two other replay rules, per MATH model: accuracy = P(finish) x P(correct | finished),
# so each paired difference splits into a finishing term dF x mean Q and a quality term dQ x mean F
MODELS = [("math_qwen4b", "Qwen3-4B"), ("math_llama8b", "R1-Llama-8B")]
OTHERS_C = ["R", "L"]
TERMS = [("net", "Δ accuracy"), ("fin", "finishing\nwithin the cap"), ("qual", "quality of\nfinished answers")]
dec = {}
for key, _ in MODELS:
    arms, cons = STATS[key]["arms"], STATS[key]["contrasts"]
    for o in OTHERS_C:
        c = cons[f"S-{o}"]
        qbar = (arms["S"]["correct_given_finished"] + arms[o]["correct_given_finished"]) / 200.0
        fbar = (arms["S"]["finish"] + arms[o]["finish"]) / 200.0
        e = {"net": [c["acc"][k] for k in ("diff", "lo", "hi")],
             "fin": [c["finish"][k] * qbar for k in ("diff", "lo", "hi")],
             "qual": [c["correct_given_finished"][k] * fbar for k in ("diff", "lo", "hi")]}
        resid = e["fin"][0] + e["qual"][0] - e["net"][0]
        if abs(resid) > 0.05:
            warn(f"decomposition residual {resid:+.3f} for {key} S-{o}")
        dec[key, o] = e
# heading claim: against both other replay rules, shortest gains by finishing (interval above 0), not by answer quality
for key, _ in MODELS:
    for o in OTHERS_C:
        if not (dec[key, o]["fin"][1] > 0 and dec[key, o]["qual"][0] <= 0.25):
            warn(f"panel c reading may no longer hold on {key} S-{o}: {dec[key, o]}")

XC0 = 4.98                                                          # panel c left edge (in)
heading(XC0, "c", "Mechanism: finishing within the cap")
ftext(XC0 + 0.16, HEAD_Y - 0.19, "MATH-500, shortest minus random or longest replay",
      fontsize=FS_SMALL, color=INK2, va="baseline")
COL_W, COL_GAP, LAB_W = 0.50, 0.165, 0.30
CX = [XC0 + LAB_W + j * (COL_W + COL_GAP) for j in range(3)]
C_TOP, C_BOT = HEAD_Y - 0.58, 1.30
# rows: per model a header slot, then shortest minus random, shortest minus longest
rows_c, yy = [], 0.0
for key, model in MODELS:
    rows_c.append(dict(kind="head", key=key, model=model, y=yy))
    yy -= 1.0
    for o in OTHERS_C:
        rows_c.append(dict(kind="arm", key=key, arm=o, y=yy))
        yy -= 1.0
    yy -= 0.35
ylo_c, yhi_c = yy + 0.35 - 0.45, 0.45
allv = [v for e in dec.values() for t in e.values() for v in t[1:]] + [0.0]
lim_lo, lim_hi = min(allv) - 0.3, max(allv) + 0.3
step_c = 2.0 if (lim_hi - lim_lo) > 3.5 else 1.0
axes_c = []
for j, (term, title) in enumerate(TERMS):
    ax = ax_at(CX[j], C_BOT, COL_W, C_TOP - C_BOT)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(ylo_c, yhi_c)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    if term == "fin":
        bg(ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc=BAND, ec="none", zorder=0)))
    for r in rows_c:
        if r["kind"] == "head":
            ax.plot([0, 0], [r["y"] - 0.55, r["y"] - 1.0 * len(OTHERS_C) - 0.45], color=AXIS, lw=0.8,
                    zorder=1, solid_capstyle="butt")
            continue
        d, lo, hi = dec[r["key"], r["arm"]][term]
        col = C["S"]
        sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [r["y"], r["y"]], color=col, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot([d], [r["y"]], "o", ms=4.8, mfc=col if sig else "white", mec=col, mew=1.0, zorder=3)
    xt_c = np.arange(np.ceil(lim_lo / step_c) * step_c, lim_hi + 1e-9, step_c)
    ax.set_xticks(xt_c)
    ax.set_xticklabels([num(f"c x tick ({term})", tick_txt(v)) for v in xt_c])
    ax.text(0.5, 1.0 + 0.10 / (C_TOP - C_BOT), title, transform=ax.transAxes, fontsize=FS_SMALL, color=INK,
            ha="center", va="bottom", linespacing=1.1, fontweight="bold" if term == "fin" else "normal")
    axes_c.append(ax)
ax0 = axes_c[0]
ytr0 = blended_transform_factory(fig.transFigure, ax0.transData)
for key, _ in MODELS:                                               # one equation per model block, between its rows
    ym = np.mean([r["y"] for r in rows_c if r["kind"] == "arm" and r["key"] == key])
    for j, glyph in ((1, "="), (2, "+")):
        ax0.text((CX[j] - COL_GAP / 2) / W, ym, glyph, transform=ytr0, fontsize=10, color=INK2, ha="center",
                 va="center")
for r in rows_c:
    if r["kind"] == "head":
        ax0.text((XC0 + 0.02) / W, r["y"], r["model"], transform=ytr0, fontsize=FS, fontweight="bold", color=INK,
                 ha="left", va="center")
    else:
        ax0.text((CX[0] - 0.05) / W, r["y"], "vs " + r["arm"], transform=ytr0, fontsize=FS, color=INK,
                 ha="right", va="center")
ftext((CX[0] + CX[2] + COL_W) / 2, C_BOT - 0.20, "shortest minus the other rule (points of accuracy)",
      fontsize=FS_SMALL,
      color=INK2, ha="center", va="top")

# --- ARC agent: messages cut at the cap and restart cost
ps = [two_sided(arc["contrasts"][f"S-{o}"][m]["p"]) for o in ("R", "L") for m in ("cut_msgs", "restart_cost_k")]
holds = all(arc["contrasts"][f"S-{o}"][m]["diff"] < 0 for o in ("R", "L") for m in ("cut_msgs", "restart_cost_k"))
if not holds:
    warn("ARC: shortest no longer has the fewest cut messages / lowest restart cost among replay rules")
T_TOP = C_BOT - 0.47
XT = XC0 + 0.02
ftext(XT, T_TOP, f"ARC-AGI-2 agent, {num('c ARC cap (k tokens)', f'{CAP / 1000:g}k')}-token cap per message",
      fontsize=FS, fontweight="bold", color=INK, va="baseline")
T_ROWS = (("cut_msgs", "messages cut at the cap", "{:.1f}%"),
          ("restart_cost_k", "tokens per code-carrying message", "{:.1f}k"))
lab_w = max(measure(lab, fontsize=FS_SMALL) for _, lab, _ in T_ROWS)
TCOL = ["S", "R", "L"]
TXC = [XT + lab_w + 0.16 + 0.28 * i for i in range(len(TCOL))]    # column centres
axt = ax_at(0, 0, W, H)                                             # overlay in inches, for the header chips
axt.set_xlim(0, W)
axt.set_ylim(0, H)
axt.axis("off")
for i, a in enumerate(TCOL):
    chip(axt, TXC[i] - 0.065, T_TOP - 0.115, 0.13, C[a], a, fs=6.4)
for k, (field, lab, fmt) in enumerate(T_ROWS):
    yrow = T_TOP - 0.255 - 0.115 * k
    ftext(XT, yrow, lab, fontsize=FS_SMALL, color=INK2, va="baseline")
    for i, a in enumerate(TCOL):
        ftext(TXC[i], yrow, num(f"c ARC {field} {a}", fmt.format(A[a][field])), fontsize=FS_SMALL, color=INK,
              ha="center", va="baseline", fontweight="bold" if a == "S" else "normal")
ftext(XT, T_TOP - 0.50, "shortest vs random and vs longest, both rows: "
      + num("c ARC S vs R/L, max two-sided P", fmt_p(max(ps)) if max(ps) >= 0.01 else "P < 0.01"),
      fontsize=FS_SMALL, color=INK2, va="baseline")


# ================================================================ checks
def text_artists():
    out = list(fig.texts)
    for ax in fig.axes:
        out += list(ax.texts)
        if ax.axison:
            out += [t for t in ax.get_xticklabels() + ax.get_yticklabels()] + [ax.xaxis.label, ax.yaxis.label]
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def mark_points(ax):
    """Display-space sample points of every data mark in ax: lines, markers (as discs), patches, arrows."""
    pts = []
    seg = np.linspace(0, 1, 30)[:, None]
    for ln in ax.lines:
        if ln.get_gid() == "bg":
            continue
        xy = ln.get_transform().transform(np.column_stack([np.asarray(ln.get_xdata(), float),
                                                           np.asarray(ln.get_ydata(), float)]))
        if ln.get_linestyle() not in ("None", "", " ") and len(xy) > 1:
            pts += [xy[k] + seg * (xy[k + 1] - xy[k]) for k in range(len(xy) - 1)]
        if ln.get_marker() not in (None, "", "None", " "):
            rad = ln.get_markersize() / 2 * fig.dpi / 72
            ang = np.linspace(0, 2 * np.pi, 16)[:, None]
            pts += [p0 + rad * np.hstack([np.cos(ang), np.sin(ang)]) for p0 in xy] + [xy]
    for p in ax.patches:
        if p.get_gid() != "bg":
            v = p.get_path().transformed(p.get_transform()).vertices
            pts.append(v)
            b = Bbox.from_extents(v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max())
            gx, gy = np.meshgrid(np.linspace(b.x0, b.x1, 12), np.linspace(b.y0, b.y1, 6))
            pts.append(np.column_stack([gx.ravel(), gy.ravel()]))
    for ax2, p0, p1 in SEGMENTS:
        if ax2 is ax:
            xy = ax.transData.transform([p0, p1])
            pts.append(xy[0] + seg * (xy[1] - xy[0]))
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def check_layout():
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = text_artists()
    boxes = [Text.get_window_extent(t, rend) for t in texts]       # an Annotation's text only, not its arrow
    page, issues = fig.bbox, []
    gap = 0.6 * fig.dpi / 72                                        # texts at least ~1 pt apart
    for t, b in zip(texts, boxes):
        if t.get_fontsize() < 6.0 - 1e-9:
            issues.append(f"text below 6 pt: {t.get_text()!r} ({t.get_fontsize()} pt)")
        if b.x0 < page.x0 or b.y0 < page.y0 or b.x1 > page.x1 or b.y1 > page.y1:
            issues.append(f"outside the page: {t.get_text()!r}")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            b1, b2 = boxes[i], boxes[j]
            if (b1.x0 < b2.x1 + gap and b2.x0 < b1.x1 + gap and b1.y0 < b2.y1 + gap and b2.y0 < b1.y1 + gap):
                issues.append(f"text overlap: {texts[i].get_text()!r} x {texts[j].get_text()!r}")
    cache = {}
    for t, b in zip(texts, boxes):
        if t.get_gid() == "inmark":
            continue
        pad = 0.5 * fig.dpi / 72
        for ax in fig.axes:
            if ax not in cache:
                cache[ax] = mark_points(ax)
            P = cache[ax]
            if len(P) and ((P[:, 0] > b.x0 + pad) & (P[:, 0] < b.x1 - pad) &
                           (P[:, 1] > b.y0 + pad) & (P[:, 1] < b.y1 - pad)).any():
                issues.append(f"text on a data mark: {t.get_text()!r}")
                break
    return issues


def seeds_phrase(key):
    """Seeds of the three replay arms and the untouched model (panel b shows no round-2 arm)."""
    n = {a: len(v["seeds"]) for a, v in STATS[key]["arms"].items()}
    core = [n[a] for a in ("S", "R", "L", "base")]
    return (f"{core[0]} seeds per arm" if len(set(core)) == 1
            else ", ".join(f"{NAME[a]} {n[a]}" for a in ("S", "R", "L", "base")) + " seeds")


def caption():
    cut_p = two_sided(arc["contrasts"]["S-base"]["cut_msgs"]["p"])
    rc_p = two_sided(arc["contrasts"]["S-base"]["restart_cost_k"]["p"])
    base_note = (f"untouched agent, whose message logs cover {LOGGED['base']} of its {len(A['base']['seeds'])} seeds: "
                 f"{A['base']['cut_msgs']:.1f}% cut, {A['base']['restart_cost_k']:.1f}k; S vs untouched two-sided "
                 f"P = {two_sided_fmt(cut_p)} and {two_sided_fmt(rc_p)}")
    pr, pl = POOLED["R"], POOLED["L"]
    return (
        "**Replaying the shortest verified solution helps a model finish within its token cap.** "
        f"**a**, Identical copies of each model are fine-tuned (LoRA r{LORA_RANK}, {EPOCHS} epoch) on one of its own "
        "verified solutions per training problem: the shortest (S), a random one (R) or the longest (L); the untouched "
        "model is the control. "
        f"**b**, Accuracy of shortest replay minus random (vs R) and minus longest replay (vs L) on ARC-AGI-2 "
        f"(Qwen3.8-27B program-writing agent, {N_TASKS} evaluation puzzles, {seeds_phrase('arc')}; puzzles × "
        f"100/{N_TASKS} gives percentage points of puzzles solved), MATH-500 (Qwen3-4B, {seeds_phrase('math_qwen4b')}; "
        f"DeepSeek-R1-Distill-Llama-8B, {seeds_phrase('math_llama8b')}) and APPS (Qwen3-4B, "
        f"{seeds_phrase('apps_qwen4b')}), per-sample accuracy points otherwise (the units are not comparable across "
        "settings): dots are mean differences and bars paired-bootstrap 95% intervals over problems, filled when the "
        "interval excludes 0. "
        + ("Shortest replay has the highest accuracy of the three rules in every setting. " if RANK_OK else "")
        + f"Pooled over the four settings (Stouffer's method on the one-sided bootstrap tails), z = {pr[0]:.2f} against "
        f"random replay ({fmt_p(pr[1])}) and z = {pl[0]:.2f} against longest replay ({fmt_p(pl[1])}), two-sided. "
        "**c**, Because accuracy = P(finish within the token cap) × P(correct | finished), the paired difference "
        "between shortest replay and random or longest replay splits exactly into a finishing term, ΔF × mean Q, and "
        "a quality term, ΔQ × mean F (means over the two arms; intervals are the bootstrap intervals of ΔF and ΔQ "
        f"times the same factors; caps {MATH_CAP_K}k tokens per MATH sample and {CAP / 1000:g}k per ARC agent "
        "message): the whole gain comes from finishing, and finished answers are not better. The ARC rows come from "
        "the agent's message logs, where restart cost is the tokens spent per code-carrying message "
        f"({base_note}).\n")


def two_sided_fmt(p):
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


issues = check_layout()
os.makedirs(FIGS, exist_ok=True)
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
with open(OUT + "_caption.md", "w") as fh:
    fh.write(caption())
print("wrote", OUT + ".pdf,", OUT + ".png and", OUT + "_caption.md")
print("layout:", "clean (no overlaps, no text on marks, nothing clipped, nothing below 6 pt)" if not issues else "")
for s in issues:
    print("   ", s)
sizes = sorted((t.get_fontsize(), t.get_text()) for t in text_artists())
print(f"smallest text: {sizes[0][0]:.2f} pt")
if shutil.which("pdffonts"):
    rep = subprocess.run(["pdffonts", OUT + ".pdf"], capture_output=True, text=True).stdout
    print(rep)
else:
    fonts = sorted(set(re.findall(rb"/BaseFont\s*/([^\s/\]>]+)", open(OUT + ".pdf", "rb").read())))
    print("PDF fonts:", [f.decode() for f in fonts])
print("numbers printed on the figure:")
for what, s in SHOWN:
    print(f"    {s:>24}  {what}")
print("plotted (not printed), b, pp:")
for r in rows_b:
    print(f"    {r['key']:<13} S-{r['other']:<2} {r['d']:+7.3f} [{r['lo']:+7.3f}, {r['hi']:+7.3f}] "
          f"{'filled' if r['sig'] else 'hollow'}")
print("plotted (not printed), c, points of accuracy (diff [lo, hi]):")
for (key, a), e in dec.items():
    print(f"    {key:<13} {a}  " + "  ".join(f"{t} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for t, v in e.items()))
print("pooled (Stouffer) z, two-sided P:", {o: (round(z, 3), p) for o, (z, p) in POOLED.items()})
print("claims made by the headings and caption:",
      "all hold" if not WARNINGS else f"{len(WARNINGS)} WARNING(S) above - review before using the figure")
