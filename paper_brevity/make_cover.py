"""Cover figure for the brevity paper: one 183 mm wide strip, three panels a | b | c.

    env -u PYTHONPATH .venv_mps/bin/python paper_brevity/make_cover.py
    -> paper_brevity/figures/cover.pdf      vector, Arial embedded as TrueType (pdf.fonttype 42)
       paper_brevity/figures/cover.png      300 dpi
       paper_brevity/figures/cover_caption.md   caption; its numbers come from the same data
       paper_brevity/arc/figures/writeup_cover.png   2:1 cover image of the Kaggle Writeup (2400 x 1200 px): the title
                                                     and panel b in large type (S, R and L only), legible as a thumbnail

Wording follows the rescoped arc/main.tex (2026-11-02): panel b says "highest point estimate", not "ranks first"; its
banner gives shortest minus untouched in puzzles with P over puzzles and over runs, and tokens per puzzle, instead of
the relative changes (+27% puzzles, -25% tokens per solved puzzle); panel c is the "accounting split", not a mechanism.

a  The selection rule (a schematic, not data): one problem's verified solutions in sampling order; keep the shortest,
   a random or the longest one; fine-tune identical copies; the untouched model is the control. Only the selection rule
   differs between the arms (the replayed solutions differ in content, not only in length).
b  Difference from the untouched model in all four settings (forest plot; paired-bootstrap 95% intervals; ARC converted
   from puzzles of 72 to percentage points), led by the ARC-AGI-2 banner: shortest minus untouched in puzzles solved,
   its two-sided P over puzzles (paired bootstrap, stored in the JSON) and over runs (exact permutation test on the
   five run totals per arm, all 252 splits, as derived_stats.py), and the change in tokens per puzzle (tokens per
   solved puzzle x puzzles solved / 72, as derived_stats.py).
c  Accounting split. With F the share of samples that finish within the cap and Q = A/F the accuracy per finished
   sample, accuracy A = F x Q, so a change in accuracy splits exactly into a finishing term and a quality term (midpoint
   form, below), shown for shortest minus random and longest replay on both MATH-500 model families with 95% intervals.
   Under it, the ARC agent's share of messages cut at the cap and its restart cost (tokens per code-carrying message)
   for S, R and L, read from the message logs.

       dAcc = dF * (Q_arm + Q_base) / 200  +  dQ * (F_arm + F_base) / 200        (F, Q in %, dAcc in points)
              '--- finishing term ---'        '--- quality term ---'

   dF and dQ are the paired-bootstrap contrasts stored in the JSON. The two terms add up to the stored accuracy
   contrast up to rounding (checked at run time), and their intervals are the contrasts' intervals times the same
   positive factors, so a term is "significant" exactly when its contrast is.

Every number is read at run time from paper_brevity/data/paper_stats.json and paper_brevity/data/arc_msgs.json and
nothing else, so re-running after the JSON changes (e.g. the third untouched R1-Llama-8B seed) refreshes values, axis
limits, label positions, the footnote and the caption. The only hard-coded numbers are method constants that the data
files do not hold: LoRA r32, one epoch (panel a) and the 8k-token cap per MATH/APPS sample (caption).
The script prints every number it put on the figure, warns when a claim made by a heading stops holding, and checks
the layout (text overlaps, text on data marks, clipping, text below 6 pt) and the fonts embedded in the PDF.
"""
import itertools
import json
import os
import re
import shutil
import subprocess

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
OUT = os.path.join(FIGS, "cover")
with open(os.path.join(DATA, "paper_stats.json")) as fh:
    STATS = json.load(fh)
with open(os.path.join(DATA, "arc_msgs.json")) as fh:
    MSGS = json.load(fh)

# method constants (not in the data files)
LORA_RANK, EPOCHS, MATH_CAP_K = 32, 1, 8

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


def p_num(p):
    """A P value without its 'P =' prefix: '< 0.001', three decimals below 0.01, else two (as fmt_p)."""
    return "< 0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def perm_p(a, b):
    """Exact two-sided permutation test on two sets of run totals (all splits of the pooled runs), as derived_stats.py:
    the share of splits whose absolute mean difference is at least the observed one."""
    pool, n = list(a) + list(b), len(a)
    obs = abs(np.mean(a) - np.mean(b))
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(len(pool)) if i not in idx]
        hits += abs(np.mean(x) - np.mean(y)) >= obs - 1e-9
        tot += 1
    return hits / tot


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
         "only the selection rule differs."
         + ("\nS2: a second round of shortest replay, from S." if HAS_S2 else ""), fontsize=FS_SMALL, color=INK2,
         va="center", linespacing=1.25, style="italic")

# ================================================================ b: every setting vs the untouched model
GROUPS = [  # data key, benchmark, model, accuracy metric
    ("arc", "ARC-AGI-2", "Qwen3.8-27B\nagent", "solved"),
    ("math_qwen4b", "MATH-500", "Qwen3-4B", "acc"),
    ("math_llama8b", "MATH-500", "R1-Llama-8B", "acc"),
    ("apps_qwen4b", "APPS", "Qwen3-4B", "acc"),
]
ARMS_B = ["S", "S2", "R", "L"]
rows_b, groups_b = [], []
yy = 0.0
for key, bench, model, metric in GROUPS:
    blk = STATS[key]
    scale = 100.0 / blk["n_tasks"] if key == "arc" else 1.0        # ARC: puzzles of 72 -> percentage points
    ys = []
    for a in ARMS_B:
        con = blk["contrasts"].get(f"{a}-base")
        if not con:
            continue
        c = con[metric]
        rows_b.append(dict(key=key, arm=a, y=yy, d=c["diff"] * scale, lo=c["lo"] * scale, hi=c["hi"] * scale,
                           raw=c, sig=(c["lo"] > 0) or (c["hi"] < 0)))
        ys.append(yy)
        yy -= 1.0
    groups_b.append(dict(key=key, bench=bench, model=model, y0=ys[0], y1=ys[-1]))
    yy -= 0.55

# heading claim: shortest has the highest point estimate of the three replay rules in every setting (not a test; the
# selection contrasts are significant only in some settings, arc/main.tex Sec. 4.2)
RANK_OK = True
for key, *_ in GROUPS:
    arms = STATS[key]["arms"]
    m = {a: (arms[a]["solved"] if key == "arc" else arms[a]["acc"]) for a in "SRL"}
    if not (m["S"] > m["R"] and m["S"] > m["L"]):
        RANK_OK = False
        warn(f"panel b heading no longer holds (neutral heading used): shortest does not have the highest point "
             f"estimate among the replay rules on {key}: {m}")

# --- headline numbers (ARC-AGI-2, shortest vs untouched): puzzles solved with P over puzzles (paired bootstrap, from
# the JSON) and over runs (exact permutation test on the five run totals per arm), and tokens per puzzle
sb = arc["contrasts"]["S-base"]["solved"]
TOK_PER_PUZZLE = {a: A[a]["tokens_per_solved_k"] * A[a]["solved"] / N_TASKS for a in A}   # k tokens per puzzle
tok = 100.0 * (TOK_PER_PUZZLE["S"] / TOK_PER_PUZZLE["base"] - 1.0)
p_arc = two_sided(sb["p"])
p_runs = perm_p(A["S"]["seeds"], A["base"]["seeds"])
if not (sb["lo"] > 0 and tok < 0):
    warn(f"ARC headline changed sign or lost significance: {sb}, tokens per puzzle {tok:+.1f}%")
XB0 = 2.38                                                          # panel b left edge (in)
heading(XB0, "b", "Shortest replay has the highest point estimate" if RANK_OK else "Difference from the untouched model")
Y_HERO = HEAD_Y - 0.39                                              # hero numerals baseline
ftext(XB0 + 0.16, HEAD_Y - 0.19, "ARC-AGI-2 agent, shortest vs untouched", fontsize=FS_SMALL, color=INK2,
      va="baseline")
x = XB0 + 0.16
for big, sub in ((num("hero: ARC S - untouched, puzzles", signed(sb["diff"])),
                  "puzzles solved\n(of " + num("hero: ARC puzzles", str(N_TASKS)) + ")"),
                 (num("hero: ARC S vs untouched, tokens per puzzle", signed(tok, 1) + "%"),
                  "tokens per\npuzzle")):
    t = ftext(x, Y_HERO, big, fontsize=FS_HERO, fontweight="bold", color=INK, va="baseline")
    x += width_in(t) + 0.05
    t = ftext(x, Y_HERO + 0.004, sub, fontsize=FS_SMALL, color=INK2, va="baseline", linespacing=1.15)
    x += width_in(t) + 0.12
ftext(XB0 + 0.16, HEAD_Y - 0.505,
      ("P < 0.001" if p_arc < 0.001 else "P = " + num("hero: ARC S vs untouched, two-sided P over puzzles",
                                                      p_num(p_arc)))
      + " over puzzles, " + num("hero: ARC S vs untouched, two-sided P over runs (permutation)", p_num(p_runs))
      + " over runs", fontsize=FS_SMALL, color=INK2, va="baseline")

# --- forest
BX, BW, BY = 3.06, 1.74, 0.50
BTOP = HEAD_Y - 0.56                                                # forest top (as before the 2026-11-02 banner)
axb = ax_at(BX, BY, BW, BTOP - BY)
x_lo, x_hi = min(r["lo"] for r in rows_b), max(r["hi"] for r in rows_b)
xmin, xmax = x_lo - 0.45, x_hi + 0.45
step = 2.0 if (x_hi - x_lo) > 6 else 1.0
axb.set_xlim(xmin, xmax)
axb.set_ylim(min(r["y"] for r in rows_b) - 0.65, rows_b[0]["y"] + 1.05)
xt = np.arange(np.ceil(xmin / step) * step, xmax + 1e-9, step)
for v in xt:
    if abs(v) > 1e-9:
        bg(axb.axvline(v, color=GRID, lw=0.5, zorder=0))
axb.axvline(0, color=C["base"], lw=0.9, zorder=1)
axb.text(-0.012 * (xmax - xmin), rows_b[0]["y"] + 0.62, "untouched", fontsize=FS_SMALL, color=INK2, ha="right",
         va="center")
for r in rows_b:
    col, emph = C[r["arm"]], r["arm"] == "S"
    axb.plot([r["lo"], r["hi"]], [r["y"], r["y"]], color=col, lw=1.6 if emph else 1.2, solid_capstyle="round",
             zorder=2)
    axb.plot([r["d"]], [r["y"]], "o", ms=5.0 if emph else 4.4, mfc=col if r["sig"] else "white", mec=col,
             mew=1.0, zorder=3)
axb.set_yticks([r["y"] for r in rows_b])
axb.set_yticklabels([r["arm"] for r in rows_b])
for lab, r in zip(axb.get_yticklabels(), rows_b):
    if r["arm"] == "S":
        lab.set_fontweight("bold")
        lab.set_color(INK)
axb.tick_params(axis="y", length=0, pad=3)
axb.spines["left"].set_visible(False)
axb.set_xticks(xt)
axb.set_xticklabels([num("b x tick (pp)", tick_txt(v)) for v in xt])
axb.set_xlabel(f"difference from untouched (percentage points)\nARC: % of its {num('b ARC puzzles', str(N_TASKS))}"
               " puzzles", labelpad=2, linespacing=1.2)
GL_X = -0.155                                                       # group labels, axes fraction
ytr = axb.get_yaxis_transform()
for g in groups_b:
    yc = (g["y0"] + g["y1"]) / 2
    axb.text(GL_X, yc + 0.28, g["bench"], transform=ytr, fontsize=FS, fontweight="bold", color=INK, ha="right",
             va="bottom")
    axb.text(GL_X, yc + 0.18, g["model"], transform=ytr, fontsize=FS_SMALL, color=INK2, ha="right", va="top",
             linespacing=1.1)
for r in rows_b:
    if r["arm"] == "S2":
        axb.annotate("round 2", xy=(r["hi"], r["y"]), xytext=(4, 0), textcoords="offset points", ha="left",
                     va="center", fontsize=FS_SMALL, color=INK2)
# significance key in the empty space right of the last group's intervals
last = groups_b[-1]
key_rows = [r for r in rows_b if r["key"] == last["key"]]
kx = max(xmin + 0.47 * (xmax - xmin), max(r["hi"] for r in key_rows) + 0.08 * (xmax - xmin))
for i, (filled, txt) in enumerate(((True, "95% CI excludes 0"), (False, "95% CI includes 0"))):
    yk = key_rows[0]["y"] - 0.45 - i * 0.95
    axb.plot([kx], [yk], "o", ms=4.2, mfc=INK2 if filled else "white", mec=INK2, mew=0.9)
    axb.annotate(num("b key", txt), xy=(kx, yk), xytext=(5, 0), textcoords="offset points", ha="left",
                 va="center", fontsize=FS_SMALL, color=INK2)

# ================================================================ c: accounting split
# shortest replay against the two other replay rules, per MATH model: accuracy A = F x Q, with F the share of samples
# that finish within the cap and Q = A/F the accuracy per finished sample (P(correct | finished) only if no cut sample
# is scored correct), so each paired difference splits into a finishing term dF x mean Q and a quality term dQ x mean F
MODELS = [("math_qwen4b", "Qwen3-4B"), ("math_llama8b", "R1-Llama-8B")]
OTHERS_C = ["R", "L"]
TERMS = [("net", "Δ accuracy"), ("fin", "finishing\nwithin the cap"), ("qual", "accuracy per\nfinished sample")]
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
heading(XC0, "c", "Accounting split: finishing within the cap")
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
def text_artists(f=None):
    f = f or fig
    out = list(f.texts)
    for ax in f.axes:
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
            rad = ln.get_markersize() / 2 * ax.figure.dpi / 72
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


def check_layout(f=None):
    f = f or fig
    f.canvas.draw()
    rend = f.canvas.get_renderer()
    texts = text_artists(f)
    boxes = [Text.get_window_extent(t, rend) for t in texts]       # an Annotation's text only, not its arrow
    page, issues = f.bbox, []
    gap = 0.6 * f.dpi / 72                                          # texts at least ~1 pt apart
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
        pad = 0.5 * f.dpi / 72
        for ax in f.axes:
            if ax not in cache:
                cache[ax] = mark_points(ax)
            P = cache[ax]
            if len(P) and ((P[:, 0] > b.x0 + pad) & (P[:, 0] < b.x1 - pad) &
                           (P[:, 1] > b.y0 + pad) & (P[:, 1] < b.y1 - pad)).any():
                issues.append(f"text on a data mark: {t.get_text()!r}")
                break
    return issues


def seeds_phrase(key):
    n = {a: len(v["seeds"]) for a, v in STATS[key]["arms"].items()}
    core = [n[a] for a in ("S", "R", "L", "base")]
    s = (f"{core[0]} seeds per arm" if len(set(core)) == 1
         else ", ".join(f"{NAME[a]} {n[a]}" for a in ("S", "R", "L", "base")) + " seeds")
    if "S2" in n and n["S2"] != core[0]:
        s += f" and {n['S2']} for S2"
    return s


def caption():
    """Caption draft in the wording of the rescoped arc/main.tex (Fig. 1), with its numbers read from the data."""
    cut_p = two_sided(arc["contrasts"]["S-base"]["cut_msgs"]["p"])
    rc_p = two_sided(arc["contrasts"]["S-base"]["restart_cost_k"]["p"])
    sel = {o: arc["contrasts"][f"S-{o}"]["solved"] for o in ("R", "L")}
    sel_d = sorted({signed(c["diff"]) for c in sel.values()})
    base_note = (f"the untouched agent's logs cover {LOGGED['base']} of its {len(A['base']['seeds'])} runs: "
                 f"{A['base']['cut_msgs']:.1f}% cut, {A['base']['restart_cost_k']:.1f}k tokens per code-carrying "
                 f"message; shortest vs untouched P = {p_num(cut_p)} and {p_num(rc_p)}")
    return (
        ("**Shortest replay has the highest point estimate of three replay rules in every setting; on mathematics an "
         "accounting split places its advantage in finishing within the token cap.** "
         if RANK_OK else "**Three rules for replaying a model's own verified solutions in self-training.** ")
        + f"**a**, Identical copies of each model are fine-tuned (LoRA rank {LORA_RANK}, {EPOCHS} epoch) on one of its "
        "own verified solutions per training problem: the shortest (S), a random one (R) or the longest (L). The arms "
        "differ only in the selection rule, so the replayed solutions differ in length and also in content; S2 repeats "
        "shortest replay starting from S; the untouched model is the control. "
        f"**b**, Difference from the untouched model on ARC-AGI-2 (Qwen3.8-27B agent, {N_TASKS} puzzles, "
        f"{seeds_phrase('arc').replace('seeds', 'runs')}; puzzles × 100/{N_TASKS} gives percentage points), MATH-500 "
        f"(Qwen3-4B, {seeds_phrase('math_qwen4b')}; DeepSeek-R1-Distill-Llama-8B, {seeds_phrase('math_llama8b')}) "
        f"and APPS (Qwen3-4B, {seeds_phrase('apps_qwen4b')}). Dots: mean differences, filled when the paired-bootstrap "
        "95% interval over problems (bar) excludes 0; P values are two-sided. The banner compares shortest replay "
        f"with the untouched agent ({signed(sb['diff'])} puzzles; P = {p_num(p_arc)} over puzzles, {p_num(p_runs)} "
        f"over runs, exact permutation test on the run totals; tokens per puzzle, {signed(tok, 1)}%); against random "
        f"and longest replay the ARC difference is {' and '.join(sel_d)} puzzles (P = {p_num(two_sided(sel['R']['p']))} "
        f"and {p_num(two_sided(sel['L']['p']))}). "
        f"**c**, Accounting split. With F the share of samples that finish within the cap ({MATH_CAP_K}k tokens) and "
        "Q = A/F the accuracy per finished sample, accuracy is A = F × Q, and the difference between shortest and "
        "random or longest replay splits exactly into a finishing term, ΔF × mean Q, and a quality term, ΔQ × mean F "
        "(Kitagawa's decomposition; means over the two arms; intervals: those of ΔF and ΔQ times the same factors): "
        "on MATH the finishing term carries the gain. ARC rows: the agent's message logs "
        f"({CAP / 1000:g}k-token cap per message; {base_note}); 'cut at the cap' counts messages cut while still "
        "reasoning; tokens per code-carrying message is the restart cost E[min(L, t)] / P(code).\n")


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
    print(f"    {r['key']:<13} {r['arm']:<3} {r['d']:+7.3f} [{r['lo']:+7.3f}, {r['hi']:+7.3f}] "
          f"{'filled' if r['sig'] else 'hollow'}")
print("plotted (not printed), c, points of accuracy (diff [lo, hi]):")
for (key, a), e in dec.items():
    print(f"    {key:<13} {a}  " + "  ".join(f"{t} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for t, v in e.items()))
print("claims made by the headings, hero and caption:",
      "all hold" if not WARNINGS else f"{len(WARNINGS)} WARNING(S) above - review before using the figure")


# ================================================================ Kaggle Writeup cover image (2:1, large type)
WC_PATH = os.path.join(ROOT, "arc", "figures", "writeup_cover.png")


def writeup_cover():
    """Cover image of the Kaggle Writeup (the Media Gallery requires one; no size is published, so 2:1 at 2400 x 1200
    px): the paper's title, the ARC-AGI-2 banner of panel b and panel b's forest for S, R and L in large type, so that
    the title, the banner and the pattern stay legible as a thumbnail. Same data, colours, wording and claim checks as
    panel b; S2 (a second round of shortest replay, not one of the three rules) is left out for legibility."""
    WW, WH = 8.0, 4.0
    g = plt.figure(figsize=(WW, WH), dpi=300)
    rend = g.canvas.get_renderer()

    def gt(x, y, s, **kw):                                          # figure text placed in inches
        return g.text(x / WW, y / WH, s, **kw)

    def wid(t):
        return t.get_window_extent(rend).width / g.dpi

    # ---- left: title, then the ARC-AGI-2 banner of panel b (shortest vs untouched)
    X0 = 0.42
    gt(X0, 3.64, "Think less,\nact more", fontsize=31, fontweight="bold", color=INK, va="top", linespacing=1.0)
    gt(X0, 2.47, "Replaying the shortest verified solutions\nfor a program-writing ARC-AGI-2 agent", fontsize=12.5,
       color=INK2, va="top", linespacing=1.3)
    YH = 1.40                                                       # hero baseline
    # 2026-11-09: the gain as a share of the 72 puzzles (percentage points), no P value, no author line
    pct = {a: 100.0 * A[a]["solved"] / N_TASKS for a in ("S", "base")}
    assert abs((pct["S"] - pct["base"]) - 100.0 * sb["diff"] / N_TASKS) < 0.05
    t = gt(X0, YH, signed(pct["S"] - pct["base"]) + "%", fontsize=34, fontweight="bold", color=C["S"], va="baseline")
    gt(X0 + wid(t) + 0.10, YH + 0.01, f"of the {N_TASKS} ARC-AGI-2 puzzles,\nshortest replay vs untouched",
       fontsize=10.5, color=INK, va="baseline", linespacing=1.25)
    gt(X0, YH - 0.36, f"{pct['S']:.1f}% vs {pct['base']:.1f}% of puzzles solved (five-run means)", fontsize=10.5,
       color=INK2, va="baseline")
    gt(X0, 0.50, "ARC Prize 2026, paper track (ARC-AGI-2)", fontsize=9, color=MUTED, va="baseline")

    # ---- right: panel b in large type, S, R and L only
    rows, groups, yy = [], [], 0.0
    for key, bench, model, _ in GROUPS:
        ys = []
        for r in rows_b:
            if r["key"] == key and r["arm"] in ("S", "R", "L"):
                rows.append(dict(r, y=yy))
                ys.append(yy)
                yy -= 1.0
        groups.append(dict(key=key, bench=bench, model=model, y0=ys[0], y1=ys[-1]))
        yy -= 0.55
    XR = 4.12                                                       # right column, left edge (in)
    gt(XR, 3.64, "Shortest replay has the highest point estimate\nof three replay rules in every setting" if RANK_OK
       else "Difference from the untouched model\nper replay rule in every setting", fontsize=11, fontweight="bold",
       color=INK, va="top", linespacing=1.2)
    AX_L, AX_R, AX_B, AX_T = 5.32, 7.62, 0.98, 3.02
    ax = g.add_axes([AX_L / WW, AX_B / WH, (AX_R - AX_L) / WW, (AX_T - AX_B) / WH])
    lo, hi = min(r["lo"] for r in rows), max(r["hi"] for r in rows)
    x0, x1 = lo - 0.45, hi + 0.45
    ax.set_xlim(x0, x1)
    ax.set_ylim(min(r["y"] for r in rows) - 0.6, 1.0)
    stp = 2.0
    xt = np.arange(np.ceil(x0 / stp) * stp, x1 + 1e-9, stp)
    for v in xt:
        if abs(v) > 1e-9:
            bg(ax.axvline(v, color=GRID, lw=0.8, zorder=0))
    ax.axvline(0, color=C["base"], lw=1.4, zorder=1)
    ax.text(-0.015 * (x1 - x0), 0.55, "untouched", fontsize=9, color=INK2, ha="right", va="center")
    for r in rows:
        col, emph = C[r["arm"]], r["arm"] == "S"
        ax.plot([r["lo"], r["hi"]], [r["y"], r["y"]], color=col, lw=2.6 if emph else 1.9, solid_capstyle="round",
                zorder=2)
        ax.plot([r["d"]], [r["y"]], "o", ms=8.0 if emph else 7.0, mfc=col if r["sig"] else "white", mec=col, mew=1.5,
                zorder=3)
    ax.set_yticks([r["y"] for r in rows])
    ax.set_yticklabels([r["arm"] for r in rows], fontsize=9)
    for lab, r in zip(ax.get_yticklabels(), rows):
        lab.set_color(INK if r["arm"] == "S" else INK2)
        if r["arm"] == "S":
            lab.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", labelsize=9, width=0.9, length=3.5)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.set_xticks(xt)
    ax.set_xticklabels([tick_txt(v) for v in xt])
    ax.set_xlabel(f"difference from untouched (percentage points;\nARC: % of its {N_TASKS} puzzles)", fontsize=9,
                  labelpad=3, linespacing=1.2)
    ytr = blended_transform_factory(g.transFigure, ax.transData)
    for grp in groups:
        yc = (grp["y0"] + grp["y1"]) / 2
        ax.text((AX_L - 0.28) / WW, yc + 0.22, grp["bench"], transform=ytr, fontsize=10.5, fontweight="bold",
                color=INK, ha="right", va="bottom")
        ax.text((AX_L - 0.28) / WW, yc + 0.10, grp["model"].replace("\n", " "), transform=ytr, fontsize=9,
                color=INK2, ha="right", va="top")
    # significance key and S/R/L names, in the empty space right of the lower groups' intervals (as panel b)
    low = [r for r in rows if r["key"] in (groups[-2]["key"], groups[-1]["key"])]
    kx = max(r["hi"] for r in low) + 0.10 * (x1 - x0)
    keys = [("dot", True, "95% CI excludes 0"), ("dot", False, "95% CI includes 0"),
            ("txt", None, "S shortest, R random,"), ("txt", None, "L longest replay")]
    yk0 = groups[-2]["y0"]
    for i, (kind, filled, txt) in enumerate(keys):
        yk = yk0 - i * 1.25 - (0.5 if kind == "txt" else 0.0)
        if kind == "dot":
            ax.plot([kx], [yk], "o", ms=6.5, mfc=INK2 if filled else "white", mec=INK2, mew=1.3)
            ax.annotate(txt, xy=(kx, yk), xytext=(7, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=9, color=INK2)
        else:
            ax.annotate(txt, xy=(kx, yk), xytext=(-4, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=9, color=INK2)
    probs = check_layout(g)
    os.makedirs(os.path.dirname(WC_PATH), exist_ok=True)
    g.savefig(WC_PATH, dpi=300)
    plt.close(g)
    return probs


wc_issues = writeup_cover()
print("wrote", WC_PATH, "(2400 x 1200 px)")
print("writeup cover layout:", "clean (no overlaps, no text on marks, nothing clipped, nothing below 6 pt)"
      if not wc_issues else f"{len(wc_issues)} issue(s)")
for s in wc_issues:
    print("   ", s)
