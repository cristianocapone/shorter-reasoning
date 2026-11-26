"""Figure 1 of the Nature Communications revision (natcomm/main_v2.tex): panel a from make_cover_nc.py (the selection
rule); b, the pre-registered test on 4,500 held-out MATH problems (data/conf4500_result.json); c, shortest minus random
and minus longest replay in the exploratory single-sample settings (data/paper_stats.json; ARC moved to the SI).

    env -u PYTHONPATH .venv_mps/bin/python paper_brevity/make_cover_nc2.py  -> figures/cover_nc2.{pdf,png}
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
OUT = os.path.join(FIGS, "cover_nc2")
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

# ================================================================ b: the pre-registered test (4,500 held-out MATH problems)
with open(os.path.join(DATA, "conf4500_result.json")) as fh:
    CONF = json.load(fh)
H1, H2 = CONF["H1"], CONF["H2"]
if not (H1["supported"] and H2["supported"]):
    warn("pre-registered hypotheses no longer both supported")
XB0 = 2.38
heading(XB0, "b", "Pre-registered test: shortest minus random")
ftext(XB0 + 0.16, HEAD_Y - 0.19, f"{CONF['n_problems']:,} held-out MATH problems, Qwen3-4B, 8k cap,", fontsize=FS_SMALL,
      color=INK2, va="baseline")
ftext(XB0 + 0.16, HEAD_Y - 0.31, f"{CONF['samples_per_arm']} samples per problem and arm", fontsize=FS_SMALL,
      color=INK2, va="baseline")
B_ROWS = [("Δ accuracy", H1, True), ("finishing\nwithin the cap", H2["finishing"], False),
          ("quality of\nfinished answers", H2["quality"], False), ("cut samples\ncredited", H2["cut_credited_residual"], False)]
BX, BW, BY = 3.30, 1.45, 0.62
BTOP = HEAD_Y - 0.50
axb = ax_at(BX, BY, BW, BTOP - BY)
ys_b = [0.0, -1.25, -2.25, -3.25]
lo_all = min(r[1]["ci95"][0] for r in B_ROWS); hi_all = max(r[1]["ci95"][1] for r in B_ROWS)
xmin, xmax = min(lo_all, 0) - 0.35, hi_all + 0.35
axb.set_xlim(xmin, xmax)
axb.set_ylim(ys_b[-1] - 0.7, ys_b[0] + 0.7)
bg(axb.add_patch(Rectangle((xmin, ys_b[1] - 0.45), xmax - xmin, 0.9, fc=BAND, ec="none", zorder=0)))
axb.axvline(0, color=AXIS, lw=0.8, zorder=1)
for v in np.arange(np.ceil(xmin * 2) / 2, xmax + 1e-9, 0.5):
    if abs(v) > 1e-9:
        bg(axb.axvline(v, color=GRID, lw=0.5, zorder=0))
col = C["S"]
for (lab, st, bold), y in zip(B_ROWS, ys_b):
    lo, hi = st["ci95"]
    sig = lo > 0 or hi < 0
    axb.plot([lo, hi], [y, y], color=col, lw=1.4, solid_capstyle="round", zorder=2)
    axb.plot([st["est"]], [y], "o", ms=4.8, mfc=col if sig else "white", mec=col, mew=1.0, zorder=3)
axb.set_yticks(ys_b)
axb.set_yticklabels([r[0] for r in B_ROWS])
for lab, (_, _, bold) in zip(axb.get_yticklabels(), B_ROWS):
    lab.set_color(INK); lab.set_fontsize(FS_SMALL); lab.set_linespacing(1.05)
    if bold:
        lab.set_fontweight("bold")
axb.tick_params(axis="y", length=0, pad=3)
axb.spines["left"].set_visible(False)
xt = np.arange(np.ceil(xmin), xmax + 1e-9, 1.0)
axb.set_xticks(xt)
axb.set_xticklabels([num("b x tick", tick_txt(v)) for v in xt])
axb.set_xlabel("points of accuracy", labelpad=2)
axb.text(H1["est"], ys_b[0] + 0.38, num("b H1", f"{signed(H1['est'], 2)} [{H1['ci95'][0]:.2f}, {H1['ci95'][1]:.2f}], ")
         + num("b H1 P", r"P < 2×10$^{-4}$"), fontsize=FS_SMALL, color=INK, ha="center", va="bottom")
ftext(XB0 + 0.16, 0.20, f"tokens per sample: {CONF['tokens_per_sample']['S']:,.0f} vs "
      f"{CONF['tokens_per_sample']['R']:,.0f} "
      f"({100 * (CONF['tokens_per_sample']['S'] / CONF['tokens_per_sample']['R'] - 1):+.1f}%)".replace("-", "−"),
      fontsize=FS_SMALL, color=INK2, va="baseline")

# ================================================================ c: the ordering in the exploratory settings
GROUPS = [("math_qwen4b", "MATH-500", "Qwen3-4B", "acc"), ("math_llama8b", "MATH-500", "R1-Llama-8B", "acc"),
          ("apps_qwen4b", "APPS", "Qwen3-4B", "acc")]
OTHERS_B = ["R", "L"]
rows_b, groups_b, yy = [], [], 0.0
for key, bench, model, metric in GROUPS:
    blk, ys = STATS[key], []
    for o in OTHERS_B:
        c = blk["contrasts"][f"S-{o}"][metric]
        rows_b.append(dict(key=key, other=o, y=yy, d=c["diff"], lo=c["lo"], hi=c["hi"], sig=(c["lo"] > 0) or (c["hi"] < 0)))
        ys.append(yy)
        yy -= 1.0
    groups_b.append(dict(key=key, bench=bench, model=model, y0=ys[0], y1=ys[-1]))
    yy -= 0.55
RANK_OK = all(r["d"] > 0 for r in rows_b)
if not RANK_OK:
    warn("shortest no longer first in every exploratory setting")
XC0 = 4.98
heading(XC0, "c", "Exploratory settings")
ftext(XC0 + 0.16, HEAD_Y - 0.19, "shortest minus random or longest replay", fontsize=FS_SMALL, color=INK2,
      va="baseline")
CXc, CW, CY = 6.02, 1.10, 0.62
axc = ax_at(CXc, CY, CW, BTOP - CY)
x_lo, x_hi = min(r["lo"] for r in rows_b), max(r["hi"] for r in rows_b)
cmin, cmax = min(x_lo, 0) - 0.3, x_hi + 0.3
axc.set_xlim(cmin, cmax)
axc.set_ylim(min(r["y"] for r in rows_b) - 0.65, rows_b[0]["y"] + 0.75)
for v in np.arange(np.ceil(cmin), cmax + 1e-9, 1.0):
    if abs(v) > 1e-9:
        bg(axc.axvline(v, color=GRID, lw=0.5, zorder=0))
axc.axvline(0, color=AXIS, lw=0.8, zorder=1)
for r in rows_b:
    axc.plot([r["lo"], r["hi"]], [r["y"], r["y"]], color=col, lw=1.4, solid_capstyle="round", zorder=2)
    axc.plot([r["d"]], [r["y"]], "o", ms=4.8, mfc=col if r["sig"] else "white", mec=col, mew=1.0, zorder=3)
axc.set_yticks([r["y"] for r in rows_b])
axc.set_yticklabels(["vs " + r["other"] for r in rows_b])
for lab in axc.get_yticklabels():
    lab.set_color(INK); lab.set_fontsize(FS_SMALL)
axc.tick_params(axis="y", length=0, pad=3)
axc.spines["left"].set_visible(False)
xtc = np.arange(np.ceil(cmin), cmax + 1e-9, 1.0)
axc.set_xticks(xtc)
axc.set_xticklabels([num("c x tick", tick_txt(v)) for v in xtc])
axc.set_xlabel("points of accuracy", labelpad=2)
ytr = axc.get_yaxis_transform()
for g in groups_b:
    yc = (g["y0"] + g["y1"]) / 2
    axc.text(-0.30, yc + 0.22, g["bench"], transform=ytr, fontsize=FS_SMALL, fontweight="bold", color=INK, ha="right",
             va="bottom")
    axc.text(-0.30, yc + 0.12, g["model"], transform=ytr, fontsize=FS_SMALL, color=INK2, ha="right", va="top")
for i, (filled, txt) in enumerate(((True, "CI excludes 0"), (False, "CI includes 0"))):
    ftext(XC0 + 0.30, 0.32 - 0.13 * i, txt, fontsize=FS_SMALL, color=INK2, va="center")
    fig.add_artist(matplotlib.lines.Line2D([(XC0 + 0.22) / W], [(0.32 - 0.13 * i) / H], marker="o", ms=4.2,
                                           mfc=INK2 if filled else "white", mec=INK2, mew=0.9, transform=fig.transFigure))

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



issues = check_layout()
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
print("wrote", OUT)
print("layout:", "clean" if not issues else "")
for s in issues:
    print("   ", s)
print("numbers printed on the figure:")
for what, s in SHOWN:
    print(f"    {s:>24}  {what}")
print("claims:", "all hold" if not WARNINGS else f"{len(WARNINGS)} WARNING(S)")
