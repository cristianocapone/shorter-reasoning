"""Figures for the brevity paper (both formats), except Figure 1, the cover (make_cover.py).

    env -u PYTHONPATH .venv_mps/bin/python paper_brevity/make_figs.py
    -> paper_brevity/figures/<name>.pdf           vector, Arial embedded as TrueType (pdf.fonttype 42)
       paper_brevity/figures/<name>.png           300 dpi
       paper_brevity/figures/<name>_caption.md    caption draft; its numbers come from the same data

    fig_restart  the restart-view prediction and its test: loosening the MATH cap from 8k to 16k tokens reduces the
                 advantage of shortest over longest replay to a non-significant difference (a accuracy and the cap x
                 rule interaction, b finishing, c tokens)
    fig1         design schematic + ARC-AGI-2 puzzles solved per seed (panel a repeats cover panel a)
    fig3         a MATH round 2, b ARC agent messages, c the 4B model on ARC-1
    fig4         the direct-answering regime of the 27B model
    figS1        supplementary (was fig2a): difference from the untouched model per replay rule in each setting's
                 own unit; cover panel b shows the same contrasts (plus R1-Llama-8B and S2) in percentage points
    figS2        supplementary (was fig2b): token cost of every arm in the four settings

Data. Every number plotted or printed is read at run time from paper_brevity/data/paper_stats.json (plus the ARC
message-log cap and seed coverage from paper_brevity/data/arc_msgs.json), except the two blocks marked
NOT IN THE JSON below (fig3c and fig4), which are hard-coded by hand. P values are two-sided, 2 min(p, 1 - p), from
the one-sided bootstrap tails stored in the JSON. Formatting and rounding are those of make_cover.py (Python's
format(), i.e. the stored binary value rounded half-to-even; e.g. 0.125 -> 0.12, 63.25 -> 63.2).
The script prints every number it put on each figure, warns when a claim made by a heading stops holding, checks
each layout (text overlaps, text on data marks, clipping, text below 6 pt) and lists the fonts embedded in each PDF.
Wording follows the rescoped arc/main.tex (2026-11-02): the 16k cap "reduces" the advantage over longest replay "to a
non-significant" difference (not "removes the gain"); finishing differences "shrink" (not "equalise"); fig1b states
the ARC-AGI-2 gain with its P over puzzles and over runs (exact permutation test on the five run totals per arm, as
derived_stats.py); fig4b says that each intervention was run once.
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
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import Bbox, blended_transform_factory  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))                 # .../paper_brevity
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures")
with open(os.path.join(DATA, "paper_stats.json")) as fh:
    STATS = json.load(fh)
with open(os.path.join(DATA, "arc_msgs.json")) as fh:
    MSGS = json.load(fh)

# method constants (not data): the two MATH caps of the cap test, in thousand tokens per sample. The cap_interaction
# keys of the JSON (acc_8k, acc_16k, acc_8k_minus_16k, ...) are built from them and checked in make_restart().
CAP_LO, CAP_HI = 8, 16

# ================================================================ NOT IN THE JSON: hard-coded by hand
# (1) fig3c. Qwen3-4B on ARC-1 (60 puzzles, 16k-token cap): the replication in which behaviour moved and accuracy did
#     not. Source: natcomm/main.tex ("cut 32% of its messages and wrote code in 65% of them, against 62-68% and 30-36%
#     for the random and longest arms ... 13, 12 and 12 of 60 puzzles at a 16,000-token cap; untouched 13"), identical
#     to SMALL['16k'] / SMALL_MECH of the previous make_figs.py (backup_20261029/make_figs.py). The untouched 4B model
#     has no message statistics in that source.
ARC1_4B = {"n": 60,
           "solved": {"S": 13, "R": 12, "L": 12, "base": 13},           # puzzles solved (of n)
           "cut_msgs": {"S": 32, "R": 62, "L": 68},                     # % of messages cut at the cap
           "code_msgs": {"S": 65, "R": 36, "L": 30}}                    # % of messages with code
# (2) fig4. Qwen3.8-27B answering ARC directly (no tools: hidden reasoning, then the output grid; two samples per test
#     input) on the 40 development puzzles. Source: natcomm/main.tex (regime section), identical to DIRECT_BUDGET /
#     DIRECT_INTERV of the previous make_figs.py. 'greedy answer phase' is 10.3 there (the text rounds it to 10).
DIRECT = {"n": 40,
          "budget": [(32, [4]), (48, [10]), (64, [17, 16]), (96, [14])],   # k tokens per answer: solved, per seed
          "plain_budget": 64,                                              # budget of the intervention runs
          "interventions": [("terse prompt", 15), ("forced answer at cap", 16), ("force + filter + resample", 15),
                            ("self-check 2nd sample", 14), ("self-distilled (direct)", 14), ("temperature 1.0", 12),
                            ("transposed 2nd sample", 10), ("greedy answer phase", 10.3)]}

# ================================================================ style (same tokens as make_cover.py)
# Paper palette with R #8a8f98 -> #90959e (the dataviz validator hard-fails S vs #8a8f98 on the normal-vision floor,
# dE 14.4 < 15; #90959e gives 16.0 and still 3.0:1 on white). S vs L sits in the CVD warn band (dE 7.4, protan), so no
# S/L mark relies on colour alone: every one carries its arm's name or letter. Navy `base` (the reference) and grey R
# (the neutral control) sit outside the categorical bands by role; S2 is an ordinal step of S, always labelled.
C = {"S": "#1b7f6e", "R": "#90959e", "L": "#c4573a", "base": "#2f3e5c", "S2": "#0b4f43"}
NAME = {"S": "shortest", "R": "random", "L": "longest", "base": "untouched", "S2": "round 2"}
INK, INK2, MUTED = "#1d1d1b", "#4f4e4a", "#6f6e69"                # 16.9:1, 8.3:1, 5.1:1 on white
AXIS, GRID, PALE, BAND = "#a3a29b", "#ebeae6", "#dcdde1", "#f1f0ec"
NEUTRAL = "#b4b7bd"   # fig4b de-emphasis bars (not an arm colour); 2.0:1 on white, so every bar carries its value
FS, FS_SMALL, FS_HEAD, FS_LETTER = 6.6, 6.0, 7.4, 9.0
MINUS = "−"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS, "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.labelsize": FS_SMALL, "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


# ================================================================ number formatting (as make_cover.py)
def signed(v, nd=1):
    s = f"{v:+.{nd}f}"
    if float(s) == 0.0:
        s = f"{0.0:.{nd}f}"
    return s.replace("-", MINUS)


def unsigned(v, nd=1):
    s = f"{v:.{nd}f}"
    if float(s) == 0.0:
        s = f"{0.0:.{nd}f}"
    return s.replace("-", MINUS)


def ci_txt(c, nd=2):
    return f"{signed(c['diff'], nd)} [{unsigned(c['lo'], nd)}, {unsigned(c['hi'], nd)}]"


def tick_txt(v, nd=0):
    return "0" if abs(v) < 1e-9 else signed(v, nd)


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


def is_sig(c):
    return c["lo"] > 0 or c["hi"] < 0


def pct_change(a, b):
    return 100.0 * (a / b - 1.0)


def pc_txt(a, b, nd=0):
    return signed(pct_change(a, b), nd) + "%"


def nice_ticks(lo, hi, step):
    return np.arange(np.ceil(lo / step - 1e-9) * step, hi + 1e-9, step) + 0.0


def seeds(key, arm):
    return len(STATS[key]["arms"][arm]["seeds"])


def bg(*artists):
    """Background artists (grid, bands): the text-on-mark check ignores them."""
    for a in artists:
        for b in (a if isinstance(a, (list, tuple)) else [a]):
            b.set_gid("bg")
    return artists[0] if len(artists) == 1 else artists


# ================================================================ a figure laid out in inches, with layout checks
class Fig:
    def __init__(self, name, w, h):
        self.name, self.W, self.H = name, w, h
        self.fig = plt.figure(figsize=(w, h), dpi=300)            # measure text at output resolution
        self.head_y = h - 0.15                                     # heading baseline
        self.shown, self.warnings, self.segments = [], [], []
        self._overlay = None

    # ---- placement
    def ax(self, x, y, w, h):
        return self.fig.add_axes([x / self.W, y / self.H, w / self.W, h / self.H])

    def overlay(self):
        """An invisible axes over the whole figure, in inches (key markers, chips, brackets between axes)."""
        if self._overlay is None:
            a = self.ax(0, 0, self.W, self.H)
            a.set_xlim(0, self.W)
            a.set_ylim(0, self.H)
            a.axis("off")
            self._overlay = a
        return self._overlay

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FS_SMALL)
        kw.setdefault("color", INK2)
        return self.fig.text(x / self.W, y / self.H, s, **kw)

    def heading(self, x, letter, title, sub=None, title2=None):
        """Panel letter and bold title on the heading baseline; an optional second title line (title2) pushes the
        subtitle down by one line."""
        y = self.head_y
        dx = 0.16 if letter else 0.0
        if letter:
            self.text(x, y, letter, fontsize=FS_LETTER, fontweight="bold", color=INK, va="baseline")
        self.text(x + dx, y, title, fontsize=FS_HEAD, fontweight="bold", color=INK, va="baseline")
        dy = 0.0
        if title2:
            dy = 0.125
            self.text(x + dx, y - dy, title2, fontsize=FS_HEAD, fontweight="bold", color=INK, va="baseline")
        if sub:
            self.text(x + dx, y - 0.13 - dy, sub, fontsize=FS_SMALL, color=INK2, va="top", linespacing=1.25)

    def measure(self, s, **kw):
        """Width (in) of a string set with these text properties, without leaving it on the figure."""
        t = self.fig.text(0, 0, s, **kw)
        w = t.get_window_extent(self.fig.canvas.get_renderer()).width / self.fig.dpi
        t.remove()
        return w

    def per_unit(self, ax):
        """Inches per data unit along x and y (limits must be set)."""
        bb = ax.get_position()
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        return bb.width * self.W / (x1 - x0), bb.height * self.H / (y1 - y0)

    def bar(self, ax, pos, v0, v1, thick, color, horizontal=True, r=0.022, zorder=2):
        """A bar from the baseline v0 to v1 (data units), `thick` data units across, drawn as one path: rounded data
        end (radius r in inches, circular on paper), square at the baseline."""
        if v1 == v0:
            return
        px, py = self.per_unit(ax)
        pv, pa = (px, py) if horizontal else (py, px)               # inches per unit: value axis, across axis
        s = 1.0 if v1 > v0 else -1.0
        r_in = min(r, abs(v1 - v0) * pv / 2, thick * pa / 2)
        rv, ra, k = r_in / pv, r_in / pa, 0.5522847498               # radii in data units; Bezier circle constant
        a0, a1 = pos - thick / 2, pos + thick / 2
        e = v1 - s * rv
        va = [(v0, a0), (e, a0), (e + s * k * rv, a0), (v1, a0 + ra - k * ra), (v1, a0 + ra), (v1, a1 - ra),
              (v1, a1 - ra + k * ra), (e + s * k * rv, a1), (e, a1), (v0, a1), (v0, a0)]
        codes = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CLOSEPOLY]
        verts = va if horizontal else [(b, a) for a, b in va]
        ax.add_patch(PathPatch(Path(verts, codes), fc=color, ec="none", lw=0, zorder=zorder))

    def key_sig(self, x, y, color=INK2):
        """Filled / hollow key for the significance convention of the cover (95% interval excludes / includes 0)."""
        ov = self.overlay()
        for filled, txt in ((True, "95% interval excludes 0"), (False, "includes 0")):
            ov.plot([x + 0.035], [y], "o", ms=4.2, mfc=color if filled else "white", mec=color, mew=0.9)
            self.text(x + 0.10, y, txt, va="center")
            x += 0.10 + self.measure(txt, fontsize=FS_SMALL) + 0.18

    def num(self, what, s):
        self.shown.append((what, s))
        return s

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"WARNING [{self.name}]: {msg}")

    # ---- checks (ported from make_cover.py)
    def text_artists(self):
        out = list(self.fig.texts)
        for ax in self.fig.axes:
            out += list(ax.texts)
            if ax.axison:
                out += list(ax.get_xticklabels()) + list(ax.get_yticklabels()) + [ax.xaxis.label, ax.yaxis.label]
        return [t for t in out if t.get_visible() and t.get_text().strip()]

    def mark_points(self, ax):
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
                rad = ln.get_markersize() / 2 * self.fig.dpi / 72
                ang = np.linspace(0, 2 * np.pi, 16)[:, None]
                pts += [p0 + rad * np.hstack([np.cos(ang), np.sin(ang)]) for p0 in xy] + [xy]
        for p in ax.patches:
            if p.get_gid() != "bg":
                v = p.get_path().transformed(p.get_transform()).vertices
                pts.append(v)
                b = Bbox.from_extents(v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max())
                gx, gy = np.meshgrid(np.linspace(b.x0, b.x1, 12), np.linspace(b.y0, b.y1, 6))
                pts.append(np.column_stack([gx.ravel(), gy.ravel()]))
        for ax2, p0, p1 in self.segments:
            if ax2 is ax:
                xy = ax.transData.transform([p0, p1])
                pts.append(xy[0] + seg * (xy[1] - xy[0]))
        return np.concatenate(pts) if pts else np.zeros((0, 2))

    def check(self):
        fig = self.fig
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        texts = self.text_artists()
        boxes = [Text.get_window_extent(t, rend) for t in texts]   # an Annotation's text only, not its arrow
        page, issues = fig.bbox, []
        gap = 0.6 * fig.dpi / 72                                     # texts at least ~1 pt apart
        for t, b in zip(texts, boxes):
            if t.get_fontsize() < 6.0 - 1e-9:
                issues.append(f"text below 6 pt: {t.get_text()!r} ({t.get_fontsize()} pt)")
            if b.x0 < page.x0 or b.y0 < page.y0 or b.x1 > page.x1 or b.y1 > page.y1:
                issues.append(f"outside the page: {t.get_text()!r}")
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                b1, b2 = boxes[i], boxes[j]
                if b1.x0 < b2.x1 + gap and b2.x0 < b1.x1 + gap and b1.y0 < b2.y1 + gap and b2.y0 < b1.y1 + gap:
                    issues.append(f"text overlap: {texts[i].get_text()!r} x {texts[j].get_text()!r}")
        cache = {}
        pad = 0.5 * fig.dpi / 72
        for t, b in zip(texts, boxes):
            if t.get_gid() == "inmark":
                continue
            for ax in fig.axes:
                if ax not in cache:
                    cache[ax] = self.mark_points(ax)
                P = cache[ax]
                if len(P) and ((P[:, 0] > b.x0 + pad) & (P[:, 0] < b.x1 - pad) &
                               (P[:, 1] > b.y0 + pad) & (P[:, 1] < b.y1 - pad)).any():
                    issues.append(f"text on a data mark: {t.get_text()!r}")
                    break
        return issues

    def save(self, caption):
        issues = self.check()
        base = os.path.join(OUT, self.name)
        self.fig.savefig(base + ".pdf")
        self.fig.savefig(base + ".png", dpi=300)
        plt.close(self.fig)
        with open(base + "_caption.md", "w") as fh:
            fh.write(caption.strip() + "\n")
        return issues


def row_label(f, ax, x_in, y, s, **kw):
    """Text at a figure x (inches) and a data y of `ax` (row labels shared by several column axes)."""
    tr = blended_transform_factory(f.fig.transFigure, ax.transData)
    kw.setdefault("fontsize", FS)
    kw.setdefault("color", INK)
    kw.setdefault("va", "center")
    return ax.text(x_in / f.W, y, s, transform=tr, **kw)


# ================================================================ shared data checks
ARC = STATS["arc"]
ARC_A = ARC["arms"]
ARC_N = ARC["n_tasks"]
ARC_CAP = MSGS["cap"]                                                # tokens per agent message


def arc_log_coverage():
    """Seeds covered by the ARC per-message logs (arc_msgs.json), per arm; checks the logs against paper_stats."""
    msg_tok, per_seed = {}, {}
    for a in ("S", "R", "L", "base"):
        X = np.asarray(MSGS["arms"][a], dtype=float)
        tok, cut, code = X[:, 0], X[:, 3], X[:, 4]
        msg_tok[a] = tok.sum()
        assert abs(100 * cut.mean() - ARC_A[a]["cut_msgs"]) < 0.05, f"{a}: cut share differs between the files"
        assert abs(100 * code.mean() - ARC_A[a]["code_msgs"]) < 0.05, f"{a}: code share differs between the files"
        assert abs(np.minimum(tok, ARC_CAP).mean() / code.mean() / 1000 - ARC_A[a]["restart_cost_k"]) < 0.05, \
            f"{a}: restart cost differs between the files"
    for a in msg_tok:   # complete logs reproduce tokens_per_solved_k from all seeds' solves
        full = abs(msg_tok[a] / sum(ARC_A[a]["seeds"]) / 1000 / ARC_A[a]["tokens_per_solved_k"] - 1) < 0.005
        per_seed[a] = len(MSGS["arms"][a]) / len(ARC_A[a]["seeds"]) if full else None
    mps = np.mean([v for v in per_seed.values() if v])
    return {a: (len(ARC_A[a]["seeds"]) if per_seed[a] else int(round(len(MSGS["arms"][a]) / mps))) for a in per_seed}


ARC_LOGGED = arc_log_coverage()

SETTINGS = [  # data key, benchmark, model, accuracy metric
    ("arc", "ARC-AGI-2", "Qwen3.8-27B agent", "solved"),
    ("math_qwen4b", "MATH-500", "Qwen3-4B", "acc"),
    ("math_llama8b", "MATH-500", "R1-Llama-8B", "acc"),
    ("apps_qwen4b", "APPS", "Qwen3-4B", "acc"),
]


def seeds_phrase(key):
    n = {a: len(v["seeds"]) for a, v in STATS[key]["arms"].items()}
    core = [n[a] for a in ("S", "R", "L", "base")]
    s = (f"{core[0]} seeds per arm" if len(set(core)) == 1
         else ", ".join(f"{NAME[a]} {n[a]}" for a in ("S", "R", "L", "base")) + " seeds")
    if "S2" in n and n["S2"] != core[0]:
        s += f", {n['S2']} for round 2"
    return s


# ================================================================ fig_restart: the restart prediction and its test
def make_restart():
    q8, q16 = STATS["math_qwen4b"], STATS["math_qwen4b_cap16k"]
    ci = STATS["cap_interaction_qwen4b"]["contrasts"]
    k8, k16 = f"{CAP_LO}k", f"{CAP_HI}k"
    INTER = f"acc_{k8}_minus_{k16}"
    # the interaction block re-uses the per-cap contrasts of math_qwen4b (8k) and math_qwen4b_cap16k (16k)
    for con, blk in ci.items():
        assert INTER in blk, f"cap_interaction {con}: no {INTER}"
        for m in ("acc", "finish"):
            for k, src in ((k8, q8), (k16, q16)):
                a, b = blk[f"{m}_{k}"], src["contrasts"][con][m]
                assert all(abs(a[x] - b[x]) < 1e-9 for x in ("diff", "lo", "hi", "p")), \
                    f"cap_interaction {con} {m}_{k} differs from the per-cap block"
    N = STATS["cap_interaction_qwen4b"]["n_problems"]
    GROUPS = [("S-L", "vs longest"), ("S-base", "vs untouched")]
    f = Fig("fig_restart", 7.2, 2.72)

    # ---- claims made by the headings: a, the advantage over longest replay is significant at 8k and falls to a smaller,
    # non-significant value at 16k (the change significant); b, both finishing differences are significant at 8k and
    # smaller and non-significant at 16k; c, shortest replay generates the fewest tokens at both caps
    sl = ci["S-L"]
    ok_a = (sl[f"acc_{k8}"]["lo"] > 0 and not is_sig(sl[f"acc_{k16}"]) and sl[INTER]["lo"] > 0
            and 0 < sl[f"acc_{k16}"]["diff"] < sl[f"acc_{k8}"]["diff"])
    ok_b = all(ci[g][f"finish_{k8}"]["lo"] > 0 and not is_sig(ci[g][f"finish_{k16}"])
               and ci[g][f"finish_{k16}"]["diff"] < ci[g][f"finish_{k8}"]["diff"] for g, _ in GROUPS)
    ok_c = all(src["arms"]["S"]["tokens"] < min(src["arms"][o]["tokens"] for o in ("L", "base")) for src in (q8, q16))
    if not ok_a:
        f.warn(f"panel a heading no longer holds (neutral heading used): S-L {sl}")
    if not ok_b:
        f.warn("panel b heading no longer holds (neutral heading used)")
    if not ok_c:
        f.warn("panel c heading no longer holds (neutral heading used): shortest is not the shortest at both caps")

    # ---- geometry (inches)
    HY = f.head_y
    C_TOP, C_BOT = HY - 0.66, 0.72
    X_LAB, X1, COL_W = 0.06, 0.76, 0.90
    X2 = X1 + COL_W + 0.16
    X3 = X2 + COL_W + 0.42
    XC = 4.80                                                       # panel c heading
    rows, yy = [], 0.0                                              # per group: a header slot, then the two caps
    for con, lab in GROUPS:
        rows.append(dict(kind="head", con=con, lab=lab, y=yy))
        yy -= 1.0
        for k in (k8, k16):
            rows.append(dict(kind="cap", con=con, cap=k, y=yy))
            yy -= 1.0
        yy -= 0.35
    ylo, yhi = yy + 0.35 - 0.45, 0.45
    vals = [ci[g][f"{m}_{k}"][e] for g, _ in GROUPS for m in ("acc", "finish") for k in (k8, k16) for e in ("lo", "hi")]
    vals += [ci[g][INTER][e] for g, _ in GROUPS for e in ("lo", "hi")] + [0.0]
    xmin, xmax = min(vals) - 0.3, max(vals) + 0.3
    xt = nice_ticks(xmin, xmax, 1.0)

    def column(x_in, title=None, band=False):
        ax = f.ax(x_in, C_BOT, COL_W, C_TOP - C_BOT)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ylo, yhi)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        if band:
            bg(ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc=BAND, ec="none", zorder=0)))
        for r in rows:
            if r["kind"] == "head":
                ax.plot([0, 0], [r["y"] - 0.55, r["y"] - 2.45], color=AXIS, lw=0.8, zorder=1, solid_capstyle="butt")
        ax.set_xticks(xt)
        ax.set_xticklabels([f.num("x tick (points)", tick_txt(v)) for v in xt])
        if title:
            ax.text(0.5, 1.0 + 0.07 / (C_TOP - C_BOT), title, transform=ax.transAxes, fontsize=FS_SMALL, color=INK,
                    ha="center", va="bottom", fontweight="bold" if band else "normal")
        return ax

    def interval(ax, c, y):
        col = C["S"]
        ax.plot([c["lo"], c["hi"]], [y, y], color=col, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot([c["diff"]], [y], "o", ms=4.8, mfc=col if is_sig(c) else "white", mec=col, mew=1.0, zorder=3)

    # ---- a: accuracy at each cap, and the change between caps
    f.heading(0.04, "a", "Loosening the cap reduces the gain" if ok_a else "Accuracy at two caps",
              "MATH-500, Qwen3-4B: shortest replay minus the other arm; the untouched\nmodel finishes "
              + f.num("a untouched finish 8k", f"{q8['arms']['base']['finish']:.0f}%") + f" of samples within {k8}, "
              + f.num("a untouched finish 16k", f"{q16['arms']['base']['finish']:.0f}%") + f" within {k16}",
              title2=("over longest replay to a non-significant "
                      + f.num(f"a S-L at {k16} (points)", unsigned(sl[f"acc_{k16}"]["diff"], 2)) + " points")
              if ok_a else None)
    ax1 = column(X1, "at each cap")
    ax2 = column(X2, f"{k8} minus {k16}", band=True)
    for r in rows:
        if r["kind"] == "head":
            row_label(f, ax1, X_LAB, r["y"], r["lab"], fontweight="bold", ha="left")
        else:
            row_label(f, ax1, X1 - 0.05, r["y"], f"{r['cap']} cap", ha="right")
            interval(ax1, ci[r["con"]][f"acc_{r['cap']}"], r["y"])
    px2, _ = f.per_unit(ax2)
    for con, _ in GROUPS:
        yh = next(r["y"] for r in rows if r["kind"] == "head" and r["con"] == con)
        c = ci[con][INTER]
        interval(ax2, c, yh - 1.5)
        txt = f.num(f"a {con} change 8k-16k, two-sided P", fmt_p(two_sided(c["p"])))
        w = f.measure(txt, fontsize=FS_SMALL, fontweight="bold") / px2
        x_left = min(max(c["diff"] - w / 2, 0.15), xmax - w - 0.05)  # above the dot, clear of the zero line
        ax2.text(x_left, yh - 0.98, txt, fontsize=FS_SMALL, ha="left", va="center",
                 color=INK if is_sig(c) else INK2, fontweight="bold" if is_sig(c) else "normal")
    f.text((X1 + X2 + COL_W) / 2, C_BOT - 0.17, "difference in accuracy (points)", ha="center", va="top")

    # ---- b: finishing within the cap (same rows as a)
    f.heading(X3 - 0.04, "b", "Finishing differences shrink" if ok_b else "Finishing within the cap",
              "share of samples that finish\nwithin the cap, same rows as a")
    ax3 = column(X3)
    for r in rows:
        if r["kind"] == "cap":
            interval(ax3, ci[r["con"]][f"finish_{r['cap']}"], r["y"])
            row_label(f, ax3, X3 - 0.05, r["y"], r["cap"], ha="right", fontsize=FS_SMALL, color=INK2)
    f.text(X3 + COL_W / 2, C_BOT - 0.17, "difference in samples\nfinished (points)", ha="center", va="top",
           linespacing=1.2)

    # ---- c: tokens per sample at both caps
    f.heading(XC, "c", "Shortest replay stays shorter" if ok_c else "Tokens per sample at two caps",
              "tokens per sample, mean over seeds")
    XCA, CW, CB, CT = XC + 0.64, 1.12, C_BOT + 0.30, HY - 0.42
    axc = f.ax(XCA, CB, CW, CT - CB)
    caps = [(k8, q8), (k16, q16)]
    arms_c = ["L", "base", "S"]
    tok = {a: [src["arms"][a]["tokens"] for _, src in caps] for a in arms_c}
    t_lo, t_hi = min(min(v) for v in tok.values()), max(max(v) for v in tok.values())
    axc.set_xlim(-0.28, 1.28)
    axc.set_ylim(np.floor((t_lo - 300) / 500) * 500, t_hi + 350)
    yt = nice_ticks(*axc.get_ylim(), 1000)
    for v in yt:
        bg(axc.axhline(v, color=GRID, lw=0.5, zorder=0))
    for a in arms_c:
        axc.plot([0, 1], tok[a], color=C[a], lw=1.6 if a == "S" else 1.2, solid_capstyle="round", zorder=2)
        axc.plot([0, 1], tok[a], "o", ms=5.0, mfc=C[a], mec="white", mew=0.8, zorder=3)
        axc.annotate(NAME[a], xy=(1, tok[a][1]), xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                     fontsize=FS, color=INK, fontweight="bold" if a == "S" else "normal")
    axc.set_xticks([0, 1])
    axc.set_xticklabels([f"{k} cap" for k, _ in caps])
    axc.tick_params(axis="x", length=0, pad=3)
    axc.spines["bottom"].set_visible(False)
    axc.set_yticks(yt)
    axc.set_yticklabels([f.num("c y tick (tokens)", f"{v / 1000:g}k") for v in yt])
    axc.set_ylabel("tokens per sample", labelpad=3)
    # under the axis: shortest vs the two other arms, per cap
    pxc, _ = f.per_unit(axc)
    x_cap = [XCA + (x - axc.get_xlim()[0]) * pxc for x in (0, 1)]
    for i, o in enumerate(("L", "base")):
        y_in = CB - 0.25 - 0.13 * i
        f.text(x_cap[0] - 0.22, y_in, f"shortest vs {NAME[o]}", ha="right", va="baseline")
        for j, (k, src) in enumerate(caps):
            f.text(x_cap[j], y_in, f.num(f"c S vs {o} tokens {k}", pc_txt(src["arms"]["S"]["tokens"],
                                                                         src["arms"][o]["tokens"])),
                   ha="center", va="baseline", color=INK, fontweight="bold" if o == "L" else "normal")

    f.key_sig(X_LAB - 0.02, 0.16)

    # ---- caption
    def pair(con, m):
        return ci[con][f"{m}_{k8}"], ci[con][f"{m}_{k16}"]
    a_l8, a_l16 = pair("S-L", "acc")
    a_b8, a_b16 = pair("S-base", "acc")
    f_l8, f_l16 = pair("S-L", "finish")
    f_b8, f_b16 = pair("S-base", "finish")
    il, ib = ci["S-L"][INTER], ci["S-base"][INTER]
    n8 = {seeds("math_qwen4b", a) for a in ("S", "L", "base")}
    n16 = {seeds("math_qwen4b_cap16k", a) for a in ("S", "L", "base")}
    assert len(n8) == 1 and len(n16) == 1
    tk = {k: {a: src["arms"][a]["tokens"] for a in arms_c} for k, src in caps}
    cap = (
        ("**Loosening the token cap reduces the advantage of shortest over longest replay to a non-significant "
         f"{unsigned(a_l16['diff'], 2)} points but not its brevity.** " if ok_a and ok_c else
         "**Accuracy, finishing and tokens of shortest replay at two token caps.** ")
        + "If the cap t on a sample acts as a restart cutoff, the expected cost of a solution is E[min(L, t)] / "
        "P(L ≤ t) (Luby, Sinclair & Zuckerman, 1993), so replaying short solutions, which shifts the length "
        "distribution L to the left, should pay only while the cap binds; any account in which the gain works through "
        "finishing within the cap predicts the same. MATH-500, Qwen3-4B, with a cap of "
        f"{k8} tokens per sample ({n8.pop()} seeds per arm) or {k16} tokens ({n16.pop()} seeds per arm). Dots: mean "
        f"differences, filled when the paired-bootstrap 95% interval over the {N} problems (bar) excludes 0; "
        "P values are two-sided. "
        f"**a**, Accuracy of shortest replay minus longest replay and minus the untouched model at each cap, and the "
        f"change between the caps ({k8} minus {k16}, shaded). Against longest replay the gain falls from "
        f"{ci_txt(a_l8)} points at {k8} to {ci_txt(a_l16)} at {k16} (change {ci_txt(il)}, "
        f"{fmt_p(two_sided(il['p']))}); against the untouched model it falls from {ci_txt(a_b8)} to {ci_txt(a_b16)} "
        f"(change {ci_txt(ib)}, {fmt_p(two_sided(ib['p']))}). "
        f"**b**, The same differences in the share of samples that finish within the cap: {signed(f_l8['diff'], 2)} "
        f"points at {k8} and {signed(f_l16['diff'], 2)} at {k16} against longest replay, {signed(f_b8['diff'], 2)} "
        f"and {signed(f_b16['diff'], 2)} against the untouched model; the untouched model finishes "
        f"{q8['arms']['base']['finish']:.0f}% of its samples within {k8} and {q16['arms']['base']['finish']:.0f}% "
        f"within {k16}. **c**, Tokens per sample, mean over seeds: shortest replay stays the shortest at both caps, "
        f"{tk[k8]['S']} at {k8} against {tk[k8]['L']} for longest replay ({pc_txt(tk[k8]['S'], tk[k8]['L'])}) and "
        f"{tk[k8]['base']} for the untouched model ({pc_txt(tk[k8]['S'], tk[k8]['base'])}), and {tk[k16]['S']} at "
        f"{k16} against {tk[k16]['L']} ({pc_txt(tk[k16]['S'], tk[k16]['L'])}) and {tk[k16]['base']} "
        f"({pc_txt(tk[k16]['S'], tk[k16]['base'])}): loosening the cap reduces the gain but not the brevity."
    )
    return f, cap


# ================================================================ fig1: design + ARC-AGI-2 per seed
def make_fig1():
    f = Fig("fig1", 7.2, 2.62)
    order = ["base", "L", "R", "S"]
    for a in order:
        assert abs(np.mean(ARC_A[a]["seeds"]) - ARC_A[a]["solved"]) < 0.05, f"ARC {a}: seed mean != solved"
    best = max(order, key=lambda a: ARC_A[a]["solved"])
    ok_b = best == "S" and all(ARC_A["S"]["solved"] > ARC_A[a]["solved"] for a in order if a != "S")
    if not ok_b:
        f.warn("panel b heading no longer holds (neutral heading used)")

    # ---- a: schematic (not data)
    f.heading(0.04, "a", "Three replay rules, one recipe",
              "for every training problem, keep one of the model's own verified solutions")
    AH = f.head_y - 0.42
    ax = f.ax(0, 0, 3.7, AH)
    ax.set_xlim(0, 3.7)
    ax.set_ylim(0, AH)
    ax.axis("off")

    def box(x, y, w, h, txt, ec, fc="white", lw=0.8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, txt, fontsize=FS, color=INK, ha="center", va="center", linespacing=1.2,
                fontweight="bold" if bold else "normal", gid="inmark")

    def arrow(p0, p1, color=INK2, lw=0.8):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15",
                                                          color=color, lw=lw, shrinkA=0, shrinkB=0))
        f.segments.append((ax, p0, p1))

    y_rows = [AH - 0.36, AH - 0.92, AH - 1.48]
    bx, bw, bh = 0.08, 1.08, 0.50
    box(bx, y_rows[0] - bh / 2, bw, bh, "the model solves\ntraining problems,\nseveral runs", AXIS, fc="#f6f6f4")
    box(bx, y_rows[1] - bh / 2 - 0.28, bw, bh, "verified solutions\nper problem,\nlengths differ", AXIS, fc="#f6f6f4")
    arrow((bx + bw / 2, y_rows[0] - bh / 2), (bx + bw / 2, y_rows[1] - 0.28 + bh / 2))
    y_src = y_rows[1] - 0.28
    kx, kw, kh = 1.50, 0.92, 0.34
    cx, cs = 2.80, 0.30
    for a, y in zip(("S", "R", "L"), y_rows):
        arrow((bx + bw, y_src), (kx - 0.02, y))
        box(kx, y - kh / 2, kw, kh, f"keep {NAME[a]}", C[a], lw=1.1 if a == "S" else 0.9, bold=a == "S")
        arrow((kx + kw + 0.02, y), (cx - 0.04, y), color=C[a], lw=1.1 if a == "S" else 0.9)
        ax.add_patch(FancyBboxPatch((cx, y - cs / 2), cs, cs, boxstyle="round,pad=0,rounding_size=0.04", fc=C[a],
                                    ec="none"))
        ax.text(cx + cs / 2, y - 0.004, a, fontsize=7.4, fontweight="bold", color="white", ha="center", va="center",
                gid="inmark")
    ax.text(cx + cs / 2, y_rows[0] + 0.30, "fine-tuned\ncopy", fontsize=FS_SMALL, color=INK2, ha="center",
            va="bottom", linespacing=1.15)
    ax.text(bx, 0.10, "same problems, same final answers, same recipe;\nonly the replayed reasoning differs",
            fontsize=FS_SMALL, color=INK2, style="italic", va="bottom", linespacing=1.25)

    # ---- b: puzzles solved per seed
    n_seeds = {len(ARC_A[a]["seeds"]) for a in order}
    f.heading(4.02, "b", "Shortest replay has the highest mean" if ok_b else "ARC-AGI-2 puzzles solved",
              f"ARC-AGI-2, Qwen3.8-27B program-writing agent, {ARC_N} puzzles, "
              + (f"{n_seeds.pop()} seeds per arm" if len(n_seeds) == 1 else "seeds per arm vary"))
    axb = f.ax(4.50, 0.40, 2.62, f.head_y - 0.62 - 0.40)
    allv = [v for a in order for v in ARC_A[a]["seeds"]]
    sb = ARC["contrasts"]["S-base"]["solved"]
    p_runs = perm_p(ARC_A["S"]["seeds"], ARC_A["base"]["seeds"])     # exact permutation test on the run totals
    y_br = max(allv) + 1.3
    axb.set_xlim(-0.55, len(order) - 0.45)
    axb.set_ylim(min(allv) - 2.0, y_br + 1.6)
    yt = nice_ticks(*axb.get_ylim(), 5)
    for v in yt:
        bg(axb.axhline(v, color=GRID, lw=0.5, zorder=0))
    for x, a in enumerate(order):
        v = ARC_A[a]["seeds"]
        xs = [x + (j - (len(v) - 1) / 2) * 0.075 for j in range(len(v))]
        axb.plot(xs, v, "o", ms=4.2, mfc=C[a], mec="white", mew=0.5, zorder=3, alpha=0.9)
        axb.plot([x - 0.27, x + 0.27], [ARC_A[a]["solved"]] * 2, color=C[a], lw=2.0, solid_capstyle="butt",
                 zorder=4)
    xs0, xs1 = order.index("base"), order.index("S")
    axb.plot([xs0, xs0, xs1, xs1], [y_br - 0.45, y_br, y_br, y_br - 0.45], color=INK2, lw=0.7, zorder=2)
    axb.text((xs0 + xs1) / 2, y_br + 0.35,
             f.num("b S - untouched (puzzles)", signed(sb["diff"])) + " puzzles, "
             + f.num("b S vs untouched, two-sided P over puzzles", fmt_p(two_sided(sb["p"]))) + " over puzzles, "
             + f.num("b S vs untouched, two-sided P over runs (permutation)", p_num(p_runs)) + " over runs",
             fontsize=FS, color=INK, ha="center", va="bottom")
    axb.set_xticks(range(len(order)))
    axb.set_xticklabels([NAME[a] for a in order])
    for lab, a in zip(axb.get_xticklabels(), order):
        if a == "S":
            lab.set_fontweight("bold")
            lab.set_color(INK)
    axb.tick_params(axis="x", length=0, pad=3)
    axb.spines["bottom"].set_visible(False)
    axb.set_yticks(yt)
    axb.set_yticklabels([f.num("b y tick", f"{v:g}") for v in yt])
    axb.set_ylabel(f"puzzles solved (of {f.num('b n puzzles', str(ARC_N))})", labelpad=3)

    rng = {a: (min(ARC_A[a]["seeds"]), max(ARC_A[a]["seeds"])) for a in order}
    sel = {o: ARC["contrasts"][f"S-{o}"]["solved"] for o in ("R", "L")}
    cap = (
        "**Design and per-seed result on ARC-AGI-2.** **a**, For each training problem one of the model's own "
        "verified solutions is kept by one of three rules (shortest, random, longest), and three copies of the model "
        "are fine-tuned with the same recipe; only the replayed reasoning differs. **b**, ARC-AGI-2 puzzles solved "
        f"(of {ARC_N}) by the Qwen3.8-27B program-writing agent, {seeds_phrase('arc')} (dots: seeds; bars: means, "
        + ", ".join(f"{ARC_A[a]['solved']:.1f} {NAME[a]}" for a in order)
        + f"). Shortest replay has the highest mean and solves {sb['diff']:.1f} more puzzles than the untouched model "
        f"(paired bootstrap over puzzles, 95% interval {sb['lo']:.1f} to {sb['hi']:.1f}, two-sided "
        f"{fmt_p(two_sided(sb['p']))}; exact permutation test on the run totals, P = {p_num(p_runs)}); its seeds "
        f"span {rng['S'][0]}–{rng['S'][1]} puzzles, those of the untouched model {rng['base'][0]}–{rng['base'][1]}. "
        + ("Its lead over random and longest replay is not resolved (" if not any(is_sig(c) for c in sel.values())
           else "Against random and longest replay: (")
        + ", ".join(f"{NAME[o]} replay {signed(c['diff'])} puzzles, {fmt_p(two_sided(c['p']))}" for o, c in sel.items())
        + ")."
    )
    return f, cap


# ================================================================ figS1 (was fig2a): difference from untouched, own units
def make_figS1():
    arms = ["S", "S2", "R", "L"]
    used = [a for a in arms if any(f"{a}-base" in STATS[k]["contrasts"] for k, *_ in SETTINGS)]
    f = Fig("figS1", 7.2, 2.02)
    HY = f.head_y
    TOP, BOT = HY - 0.42, 0.42
    LAB_W, GAP = 0.60, 0.20
    CW = (7.2 - 0.04 - LAB_W - GAP * (len(SETTINGS) - 1)) / len(SETTINGS)
    ys = {a: -i for i, a in enumerate(used)}
    ylo, yhi = -len(used) + 0.45, 0.55
    # the three accuracy-point settings share one scale; ARC keeps its native unit (puzzles of 72)
    pts = [c[e] for k, *_, m in SETTINGS if k != "arc" for a in used
           for c in [STATS[k]["contrasts"].get(f"{a}-base", {}).get(m)] if c for e in ("lo", "hi")]
    arcv = [ARC["contrasts"][f"{a}-base"]["solved"][e] for a in used if f"{a}-base" in ARC["contrasts"]
            for e in ("lo", "hi")]
    lim = {"pts": (min(pts + [0]) - 0.35, max(pts + [0]) + 0.35), "arc": (min(arcv + [0]) - 0.6, max(arcv + [0]) + 0.6)}
    axes = []
    for j, (key, bench, model, metric) in enumerate(SETTINGS):
        x_in = 0.04 + LAB_W + j * (CW + GAP)
        f.heading(x_in - 0.02, "abcd"[j], bench, f"{model}, {seeds_phrase(key)}")
        ax = f.ax(x_in, BOT, CW, TOP - BOT)
        x0, x1 = lim["arc" if key == "arc" else "pts"]
        ax.set_xlim(x0, x1)
        ax.set_ylim(ylo, yhi)
        step = 2.0 if key == "arc" else 1.0
        xt = nice_ticks(x0, x1, step)
        for v in xt:
            if abs(v) > 1e-9:
                bg(ax.axvline(v, color=GRID, lw=0.5, zorder=0))
        ax.axvline(0, color=C["base"], lw=0.9, zorder=1)
        for a in used:
            c = STATS[key]["contrasts"].get(f"{a}-base", {}).get(metric)
            if not c:
                continue
            col, emph = C[a], a == "S"
            ax.plot([c["lo"], c["hi"]], [ys[a]] * 2, color=col, lw=1.6 if emph else 1.2, solid_capstyle="round",
                    zorder=2)
            ax.plot([c["diff"]], [ys[a]], "o", ms=5.0 if emph else 4.4, mfc=col if is_sig(c) else "white", mec=col,
                    mew=1.0, zorder=3)
        ax.set_xticks(xt)
        ax.set_xticklabels([f.num(f"{key} x tick", tick_txt(v)) for v in xt])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(f"difference in puzzles solved (of {ARC_N})" if key == "arc" else "difference in accuracy "
                      "(points)", labelpad=2)
        axes.append(ax)
    for a in used:
        row_label(f, axes[0], 0.04 + LAB_W - 0.08, ys[a], NAME[a], ha="right",
                  fontweight="bold" if a == "S" else "normal")
    axes[0].annotate("untouched", xy=(0, yhi), xytext=(0, 1.5), textcoords="offset points", fontsize=FS_SMALL,
                     color=INK2, ha="center", va="bottom")
    f.key_sig(0.04 + LAB_W + 2 * (CW + GAP) + 0.02, 0.06)
    cap = (
        "**Difference from the untouched model for each replay rule, in each setting's own unit.** "
        f"**a**, ARC-AGI-2, Qwen3.8-27B program-writing agent, puzzles solved of {ARC_N} ({seeds_phrase('arc')}). "
        f"**b**, MATH-500, Qwen3-4B ({seeds_phrase('math_qwen4b')}). **c**, MATH-500, DeepSeek-R1-Distill-Llama-8B "
        f"({seeds_phrase('math_llama8b')}). **d**, APPS, Qwen3-4B ({seeds_phrase('apps_qwen4b')}). Accuracy in "
        "points (b–d, one scale); round 2 is a second round of shortest replay starting from the shortest-replay "
        "model. Dots: mean differences, filled when the paired-bootstrap 95% interval over puzzles or problems (bar) "
        "excludes 0. The same contrasts appear in percentage points in Fig. 1b."
    )
    return f, cap


# ================================================================ figS2 (was fig2b): token cost per arm
def make_figS2():
    arms = ["base", "S", "S2", "R", "L"]
    used = [a for a in arms if any(a in STATS[k]["arms"] for k, *_ in SETTINGS)]
    f = Fig("figS2", 7.2, 2.22)
    HY = f.head_y
    TOP, BOT = HY - 0.42, 0.40
    LAB_W, GAP = 0.60, 0.20
    CW = (7.2 - 0.04 - LAB_W - GAP * (len(SETTINGS) - 1)) / len(SETTINGS)
    ys = {a: -i for i, a in enumerate(used)}
    ylo, yhi = -len(used) + 0.5, 0.5

    def tokens(key, a):
        arm = STATS[key]["arms"].get(a)
        if not arm:
            return None
        return arm["tokens_per_solved_k"] if key == "arc" else arm["tokens"]

    per_sample_max = max(tokens(k, a) for k, *_ in SETTINGS if k != "arc" for a in used if tokens(k, a))
    claims = []
    axes = []
    for j, (key, bench, model, _) in enumerate(SETTINGS):
        x_in = 0.04 + LAB_W + j * (CW + GAP)
        f.heading(x_in - 0.02, "abcd"[j], bench, model + (", per solved puzzle" if key == "arc" else ", per sample"))
        ax = f.ax(x_in, BOT, CW, TOP - BOT)
        vmax = max(tokens(key, a) for a in used if tokens(key, a)) if key == "arc" else per_sample_max
        ax.set_xlim(0, vmax * 1.42)
        ax.set_ylim(ylo, yhi)
        step = 200 if key == "arc" else 2000
        xt = nice_ticks(0, vmax * 1.42, step)
        for v in xt:
            if v > 0:
                bg(ax.axvline(v, color=GRID, lw=0.5, zorder=0))
        ref = tokens(key, "base")
        px, _ = f.per_unit(ax)
        for a in used:
            t = tokens(key, a)
            if t is None:
                continue
            f.bar(ax, ys[a], 0, t, 0.56, C[a])
            lab = (f"{t:.1f}k" if key == "arc" else f"{t}") if a == "base" else signed(pct_change(t, ref), 1) + "%"
            ax.text(t + 0.04 / px, ys[a], f.num(f"{key} {a} tokens label", lab), fontsize=FS_SMALL, va="center",
                    ha="left", color=INK2 if a == "base" else INK, fontweight="bold" if a == "S" else "normal")
        ax.set_xticks(xt)
        ax.set_xticklabels([f.num(f"{key} x tick", "0" if v == 0 else (f"{v:g}" if key == "arc" else f"{v / 1000:g}k"))
                            for v in xt])
        ax.set_yticks([])
        ax.spines["left"].set_color(AXIS)
        ax.set_xlabel("thousand tokens per solved puzzle" if key == "arc" else "tokens per sample", labelpad=2)
        rules = {a: tokens(key, a) for a in ("S", "R", "L") if tokens(key, a)}
        claims.append((key, min(rules, key=rules.get) == "S"))
        axes.append(ax)
    for a in used:
        row_label(f, axes[0], 0.04 + LAB_W - 0.08, ys[a], NAME[a], ha="right",
                  fontweight="bold" if a == "S" else "normal")
    for key, ok in claims:
        if not ok:
            f.warn(f"caption claim fails: shortest is not the cheapest replay rule on {key}")
    pc = {k: {a: pct_change(tokens(k, a), tokens(k, "base")) for a in used if tokens(k, a)} for k, *_ in SETTINGS}
    cap = (
        "**Token cost of every arm.** Tokens per solved puzzle for the ARC-AGI-2 agent (thousands; all tokens of all "
        "seeds over all solves) and generated tokens per sample on MATH-500 and APPS (mean over seeds; b–d share one "
        "scale); labels give the change from the untouched model, whose own value is printed. Shortest replay is the "
        "cheapest of the three replay rules in every setting and changes the cost relative to the untouched model by "
        f"{signed(pc['arc']['S'], 1)}% (ARC-AGI-2), {signed(pc['math_qwen4b']['S'], 1)}% (MATH-500, Qwen3-4B"
        + (f"; {signed(pc['math_qwen4b']['S2'], 1)}% after round 2" if "S2" in pc["math_qwen4b"] else "")
        + f"), {signed(pc['math_llama8b']['S'], 1)}% (MATH-500, R1-Llama-8B) and "
        f"{signed(pc['apps_qwen4b']['S'], 1)}% (APPS)."
    )
    return f, cap


# ================================================================ fig3: round 2, ARC agent messages, 4B on ARC-1
def make_fig3():
    f = Fig("fig3", 7.2, 2.62)
    HY = f.head_y
    m = STATS["math_qwen4b"]
    s2s = m["contrasts"]["S2-S"]["acc"]
    ok_a = m["arms"]["S2"]["acc"] > m["arms"]["S"]["acc"] > m["arms"]["base"]["acc"] and s2s["lo"] > 0
    ok_b = (ARC_A["S"]["cut_msgs"] < min(ARC_A[a]["cut_msgs"] for a in ("R", "L", "base"))
            and ARC_A["S"]["code_msgs"] > max(ARC_A[a]["code_msgs"] for a in ("R", "L", "base")))
    s4 = ARC1_4B
    ok_c = (s4["cut_msgs"]["S"] < min(s4["cut_msgs"][a] for a in ("R", "L"))
            and s4["code_msgs"]["S"] > max(s4["code_msgs"][a] for a in ("R", "L"))
            and max(s4["solved"].values()) - min(s4["solved"].values()) <= 1)
    for ok, p in ((ok_a, "a"), (ok_b, "b"), (ok_c, "c")):
        if not ok:
            f.warn(f"panel {p} heading no longer holds (neutral heading used)")

    # ---- a: rounds compound (MATH-500, Qwen3-4B)
    f.heading(0.04, "a", "Rounds compound" if ok_a else "Rounds of shortest replay",
              "MATH-500, Qwen3-4B; dots: seeds, bars: means")
    arms_a = ["base", "S", "S2"]
    lab_a = {"base": "untouched", "S": "round 1", "S2": "round 2"}
    AX0 = 0.66
    axa = f.ax(AX0, 0.62, 1.50, HY - 0.48 - 0.62)
    allv = [v for a in arms_a for v in m["arms"][a]["seeds"]]
    y_br = max(allv) + 0.35
    axa.set_xlim(-0.5, len(arms_a) - 0.5)
    axa.set_ylim(min(allv) - 0.4, y_br + 0.45)
    yt = nice_ticks(*axa.get_ylim(), 1)
    for v in yt:
        bg(axa.axhline(v, color=GRID, lw=0.5, zorder=0))
    for x, a in enumerate(arms_a):
        v = m["arms"][a]["seeds"]
        assert abs(np.mean(v) - m["arms"][a]["acc"]) < 0.01, f"MATH {a}: seed mean != acc"
        xs = [x + (j - (len(v) - 1) / 2) * 0.075 for j in range(len(v))]
        axa.plot(xs, v, "o", ms=4.2, mfc=C[a], mec="white", mew=0.5, zorder=3, alpha=0.9)
        axa.plot([x - 0.27, x + 0.27], [m["arms"][a]["acc"]] * 2, color=C[a], lw=2.0, solid_capstyle="butt", zorder=4)
    axa.plot([1, 1, 2, 2], [y_br - 0.12, y_br, y_br, y_br - 0.12], color=INK2, lw=0.7, zorder=2)
    axa.text(1.5, y_br + 0.08, f.num("a S2 - S (points)", signed(s2s["diff"], 2)) + ", "
             + f.num("a S2 vs S, two-sided P", fmt_p(two_sided(s2s["p"]))), fontsize=FS_SMALL, color=INK,
             ha="center", va="bottom")
    axa.set_xticks(range(len(arms_a)))
    axa.set_xticklabels([lab_a[a] for a in arms_a])
    axa.tick_params(axis="x", length=0, pad=3)
    axa.spines["bottom"].set_visible(False)
    axa.set_yticks(yt)
    axa.set_yticklabels([f.num("a y tick", f"{v:g}") for v in yt])
    axa.set_ylabel("accuracy (%)", labelpad=3)
    pxa, _ = f.per_unit(axa)
    for x, a in enumerate(arms_a):
        f.text(AX0 + (x - axa.get_xlim()[0]) * pxa, 0.30, f.num(f"a tokens {a}", f"{m['arms'][a]['tokens']}"),
               ha="center", va="baseline", color=INK)
    f.text(AX0 - 0.08, 0.30, "tokens per\nsample", ha="right", va="baseline", linespacing=1.15)

    # ---- b, c: message statistics as a table of bars sharing the rows
    rows = ["S", "R", "L", "base"]
    ys = {a: -i for i, a in enumerate(rows)}
    TOP, BOT = HY - 0.66, 0.34
    LAB_X = 2.84                                                    # right edge of the row labels
    BW_, SW_, GAPC = 0.66, 0.36, 0.08                               # bar column, solved column, gap (in)

    def group(x0, title_letter, title, sub, data, n_solved, solved_fmt):
        f.heading(x0 - 0.04, title_letter, title, sub)
        axs = []
        for j, (field, ctitle) in enumerate((("cut_msgs", "cut at\nthe cap (%)"), ("code_msgs", "contain\ncode (%)"))):
            ax = f.ax(x0 + j * (BW_ + GAPC), BOT, BW_, TOP - BOT)
            ax.set_xlim(0, 100)
            ax.set_ylim(-len(rows) + 0.5, 0.5)
            ax.axis("off")
            ax.plot([0, 0], [-len(rows) + 0.55, 0.45], color=AXIS, lw=0.6, zorder=1, solid_capstyle="butt")
            px, _ = f.per_unit(ax)
            for a in rows:
                v = data[field].get(a)
                if v is None:
                    continue
                f.bar(ax, ys[a], 0, v, 0.52, C[a])
                ax.text(v + 0.04 / px, ys[a], f.num(f"{title_letter} {field} {a}", f"{v:.1f}" if isinstance(v, float)
                        else f"{v}"), fontsize=FS_SMALL, ha="left", va="center",
                        color=INK, fontweight="bold" if a == "S" else "normal")
            ax.text(0.0, 1.0 + 0.05 / (TOP - BOT), ctitle, transform=ax.transAxes, fontsize=FS_SMALL, color=INK2,
                    ha="left", va="bottom", linespacing=1.15)
            axs.append(ax)
        xs_ = x0 + 2 * (BW_ + GAPC)
        ax = f.ax(xs_, BOT, SW_, TOP - BOT)
        ax.set_xlim(0, 1)
        ax.set_ylim(-len(rows) + 0.5, 0.5)
        ax.axis("off")
        for a in rows:
            v = data["solved"].get(a)
            if v is not None:
                ax.text(0.5, ys[a], f.num(f"{title_letter} solved {a}", solved_fmt(v)), fontsize=FS_SMALL,
                        ha="center", va="center", color=INK, fontweight="bold" if a == "S" else "normal")
        ax.text(0.5, 1.0 + 0.05 / (TOP - BOT), f"solved\n(of {n_solved})", transform=ax.transAxes,
                fontsize=FS_SMALL, color=INK2, ha="center", va="bottom", linespacing=1.15)
        return axs

    arc_data = {"cut_msgs": {a: ARC_A[a]["cut_msgs"] for a in rows}, "code_msgs": {a: ARC_A[a]["code_msgs"] for a in rows},
                "solved": {a: ARC_A[a]["solved"] for a in rows}}
    axs_b = group(LAB_X + 0.10, "b", "ARC agent: fewer cuts, more code" if ok_b else "ARC agent messages",
                  f"Qwen3.8-27B on ARC-AGI-2,\n{f.num('b ARC cap', f'{ARC_CAP / 1000:g}k')}-token cap per message",
                  arc_data, ARC_N, lambda v: f"{v:.1f}")
    small = {"cut_msgs": s4["cut_msgs"], "code_msgs": s4["code_msgs"], "solved": s4["solved"]}
    group(LAB_X + 0.10 + 2 * (BW_ + GAPC) + SW_ + 0.30, "c",
          "4B on ARC-1: same shift, no gain" if ok_c else "4B on ARC-1",
          "Qwen3-4B on ARC-1,\n16k-token cap per message", small, s4["n"], lambda v: f"{v}")
    for a in rows:
        row_label(f, axs_b[0], LAB_X, ys[a], NAME[a], ha="right", fontweight="bold" if a == "S" else "normal")

    base_cov = f"{ARC_LOGGED['base']} of its {len(ARC_A['base']['seeds'])} seeds" \
        if ARC_LOGGED["base"] < len(ARC_A["base"]["seeds"]) else "all seeds"
    cap = (
        "**Compounding and behaviour.** **a**, MATH-500 accuracy of Qwen3-4B untouched, after one round and after "
        f"two rounds of shortest replay (dots: seeds; bars: means; {seeds_phrase('math_qwen4b')}) with tokens per "
        f"sample below. Round 2 adds {signed(s2s['diff'], 2)} points to round 1 (95% interval {s2s['lo']:.2f} to "
        f"{s2s['hi']:.2f}, two-sided {fmt_p(two_sided(s2s['p']))}) and uses {m['arms']['S2']['tokens']} tokens per "
        f"sample against {m['arms']['base']['tokens']} for the untouched model. **b**, ARC-AGI-2 agent "
        f"(Qwen3.8-27B): share of its messages cut at the {ARC_CAP / 1000:g}k-token cap and share containing code to "
        "execute, from the per-message logs (all seeds of the three replay arms; the untouched agent's logs cover "
        f"{base_cov}), and puzzles solved (of {ARC_N}, mean over seeds). **c**, Qwen3-4B on ARC-1 ({s4['n']} "
        "puzzles, 16k-token cap): the same shift in behaviour with no change in puzzles solved (no message "
        "statistics are reported for the untouched 4B model)."
    )
    return f, cap


# ================================================================ fig4: the direct-answering regime (NOT IN THE JSON)
def make_fig4():
    f = Fig("fig4", 7.2, 2.5)
    HY = f.head_y
    d = DIRECT
    plain = [s for b, s in d["budget"] if b == d["plain_budget"]][0]
    plain_mean = float(np.mean(plain))
    ok_b = all(v <= plain_mean for _, v in d["interventions"])
    if not ok_b:
        f.warn("panel b heading no longer holds (neutral heading used)")

    f.heading(0.04, "a", "Direct answering needs long reasoning",
              f"Qwen3.8-27B, no tools, {d['n']} development puzzles")
    axa = f.ax(0.46, 0.46, 2.62, HY - 0.48 - 0.46)
    xs = [b for b, _ in d["budget"]]
    means = [float(np.mean(v)) for _, v in d["budget"]]
    axa.set_xlim(xs[0] - 8, xs[-1] + 8)
    axa.set_ylim(0, 20)
    for v in (5, 10, 15, 20):
        bg(axa.axhline(v, color=GRID, lw=0.5, zorder=0))
    axa.plot(xs, means, color=C["base"], lw=1.4, zorder=2, solid_capstyle="round")
    axa.plot(xs, means, "o", ms=5.0, mfc=C["base"], mec="white", mew=0.8, zorder=4)
    for (b, v), mu in zip(d["budget"], means):   # a budget with several seeds: mean, then the seeds
        lab = f"{mu:g}" + (f" (seeds {' and '.join(str(s) for s in v)})" if len(v) > 1 else "")
        axa.annotate(f.num(f"a solved at {b}k", lab), xy=(b, mu), xytext=(0, 7), textcoords="offset points",
                     ha="center", va="bottom", fontsize=FS_SMALL, color=INK)
    axa.set_xticks(xs)
    axa.set_xticklabels([f"{b}k" for b in xs])
    axa.set_yticks([0, 5, 10, 15, 20])
    axa.set_xlabel("reasoning budget per answer (tokens)", labelpad=2)
    axa.set_ylabel(f"puzzles solved (of {d['n']})", labelpad=3)

    f.heading(3.42, "b", "Every intervention is at or below plain" if ok_b else "Interventions at a fixed budget",
              f"{d['plain_budget']}k-token budget per answer; one run per intervention")
    items = [(f"plain ({len(plain)} seeds)", plain_mean)] + d["interventions"]
    BX = 4.98                                                       # bars start; values sit just left of them
    axb = f.ax(BX, 0.46, 1.98, HY - 0.48 - 0.46)
    axb.set_xlim(0, 20)
    axb.set_ylim(-len(items) + 0.4, 0.6)
    for v in (5, 10, 15, 20):
        bg(axb.axvline(v, color=GRID, lw=0.5, zorder=0))
    for i, (name, v) in enumerate(items):
        col = C["base"] if i == 0 else NEUTRAL
        f.bar(axb, -i, 0, v, 0.58, col)
        row_label(f, axb, BX - 0.06, -i, f.num(f"b {name}", f"{v:g}"), ha="right", fontsize=FS_SMALL,
                  fontweight="bold" if i == 0 else "normal")
        row_label(f, axb, BX - 0.34, -i, name, ha="right", fontsize=FS_SMALL, color=INK2 if i else INK,
                  fontweight="bold" if i == 0 else "normal")
    axb.plot([plain_mean, plain_mean], [-len(items) + 0.55, 0.45], color=C["base"], lw=0.7, zorder=3)
    axb.set_yticks([])
    axb.spines["left"].set_color(AXIS)
    axb.set_xticks([0, 5, 10, 15, 20])
    axb.set_xlabel(f"puzzles solved (of {d['n']})", labelpad=2)
    cap = (
        "**Regime dependence.** The same Qwen3.8-27B model answering ARC puzzles directly, without tools (hidden "
        f"reasoning, then the output grid; two samples per test input; {d['n']} development puzzles). **a**, Puzzles "
        "solved against the reasoning budget per answer (mean of two seeds at "
        f"{d['plain_budget']}k). **b**, At a {d['plain_budget']}k-token budget, every intervention that "
        "shortens or redirects the reasoning (one run each, no intervals) solves as many puzzles as the plain model "
        f"({plain_mean:g}, mean of {len(plain)} seeds) or fewer."
    )
    return f, cap


# ================================================================ main
def main():
    os.makedirs(OUT, exist_ok=True)
    print("ARC message logs, seeds covered:", ARC_LOGGED)
    report = []
    for make in (make_restart, make_fig1, make_fig3, make_fig4, make_figS1, make_figS2):
        f, cap = make()
        issues = f.save(cap)
        report.append((f, issues))
    for f, issues in report:
        print(f"\n=== {f.name}  ({f.W:g} x {f.H:g} in)")
        print("layout:", "clean (no overlaps, no text on marks, nothing clipped, nothing below 6 pt)" if not issues
              else f"{len(issues)} issue(s)")
        for s in issues:
            print("   ", s)
        pdf = os.path.join(OUT, f.name + ".pdf")
        if shutil.which("pdffonts"):
            rep = subprocess.run(["pdffonts", pdf], capture_output=True, text=True).stdout.strip().splitlines()[2:]
            print("fonts:", ", ".join(sorted({ln.split()[0] for ln in rep})))
        else:
            fonts = sorted(set(re.findall(rb"/BaseFont\s*/([^\s/\]>]+)", open(pdf, "rb").read())))
            print("fonts:", [x.decode() for x in fonts])
        print("numbers printed on the figure:")
        for what, s in f.shown:
            if "tick" not in what:
                print(f"    {s:>24}  {what}")
        print("claims:", "all hold" if not f.warnings else f"{len(f.warnings)} WARNING(S)")
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
