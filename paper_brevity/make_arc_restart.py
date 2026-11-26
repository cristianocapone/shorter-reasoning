"""ARC-AGI-2 restart cost against the per-message cap, by post-hoc truncation of the agent's message logs.

    env -u PYTHONPATH .venv_mps/bin/python paper_brevity/make_arc_restart.py
    -> paper_brevity/figures/fig_arc_restart.pdf   vector, Arial embedded as TrueType (pdf.fonttype 42)
       paper_brevity/figures/fig_arc_restart.png   300 dpi

Data: paper_brevity/data/arc_msgs.json only (per arm S, R, L, base: one row per agent message with fields tokens,
thinking_chars, visible_chars, cut, has_code; cap 24000 generated tokens per message). The untouched arm's logs cover
3 of its 5 runs (make_figs.arc_log_coverage()).

For a cap t <= 24000, cutting a message at t leaves its first t tokens unchanged, so with L the message's length:
    C(t) = E[min(L, t)] / P(the message carries a program and L <= t)
is the expected number of generated tokens per code-carrying message under a cap t (Luby et al.'s fixed-cutoff cost,
main.tex Eq. 1, with 'finished' read as 'carries a program'). No message cut at 24k carries a program (checked below),
so at t = 24000 the formula is the 'tokens/code msg' column of Table 1 (paper_stats arc.arms.*.restart_cost_k);
the script asserts this. Curves are evaluated exactly at every integer t from 4000 to 24000.
Caveats (stated in the caption): the logs are censored at 24k; messages of one thread are not independent (after a cut
the agent is told to think less, and short follow-up messages depend on the long ones before them), so the curve for
t < 24k is a counterfactual that ignores this feedback; a cost is not an accuracy.

Style, palette, layout checks and number formatting are those of make_figs.py (imported; its main() is not run).
"""
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import make_figs as mf  # noqa: E402  (sets the rcParams: Arial, pdf.fonttype 42)

T_LO, T_HI = 4000, 24000
ARMS = ["S", "R", "L", "base"]
LS = {"S": "-", "R": "-", "L": "-", "base": (0, (3.2, 1.6))}
LW = {"S": 1.3, "R": 1.0, "L": 1.0, "base": 1.0}


def curves():
    cap = mf.MSGS["cap"]
    assert cap == T_HI, f"message cap {cap} != {T_HI}"
    t = np.arange(T_LO, T_HI + 1)
    out = {}
    for a in ARMS:
        X = np.asarray(mf.MSGS["arms"][a], dtype=float)
        L, code = X[:, 0], X[:, 4].astype(bool)
        assert L.max() <= cap
        assert not (code & (L >= cap)).any(), f"{a}: a message at the cap carries a program"
        Ls = np.sort(L)
        csum = np.concatenate([[0.0], np.cumsum(Ls)])
        k = np.searchsorted(Ls, t, side="right")                   # messages with L <= t
        e_min = (csum[k] + t * (len(Ls) - k)) / len(Ls)             # E[min(L, t)]
        Lc = np.sort(L[code])
        p_code = np.searchsorted(Lc, t, side="right") / len(Ls)      # P(code and L <= t)
        out[a] = dict(t=t, e_min=e_min, p_code=p_code, C=e_min / p_code, n=len(Ls))
        # t = 24k reproduces Table 1 (paper_stats arc.arms.*.restart_cost_k and code_msgs)
        assert abs(out[a]["C"][-1] / 1000 - mf.ARC_A[a]["restart_cost_k"]) < 0.05, f"{a}: C(24k) != Table 1"
        assert abs(100 * p_code[-1] - mf.ARC_A[a]["code_msgs"]) < 0.05, f"{a}: P(code) != Table 1"
    return out


def repel(ys, gap):
    """Vertical positions for end labels: original order kept, at least `gap` apart, centred on the originals."""
    order = np.argsort(ys)
    pos = np.asarray(ys, float)[order].copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    pos += (np.asarray(ys, float)[order].mean() - pos.mean())
    for i in range(1, len(pos)):                                    # re-enforce after centring
        pos[i] = max(pos[i], pos[i - 1] + gap)
    res = np.empty_like(pos)
    res[order] = pos
    return res


def main():
    cv = curves()
    t = cv["S"]["t"]
    logged = mf.ARC_LOGGED                                          # runs covered by the logs, per arm
    n_runs = {a: len(mf.ARC_A[a]["seeds"]) for a in ARMS}

    # ---- claims (warn if one stops holding)
    others = np.minimum.reduce([cv[a]["C"] for a in ("R", "L", "base")])
    s_lowest = bool((cv["S"]["C"] < others).all())
    falling = {a: bool(cv[a]["C"][t == 23000][0] > cv[a]["C"][-1]) for a in ARMS}
    p_others = np.maximum.reduce([cv[a]["p_code"] for a in ("R", "L", "base")])
    s_most = bool((cv["S"]["p_code"] > p_others).all())             # panel a heading: more programs at every t
    peak = {a: int(t[cv[a]["C"].argmax()]) for a in ARMS}
    e_spread = max((max(cv[a]["e_min"][i] for a in ARMS) / min(cv[a]["e_min"][i] for a in ARMS) - 1)
                   for i in range(len(t)))
    print("C(t) in thousand tokens per code-carrying message (Table 1 at 24k):")
    for a in ARMS:
        c = cv[a]["C"] / 1000
        pts = "  ".join(f"{tt // 1000}k {c[t == tt][0]:.1f}" for tt in (4000, 8000, 12000, 16000, 20000, 24000))
        print(f"   {mf.NAME[a]:9s} {pts} | peak at {peak[a]} ({c.max():.1f}k) | "
              f"P(program within 8k/16k/24k) {100 * cv[a]['p_code'][t == 8000][0]:.1f}/"
              f"{100 * cv[a]['p_code'][t == 16000][0]:.1f}/{100 * cv[a]['p_code'][-1]:.1f}% | "
              f"messages {cv[a]['n']}, runs logged {logged[a]} of {n_runs[a]}")
    gap = (others - cv["S"]["C"]) / 1000
    print(f"shortest lowest at every t in [{T_LO}, {T_HI}]: {s_lowest} (smallest margin {gap.min():.1f}k at "
          f"t = {t[gap.argmin()]})")
    print("C(23k) > C(24k), i.e. still falling at the cap:", falling)
    print(f"shortest has the largest share of messages with a program at every t in [{T_LO}, {T_HI}]: {s_most}")
    print(f"largest spread of E[min(L, t)] between arms over t: {100 * e_spread:.1f}%")
    warnings = []
    if not s_lowest:
        warnings.append("shortest replay is not the lowest at every t")
    if not s_most:
        warnings.append("shortest replay does not have the most programs at every t (panel a heading)")
    if not all(falling.values()):
        warnings.append("not every curve is still falling at 24k")

    # ---- figure
    f = mf.Fig("fig_arc_restart", 7.2, 1.92)
    HY = f.head_y
    BOT, TOP = 0.38, HY - 0.42
    XA, WA = 0.50, 2.35
    XB, WB = 4.12, 2.18
    f.heading(0.04, "a", "Shortest replay delivers programs earlier",
              "share of messages that end with a program within t tokens;\ndashed: untouched agent "
              f"(logs of {logged['base']} of its {n_runs['base']} runs)")
    f.heading(XB - 0.46, "b", "Restart cost under a per-message cap t",
              "E[min(L, t)] / P(program within t): expected tokens\nper code-carrying message")

    def axes(x, w, xlo, ylo, yhi, ystep, yfmt):
        ax = f.ax(x, BOT, w, TOP - BOT)
        ax.set_xlim(xlo, T_HI)
        ax.set_ylim(ylo, yhi)
        yt = mf.nice_ticks(ylo, yhi, ystep)
        for v in yt:
            mf.bg(ax.axhline(v, color=mf.GRID, lw=0.5, zorder=0))
        ax.set_yticks(yt)
        ax.set_yticklabels([f.num("y tick", yfmt(v)) for v in yt])
        xt = np.arange(np.ceil(xlo / 4000) * 4000, T_HI + 1, 4000)
        ax.set_xticks(xt)
        ax.set_xticklabels([f.num("x tick", "0" if v == 0 else f"{v / 1000:g}k") for v in xt])
        ax.set_xlabel("cap t (generated tokens per message)", labelpad=2)
        return ax

    def end_labels(ax, vals, fmt, gap_in):
        _, py = f.per_unit(ax)
        ys = repel([vals[a] for a in ARMS], gap_in / py)
        for a, y in zip(ARMS, ys):
            txt = f.num(f"{ax.get_label()} {a} at 24k", fmt(vals[a]))
            ax.annotate(f"{mf.NAME[a]} {txt}", xy=(T_HI, vals[a]), xytext=(T_HI + 0.045 * (T_HI - ax.get_xlim()[0]), y),
                        textcoords="data", annotation_clip=False, ha="left", va="center", fontsize=mf.FS,
                        color=mf.INK, fontweight="bold" if a == "S" else "normal",
                        arrowprops=dict(arrowstyle="-", color=mf.AXIS, lw=0.5, shrinkA=0, shrinkB=1.5))

    # a: P(program within t), from 0
    axa = axes(XA, WA, 0, 0, 0.42, 0.1, lambda v: f"{100 * v:.0f}%")
    axa.set_label("a")
    X0 = {a: np.asarray(mf.MSGS["arms"][a], dtype=float) for a in ARMS}
    tf = np.arange(0, T_HI + 1, 20)
    for a in ARMS:
        L, code = X0[a][:, 0], X0[a][:, 4].astype(bool)
        Lc = np.sort(L[code])
        p = np.searchsorted(Lc, tf, side="right") / len(L)
        axa.plot(tf, p, color=mf.C[a], lw=LW[a], ls=LS[a], zorder=3 if a == "S" else 2, solid_capstyle="butt")
    end_labels(axa, {a: cv[a]["p_code"][-1] for a in ARMS}, lambda v: f"{100 * v:.1f}%", 0.095)
    axa.set_ylabel("messages with a program", labelpad=3)

    # b: C(t) from 4k
    yhi = np.ceil(max(cv[a]["C"].max() for a in ARMS) / 10000) * 10000 + 2000
    axb = axes(XB, WB, T_LO, 40000, yhi, 20000, lambda v: f"{v / 1000:.0f}k")
    axb.set_label("b")
    for a in ARMS:
        axb.plot(t, cv[a]["C"], color=mf.C[a], lw=LW[a], ls=LS[a], zorder=3 if a == "S" else 2, solid_capstyle="butt")
        axb.plot([T_HI], [cv[a]["C"][-1]], "o", ms=3.6, mfc=mf.C[a], mec="white", mew=0.6, zorder=4)
    end_labels(axb, {a: cv[a]["C"][-1] for a in ARMS}, lambda v: f"{v / 1000:.1f}k", 0.095)
    axb.set_ylabel("tokens per program", labelpad=3)

    issues = f.check()
    base = os.path.join(mf.OUT, f.name)
    f.fig.savefig(base + ".pdf")
    f.fig.savefig(base + ".png", dpi=300)
    mf.plt.close(f.fig)
    print(f"\n=== {f.name} ({f.W:g} x {f.H:g} in) -> {base}.pdf/.png")
    print("layout:", "clean (no overlaps, no text on marks, nothing clipped, nothing below 6 pt)" if not issues
          else f"{len(issues)} issue(s)")
    for s in issues:
        print("   ", s)
    if shutil.which("pdffonts"):
        rep = subprocess.run(["pdffonts", base + ".pdf"], capture_output=True, text=True).stdout.strip().splitlines()[2:]
        print("fonts:", ", ".join(sorted({ln.split()[0] for ln in rep})))
    print("numbers printed on the figure:")
    for what, s in f.shown:
        if "tick" not in what:
            print(f"    {s:>10}  {what}")
    print("claims:", "all hold" if not warnings else "; ".join(warnings))


if __name__ == "__main__":
    main()
