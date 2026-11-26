"""Secondary statistics quoted in natcomm/main.tex that are derived from, but not stored in, the two data files.

    env -u PYTHONPATH .venv_mps/bin/python paper_brevity/derived_stats.py

Reads only paper_brevity/data/paper_stats.json and paper_brevity/data/arc_msgs.json and prints:
  1. the eight selection contrasts (shortest minus random / longest replay in each setting) with two-sided P,
     their Holm adjustment, and a test pooled over the four settings (Stouffer's z from the one-sided bootstrap
     tails P(diff <= 0); a stored tail of 0 is floored at 1/10,000, the resolution of the bootstrap, which can only
     make the pooled P larger);
  2. ARC-AGI-2 seed-level checks: seed totals per arm, their variance, and exact permutation tests on the five seed
     totals per arm (all 252 splits), which treat seeds rather than puzzles as the unit;
  3. ARC-AGI-2 tokens per puzzle (tokens per solved puzzle x mean puzzles solved / 72);
  4. ARC message logs: unrounded cut shares and their differences, the share of tokens spent in cut messages, the
     restart cost at hypothetical caps (sum of min(tokens, t) over messages / messages with code within t tokens), and
     the share of messages that had delivered a program within t tokens.
P values follow the paper's rule: two-sided, 2 min(p, 1 - p); below 0.001 reported as P < 0.001.
"""
import itertools
import json
import os
from statistics import NormalDist

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(ROOT, "data", "paper_stats.json")))
M = json.load(open(os.path.join(ROOT, "data", "arc_msgs.json")))
N_BOOT = 10000
ND = NormalDist()
SETTINGS = [("arc", "solved"), ("math_qwen4b", "acc"), ("math_llama8b", "acc"), ("apps_qwen4b", "acc")]


def two_sided(p):
    return 2 * min(p, 1 - p)


def fmt(p):
    """The paper's rule: two significant digits, at most three decimal places, '<0.001' below 0.001."""
    if p < 0.001:
        return "<0.001"
    x = round(float(f"{p:.2g}"), 3)
    return f"{x:.{min(3, max(1, 1 - int(np.floor(np.log10(x)))))}f}"


print("1. selection contrasts (diff [95% CI], two-sided P, Holm-adjusted P over the eight)")
rows = []
for key, m in SETTINGS:
    for o in "RL":
        c = S[key]["contrasts"][f"S-{o}"][m]
        rows.append(dict(key=key, o=o, c=c, P=two_sided(c["p"])))
order = sorted(range(len(rows)), key=lambda i: max(rows[i]["P"], 2 / N_BOOT))
run, alive = 0.0, True
for rank, i in enumerate(order):
    P = max(rows[i]["P"], 2 / N_BOOT)
    run = max(run, min(1.0, (len(rows) - rank) * P))
    alive = alive and P <= 0.05 / (len(rows) - rank)
    rows[i]["holm"], rows[i]["holm_ok"] = run, alive
for r in rows:
    c = r["c"]
    print(f"   {r['key']:13s} S-{r['o']}  {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]  P {fmt(r['P']):>6}  "
          f"Holm {fmt(r['holm']):>6}  {'significant after Holm' if r['holm_ok'] else ''}")
for o in "RL":
    tails = [max(S[k]["contrasts"][f"S-{o}"][m]["p"], 1 / N_BOOT) for k, m in SETTINGS]
    z = sum(ND.inv_cdf(1 - p) for p in tails) / len(tails) ** 0.5
    print(f"   pooled S-{o} (Stouffer, 4 settings): z = {z:.2f}, two-sided P = {two_sided(1 - ND.cdf(z)):.2e} "
          f"({fmt(two_sided(1 - ND.cdf(z)))})")

print("2. ARC-AGI-2 seed totals and seed-level permutation tests")
A = S["arc"]["arms"]
for a in ("S", "R", "L", "base"):
    x = np.array(A[a]["seeds"], float)
    print(f"   {a:4s} seeds {A[a]['seeds']}  mean {x.mean():.1f}  range {x.min():.0f}-{x.max():.0f}  "
          f"variance {x.var(ddof=1):.1f}")


def perm(a, b):
    pool, n = list(a) + list(b), len(a)
    obs = np.mean(a) - np.mean(b)
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(len(pool)) if i not in idx]
        hits += abs(np.mean(x) - np.mean(y)) >= abs(obs) - 1e-9
        tot += 1
    return obs, hits, tot


for a, b in (("S", "base"), ("R", "base"), ("L", "base"), ("S", "R"), ("S", "L")):
    obs, hits, tot = perm(A[a]["seeds"], A[b]["seeds"])
    print(f"   {a}-{b}: {obs:+.1f} puzzles, exact two-sided P = {hits}/{tot} = {hits / tot:.3f} ({fmt(hits / tot)})")

print("3. ARC-AGI-2 tokens per puzzle (k) = tokens per solved puzzle x puzzles solved / 72")
tp = {a: A[a]["tokens_per_solved_k"] * A[a]["solved"] / S["arc"]["n_tasks"] for a in A}
for a in ("S", "R", "L", "base"):
    print(f"   {a:4s} {tp[a]:.1f}k per puzzle ({100 * (tp[a] / tp['base'] - 1):+.1f}% vs untouched), "
          f"{A[a]['tokens_per_solved_k']:.1f}k per solved puzzle "
          f"({100 * (A[a]['tokens_per_solved_k'] / A['base']['tokens_per_solved_k'] - 1):+.1f}%)")

print("4. ARC message logs")
X = {a: np.asarray(v, float) for a, v in M["arms"].items()}
cut = {a: 100 * x[:, 3].mean() for a, x in X.items()}
for a, x in X.items():
    tok, c, code = x[:, 0], x[:, 3], x[:, 4]
    print(f"   {a:4s} {len(x)} messages, {int(c.sum())} cut ({cut[a]:.3f}%), code {100 * code.mean():.2f}%, "
          f"tokens in cut messages {100 * tok[c == 1].sum() / tok.sum():.1f}%, "
          f"finished messages with code {100 * code[c == 0].mean():.1f}%")
print(f"   cut share, S-R {cut['S'] - cut['R']:+.4f}, S-L {cut['S'] - cut['L']:+.4f}, S-base {cut['S'] - cut['base']:+.4f}")
print("   restart cost (k tokens per code-carrying message) at hypothetical caps t:")
for a, x in X.items():
    tok, code = x[:, 0], x[:, 4]
    cost = {t: np.minimum(tok, t).sum() / max(1, ((code == 1) & (tok <= t)).sum()) / 1000
            for t in range(8000, M["cap"] + 1, 250)}
    best = min(cost, key=cost.get)
    print(f"     {a:4s} " + "  ".join(f"{t // 1000}k {cost[t]:.1f}" for t in range(8000, M["cap"] + 1, 4000))
          + f"   minimum over 8k-{M['cap'] // 1000}k (250-token grid) at {best / 1000:g}k")
print("   share of messages that had delivered a program within t tokens (%):")
for t in (500, 750, 4000, 8000, 12000, 16000, 20000, 24000):
    print(f"     {t:>6}: " + "  ".join(f"{a} {100 * ((x[:, 4] == 1) & (x[:, 0] <= t)).mean():.1f}" for a, x in X.items()))
