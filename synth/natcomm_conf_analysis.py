#!/usr/bin/env python3
"""Pre-registered analysis of the confirmatory MATH-4500 test (paper_brevity/natcomm/PREREG_MATH4500.md), exactly as
fixed there. Run on Leonardo from the repository root once the four runs nc8c_q4_{S,R}_s{0,1} are complete:

    $WORK/ccapone0/venv_vllm2/bin/python synth/natcomm_conf_analysis.py

H1: mean over problems of (acc S - acc R), per-problem accuracy = mean correctness over the 8 samples (K=4 x seeds 0,1);
    two-sided paired bootstrap over problems, B = 10,000, seed 0, P = 2 min(share <= 0, share >= 0), 95% percentile
    interval; paired sign-flip test (10,000 flips) reported alongside. Supported iff P < 0.05 and difference > 0.
H2 (only if H1): Kitagawa split of the same difference, accuracy = F*Q + C (F = share finished, Q = share correct among
    finished, C = share cut but credited); dA = dF*Qbar + dQ*Fbar + dC, terms of pooled shares, each bootstrapped with
    the same resamples. Supported iff the finishing term's interval lies above 0 and the quality term's estimate <= 0.
Writes paper_brevity/data/conf4500_result.json and prints a summary.
"""
import glob, json, os
import numpy as np

B, SEED, ARMS, SEEDS, K = 10_000, 0, ("S", "R"), (0, 1), 4


def load(arm, seed):
    out = {}
    for f in sorted(glob.glob(f"runs_synth/agent_nc8c_q4_{arm}_s{seed}/shard_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            out[r["task_id"]] = r
    return out


ids = [l.strip() for l in open("synth/math/conf4500_ids.txt") if l.strip()]
runs = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
keep, excluded = [], []
for t in ids:
    ok = all(t in runs[k] and len(runs[k][t]["oks"]) == K and len(runs[k][t]["fins"]) == K for k in runs)
    (keep if ok else excluded).append(t)

# per problem and arm: counts over the 8 samples
def counts(arm):
    ok = np.array([[o for s in SEEDS for o in runs[(arm, s)][t]["oks"]] for t in keep], dtype=float)
    fin = np.array([[f for s in SEEDS for f in runs[(arm, s)][t]["fins"]] for t in keep], dtype=float)
    return dict(ok=ok.sum(1), fin=fin.sum(1), okfin=(ok * fin).sum(1), okcut=(ok * (1 - fin)).sum(1))

c = {a: counts(a) for a in ARMS}
n, N = len(keep), 2 * K
d = (c["S"]["ok"] - c["R"]["ok"]) / N            # per-problem accuracy difference


def terms(sumf):
    """Pooled shares from column sums -> (dA, finishing, quality, residual) in points."""
    F = {a: sumf(c[a]["fin"]) / (sumf(np.ones(n)) * N) for a in ARMS}
    Q = {a: sumf(c[a]["okfin"]) / sumf(c[a]["fin"]) for a in ARMS}
    C = {a: sumf(c[a]["okcut"]) / (sumf(np.ones(n)) * N) for a in ARMS}
    A = {a: sumf(c[a]["ok"]) / (sumf(np.ones(n)) * N) for a in ARMS}
    dF, dQ, dC = F["S"] - F["R"], Q["S"] - Q["R"], C["S"] - C["R"]
    fin_t = dF * (Q["S"] + Q["R"]) / 2
    qual_t = dQ * (F["S"] + F["R"]) / 2
    return np.array([A["S"] - A["R"], fin_t, qual_t, dC]) * 100, dict(A=A, F=F, Q=Q, C=C)


obs, shares = terms(lambda v: v.sum())
rng = np.random.default_rng(SEED)
bs = np.empty((B, 4))
for i0 in range(0, B, 500):
    idx = rng.integers(0, n, size=(min(500, B - i0), n))
    bs[i0:i0 + len(idx)] = np.stack([terms(lambda v, ix=ix: v[ix].sum())[0] for ix in idx])
bs = np.round(bs, 9)


def summ(j):
    x = bs[:, j]
    p = 2 * min((x <= 0).mean(), (x >= 0).mean())
    return dict(est=round(float(obs[j]), 4), ci95=[round(float(np.percentile(x, 2.5)), 4),
                round(float(np.percentile(x, 97.5)), 4)], P_boot=round(float(min(1.0, p)), 5))


h1 = summ(0)
assert abs(obs[0] - d.mean() * 100) < 1e-9
# sign flip on the per-problem differences
frng = np.random.default_rng(SEED)
T = d.mean()
flips = np.array([(d * frng.choice((-1.0, 1.0), size=n)).mean() for _ in range(10_000)])
h1["P_signflip"] = round(float((1 + (np.abs(flips) >= abs(T) - 1e-12).sum()) / (1 + len(flips))), 5)
h1_supported = bool(h1["P_boot"] < 0.05 and h1["est"] > 0)

fin_t, qual_t, res_t = summ(1), summ(2), summ(3)
h2_supported = bool(h1_supported and fin_t["ci95"][0] > 0 and qual_t["est"] <= 0) if h1_supported else None

res = dict(
    prereg="paper_brevity/natcomm/PREREG_MATH4500.md", n_problems=n, n_excluded=len(excluded), samples_per_arm=N,
    B=B, seed=SEED,
    arms={a: {k: round(float(v[a]) * 100, 3) for k, v in shares.items()} for a in ARMS},
    tokens_per_sample={a: round(float(np.mean([l for s in SEEDS for t in keep for l in runs[(a, s)][t]["lens"]])), 1)
                       for a in ARMS},
    H1=dict(**h1, supported=h1_supported),
    H2=dict(finishing=fin_t, quality=qual_t, cut_credited_residual=res_t,
            tested=h1_supported, supported=h2_supported),
)
os.makedirs("paper_brevity/data", exist_ok=True)
json.dump(res, open("paper_brevity/data/conf4500_result.json", "w"), indent=1)
print(json.dumps(res, indent=1))
