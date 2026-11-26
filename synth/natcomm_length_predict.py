#!/usr/bin/env python3
"""Does length alone predict the accuracy of a trained arm at every cap? (NatComm revision, exploratory, 2026-11-22.)

    env -u PYTHONPATH .venv_mps/bin/python synth/natcomm_length_predict.py [--root paper_brevity/data/nc32_records]

Data: the 32k MATH-500 Qwen3-4B generations (agent_nc32_q4_<arm>_s<seed>/shard_*.jsonl; per problem the lengths,
correctness and finish flags of 4 samples per seed). Scoring as in the paper: at cap t a sample is correct iff it
finished within 32k, is correct and has length <= t (cut samples scored as wrong; no credited cut samples here).

Rank-transplant prediction of arm X from a reference arm Y (the untouched model unless stated): per problem, sort Y's
samples and X's samples by length (samples unfinished at 32k last); the k-th shortest sample of X is predicted to be
correct at cap t iff the k-th shortest sample of Y was correct and X's k-th length is <= t. The prediction uses X's
lengths but never X's correctness: it asks whether X is Y with its attempts shortened in rank order. Gap explained
= (pred - Y) / (obs - Y). Residual obs - pred is bootstrapped over problems (B = 10,000, default_rng(0), 95% percentile
interval, P = 2 min). Seeds: those both arms have, samples pooled. Writes paper_brevity/data/natcomm_length_predict.json.
"""
import argparse, glob, json, os
import numpy as np

B = 10_000
CAPS = (2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768)
PAIRS = (("S", "base"), ("R", "base"), ("L", "base"), ("KA", "base"), ("S", "R"), ("S", "L"), ("R", "L"),
         ("Rtm", "base"), ("Ltm", "base"), ("q4bRtm", "base"), ("q4bLtm", "base"),
         # calibration: arms whose answer quality cannot have changed (zero-step adapter with bit-identical weights,
         # lower temperature) and the untouched model predicted from its other sampling seed
         ("sham", "base"), ("baseT04", "base"), ("base@0", "base@1"), ("base@1", "base@0"))


def seeds_of(root, arm):
    return [s for s in (0, 1, 2) if glob.glob(f"{root}/agent_nc32_q4_{arm}_s{s}/shard_*.jsonl")]


def load(root, arm, seeds):
    """task -> list of (len, correct_and_finished) over samples of the given seeds."""
    out = {}
    for s in seeds:
        for f in sorted(glob.glob(f"{root}/agent_nc32_q4_{arm}_s{s}/shard_*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                for L, o, fi in zip(r["lens"], r["oks"], r["fins"]):
                    out.setdefault(r["task_id"], []).append((L if fi else np.inf, int(o and fi)))
    return out


def boot(d):
    rng = np.random.default_rng(0)
    n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return dict(est=round(float(d.mean()), 3), ci95=[round(float(np.percentile(bs, q)), 3) for q in (2.5, 97.5)],
                P=round(float(p), 4))


def arm_seeds(root, spec, other):
    """'arm' -> (arm, seeds shared with the other arm); 'arm@s' -> (arm, [s])."""
    if "@" in spec:
        arm, s = spec.split("@")
        return arm, [int(s)]
    o = other.split("@")[0]
    return spec, sorted(set(seeds_of(root, spec)) & set(seeds_of(root, o)))


def curves(X, Y, ids):
    """per-problem arrays (n_ids, n_caps): observed X, observed Y, predicted X."""
    obs, ref, pred, flat, shuf, glob_ = (np.zeros((len(ids), len(CAPS))) for _ in range(6))
    allf = [o for t in ids for L, o in Y[t] if np.isfinite(L)]
    qg = np.mean(allf)  # one answer quality for all problems (misspecified control: ignores problem difficulty)
    rng = np.random.default_rng(1)
    for i, t in enumerate(ids):
        x = sorted(X[t], key=lambda v: v[0]); y = sorted(Y[t], key=lambda v: v[0])
        if len(x) != len(y):  # map ranks by quantile when sample counts differ
            idx = np.round(np.linspace(0, len(y) - 1, len(x))).astype(int)
            y_on_x = [y[j] for j in idx]
        else:
            y_on_x = y
        for c, cap in enumerate(CAPS):
            obs[i, c] = np.mean([o and L <= cap for L, o in x])
            ref[i, c] = np.mean([o and L <= cap for L, o in y])
            pred[i, c] = np.mean([yo and xl <= cap for (xl, _), (_, yo) in zip(x, y_on_x)])
        # controls: rank-free (Y's per-problem share correct among its finished samples, for every finished X sample)
        # and shuffled ranks (Y's correctness permuted within the problem before the transplant)
        yf = [o for L, o in y if np.isfinite(L)]
        q = np.mean(yf) if yf else 0.0
        perm = [y_on_x[j][1] for j in rng.permutation(len(y_on_x))]
        for c, cap in enumerate(CAPS):
            flat[i, c] = np.mean([q * (np.isfinite(xl) and xl <= cap) for xl, _ in x])
            shuf[i, c] = np.mean([yo and xl <= cap for (xl, _), yo in zip(x, perm)])
            glob_[i, c] = np.mean([qg * (np.isfinite(xl) and xl <= cap) for xl, _ in x])
    return obs * 100, ref * 100, pred * 100, flat * 100, shuf * 100, glob_ * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="paper_brevity/data/nc32_records")
    ap.add_argument("--out", default="paper_brevity/data/natcomm_length_predict.json")
    a = ap.parse_args()
    res = dict(caps=CAPS, pairs=[])
    # composition of the reference: per-problem share correct among finished samples of the untouched model
    Yb = load(a.root, "base", seeds_of(a.root, "base"))
    q = np.array([np.mean([o for L, o in v if np.isfinite(L)]) if any(np.isfinite(L) for L, _ in v) else 0.0
                  for v in Yb.values()])
    res["reference_composition"] = dict(n=len(q), q_eq_1=int((q == 1).sum()), q_eq_0=int((q == 0).sum()),
                                        mixed=int(((q > 0) & (q < 1)).sum()))
    print("untouched per-problem quality: q=1", res["reference_composition"]["q_eq_1"], "q=0",
          res["reference_composition"]["q_eq_0"], "mixed", res["reference_composition"]["mixed"])
    for xs, ys in PAIRS:
        x, sx = arm_seeds(a.root, xs, ys)
        y, sy = arm_seeds(a.root, ys, xs)
        if "@" not in xs and "@" not in ys:
            sx = sy = sorted(set(sx) & set(sy))
        if not sx or not sy:
            continue
        X, Y = load(a.root, x, sx), load(a.root, y, sy)
        ids = sorted(set(X) & set(Y))
        obs, ref, pred, flat, shuf, glob_ = curves(X, Y, ids)
        x, y, seeds = xs, ys, sorted(set(sx) | set(sy))
        rows = []
        for c, cap in enumerate(CAPS):
            gap, expl = obs[:, c].mean() - ref[:, c].mean(), pred[:, c].mean() - ref[:, c].mean()
            rows.append(dict(cap=cap, obs=round(obs[:, c].mean(), 2), ref=round(ref[:, c].mean(), 2),
                             pred=round(pred[:, c].mean(), 2), gap=round(gap, 2), pred_gap=round(expl, 2),
                             explained=round(expl / gap, 2) if abs(gap) > 0.3 else None,
                             gap_boot=boot(obs[:, c] - ref[:, c]), residual=boot(obs[:, c] - pred[:, c]),
                             resid_rankfree=round(obs[:, c].mean() - flat[:, c].mean(), 2),
                             resid_shuffled=round(obs[:, c].mean() - shuf[:, c].mean(), 2),
                             resid_global=round(obs[:, c].mean() - glob_[:, c].mean(), 2)))
        res["pairs"].append(dict(x=x, ref=y, seeds=seeds, n=len(ids), rows=rows))
        print(f"\n{x} predicted from {y} (seeds {seeds}, n={len(ids)})")
        print("   cap   obs    ref   pred |  gap  predgap expl | residual obs-pred [95% CI] P | resid rank-free, shuffled, global")
        for r in rows:
            e = f"{r['explained']:.2f}" if r["explained"] is not None else "  - "
            print(f"{r['cap']:6d} {r['obs']:6.2f} {r['ref']:6.2f} {r['pred']:6.2f} | {r['gap']:+5.2f} {r['pred_gap']:+6.2f}"
                  f"  {e} | {r['residual']['est']:+.2f} {r['residual']['ci95']} P {r['residual']['P']}"
                  f" | {r['resid_rankfree']:+.2f} {r['resid_shuffled']:+.2f} {r['resid_global']:+.2f}")
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
