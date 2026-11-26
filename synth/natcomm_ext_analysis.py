#!/usr/bin/env python3
"""Published length-reduction checkpoints under a token cap (NatComm revision, package B; pre-registration
paper_brevity/natcomm/PREREG_EXTERNAL.md). Written before any of these models was evaluated.

    python3 synth/natcomm_ext_analysis.py [--root runs_synth] [--out paper_brevity/data/natcomm_ext.json]

Runs: runs_synth/agent_ncx_<arm>_s0/{shard_*.jsonl, forced.jsonl}; MATH-500, 4 samples per problem, 16k generations
(synth/math_harvest.py) and forced answers at budgets 1k/2k/4k/8k/16k (synth/forced_answer.py), as in the paper.
Per sample at budget t (rows of forced.jsonl):
  finished  status 'finished' (ended by itself within t tokens)
  cut       = 1 - finished; the sample's correctness under three scorings:
            cut-as-wrong  finished and correct (correct_t of finished rows; 0 otherwise)
            credited      truncated_t (the paper's checker on the text cut at t: credits a boxed answer already written)
            forced        correct_t (answer forced at t with Qwen's thinking-budget text; 'continued' rows close themselves)
  tokens    generated tokens with forced answers: finished n_tokens; forced prefix + |stop text| + forced answer;
            continued prefix + forced answer (|stop text| from the arm's own tokenizer, see stop_len)
Per problem, outcomes are averaged over the 4 samples. Contrasts are paired over problems; bootstrap B = 10,000 resamples
of the problems (numpy default_rng(0)), 95% percentile interval, P = 2 min(share <= 0, share >= 0).
Split of a cut-as-wrong accuracy difference: A = F Q, dA = dF Qbar + dQ Fbar (pooled shares, problems resampled).
PRIMARY (pre-registered): difference in differences D = mean over the four method contrasts M and the budgets 4k and 8k of
  [forced(M) - forced(base_M)] - [cut-as-wrong(M) - cut-as-wrong(base_M)], per problem, bootstrapped over problems.
  M: arora_a02 vs r1base, thinkprune vs r1base, dler vs r1base, l1max4k vs dsr.  Two-sided; D < 0 supports H1.
Writes the json and prints a summary.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

B = 10_000
BUDGETS = (1024, 2048, 4096, 8192, 16384)
ARMS = ("r1base", "arora_a0", "arora_a02", "thinkprune", "dler", "dsr", "l1max4k", "l1max1k")
CONTRASTS = (("arora_a02", "r1base"), ("thinkprune", "r1base"), ("dler", "r1base"), ("l1max4k", "dsr"),
             ("arora_a02", "arora_a0"), ("arora_a0", "r1base"), ("l1max1k", "dsr"))
PRIMARY = (("arora_a02", "r1base"), ("thinkprune", "r1base"), ("dler", "r1base"), ("l1max4k", "dsr"))
PRIMARY_BUDGETS = (4096, 8192)
QWEN_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly "
             "now.\n</think>\n\n")
R1_TOKENIZER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"   # every arm shares this vocabulary (R1-Distill-Qwen-1.5B family)


def stop_len():
    try:
        from transformers import AutoTokenizer
        return len(AutoTokenizer.from_pretrained(R1_TOKENIZER).encode(QWEN_STOP, add_special_tokens=False))
    except Exception as e:  # offline without the tokenizer: the token count is reported, not used for accuracy
        print("warning: stop-text length unavailable,", e)
        return None


def per_problem(root, arm, nstop):
    """task -> budget -> array [cut-as-wrong, credited, forced, finished, tokens]"""
    f = f"{root}/agent_ncx_{arm}_s0/forced.jsonl"
    acc = defaultdict(lambda: defaultdict(lambda: np.zeros(6)))
    for line in open(f):
        r = json.loads(line)
        st = r["status"]
        if st == "not_evaluable" or r["correct_t"] is None:
            continue
        fin = st == "finished"
        if fin:
            tok = r["n_tokens"]
        elif st == "forced":
            tok = r["prefix_tokens"] + (nstop or 0) + r["forced_tokens"]
        else:
            tok = r["prefix_tokens"] + r["forced_tokens"]
        acc[r["task_id"]][r["budget"]] += np.array([fin and r["correct_t"], r["truncated_t"] or 0, r["correct_t"],
                                                    fin, tok, 1.0])
    return {t: {b: v[:5] / v[5] for b, v in bb.items()} for t, bb in acc.items()}


def boot(d, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    bs = np.empty(B)
    for i0 in range(0, B, 1000):
        idx = rng.integers(0, n, size=(min(1000, B - i0), n))
        bs[i0:i0 + len(idx)] = d[idx].mean(axis=1)
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return dict(est=round(float(d.mean()), 3), ci95=[round(float(np.percentile(bs, q)), 3) for q in (2.5, 97.5)],
                P=round(float(p), 4))


def split(x, y):
    """cut-as-wrong dA = dF Qbar + dQ Fbar from pooled shares; x, y: (n, 5) per-problem arrays at one budget"""
    rng = np.random.default_rng(0)
    n = len(x)

    def terms(ix):
        Fx, Fy = x[ix, 3].mean(), y[ix, 3].mean()
        Ax, Ay = x[ix, 0].mean(), y[ix, 0].mean()
        Qx, Qy = (Ax / Fx if Fx else 0), (Ay / Fy if Fy else 0)
        return (Fx - Fy) * (Qx + Qy) / 2 * 100, (Qx - Qy) * (Fx + Fy) / 2 * 100
    est = terms(np.arange(n))
    bs = np.array([terms(rng.integers(0, n, n)) for _ in range(B)])
    out = {}
    for k, name in enumerate(("finishing", "quality")):
        p = min(1.0, 2 * min((bs[:, k] <= 0).mean(), (bs[:, k] >= 0).mean()))
        out[name] = dict(est=round(float(est[k]), 3), ci95=[round(float(np.percentile(bs[:, k], q)), 3) for q in (2.5, 97.5)],
                         P=round(float(p), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_synth")
    ap.add_argument("--out", default="paper_brevity/data/natcomm_ext.json")
    a = ap.parse_args()
    nstop = stop_len()
    arms = [x for x in ARMS if os.path.exists(f"{a.root}/agent_ncx_{x}_s0/forced.jsonl")]
    P = {x: per_problem(a.root, x, nstop) for x in arms}
    res = dict(stop_tokens=nstop, arms={}, contrasts=[], primary=None)
    for x in arms:
        res["arms"][x] = {str(b): dict(zip(("cut_as_wrong", "credited", "forced", "finished", "tokens_forced"),
                                           [round(float(np.mean([P[x][t][b][k] for t in P[x]])) * (1 if k == 4 else 100), 2)
                                            for k in range(5)])) for b in BUDGETS}
    for x, y in CONTRASTS:
        if x not in P or y not in P:
            continue
        ids = sorted(set(P[x]) & set(P[y]))
        for b in BUDGETS:
            X = np.array([P[x][t][b] for t in ids]); Y = np.array([P[y][t][b] for t in ids])
            res["contrasts"].append(dict(a=x, b=y, budget=b, n=len(ids),
                                         cut_as_wrong=boot((X[:, 0] - Y[:, 0]) * 100), credited=boot((X[:, 1] - Y[:, 1]) * 100),
                                         forced=boot((X[:, 2] - Y[:, 2]) * 100), finished=boot((X[:, 3] - Y[:, 3]) * 100),
                                         forced_minus_cut=boot(((X[:, 2] - Y[:, 2]) - (X[:, 0] - Y[:, 0])) * 100),
                                         tokens_rel_pct=round(float(X[:, 4].sum() / Y[:, 4].sum() - 1) * 100, 2),
                                         split=split(X, Y)))
    if all(x in P and y in P for x, y in PRIMARY):
        ids = sorted(set.intersection(*[set(P[x]) & set(P[y]) for x, y in PRIMARY]))
        d = np.mean([[((P[x][t][b][2] - P[y][t][b][2]) - (P[x][t][b][0] - P[y][t][b][0])) * 100 for t in ids]
                     for x, y in PRIMARY for b in PRIMARY_BUDGETS], axis=0)
        res["primary"] = dict(n=len(ids), contrasts=PRIMARY, budgets=PRIMARY_BUDGETS, D=boot(d))
    json.dump(res, open(a.out, "w"), indent=1)
    print("stop text tokens", nstop)
    for x in arms:
        print(x, " | ".join(f"{b}: cut {res['arms'][x][str(b)]['cut_as_wrong']:.1f} cred {res['arms'][x][str(b)]['credited']:.1f} "
                            f"forced {res['arms'][x][str(b)]['forced']:.1f} fin {res['arms'][x][str(b)]['finished']:.1f}"
                            for b in BUDGETS))
    for c in res["contrasts"]:
        print(f"{c['a']}-{c['b']} {c['budget']:5d}: cut {c['cut_as_wrong']['est']:+.2f} {c['cut_as_wrong']['ci95']} | "
              f"forced {c['forced']['est']:+.2f} {c['forced']['ci95']} | forced-cut {c['forced_minus_cut']['est']:+.2f} "
              f"{c['forced_minus_cut']['ci95']} | F {c['split']['finishing']['est']:+.2f} Q {c['split']['quality']['est']:+.2f} "
              f"| tokens {c['tokens_rel_pct']:+.1f}%")
    if res["primary"]:
        print("PRIMARY D", res["primary"]["D"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
