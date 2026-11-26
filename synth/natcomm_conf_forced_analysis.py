#!/usr/bin/env python3
"""Amendment 1 to the pre-registered MATH-4500 test (paper_brevity/natcomm/PREREG_MATH4500_AMEND1.md): answers forced at
the cap on the saved generations of the four pre-registered runs nc8c_q4_{S,R}_s{0,1}. Written before any forced answer
on these generations was produced.

    python3 synth/natcomm_conf_forced_analysis.py [--root runs_synth] [--out paper_brevity/data/conf4500_forced.json]

Input: runs_synth/agent_nc8c_q4_<arm>_s<seed>/forced.jsonl (synth/forced_answer.py, budgets 4096 and 8192).
Per sample at budget t: cut-as-wrong (finished within t and correct), credited (truncated_t: the pre-registered checker
on the text cut at t; at 8192 this is the pre-registered scoring), forced (correct_t), finished, tokens with forced
answers (finished: n_tokens; forced: prefix + |stop text| + answer; continued: prefix + answer).
Per problem: mean over the 8 samples (K = 4 x seeds 0, 1). A problem missing in any run or budget is excluded (reported).
Bootstrap over problems: B = 10,000, numpy default_rng(0), 95% percentile interval, P = 2 min(share <= 0, share >= 0).

H3 (primary): at 8192, D8 = mean_i [(forced_S - forced_R) - (credited_S - credited_R)] < 0, P < 0.05.
H4 (primary): at 8192, forced_S - forced_R is equivalent to 0 within +-0.5 points: the 90% bootstrap interval
    (5th, 95th percentiles) lies inside (-0.5, 0.5).
Secondary (labelled): the same quantities at 4096; cut-as-wrong contrasts; finished shares; relative tokens with forced
answers, S vs R, bootstrapped (ratio of totals over resampled problems).
"""
import argparse, json, os
from collections import defaultdict
import numpy as np

B = 10_000
BUDGETS = (4096, 8192)
ARMS, SEEDS, K = ("S", "R"), (0, 1), 4
QWEN_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly "
             "now.\n</think>\n\n")
TOKENIZER = "Qwen/Qwen3-4B-Thinking-2507"
COLS = ("cut_as_wrong", "credited", "forced", "finished", "tokens")


def stop_len():
    try:
        from transformers import AutoTokenizer
        return len(AutoTokenizer.from_pretrained(TOKENIZER).encode(QWEN_STOP, add_special_tokens=False))
    except Exception as e:
        print("warning: stop-text length unavailable,", e)
        return None


def load(root, arm, seed, nstop):
    """task -> budget -> list of per-sample [cut_as_wrong, credited, forced, finished, tokens]"""
    out = defaultdict(lambda: defaultdict(list))
    for line in open(f"{root}/agent_nc8c_q4_{arm}_s{seed}/forced.jsonl"):
        r = json.loads(line)
        if r["status"] == "not_evaluable" or r["correct_t"] is None:
            continue
        fin = r["status"] == "finished"
        if fin:
            tok = r["n_tokens"]
        elif r["status"] == "forced":
            tok = r["prefix_tokens"] + (nstop or 0) + r["forced_tokens"]
        else:
            tok = r["prefix_tokens"] + r["forced_tokens"]
        out[r["task_id"]][r["budget"]].append([float(fin and r["correct_t"]), float(r["truncated_t"] or 0),
                                               float(r["correct_t"]), float(fin), float(tok)])
    return out


def pct(bs, q):
    return round(float(np.percentile(bs, q)), 3)


def boot(d, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    n = len(d)
    bs = np.empty(B)
    for i0 in range(0, B, 1000):
        idx = rng.integers(0, n, size=(min(1000, B - i0), n))
        bs[i0:i0 + len(idx)] = d[idx].mean(axis=1)
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return dict(est=round(float(d.mean()), 3), ci95=[pct(bs, 2.5), pct(bs, 97.5)], ci90=[pct(bs, 5), pct(bs, 95)],
                P=round(float(p), 4))


def boot_ratio(x, y, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    bs = np.empty(B)
    for i0 in range(0, B, 1000):
        idx = rng.integers(0, n, size=(min(1000, B - i0), n))
        bs[i0:i0 + len(idx)] = (x[idx].sum(1) / y[idx].sum(1) - 1) * 100
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return dict(est=round(float((x.sum() / y.sum() - 1) * 100), 3), ci95=[pct(bs, 2.5), pct(bs, 97.5)], P=round(float(p), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_synth")
    ap.add_argument("--ids", default="synth/math/conf4500_ids.txt")
    ap.add_argument("--out", default="paper_brevity/data/conf4500_forced.json")
    a = ap.parse_args()
    nstop = stop_len()
    runs = {(arm, s): load(a.root, arm, s, nstop) for arm in ARMS for s in SEEDS}
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    keep, excluded = [], []
    for t in ids:
        ok = all(len(runs[k][t][b]) == K for k in runs for b in BUDGETS)
        (keep if ok else excluded).append(t)
    # per problem, arm, budget: mean over the 8 samples
    P = {arm: {b: np.array([np.mean([x for s in SEEDS for x in runs[(arm, s)][t][b]], axis=0) for t in keep])
               for b in BUDGETS} for arm in ARMS}
    res = dict(stop_tokens=nstop, n=len(keep), excluded=excluded, arms={}, budgets={})
    for arm in ARMS:
        res["arms"][arm] = {str(b): {c: round(float(P[arm][b][:, k].mean()) * (1 if c == "tokens" else 100), 3)
                                     for k, c in enumerate(COLS)} for b in BUDGETS}
    for b in BUDGETS:
        S, R = P["S"][b], P["R"][b]
        dd = {c: (S[:, k] - R[:, k]) * 100 for k, c in enumerate(COLS[:4])}
        res["budgets"][str(b)] = dict(
            **{f"S_minus_R_{c}": boot(dd[c]) for c in COLS[:4]},
            forced_minus_credited=boot(dd["forced"] - dd["credited"]),
            forced_minus_cut=boot(dd["forced"] - dd["cut_as_wrong"]),
            tokens_forced_rel_pct=boot_ratio(S[:, 4], R[:, 4]))
    b8 = res["budgets"]["8192"]
    res["H3"] = dict(D8=b8["forced_minus_credited"], supported=b8["forced_minus_credited"]["est"] < 0
                     and b8["forced_minus_credited"]["P"] < 0.05)
    lo, hi = b8["S_minus_R_forced"]["ci90"]
    res["H4"] = dict(forced_S_minus_R=b8["S_minus_R_forced"], margin=0.5, supported=bool(lo > -0.5 and hi < 0.5))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("stop text tokens", nstop, "| problems", len(keep), "| excluded", len(excluded))
    for arm in ARMS:
        print(arm, " | ".join(f"{b}: " + " ".join(f"{c} {res['arms'][arm][str(b)][c]:.2f}" for c in COLS) for b in BUDGETS))
    for b in BUDGETS:
        r = res["budgets"][str(b)]
        print(f"{b}: " + " | ".join(f"S-R {c} {r['S_minus_R_' + c]['est']:+.3f} {r['S_minus_R_' + c]['ci95']}" for c in COLS[:4])
              + f" | forced-credited {r['forced_minus_credited']['est']:+.3f} {r['forced_minus_credited']['ci95']}"
              + f" | tokens {r['tokens_forced_rel_pct']['est']:+.2f}% {r['tokens_forced_rel_pct']['ci95']}")
    print("H3", res["H3"])
    print("H4", res["H4"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
