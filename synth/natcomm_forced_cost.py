#!/usr/bin/env python3
"""Token cost of the forced-answer protocol (NatComm revision, main_v2.tex, section 'Forced answer'). Exploratory.

    env -u PYTHONPATH .venv_mps/bin/python synth/natcomm_forced_cost.py [--root paper_brevity/data/nc32_records]

Reads the forced-answer records of the 32k MATH-500 Qwen3-4B arms (agent_nc32_q4_<arm>_s<seed>/forced.jsonl, from
synth/forced_answer.py). Generated tokens of one sample at thinking budget t:
  finished       n_tokens                                   (stopped by itself within t)
  forced         prefix_tokens + |stop text| + forced_tokens (cut at t, closed with Qwen's stop text, answer written)
  continued      prefix_tokens + forced_tokens               (the reasoning had already closed inside the first t tokens)
  not_evaluable  excluded (none expected: the 32k runs have no lower cap)
|stop text| is counted with the Qwen2.5/Qwen3 BPE tokenizer (same vocabulary), tokenised on its own as forced_answer.py
does. Accuracy is correct_t. Per problem, outcomes are averaged over samples and seeds; contrasts are paired over the
problems and the seeds both arms have; bootstrap B = 10,000 resamples of problems (numpy default_rng(0)), 95% percentile
interval, P = 2 min(share <= 0, share >= 0). The same conventions as synth/natcomm_32k_analysis.py.
Writes paper_brevity/data/natcomm_forced_cost.json.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

B = 10_000
BUDGETS = (4096, 8192, 16384, 32768)
ARMS = ("base", "S", "R", "L", "KA", "Rtm", "Ltm", "sham", "baseT04")
PAIRS = (("S", "base"), ("S", "R"), ("S", "L"), ("S", "KA"), ("S", "Rtm"), ("R", "base"), ("L", "base"))
QWEN_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly "
             "now.\n</think>\n\n")


def stop_len():
    from tokenizers import Tokenizer
    f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/tokenizer.json"))
    return len(Tokenizer.from_file(f[0]).encode(QWEN_STOP, add_special_tokens=False).ids)


def seeds_of(root, arm):
    return [s for s in (0, 1, 2) if os.path.exists(f"{root}/agent_nc32_q4_{arm}_s{s}/forced.jsonl")]


def per_problem(root, arm, seeds, nstop):
    """task -> budget -> (acc, tokens, share forced), averaged over samples and seeds."""
    acc = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    for s in seeds:
        for line in open(f"{root}/agent_nc32_q4_{arm}_s{s}/forced.jsonl"):
            r = json.loads(line)
            st = r["status"]
            if st == "not_evaluable" or r["correct_t"] is None:
                continue
            if st == "finished":
                tok = r["n_tokens"]
            elif st == "forced":
                tok = r["prefix_tokens"] + nstop + r["forced_tokens"]
            elif st == "continued":
                tok = r["prefix_tokens"] + r["forced_tokens"]
            else:
                raise ValueError(st)
            acc[r["task_id"]][r["budget"]] += np.array([r["correct_t"], tok, st != "finished", 1.0])
    return {t: {b: v[:3] / v[3] for b, v in bb.items()} for t, bb in acc.items()}


def boot(x, y):
    rng = np.random.default_rng(0)
    n = len(x)
    bs = np.empty(B)
    for i0 in range(0, B, 1000):
        idx = rng.integers(0, n, size=(min(1000, B - i0), n))
        bs[i0:i0 + len(idx)] = [x[ix].mean() - y[ix].mean() for ix in idx]
    bs = np.round(bs, 9)
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return dict(est=round(float(x.mean() - y.mean()), 4), ci95=[round(float(np.percentile(bs, q)), 4) for q in (2.5, 97.5)],
                P=round(float(p), 4))


def rel(x, y):
    rng = np.random.default_rng(0)
    n = len(x)
    bs = np.empty(B)
    for i0 in range(0, B, 1000):
        idx = rng.integers(0, n, size=(min(1000, B - i0), n))
        bs[i0:i0 + len(idx)] = [x[ix].sum() / y[ix].sum() - 1 for ix in idx]
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return dict(est=round(float(x.sum() / y.sum() - 1) * 100, 2),
                ci95=[round(float(np.percentile(bs, q)) * 100, 2) for q in (2.5, 97.5)], P=round(float(p), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="paper_brevity/data/nc32_records")
    ap.add_argument("--out", default="paper_brevity/data/natcomm_forced_cost.json")
    a = ap.parse_args()
    nstop = stop_len()
    arms = [x for x in ARMS if seeds_of(a.root, x)]
    res = dict(stop_tokens=nstop, arms={}, contrasts=[])
    for arm in arms:
        s = seeds_of(a.root, arm)
        P = per_problem(a.root, arm, s, nstop)
        res["arms"][arm] = dict(seeds=s, n=len(P), **{str(b): dict(
            acc=round(float(np.mean([P[t][b][0] for t in P])) * 100, 2),
            tokens=round(float(np.mean([P[t][b][1] for t in P])), 1),
            forced_share=round(float(np.mean([P[t][b][2] for t in P])) * 100, 2)) for b in BUDGETS})
    for x, y in PAIRS:
        if x not in arms or y not in arms:
            continue
        common = sorted(set(seeds_of(a.root, x)) & set(seeds_of(a.root, y)))
        Px, Py = per_problem(a.root, x, common, nstop), per_problem(a.root, y, common, nstop)
        ids = sorted(set(Px) & set(Py))
        for b in BUDGETS:
            ax = np.array([Px[t][b] for t in ids]); ay = np.array([Py[t][b] for t in ids])
            res["contrasts"].append(dict(a=x, b=y, budget=b, seeds=common, n=len(ids),
                                         acc_pts=boot(ax[:, 0] * 100, ay[:, 0] * 100),
                                         tokens_abs=boot(ax[:, 1], ay[:, 1]), tokens_rel_pct=rel(ax[:, 1], ay[:, 1])))
    json.dump(res, open(a.out, "w"), indent=1)
    print("stop text tokens", nstop)
    for arm in arms:
        print(arm, res["arms"][arm]["seeds"], " | ".join(
            f"{b}: acc {res['arms'][arm][str(b)]['acc']:.2f} tok {res['arms'][arm][str(b)]['tokens']:.0f} "
            f"forced {res['arms'][arm][str(b)]['forced_share']:.1f}%" for b in BUDGETS))
    for c in res["contrasts"]:
        print(f"{c['a']}-{c['b']} {c['budget']:5d} seeds {c['seeds']}: acc {c['acc_pts']['est']:+.2f} {c['acc_pts']['ci95']} "
              f"P {c['acc_pts']['P']} | tokens {c['tokens_abs']['est']:+.0f} = {c['tokens_rel_pct']['est']:+.2f}% "
              f"{c['tokens_rel_pct']['ci95']} P {c['tokens_rel_pct']['P']}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
