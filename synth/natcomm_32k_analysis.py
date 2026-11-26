#!/usr/bin/env python3
"""Exploratory analyses of the NatComm revision runs of 2026-11-19/20 (package B, CORE lane). Run on Leonardo from the
repository root after the lanes have finished:

    $WORK/ccapone0/venv_vllm2/bin/python synth/natcomm_32k_analysis.py

Sections (all exploratory; the only confirmatory test is synth/natcomm_conf_analysis.py):
  m32      MATH-500, Qwen3-4B, 32k cap: per-arm accuracy, finishing, tokens; paired contrasts of S against every other
           arm and of the controls against the untouched model, over the sampling seeds both arms have.
  forced   forced-answer curves at thinking budgets 4k/8k/16k/32k for every arm (forced_answer.py output): accuracy with
           forced answers (correct_t), with cut samples scored as wrong (finished * correct, the paper's convention) and
           as an independent capped run would score them (truncated_t); paired contrasts at every budget.
  aime     AIME 2024+2025 (60 problems), 32k: S, L and the untouched model.
  l8sham   R1-Llama-8B zero-step (sham) adapter at 8k, seeds 0-2, against the untouched model of the paper.
Per problem, outcomes are averaged over samples and seeds; contrasts are paired over problems; bootstrap B = 10,000
resamples of the problems (numpy default_rng(0)), 95% percentile interval, P = 2 min(share <= 0, share >= 0).
Writes paper_brevity/data/natcomm_32k.json and prints a summary.
"""
import glob, json, os
from collections import defaultdict
import numpy as np

B = 10_000
BUDGETS = (4096, 8192, 16384, 32768)
ARMS32 = ("base", "S", "R", "L", "KA", "sham", "Rtm", "Ltm", "baseT04", "q4bRtm", "q4bLtm")


def records(tag):
    out = {}
    for f in sorted(glob.glob(f"runs_synth/agent_{tag}/shard_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            out[r["task_id"]] = r
    return out


def seeds_of(fmt, arm, seeds=(0, 1, 2)):
    return [s for s in seeds if os.path.exists(f"runs_synth/agent_{fmt.format(a=arm, s=s)}/EVAL_DONE")]


def per_problem(fmt, arm, seeds, ids=None):
    """dict task -> (acc, fin, tokens) averaged over all samples of the given seeds."""
    acc, fin, tok, n = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(int)
    for s in seeds:
        for t, r in records(fmt.format(a=arm, s=s)).items():
            k = len(r["lens"]) if "lens" in r else (round(r["tokens"] / r["mean_tokens"]) if r.get("mean_tokens") else 1)
            acc[t] += r["acc"] * k
            fin[t] += sum(r["fins"]) if "fins" in r else k - r.get("cut", 0)
            tok[t] += r["tokens"]
            n[t] += k
    keep = [t for t in n if ids is None or t in ids]
    return {t: (acc[t] / n[t], fin[t] / n[t], tok[t] / n[t]) for t in keep}


def boot(x, y=None, stat=np.mean, seed=0):
    """Paired bootstrap of stat over problems; x, y per-problem arrays (y None: one-sample)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    obs = stat(x) if y is None else stat(x) - stat(y)
    bs = np.empty(B)
    for i0 in range(0, B, 1000):
        idx = rng.integers(0, n, size=(min(1000, B - i0), n))
        bs[i0:i0 + len(idx)] = [stat(x[ix]) if y is None else stat(x[ix]) - stat(y[ix]) for ix in idx]
    bs = np.round(bs, 9)
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return dict(est=round(float(obs), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4),
                round(float(np.percentile(bs, 97.5)), 4)], P=round(float(p), 4))


def rel_tokens(x, y):
    """Relative token change of arm x vs y (ratio of total tokens - 1), in percent, bootstrapped."""
    rng = np.random.default_rng(0)
    n = len(x)
    obs = x.sum() / y.sum() - 1
    bs = np.empty(B)
    for i0 in range(0, B, 1000):
        idx = rng.integers(0, n, size=(min(1000, B - i0), n))
        bs[i0:i0 + len(idx)] = [x[ix].sum() / y[ix].sum() - 1 for ix in idx]
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return dict(est=round(float(obs) * 100, 2), ci95=[round(float(np.percentile(bs, 2.5)) * 100, 2),
                round(float(np.percentile(bs, 97.5)) * 100, 2)], P=round(float(p), 4))


def contrast(fmt, a, b, seeds=None):
    sa, sb = seeds_of(fmt, a), seeds_of(fmt, b)
    common = sorted(set(sa) & set(sb)) if seeds is None else seeds
    A, Bb = per_problem(fmt, a, common), per_problem(fmt, b, common)
    ids = sorted(set(A) & set(Bb))
    xa = np.array([A[t] for t in ids]); xb = np.array([Bb[t] for t in ids])
    return dict(a=a, b=b, seeds=common, n=len(ids),
                acc_pts=boot(xa[:, 0] * 100, xb[:, 0] * 100),
                finish_pts=boot(xa[:, 1] * 100, xb[:, 1] * 100),
                tokens_rel_pct=rel_tokens(xa[:, 2], xb[:, 2]))


def arm_summary(fmt, a):
    s = seeds_of(fmt, a)
    P = per_problem(fmt, a, s)
    x = np.array(list(P.values()))
    return dict(seeds=s, n=len(x), acc=round(float(x[:, 0].mean()) * 100, 2), finish=round(float(x[:, 1].mean()) * 100, 2),
                tokens=round(float(x[:, 2].mean()), 1))


res = {}
# ---------------------------------------------------------------- m32
F32 = "nc32_q4_{a}_s{s}"
arms = [a for a in ARMS32 if seeds_of(F32, a)]
res["m32"] = dict(arms={a: arm_summary(F32, a) for a in arms},
                  contrasts=[contrast(F32, "S", b) for b in arms if b != "S"]
                  + [contrast(F32, b, "base") for b in ("sham", "baseT04", "KA", "R", "L", "Rtm", "Ltm", "q4bRtm", "q4bLtm") if b in arms])
# ---------------------------------------------------------------- forced
def forced(tag):
    f = f"runs_synth/agent_{tag}/forced.jsonl"
    if not os.path.exists(f):
        return None
    out = defaultdict(dict)   # (task, k) -> budget -> row
    for line in open(f):
        r = json.loads(line)
        out[(r["task_id"], r["k"])][r["budget"]] = r
    return out


def forced_pp(fmt, arm, seeds):
    """task -> budget -> (forced acc, convention acc, truncated acc), averaged over samples and seeds."""
    acc = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    for s in seeds:
        fr = forced(fmt.format(a=arm, s=s))
        if fr is None:
            continue
        for (t, k), bud in fr.items():
            for b, r in bud.items():
                if r["correct_t"] is None:
                    continue
                conv = float(r["status"] == "finished" and bool(r["correct"]))
                acc[t][b] += np.array([float(r["correct_t"]), conv, float(r["truncated_t"] or 0), 1.0])
    return {t: {b: v[:3] / v[3] for b, v in bb.items()} for t, bb in acc.items()}


def forced_block(fmt, arms_, pairs):
    out = dict(arms={}, contrasts=[])
    cache = {a: forced_pp(fmt, a, seeds_of(fmt, a)) for a in arms_}
    for a in arms_:
        P = cache[a]
        out["arms"][a] = {str(b): dict(zip(("forced", "convention", "truncated"),
                                           [round(float(np.mean([P[t][b][j] for t in P if b in P[t]])) * 100, 2)
                                            for j in range(3)])) for b in BUDGETS}
    for a, b in pairs:
        common = sorted(set(seeds_of(fmt, a)) & set(seeds_of(fmt, b)))
        Pa, Pb = forced_pp(fmt, a, common), forced_pp(fmt, b, common)
        for bud in BUDGETS:
            ids = sorted(t for t in set(Pa) & set(Pb) if bud in Pa[t] and bud in Pb[t])
            row = dict(a=a, b=b, budget=bud, seeds=common, n=len(ids))
            for j, name in enumerate(("forced", "convention")):
                row[name] = boot(np.array([Pa[t][bud][j] for t in ids]) * 100, np.array([Pb[t][bud][j] for t in ids]) * 100)
            out["contrasts"].append(row)
    return out


pairs = [("S", "R"), ("S", "L"), ("S", "base"), ("S", "KA"), ("S", "Rtm"), ("S", "Ltm"), ("Rtm", "R"), ("Ltm", "L"), ("KA", "R")]
res["forced"] = forced_block(F32, arms, [p for p in pairs if p[0] in arms and p[1] in arms])

# ---------------------------------------------------------------- trunc: cut-is-wrong accuracy at t from the records
def trunc_pp(fmt, arm, seeds):
    """task -> budget -> accuracy with a sample counted correct iff it finished within t tokens and was correct."""
    acc, n = defaultdict(lambda: np.zeros(len(BUDGETS))), defaultdict(int)
    for s in seeds:
        for t, r in records(fmt.format(a=arm, s=s)).items():
            for L, o, f in zip(r["lens"], r["oks"], r["fins"]):
                acc[t] += np.array([float(o and f and L <= b) for b in BUDGETS]); n[t] += 1
    return {t: acc[t] / n[t] for t in acc}


def trunc_block(fmt, arms_, pairs_):
    out = dict(arms={}, contrasts=[])
    for a in arms_:
        P = trunc_pp(fmt, a, seeds_of(fmt, a))
        out["arms"][a] = {str(b): round(float(np.mean([v[j] for v in P.values()])) * 100, 2) for j, b in enumerate(BUDGETS)}
    for a, b, seeds in pairs_:
        common = sorted(set(seeds_of(fmt, a)) & set(seeds_of(fmt, b))) if seeds is None else seeds
        Pa, Pb = trunc_pp(fmt, a, common), trunc_pp(fmt, b, common)
        ids = sorted(set(Pa) & set(Pb))
        for j, bud in enumerate(BUDGETS[:2]):
            out["contrasts"].append(dict(a=a, b=b, budget=bud, seeds=common, n=len(ids),
                                         conv=boot(np.array([Pa[t][j] for t in ids]) * 100, np.array([Pb[t][j] for t in ids]) * 100)))
    return out


tp = [("S", "R", None), ("S", "L", None), ("Rtm", "R", None), ("Ltm", "L", None), ("q4bRtm", "R", [0]), ("q4bLtm", "L", [0]),
      ("Rtm", "R", [1]), ("Ltm", "L", [1]), ("S", "Rtm", None), ("S", "q4bRtm", [0]), ("S", "Ltm", None), ("S", "q4bLtm", [0]),
      ("q4bRtm", "Rtm", [0]), ("q4bLtm", "Ltm", [0])]
res["trunc"] = trunc_block(F32, arms, [p for p in tp if p[0] in arms and p[1] in arms])
# ---------------------------------------------------------------- aime
FA = "nc32_aime_q4_{a}_s{s}"
aarms = [a for a in ("base", "S", "L", "R") if seeds_of(FA, a)]
res["aime"] = dict(arms={a: arm_summary(FA, a) for a in aarms},
                   contrasts=[contrast(FA, x, y) for x, y in (("S", "base"), ("L", "base"), ("S", "L")) if x in aarms and y in aarms],
                   forced=forced_block(FA, aarms, [p for p in (("S", "base"), ("S", "L")) if p[0] in aarms and p[1] in aarms]))
# ---------------------------------------------------------------- l8 sham vs untouched (paper runs)
FL_paper = "math_l8_eval_{a}_s{s}"
if seeds_of(FL_paper, "base") or glob.glob("runs_synth/agent_math_l8_eval_base_s*/shard_*.jsonl"):
    sb = [s for s in (0, 1, 2) if glob.glob(f"runs_synth/agent_math_l8_eval_base_s{s}/shard_*.jsonl")]
    ss = [s for s in (0, 1, 2) if glob.glob(f"runs_synth/agent_nc8_l8_sham_s{s}/shard_*.jsonl")]
    Bp = per_problem(FL_paper, "base", sb)
    Sp = per_problem("nc8_l8_{a}_s{s}", "sham", ss)
    ids = sorted(set(Bp) & set(Sp))
    xs, xb = np.array([Sp[t] for t in ids]), np.array([Bp[t] for t in ids])
    res["l8sham"] = dict(seeds_sham=ss, seeds_base=sb, n=len(ids),
                         sham=dict(acc=round(float(xs[:, 0].mean()) * 100, 2), finish=round(float(xs[:, 1].mean()) * 100, 2)),
                         base=dict(acc=round(float(xb[:, 0].mean()) * 100, 2), finish=round(float(xb[:, 1].mean()) * 100, 2)),
                         acc_pts=boot(xs[:, 0] * 100, xb[:, 0] * 100), finish_pts=boot(xs[:, 1] * 100, xb[:, 1] * 100),
                         tokens_rel_pct=rel_tokens(xs[:, 2], xb[:, 2]))

os.makedirs("paper_brevity/data", exist_ok=True)
json.dump(res, open("paper_brevity/data/natcomm_32k.json", "w"), indent=1)


def line(c, key="acc_pts"):
    v = c[key]
    return f"{v['est']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}] P={v['P']}"


print("== m32 arms"); [print(f"  {a:8s} {v}") for a, v in res["m32"]["arms"].items()]
print("== m32 contrasts (acc pts | tokens %)")
for c in res["m32"]["contrasts"]:
    print(f"  {c['a']:>7s}-{c['b']:<8s} s{c['seeds']} n={c['n']}  acc {line(c)} | tok {line(c, 'tokens_rel_pct')}")
print("== forced arms (forced / convention / truncated)")
for a, v in res["forced"]["arms"].items():
    print(f"  {a:8s} " + "  ".join(f"{b}:{x['forced']:.1f}/{x['convention']:.1f}/{x['truncated']:.1f}" for b, x in v.items()))
print("== forced contrasts")
for c in res["forced"]["contrasts"]:
    print(f"  {c['a']}-{c['b']} {c['budget']:>5d} s{c['seeds']}  forced {line(c, 'forced')} | conv {line(c, 'convention')}")
print("== trunc arms (cut-is-wrong acc at 4k/8k/16k/32k)")
for a, v in res["trunc"]["arms"].items():
    print(f"  {a:8s} " + "  ".join(f"{b}:{x:.2f}" for b, x in v.items()))
print("== trunc contrasts")
for c in res["trunc"]["contrasts"]:
    print(f"  {c['a']}-{c['b']} {c['budget']:>5d} s{c['seeds']}  conv {line(c, 'conv')}")
print("== aime"); print(json.dumps(res["aime"]["arms"]))
for c in res["aime"]["contrasts"]:
    print(f"  {c['a']}-{c['b']} acc {line(c)} | tok {line(c, 'tokens_rel_pct')}")
for c in res["aime"]["forced"]["contrasts"]:
    print(f"  forced {c['a']}-{c['b']} {c['budget']:>5d}  {line(c, 'forced')}")
if "l8sham" in res:
    r = res["l8sham"]
    print(f"== l8 sham vs base n={r['n']} sham {r['sham']} base {r['base']} acc {line(r)} | tok {line(r, 'tokens_rel_pct')}")
