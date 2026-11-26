"""Mechanism analysis for the brevity paper (no GPU): does replaying the shortest trajectory work through the restart
channel (fewer messages/samples cut at the cap) or through the quality of finished reasoning?

ARC (agent, per-message msg_log [tokens, think_chars, visible_chars, cut, has_code]):
  per arm, pooled over seeds, on tasks present in every arm: fraction of messages cut, fraction with code, and the
  Luby-Sinclair-Zuckerman restart cost E[min(L,t)] / P(code within t) at the run cap t = 24k.
Single-turn domains (MATH, APPS; 4 samples per problem, cap 8k):
  accuracy = P(finish within cap) x P(correct | finished), assuming a cut sample is wrong (it has no final answer).
Every arm difference is a paired bootstrap over tasks/problems (10,000 resamples), P = fraction of resamples <= 0.
   python synth/mech_analysis.py
"""
import glob, json, random

B = 10000


def load(tag):
    R = {}
    for f in glob.glob(f"runs_synth/agent_{tag}/shard_*.jsonl"):
        for l in open(f):
            if l.strip():
                r = json.loads(l); R.setdefault(r["task_id"], r)
    return R


def seeds(fmt, n=5):
    return [R for R in (load(fmt.format(s)) for s in range(n)) if R]


def boot(ids, stat, rng):
    """stat(ids) -> float; returns observed value, 95% interval, P(value <= 0)."""
    obs = stat(ids)
    bs = sorted(stat([rng.choice(ids) for _ in ids]) for _ in range(B))
    return obs, bs[int(0.025 * B)], bs[int(0.975 * B)], sum(x <= 0 for x in bs) / B


def ratio(num, den):
    return lambda ids: sum(num[i] for i in ids) / max(1e-9, sum(den[i] for i in ids))


def arc():
    print("=== ARC-AGI-2 clean72, 27B agent, pooled over seeds; message-level, cap 24k ===")
    arms = {a: seeds(f"brevcl_{a}_c72_s{{}}") for a in "SRL"}
    arms["base"] = seeds("brevctl_base_c72_s{}")
    ids = sorted(set.intersection(*[set.intersection(*[set(R) for R in runs]) for runs in arms.values()]))
    print(f"tasks in every run: {len(ids)}; seeds: " + ", ".join(f"{a} {len(r)}" for a, r in arms.items()))
    agg = {}
    for a, runs in arms.items():
        n = {i: 0.0 for i in ids}; cut = dict(n); code = dict(n); spent = dict(n); solved = dict(n)
        for R in runs:
            for i in ids:
                solved[i] += bool(R[i].get("pass2")) / len(runs)
                for th in R[i].get("msg_log", []):
                    for m in th:
                        n[i] += 1; cut[i] += bool(m[3]); code[i] += bool(m[4]); spent[i] += min(m[0], 24000)
        agg[a] = dict(n=n, cut=cut, code=code, spent=spent, solved=solved)
        N = sum(n.values())
        print(f"{a:5s} messages {N:5.0f} | cut {sum(cut.values())/N:.3f} | with code {sum(code.values())/N:.3f} | "
              f"restart cost {sum(spent.values())/sum(code.values())/1e3:.1f}k tokens per code message | solved {sum(solved.values()):.1f}")
    rng = random.Random(0)
    for a, b in (("S", "base"), ("S", "R"), ("S", "L"), ("R", "base"), ("L", "base")):
        A, Bb = agg[a], agg[b]
        dcut = lambda I: ratio(A["cut"], A["n"])(I) - ratio(Bb["cut"], Bb["n"])(I)
        dcost = lambda I: (ratio(Bb["spent"], Bb["code"])(I) - ratio(A["spent"], A["code"])(I)) / 1e3
        o1, l1, h1, p1 = boot(ids, lambda I: -dcut(I), rng)
        o2, l2, h2, p2 = boot(ids, dcost, rng)
        print(f"{a} vs {b}: cut fraction {-o1:+.3f} [{-h1:+.3f}, {-l1:+.3f}] P={p1:.3f} | "
              f"restart cost {-o2:+.1f}k [{-h2:+.1f}, {-l2:+.1f}] P={p2:.3f}")
    # does the per-task change in cutting track the per-task change in solving? (S vs base)
    dc = [agg["base"]["cut"][i] / max(1, agg["base"]["n"][i]) - agg["S"]["cut"][i] / max(1, agg["S"]["n"][i]) for i in ids]
    ds = [agg["S"]["solved"][i] - agg["base"]["solved"][i] for i in ids]
    mc, ms = sum(dc) / len(dc), sum(ds) / len(ds)
    cov = sum((x - mc) * (y - ms) for x, y in zip(dc, ds))
    r = cov / ((sum((x - mc) ** 2 for x in dc) * sum((y - ms) ** 2 for y in ds)) ** 0.5 + 1e-12)
    print(f"per-task corr(fewer cuts, more solved), S vs base: r = {r:+.2f} over {len(ids)} tasks")


def single(name, fmt, arms, n_ids_min):
    print(f"=== {name}: accuracy = P(finish) x P(correct | finished) ===")
    data = {a: seeds(fmt.format(a=a, s="{}")) for a in arms}
    data = {a: [R for R in runs if len(R) >= n_ids_min] for a, runs in data.items()}
    data = {a: runs for a, runs in data.items() if runs}
    ids = sorted(set.intersection(*[set.intersection(*[set(R) for R in runs]) for runs in data.values()]))
    agg = {}
    for a, runs in data.items():
        acc = {i: sum(R[i]["acc"] for R in runs) / len(runs) for i in ids}
        fin = {i: sum((4 - R[i]["cut"]) / 4 for R in runs) / len(runs) for i in ids}
        agg[a] = (acc, fin)
        A = sum(acc.values()) / len(ids); F = sum(fin.values()) / len(ids)
        print(f"{a:5s} seeds {len(runs)} | acc {100*A:.2f} | finish {100*F:.2f} | correct|finished {100*A/F:.2f}")
    rng = random.Random(0)
    for a, b in (("S", "base"), ("S", "R"), ("S", "L"), ("R", "base"), ("L", "base"), ("S2", "S"), ("S2", "base")):
        if a not in agg or b not in agg:
            continue
        (aa, fa), (ab, fb) = agg[a], agg[b]
        o1, l1, h1, p1 = boot(ids, lambda I: 100 * (sum(fa[i] - fb[i] for i in I) / len(I)), rng)
        cond = lambda I: 100 * (ratio(aa, fa)(I) - ratio(ab, fb)(I))
        o2, l2, h2, p2 = boot(ids, cond, rng)
        print(f"{a} - {b}: finish {o1:+.2f} [{l1:+.2f}, {h1:+.2f}] P={p1:.3f} | correct|finished {o2:+.2f} [{l2:+.2f}, {h2:+.2f}] P={p2:.3f}")


arc()
single("MATH-500, Qwen3-4B", "math_eval_{a}_s{s}", ["S", "R", "L", "base", "S2"], 490)
single("MATH-500, R1-Distill-Llama-8B", "math_l8_eval_{a}_s{s}", ["S", "R", "L", "base"], 490)
single("APPS, Qwen3-4B", "code_eval_{a}_s{s}", ["S", "R", "L", "base"], 490)
