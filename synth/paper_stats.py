"""All numbers for the brevity paper's figures in one JSON (no GPU). Per domain and arm: per-seed accuracy, mean, tokens,
P(finish within cap), P(correct | finished); arm contrasts with paired-bootstrap 95% intervals over problems (seed-mean per
problem, 10,000 resamples) and P = fraction of resamples <= 0. ARC also gets message-level cut / code / restart cost.
   python synth/paper_stats.py > logs/paper_stats.json
"""
import glob, json, random, sys

B = 10000


def load(tag):
    R = {}
    for f in glob.glob(f"runs_synth/agent_{tag}/shard_*.jsonl"):
        for l in open(f):
            if l.strip():
                r = json.loads(l); R.setdefault(r["task_id"], r)
    return R


def boot(ids, stat, seed=0):
    rng = random.Random(seed)
    obs = stat(ids)
    bs = sorted(stat([rng.choice(ids) for _ in ids]) for _ in range(B))
    return dict(diff=round(obs, 3), lo=round(bs[int(0.025 * B)], 3), hi=round(bs[int(0.975 * B)], 3),
                p=round(sum(x <= 0 for x in bs) / B, 4))


def single(fmt, arms, n_min, contrasts):
    runs = {a: [R for R in (load(fmt.format(a=a, s=s)) for s in range(5)) if len(R) >= n_min] for a in arms}
    runs = {a: r for a, r in runs.items() if r}
    if not runs:
        return None
    ids = sorted(set.intersection(*[set.intersection(*[set(R) for R in rr]) for rr in runs.values()]))
    out = dict(n_problems=len(ids), arms={}, contrasts={})
    acc, fin = {}, {}
    for a, rr in runs.items():
        acc[a] = {i: sum(R[i]["acc"] for R in rr) / len(rr) for i in ids}
        fin[a] = {i: sum((4 - R[i]["cut"]) / 4 for R in rr) / len(rr) for i in ids}
        A = sum(acc[a].values()) / len(ids); F = sum(fin[a].values()) / len(ids)
        out["arms"][a] = dict(seeds=[round(100 * sum(R[i]["acc"] for i in ids) / len(ids), 2) for R in rr],
                              acc=round(100 * A, 2), finish=round(100 * F, 2), correct_given_finished=round(100 * A / F, 2),
                              tokens=round(sum(sum(R[i]["mean_tokens"] for i in ids) / len(ids) for R in rr) / len(rr)))
    for a, b in contrasts:
        if a in acc and b in acc:
            out["contrasts"][f"{a}-{b}"] = dict(
                acc=boot(ids, lambda I: 100 * sum(acc[a][i] - acc[b][i] for i in I) / len(I)),
                finish=boot(ids, lambda I: 100 * sum(fin[a][i] - fin[b][i] for i in I) / len(I)),
                correct_given_finished=boot(ids, lambda I: 100 * (sum(acc[a][i] for i in I) / sum(fin[a][i] for i in I)
                                                                   - sum(acc[b][i] for i in I) / sum(fin[b][i] for i in I))))
    return out


def arc():
    runs = {a: [R for R in (load(f"brevcl_{a}_c72_s{s}") for s in range(5)) if R] for a in "SRL"}
    runs["base"] = [R for R in (load(f"brevctl_base_c72_s{s}") for s in range(5)) if R]
    ids = sorted(set.intersection(*[set.intersection(*[set(R) for R in rr]) for rr in runs.values()]))
    out = dict(n_tasks=len(ids), arms={}, contrasts={})
    sol, n, cut, code, spent, tok = {}, {}, {}, {}, {}, {}
    for a, rr in runs.items():
        sol[a] = {i: sum(bool(R[i].get("pass2")) for R in rr) / len(rr) for i in ids}
        n[a] = {i: 0 for i in ids}; cut[a] = dict(n[a]); code[a] = dict(n[a]); spent[a] = dict(n[a])
        for R in rr:
            for i in ids:
                for th in R[i].get("msg_log", []):
                    for m in th:
                        n[a][i] += 1; cut[a][i] += bool(m[3]); code[a][i] += bool(m[4]); spent[a][i] += min(m[0], 24000)
        N = sum(n[a].values()); solved_seeds = [sum(bool(R[i].get("pass2")) for i in ids) for R in rr]
        toks = sum(R[i]["tokens"] for R in rr for i in ids)
        out["arms"][a] = dict(seeds=solved_seeds, solved=round(sum(sol[a].values()), 2),
                              tokens_per_solved_k=round(toks / max(1, sum(solved_seeds)) / 1e3, 1),
                              cut_msgs=round(100 * sum(cut[a].values()) / N, 2), code_msgs=round(100 * sum(code[a].values()) / N, 2),
                              restart_cost_k=round(sum(spent[a].values()) / sum(code[a].values()) / 1e3, 2))
    r = lambda num, den: (lambda I: sum(num[i] for i in I) / max(1e-9, sum(den[i] for i in I)))
    for a, b in (("S", "base"), ("R", "base"), ("L", "base"), ("S", "R"), ("S", "L")):
        out["contrasts"][f"{a}-{b}"] = dict(
            solved=boot(ids, lambda I: sum(sol[a][i] - sol[b][i] for i in I) * len(ids) / len(I)),
            cut_msgs=boot(ids, lambda I: 100 * (r(cut[a], n[a])(I) - r(cut[b], n[b])(I))),
            restart_cost_k=boot(ids, lambda I: (r(spent[a], code[a])(I) - r(spent[b], code[b])(I)) / 1e3))
    return out


def cap_interaction(fmt8, fmt16, arms=("S", "L", "base"), n_min=490):
    """Restart-theory test: does loosening the token cap (8k -> 16k) shrink the arm differences? Per problem, the seed-mean
    difference at 8k minus the same difference at 16k, paired bootstrap over problems."""
    acc, fin = {}, {}
    for cap, fmt in (("8k", fmt8), ("16k", fmt16)):
        for a in arms:
            rr = [R for R in (load(fmt.format(a=a, s=s)) for s in range(5)) if len(R) >= n_min]
            if not rr:
                return None
            acc[cap, a] = {i: sum(R[i]["acc"] for R in rr if i in R) / sum(1 for R in rr if i in R)
                           for i in set.intersection(*[set(R) for R in rr])}
            fin[cap, a] = {i: sum((4 - R[i]["cut"]) / 4 for R in rr if i in R) / sum(1 for R in rr if i in R)
                           for i in acc[cap, a]}
    ids = sorted(set.intersection(*[set(v) for v in acc.values()]))
    out = dict(n_problems=len(ids), contrasts={})
    for a, b in (("S", "L"), ("S", "base"), ("L", "base")):
        d = lambda X, cap: (lambda I: 100 * sum(X[cap, a][i] - X[cap, b][i] for i in I) / len(I))
        out["contrasts"][f"{a}-{b}"] = {
            "acc_8k": boot(ids, d(acc, "8k")), "acc_16k": boot(ids, d(acc, "16k")),
            "acc_8k_minus_16k": boot(ids, lambda I: d(acc, "8k")(I) - d(acc, "16k")(I)),
            "finish_8k": boot(ids, d(fin, "8k")), "finish_16k": boot(ids, d(fin, "16k"))}
    return out


def replicates(fmts, arms=("S", "R", "L"), n_min=490, seed=0):
    """Training replicates (independent fine-tuning runs per arm): fmts = {replicate: tag format with {a} and {s}}.
    Hierarchical paired bootstrap: resample replicates, then problems; the statistic is the mean over the drawn replicates
    of the per-problem (seed-mean) arm difference. Also a replicate-level t-test (df = n_rep - 1)."""
    acc, tok = {}, {}
    for rep, fmt in fmts.items():
        for a in arms:
            rr = [R for R in (load(fmt.format(a=a, s=s)) for s in range(5)) if len(R) >= n_min]
            if not rr:
                return None
            ids = set.intersection(*[set(R) for R in rr])
            acc[rep, a] = {i: sum(R[i]["acc"] for R in rr) / len(rr) for i in ids}
            tok[rep, a] = (len(rr), [round(100 * sum(R[i]["acc"] for i in ids) / len(ids), 2) for R in rr],
                           round(sum(R[i]["mean_tokens"] for R in rr for i in ids) / (len(rr) * len(ids))))
    reps = list(fmts)
    ids = sorted(set.intersection(*[set(v) for v in acc.values()]))
    out = dict(n_problems=len(ids), replicates={}, contrasts={})
    for rep in reps:
        out["replicates"][rep] = {a: dict(n_seeds=tok[rep, a][0], seeds=tok[rep, a][1],
                                          acc=round(100 * sum(acc[rep, a][i] for i in ids) / len(ids), 3), tokens=tok[rep, a][2])
                                  for a in arms}
    rng = random.Random(seed)
    for a, b in (("S", "R"), ("S", "L"), ("R", "L")):
        per_rep = [100 * sum(acc[r, a][i] - acc[r, b][i] for i in ids) / len(ids) for r in reps]
        n = len(per_rep); m = sum(per_rep) / n
        sd = (sum((x - m) ** 2 for x in per_rep) / (n - 1)) ** 0.5
        t = m / (sd / n ** 0.5) if sd > 0 else float("inf")
        p_t = (1 - abs(t) / (2 + t * t) ** 0.5) if n == 3 else None          # exact two-sided P for Student t with df = 2
        bs = []
        for _ in range(B):
            rs = [rng.choice(reps) for _ in reps]; I = [rng.choice(ids) for _ in ids]
            bs.append(sum(100 * sum(acc[r, a][i] - acc[r, b][i] for i in I) / len(I) for r in rs) / len(rs))
        bs.sort(); p1 = sum(x <= 0 for x in bs) / B
        out["contrasts"][f"{a}-{b}"] = dict(per_replicate=[round(x, 3) for x in per_rep], mean=round(m, 3), sd=round(sd, 3),
                                            t=round(t, 2), p_t_two_sided=None if p_t is None else round(p_t, 4),
                                            hier_boot=dict(lo=round(bs[int(0.025 * B)], 3), hi=round(bs[int(0.975 * B)], 3),
                                                           p=round(p1, 4), p_two_sided=round(2 * min(p1, 1 - p1), 4)))
    return out


C = (("S", "base"), ("R", "base"), ("L", "base"), ("S", "R"), ("S", "L"), ("R", "L"), ("S2", "S"), ("S2", "base"))
stats = dict(
    arc=arc(),
    math_qwen4b=single("math_eval_{a}_s{s}", ["S", "R", "L", "base", "S2"], 490, C),
    math_llama8b=single("math_l8_eval_{a}_s{s}", ["S", "R", "L", "base"], 490, C),
    apps_qwen4b=single("code_eval_{a}_s{s}", ["S", "R", "L", "base"], 490, C),
    math_qwen4b_cap16k=single("math16_eval_{a}_s{s}", ["S", "L", "base"], 490, C),
    cap_interaction_qwen4b=cap_interaction("math_eval_{a}_s{s}", "math16_eval_{a}_s{s}"),
    replicates_qwen4b=replicates({"rep0": "math_eval_{a}_s{s}", "rep1": "math_eval_{a}rep1_s{s}", "rep2": "math_eval_{a}rep2_s{s}"}),
    notes=dict(apps="finish/correct_given_finished unreliable: the code harvester's cut field also counts samples without a code block",
               arc_units="solved and its contrasts in tasks out of n_tasks; cut/code in % of assistant messages; restart cost = "
                         "E[min(L,24k)] / P(code) in thousand tokens per code-carrying message"))
json.dump(stats, sys.stdout, indent=1)
