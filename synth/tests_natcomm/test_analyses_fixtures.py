"""Synthetic fixtures for synth/natcomm_analyses.py (no real data): a repository root with runs_synth/agent_<tag>/
shard_*.jsonl eval records for every single-sample setting and the ARC agent, harvest records with verified
trajectories, id lists, sft_student caches and logs. Used by the test_analyses_*.py tests; also a sanity test of the
generator itself.

The single-sample generator draws, per problem, a difficulty d; per sample a length L = exp(N(log mu_d + shift_arm,
0.5)) (or, with probability `loop`, a non-terminating sample that runs to the cap); a sample is cut when L >= cap,
otherwise correct with probability q_d + dq_arm. Records carry exactly the fields of synth/math_harvest.py's eval
records: task_id, acc, tokens, mean_tokens, cut, pass2, level (APPS: difficulty, and a cut field that also counts
finished samples without a code block, as paper_stats.py's notes describe).
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH = os.path.dirname(HERE)
sys.path.insert(0, SYNTH)

ARMS = {"base": (0.0, 0.0, 0.0), "S": (-0.15, -0.005, 0.0), "R": (-0.04, 0.0, 0.0), "L": (0.12, 0.004, 0.0),
        "S2": (-0.25, -0.006, 0.0)}          # arm: (log-length shift, change in P(correct | finished), loop probability)


def write_jsonl(path, rows, shards=2):
    os.makedirs(path, exist_ok=True)
    for k in range(shards):
        with open(os.path.join(path, f"shard_{k}.jsonl"), "w") as fh:
            for r in rows[k::shards]:
                fh.write(json.dumps(r) + "\n")


def problems(n, seed, apps=False):
    rng = np.random.default_rng(seed)
    d = rng.random(n)
    pid = [f"te_{k:04d}" for k in range(n)]
    lev = [(["introductory", "interview"][int(x > 0.5)] if apps else int(1 + np.floor(5 * x))) for x in d]
    return pid, d, lev


def single_records(n=60, seed=0, arm="base", eval_seed=0, cap=8192, K=4, apps=False, arms=ARMS, log_mu=7.2, slope=2.0,
                   q0=0.985, qslope=0.06, nocode=0.06):
    pid, d, lev = problems(n, 1000 + seed, apps)
    shift, dq, loop = arms[arm]
    rng = np.random.default_rng([seed, eval_seed, sum(map(ord, arm))])
    rows = []
    for i in range(n):
        L = np.exp(rng.normal(log_mu + slope * d[i] + shift, 0.5, K))
        L = np.where(rng.random(K) < loop, cap, np.minimum(np.round(L), cap)).astype(int)
        cut = L >= cap
        ok = (~cut) & (rng.random(K) < q0 - qslope * d[i] + dq)
        c = int(cut.sum())
        if apps:      # the harvester's cut field also counts finished samples without a code block
            noc = (~cut) & (rng.random(K) < nocode)
            ok &= ~noc
            c = int((cut | noc).sum())
        r = dict(task_id=pid[i], acc=float(ok.mean()), tokens=int(L.sum()), mean_tokens=float(L.mean()), cut=c,
                 pass2=bool(ok[0]))
        r["difficulty" if apps else "level"] = lev[i]
        rows.append(r)
    return rows


SETTINGS = {   # fixture setting -> (tag format, arms, seeds, cap, apps, generator seed)
    "math_qwen4b": ("math_eval_{a}_s{s}", ("S", "R", "L", "base"), range(5), 8192, False, 0),
    "math_qwen4b_S2": ("math_eval_{a}_s{s}", ("S2",), range(3), 8192, False, 0),
    "math_llama8b": ("math_l8_eval_{a}_s{s}", ("S", "R", "L", "base"), range(3), 8192, False, 1),
    "apps_qwen4b": ("code_eval_{a}_s{s}", ("S", "R", "L", "base"), range(3), 8192, True, 2),
    "math_qwen4b_cap16k": ("math16_eval_{a}_s{s}", ("S", "L", "base"), range(2), 16384, False, 0),
    "rep1": ("math_eval_{a}rep1_s{s}", ("S", "R", "L"), range(2), 8192, False, 0),
    "rep2": ("math_eval_{a}rep2_s{s}", ("S", "R", "L"), range(2), 8192, False, 0),
}
LLAMA_ARMS = dict(ARMS, R=(-0.10, 0.0, 0.03))      # random replay: shorter, plus a non-terminating tail (crossing)


def build_single(root, n=60):
    for key, (fmt, arms, seeds, cap, apps, gseed) in SETTINGS.items():
        for a in arms:
            for s in seeds:
                es = s + (100 * int(key[-1]) if key.startswith("rep") else 0)
                rows = single_records(n, gseed, a, es, cap, apps=apps, arms=LLAMA_ARMS if key == "math_llama8b" else ARMS)
                write_jsonl(os.path.join(root, "runs_synth", f"agent_{fmt.format(a=a, s=s)}"), rows)


# ------------------------------------------------------------------------------------------------ ARC agent
ARC_P = {"base": 0.25, "S": 0.33, "R": 0.30, "L": 0.29}
ARC_CUT = {"base": 0.60, "S": 0.52, "R": 0.62, "L": 0.63}


def arc_records(n=24, arm="S", seed=0, cap=24000, rounds=6, log=True, seed_shift=0.0):
    rng = np.random.default_rng([7, seed, sum(map(ord, arm))])
    base_p = np.random.default_rng(3).beta(0.6, 1.4, n)
    rows = []
    for i in range(n):
        p = min(1.0, base_p[i] * ARC_P[arm] / 0.3 + seed_shift)
        solved = bool(rng.random() < p)
        msgs, k = [], 0
        while k < rounds:
            k += 1
            cut = bool(rng.random() < ARC_CUT[arm])
            tok = cap if cut else int(rng.integers(500, cap - 1000))
            code = (not cut) and bool(rng.random() < 0.96)
            msgs.append([tok, int(tok * 2.3), 0 if cut else 900, cut, code])
            if solved and code and rng.random() < 0.5:
                break
        r = dict(task_id=f"arc{i:03d}", sampled=1, consistent=int(solved), seconds=float(rng.integers(300, 3000)),
                 tokens=int(sum(m[0] for m in msgs)), explores=int(sum(m[4] for m in msgs)) - int(solved),
                 finals=int(solved), status={"consistent" if solved else "out of rounds": 1},
                 pass1=solved, pass2=solved, oracle=solved)
        if log:
            r["msg_log"] = [msgs]
        rows.append(r)
    return rows


def build_arc(root, n=24, base_seed_shift=(0.15, -0.1, 0.1, -0.1, -0.15)):
    for a in ("S", "R", "L", "base"):
        for s in range(5):
            tag = f"brevcl_{a}_c72_s{s}" if a != "base" else f"brevctl_base_c72_s{s}"
            rows = arc_records(n, a, s, log=not (a == "base" and s < 2),
                               seed_shift=base_seed_shift[s] if a == "base" else 0.0)
            write_jsonl(os.path.join(root, "runs_synth", f"agent_{tag}"), rows, shards=4)
    os.makedirs(os.path.join(root, "synth"), exist_ok=True)
    ids = [f"arc{i:03d}" for i in range(n)]
    open(os.path.join(root, "synth", "eval40_dev.txt"), "w").write("\n".join(ids[: n // 2]) + "\n")
    open(os.path.join(root, "synth", "brev_excl72.txt"), "w").write("\n".join(ids + ["trainX"]) + "\n")
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    open(os.path.join(root, "logs", "chain_brevcl_S_c72_s0.log"), "w").write(
        "2026-10-19_10:00:00 iter 1: 0/24 tasks done\n  submitted: 111 112\n2026-10-19_10:40:00 iter 2: 24/24 tasks done\n")
    open(os.path.join(root, "logs", "agent_dbg-111_s0.out"), "w").write(
        "INFO Initializing a V1 LLM engine (v0.11.2) with config: ...\nengine up; speculative=False\n"
        "chat_template_kwargs: {'enable_thinking': True, 'preserve_thinking': False}\n"
        "    round 3: llm.chat failed (ValueError: x); retrying thread by thread\n")


# ------------------------------------------------------------------------------------------------ harvests
def math_harvest_rows(n=30, seed=0, K=4):
    rng = np.random.default_rng([11, seed])
    rows = []
    for i in range(n):
        nv = int(rng.integers(0, K + 1))
        trs = []
        for k in range(nv):
            tok = int(rng.integers(300, 8000))
            body = f"<think>{'x' * (tok // 2)}</think> answer {i} v{k}"
            if k == 1 and i % 5 == 0:           # an exact duplicate of the first verified sample
                tok, body = trs[0]["tokens"], trs[0]["turns"][-1]["content"]
            trs.append(dict(rounds=1, tokens=tok, turns=[{"role": "user", "content": f"problem {i}"},
                                                         {"role": "assistant", "content": body}]))
        rows.append(dict(task_id=f"tr_{i:04d}", acc=nv / K, tokens=0, mean_tokens=0, cut=0, pass2=nv > 0, level="Level 3",
                         trajectories=trs))
    return rows


def arc_harvest_rows(prefix, n=8, seed=0, cap=24000):
    rng = np.random.default_rng([13, seed])
    rows = []
    for i in range(n):
        trs, threads = [], []
        for k in range(int(rng.integers(1, 4))):
            turns = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "grids " * 200}]
            log = []
            nm = int(rng.integers(1, 4))
            for j in range(nm):
                cut = j < nm - 1 and bool(rng.random() < 0.5)
                tok = cap if cut else int(rng.integers(800, 9000))
                raw = ("t" * int(tok * 2.5)) if cut else ("t" * int(tok * 2.0) + "</think>```python\n# FINAL\n```")
                turns.append({"role": "assistant", "content": raw})
                log.append([tok, len(raw), 0 if cut else 30, cut, not cut])
                if j < nm - 1:
                    turns.append({"role": "user", "content": ("Your message was cut off before any code (token limit)."
                                                              if cut else "Output:\n1")})
            trs.append(dict(rounds=nm, tokens=int(sum(m[0] for m in log)), turns=turns))
            threads.append(log)
        threads.insert(0, [[cap, 1, 0, True, False]] * 6)      # an unsuccessful thread: not a trajectory
        rows.append(dict(task_id=f"{prefix}{i:03d}", trajectories=trs, msg_log=threads, pass2=True))
    return rows


def build_harvests(root):
    write_jsonl(os.path.join(root, "runs_synth", "agent_math_harv_4b"), math_harvest_rows(30, 0))
    write_jsonl(os.path.join(root, "runs_synth", "agent_math_harv_4b_S"), math_harvest_rows(30, 1))
    write_jsonl(os.path.join(root, "runs_synth", "agent_math_harv_l8"), math_harvest_rows(30, 2))
    write_jsonl(os.path.join(root, "runs_synth", "agent_code_harv_4b"), math_harvest_rows(30, 3))
    write_jsonl(os.path.join(root, "runs_synth", "agent_harv_b"), arc_harvest_rows("hb", 8, 0))
    write_jsonl(os.path.join(root, "runs_synth", "agent_harvest_q38"), arc_harvest_rows("hq", 6, 1))
    write_jsonl(os.path.join(root, "runs_synth", "agent_harv_eval80_hard"), arc_harvest_rows("arc", 4, 2))   # 'arc000' excluded
    write_jsonl(os.path.join(root, "runs_synth", "agent_harv_nvarc80"), arc_harvest_rows("arc001_", 3, 3))  # NVARC seeds of arc001


def build_root(root, n=60, n_arc=24):
    build_single(root, n)
    build_arc(root, n_arc)
    build_harvests(root)
    return root


class FixtureSanity(unittest.TestCase):
    def test_single_records_shape(self):
        rows = single_records(20, 0, "S", 0)
        self.assertEqual(len(rows), 20)
        for r in rows:
            self.assertAlmostEqual(r["mean_tokens"] * 4, r["tokens"])
            self.assertTrue(0 <= r["cut"] <= 4)
            self.assertTrue(r["tokens"] >= r["cut"] * 8192)
            self.assertTrue(r["acc"] <= (4 - r["cut"]) / 4 + 1e-12)

    def test_apps_cut_counts_no_code(self):
        rows = single_records(200, 2, "base", 0, apps=True)
        self.assertTrue(any(r["cut"] > r["tokens"] // 8192 for r in rows))


if __name__ == "__main__":
    unittest.main()
