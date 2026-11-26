#!/usr/bin/env python3
"""Analyses of the EXISTING brevity-paper data for the Nature Communications revision: the editor's must-fix items 4, 5,
9, 10 and 13 of the blind review of 2026-11-01 (paper_brevity/natcomm/review_20261101/editor.md). No GPU, no new runs.

    python synth/natcomm_analyses.py                        # Leonardo, repository root (records in runs_synth/)
    python synth/natcomm_analyses.py --only item5_arc        # one section; the other sections of the output are kept
    env -u PYTHONPATH .venv_mps/bin/python synth/natcomm_analyses.py   # locally: the parts that need no records

Needs numpy (and torch only to read sft_student's caches); about 40 s and < 1 GB with B = 10,000 on records of the real
size (measured on synthetic fixtures). Tests: synth/tests_natcomm/test_analyses_*.py (unittest).
Writes paper_brevity/data/natcomm_stats.json (it refuses to write a file called paper_stats.json) and prints a summary.
A section is computed when its inputs exist; otherwise the version already in the output file is kept (or the section
is marked as skipped), so a local run (message logs, summary statistics) and a cluster run (records, harvests) fill the
same file. meta.sections records where, when and from which inputs each section was computed.

INPUTS
  records   runs_synth/agent_<tag>/shard_*.jsonl, read with paper_stats.load (compiled from synth/paper_stats.py without
            running that script's module-level analyses); the tags of paper_stats.py. Eval records hold per-problem
            aggregates: acc (mean over K samples), tokens (sum over the K samples), mean_tokens, cut (samples stopped at
            the cap), pass2, level/difficulty; ARC records hold pass2, tokens, seconds, status and msg_log.
  harvests  the harvest records distill/sft_student.py trained on (item 9), and, if present, sft_student's own caches of
            the tokenised training examples ($WORK/ccapone0/students/<arm>/samples_cache.pt) and its logs (logs/).
  summary   paper_brevity/data/paper_stats.json (consistency checks, fallbacks) and paper_brevity/data/arc_msgs.json.

CONVENTIONS (those of paper_stats.py and the paper's Methods)
  Per problem, outcomes are averaged over seeds; arm contrasts are paired over problems. Bootstrap: B = 10,000
  resamples of the problems, drawn exactly as paper_stats.boot draws them (random.Random(0), one draw per problem and
  resample), so a contrast that paper_stats.py also computes gets the very same resamples; 95% CI = the sorted
  resamples at positions int(0.025 B) and int(0.975 B); one-sided p = share of resamples <= 0 (resampled values are
  rounded to 1e-9 first, so float noise cannot move an exact zero); two-sided P = 2 min(p, 1 - p). With discrete
  outcomes some resamples are exactly 0: their share is p_tie (paper_stats.py's p lies in [p - p_tie, p]) and
  P_ties = min(1, 2 min(P(<= 0), P(>= 0))) is the tie-aware two-sided P that the multiplicity families use.
  Next to every bootstrap P, a paired sign-flip (randomisation) P: 10,000 random relabellings of the two arms within
  each problem (for ratio statistics the whole problem is swapped and the statistic recomputed); P_perm is two-sided
  (|T*| >= |T|), p_perm one-sided (T* >= T), both with the observed labelling counted (add-one). Interval statistics
  that are not antisymmetric in the arms (relative changes) take the P of their log ratio.

SECTIONS
  item4_quality       per single-sample setting (MATH-500 Qwen3-4B 8k and 16k, R1-Llama-8B, APPS, Qwen3-4B training
                      replicates 2 and 3): consistency of the aggregates (cap_check: all-cut records at K x cap tokens,
                      records proving that a cut sample was graded correct, which would make Q = A / F an upper bound);
                      arm summaries with CIs (incl. the share of samples that finish wrong, the Q
                      ceiling); every contrast's accuracy, finishing, quality and the accounting split
                      dA = dF Qbar + dQ Fbar with the product terms bootstrapped directly; (a) accuracy on the fixed set
                      of problems that both arms (and all arms) finish on every sample of every seed; (b) contrasts by
                      the untouched model's finishing rate (cross-fitted for contrasts that involve the untouched model)
                      and by MATH level / APPS difficulty; (c) the per-problem split; (d) principal-stratification
                      (Lee) bounds on dQ among samples that finish under both arms: pooled, within level, within problem.
  item5_arc           ARC-AGI-2 agent: per-seed diagnostics (records, files, chain and agent logs); every Table 1
                      contrast by puzzle bootstrap + sign flip, crossed puzzle x seed (two-way) bootstrap and exact
                      seed-level permutation tests (unpaired, and paired by seed index); seed heterogeneity (Cochran's
                      Q, exact conditional Monte Carlo P); 40 development vs fresh puzzles; relative gains with CIs;
                      message statistics on all messages and on the first message of each puzzle x seed (an outcome-
                      independent unit), by message index, and the time to the first program (Kaplan-Meier).
  item5_arc_summary   the seed-level tests that need only the seed totals of paper_stats.json (local).
  item5_arc_messages_local   arc_msgs.json (local): message-level statistics with message-level CIs and two-sample
                      permutation P (messages carry no puzzle or seed labels there; the clustered versions are in
                      item5_arc), and the design effects implied by the puzzle-clustered intervals of paper_stats.json.
  item9_training_sets per setting and arm: verified solutions per problem, problems with >= 2 distinct verified
                      solutions (the only ones where S, R and L can differ), how often the arms pick the same solution,
                      target lengths, examples, cut turns (ARC), loss tokens, optimiser steps, loss normalisation; exact
                      from sft_student's caches/logs when present, else reconstructed from the harvests.
  item10_finishing    APPS finishing recomputed from token counts (exact bounds; the cut flag's inconsistencies; exact
                      finishing and its accounting split if the records carry per-sample lengths, as the updated
                      harvesters write them); length-distribution crossing (capped mean vs cut rate, intersection-union
                      test, per-problem asymmetry), with the R1-Llama random-replay vs untouched case and where, by the
                      untouched model's finishing rate, the extra cuts arise.
  item13_*            cap interactions (acc, finishing, logit scale, tokens) with sign-flip P; replicates (paper_stats
                      hierarchical bootstrap reproduced + sign flips); a registry of all runs; token and cost claims
                      with CIs; the confirmatory family (pooled S-R and S-L, Stouffer, with leave-one-out and partial-
                      conjunction sensitivity); Holm and Benjamini-Hochberg within every secondary family.
"""
import argparse
import ast
import datetime
import glob
import hashlib
import itertools
import json
import math
import os
import platform
import random
import re
import socket
import sys
import warnings
from collections import Counter
from statistics import NormalDist

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ND = NormalDist()


# ============================================================================ reuse of synth/paper_stats.py
def compile_from_paper_stats(names, path=os.path.join(HERE, "paper_stats.py")):
    """paper_stats.py computes and prints all its statistics at import time, so it is not imported: the named top-level
    definitions, and the imports they use, are compiled from its source unchanged and without side effects."""
    with open(path) as fh:
        tree = ast.parse(fh.read(), path)
    keep = [nd for nd in tree.body
            if isinstance(nd, (ast.Import, ast.ImportFrom))
            or (isinstance(nd, ast.FunctionDef) and nd.name in names)
            or (isinstance(nd, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in nd.targets))]
    ns = {"__name__": "paper_stats_defs", "__file__": path}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    missing = [n for n in names if n not in ns]
    if missing:
        raise RuntimeError(f"{path} does not define {missing}")
    return ns


_PS = compile_from_paper_stats(("B", "load"))
load = _PS["load"]          # task_id -> first record in runs_synth/agent_<tag>/shard_*.jsonl (relative to the cwd)
B = _PS["B"]                # bootstrap resamples (10,000)
B_PERM = 10000              # sign-flip / permutation draws
N_MIN = 490                 # paper_stats.py: a single-sample run counts when it has >= 490 problems
BOOT_SEED, PERM_SEED, TWOWAY_SEED = 0, 20261102, 7

# ============================================================================ settings (tags as in paper_stats.py)
SINGLE = {   # per-arm tag formats; cut: 'length' = the cut field counts samples stopped at the cap
    "math_qwen4b": dict(arms={a: f"math_eval_{a}_s{{s}}" for a in ("S", "R", "L", "base", "S2")}, cap=8192, field="level"),
    "math_llama8b": dict(arms={a: f"math_l8_eval_{a}_s{{s}}" for a in ("S", "R", "L", "base")}, cap=8192, field="level"),
    "apps_qwen4b": dict(arms={a: f"code_eval_{a}_s{{s}}" for a in ("S", "R", "L", "base")}, cap=8192, field="difficulty",
                        cut="length_or_no_code"),
    "math_qwen4b_cap16k": dict(arms={a: f"math16_eval_{a}_s{{s}}" for a in ("S", "L", "base")}, cap=16384, field="level"),
    "math_qwen4b_rep1": dict(arms={**{a: f"math_eval_{a}rep1_s{{s}}" for a in "SRL"}, "base": "math_eval_base_s{s}"},
                             cap=8192, field="level"),
    "math_qwen4b_rep2": dict(arms={**{a: f"math_eval_{a}rep2_s{{s}}" for a in "SRL"}, "base": "math_eval_base_s{s}"},
                             cap=8192, field="level"),
}
MAIN_SINGLE = ("math_qwen4b", "math_llama8b", "apps_qwen4b", "math_qwen4b_cap16k")
CONTRASTS = (("S", "base"), ("R", "base"), ("L", "base"), ("S", "R"), ("S", "L"), ("R", "L"), ("S2", "S"), ("S2", "base"))
STRATA_CONTRASTS = (("S", "R"), ("S", "L"), ("S", "base"), ("L", "base"), ("R", "base"))
FINISH_DEF = {"length": "a sample finishes when it ends before the cap (the cut field counts samples stopped at the cap)",
              "length_or_no_code": "the harvester's cut field counts samples stopped at the cap and, per paper_stats.py's "
                                   "notes, finished samples without a code block, so here 'finish' = deliver a program "
                                   "within the cap (see item10_finishing for finishing from token counts)"}
ARC = dict(arms={**{a: f"brevcl_{a}_c72_s{{s}}" for a in "SRL"}, "base": "brevctl_base_c72_s{s}"}, cap=24000, rounds=6,
           dev="synth/eval40_dev.txt", excl="synth/brev_excl72.txt")
ARC_CONTRASTS = (("S", "base"), ("R", "base"), ("L", "base"), ("S", "R"), ("S", "L"), ("R", "L"))
SELECTION_SETTINGS = (("arc", "solved"), ("math_qwen4b", "acc"), ("math_llama8b", "acc"), ("apps_qwen4b", "acc"))
QWEN_END, LLAMA_END = "<|im_end|>\n", "<｜end▁of▁sentence｜>"
ARC_DATA = ("runs_synth/agent_harv_b/shard_*.jsonl,runs_synth/agent_harvest_q38/shard_*.jsonl,"
            "runs_synth/agent_harv_eval80_hard/shard_*.jsonl,runs_synth/agent_harv_nvarc80/shard_*.jsonl")
TRAIN = {   # the training sets of every evaluated arm, with the flags of the leonardo/run_*.sh scripts (local copies)
    "math_qwen4b": dict(data="runs_synth/agent_math_harv_4b/shard_*.jsonl", seed=0, accum=2, world=4, max_len=12288,
                        arms={"S": ("shortest", "math4b_S"), "R": ("random", "math4b_R"), "L": ("longest", "math4b_L")},
                        sft_end=QWEN_END, end_tokens=2, system_prompt_added=True, script="leonardo/run_math_brev.sh"),
    "math_qwen4b_rep1": dict(data="runs_synth/agent_math_harv_4b/shard_*.jsonl", seed=1, accum=2, world=4, max_len=12288,
                             arms={a: (s, f"math4b_{a}_rep1") for a, s in (("S", "shortest"), ("R", "random"), ("L", "longest"))},
                             sft_end=QWEN_END, end_tokens=2, system_prompt_added=True, script="leonardo/run_math_reps.sh 1"),
    "math_qwen4b_rep2": dict(data="runs_synth/agent_math_harv_4b/shard_*.jsonl", seed=2, accum=2, world=4, max_len=12288,
                             arms={a: (s, f"math4b_{a}_rep2") for a, s in (("S", "shortest"), ("R", "random"), ("L", "longest"))},
                             sft_end=QWEN_END, end_tokens=2, system_prompt_added=True, script="leonardo/run_math_reps.sh 2"),
    "math_qwen4b_S2": dict(data="runs_synth/agent_math_harv_4b_S/shard_*.jsonl", seed=0, accum=2, world=4, max_len=12288,
                           arms={"S2": ("shortest", "math4b_S2")}, sft_end=QWEN_END, end_tokens=2, system_prompt_added=True,
                           script="leonardo/run_paper_extra.sh (lane A)"),
    "math_llama8b": dict(data="runs_synth/agent_math_harv_l8/shard_*.jsonl", seed=0, accum=2, world=4, max_len=12288,
                         arms={"S": ("shortest", "mathl8_S"), "R": ("random", "mathl8_R"), "L": ("longest", "mathl8_L")},
                         sft_end=LLAMA_END, end_tokens=1, system_prompt_added=False, script="leonardo/run_math_llama.sh"),
    "apps_qwen4b": dict(data="runs_synth/agent_code_harv_4b/shard_*.jsonl", seed=0, accum=2, world=4, max_len=12288,
                        arms={"S": ("shortest", "code4b_S"), "R": ("random", "code4b_R"), "L": ("longest", "code4b_L")},
                        sft_end=QWEN_END, end_tokens=2, system_prompt_added=True, script="leonardo/run_paper_extra.sh (lane B)"),
    "arc": dict(data=ARC_DATA, seed=0, accum=8, world=1, max_len=24576, task_cap=15000, exclude="synth/brev_excl72.txt",
                arms={"S": ("shortest", "brevSc"), "R": ("random", "brevRc"), "L": ("longest", "brevLc")},
                sft_end=QWEN_END, end_tokens=2, system_prompt_added=False, script="leonardo/run_brev_clean.sh"),
}
ARC_SFT_SYSTEM = "You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles by writing Python."
CUT_NOTICE = "Your message was cut off"


# ============================================================================ small utilities
def two_sided(p):
    return 2 * min(p, 1 - p)


def sig(x, n=3):
    """n significant digits (for P values that can be far below 0.001)."""
    return None if x is None or not math.isfinite(x) else float(f"{x:.{n}g}")


def rnd(x, nd=3):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def read_ids(path):
    return [l.strip() for l in open(path) if l.strip()] if path and os.path.exists(path) else None


def level_label(v):
    if v is None:
        return "unknown"
    m = re.search(r"\d+", str(v))
    return m.group(0) if m else str(v)


# ---------------------------------------------------------------- special functions (no scipy on the cluster venv)
def gammaincc(a, x):
    """Regularised upper incomplete gamma Q(a, x) (series / continued fraction, Numerical Recipes 6.2)."""
    if x <= 0:
        return 1.0
    lg = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        ap, s = a, 1.0 / a
        d = s
        for _ in range(100000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(lg))
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return min(1.0, math.exp(lg) * h)


def chi2_sf(x, df):
    return gammaincc(df / 2.0, x / 2.0) if df > 0 else float("nan")


def holm(ps):
    m = len(ps)
    adj, run = [None] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: ps[j])):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def bh(ps):
    m = len(ps)
    adj, run = [None] * m, 1.0
    for k, i in enumerate(sorted(range(m), key=lambda j: ps[j], reverse=True)):
        run = min(run, ps[i] * m / (m - k))
        adj[i] = min(1.0, run)
    return adj


def stouffer(tails):
    """One-sided tails P(diff <= 0) -> (z, one-sided P, two-sided P); tails clipped to [1/B, 1 - 1/B] (derived_stats.py
    floors a tail of 0 at 1/B, the bootstrap's resolution)."""
    t = [min(max(p, 1 / B), 1 - 1 / B) for p in tails]
    z = sum(ND.inv_cdf(1 - p) for p in t) / math.sqrt(len(t))
    p1 = 1 - ND.cdf(z)
    return z, p1, two_sided(p1)


def partial_conjunction(tails, u):
    """Benjamini-Heller partial conjunction (Stouffer combination of the n - u + 1 largest one-sided tails): one-sided P
    for 'the effect is positive in at least u of the n settings'."""
    t = sorted(min(max(p, 1 / B), 1 - 1 / B) for p in tails)[u - 1:]
    return 1 - ND.cdf(sum(ND.inv_cdf(1 - p) for p in t) / math.sqrt(len(t)))


# ============================================================================ resampling machinery
_COUNTS, _FLIPS = {}, {}


def _cache_put(cache, key, val, cap=8):
    if len(cache) >= cap:
        cache.pop(next(iter(cache)))      # least recently used (hits are moved to the end)
    cache[key] = val
    return val


def _cache_get(cache, key):
    if key in cache:
        cache[key] = cache.pop(key)
        return cache[key]
    return None


def boot_indices(n, seed=BOOT_SEED, nboot=None):
    """(B, n) problem indices drawn as paper_stats.boot draws them: rng = random.Random(seed), then for each resample
    [rng.choice(ids) for _ in ids]; rng.choice(seq) is seq[rng._randbelow(len(seq))]."""
    nboot = nboot or B
    rng = random.Random(seed)
    rb = getattr(rng, "_randbelow", None) or (lambda m: rng.choice(range(m)))
    return np.fromiter((rb(n) for _ in range(nboot * n)), dtype=np.int64, count=nboot * n).reshape(nboot, n)


def boot_counts(n, seed=BOOT_SEED):
    """(B, n): how often each problem is drawn in each resample (paper_stats.boot's draws)."""
    key = (n, seed, B)
    C = _cache_get(_COUNTS, key)
    if C is None:
        idx = boot_indices(n, seed)
        flat = (idx + n * np.arange(B)[:, None]).ravel()
        C = _cache_put(_COUNTS, key, np.bincount(flat, minlength=B * n).reshape(B, n).astype(float))
    return C


def flip_masks(n):
    """(B_PERM, n) 0/1: 1 = swap the two arms' values for this problem (paired sign flip)."""
    key = (n, B_PERM)
    M = _cache_get(_FLIPS, key)
    if M is None:
        M = _cache_put(_FLIPS, key, np.random.default_rng([PERM_SEED, n]).integers(0, 2, size=(B_PERM, n)).astype(float))
    return M


def summarize(obs, bs, ts=None, nd=3):
    """paper_stats convention for one statistic: diff, lo, hi, p (share of resamples <= 0), P = 2 min(p, 1 - p); P_perm
    and p_perm from the sign-flip draws ts when given. Resamples whose value is exactly 0 (discrete outcomes) count as
    <= 0; their share is reported as p_tie (paper_stats.py's float sums put each of them on either side of 0, so its p
    lies in [p - p_tie, p]), with the tie-aware two-sided P_ties = min(1, 2 min(P(<= 0), P(>= 0))), which the
    multiplicity families use. When every resample is 0 there is nothing to test and P = 1."""
    obs = float(obs)
    if not math.isfinite(obs):
        return dict(diff=None, note="undefined (empty or zero denominator)")
    bs = np.round(np.asarray(bs, float), 9)
    good = bs[np.isfinite(bs)]
    if not len(good):
        return dict(diff=rnd(obs, nd), note="no defined resample")
    good = np.sort(good)
    m = len(good)
    p, p_ge = float(np.mean(good <= 0)), float(np.mean(good >= 0))
    out = dict(diff=rnd(obs, nd), lo=rnd(good[int(0.025 * m)], nd), hi=rnd(good[int(0.975 * m)], nd),
               p=round(p, 4), P=round(two_sided(p), 4))
    tie = p + p_ge - 1
    if tie > 0:
        out["p_tie"] = round(tie, 4)
        out["P_ties"] = round(min(1.0, 2 * min(p, p_ge)), 4)
        if tie >= 1:
            out["P"], out["note"] = 1.0, "no variation: every resample is 0"
    if m < len(bs):
        out["n_undefined_resamples"] = int(len(bs) - m)
    if ts is not None:
        ts = np.round(np.asarray(ts, float), 9)
        ts = ts[np.isfinite(ts)]
        o = round(obs, 9)
        out["P_perm"] = round((1 + int(np.sum(np.abs(ts) >= abs(o) - 1e-12))) / (1 + len(ts)), 4)
        out["p_perm"] = round((1 + int(np.sum(ts >= o - 1e-12))) / (1 + len(ts)), 4)
    return out


def resample(Xa, Xb, f, perm=True, seed=BOOT_SEED):
    """Statistic f of two arms' per-problem quantities: observed value, bootstrap draws, sign-flip draws.
    Xa, Xb: (k, n), row j of Xb = the same quantity as row j of Xa for the other arm (rows equal in both arms, such as
    stratum indicators, are left unchanged by a swap). f(Sa, Sb) maps column sums (shape (..., k)) to a statistic, or to
    m statistics (shape (..., m))."""
    Xa = np.atleast_2d(np.asarray(Xa, float))
    Xb = np.atleast_2d(np.asarray(Xb, float))
    n = Xa.shape[1]
    sa, sb = Xa.sum(1), Xb.sum(1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        obs = np.asarray(f(sa, sb), float)
        C = boot_counts(n, seed)
        bs = np.asarray(f(C @ Xa.T, C @ Xb.T), float)
        ts = None
        if perm:
            D = flip_masks(n) @ (Xb - Xa).T
            ts = np.asarray(f(sa + D, sb - D), float)
    return obs, bs, ts


def test2(Xa, Xb, f, perm=True, nd=3, seed=BOOT_SEED):
    obs, bs, ts = resample(Xa, Xb, f, perm, seed)
    return summarize(obs, bs, ts, nd)


def tests_multi(Xa, Xb, f, names, perm=True, nd=3, perm_from=None, seed=BOOT_SEED):
    """Several statistics from the same resamples. perm_from = {name: other}: name takes other's permutation P (a
    relative change takes the P of its log ratio). Names starting with '_' are helpers and are not returned."""
    obs, bs, ts = resample(Xa, Xb, f, perm, seed)
    out = {nm: summarize(obs[..., j], bs[:, j], None if ts is None else ts[:, j], nd) for j, nm in enumerate(names)}
    for nm, other in (perm_from or {}).items():
        if nm in out and other in out and "P_perm" in out[other]:
            out[nm]["P_perm"], out[nm]["p_perm"] = out[other]["P_perm"], out[other]["p_perm"]
            out[nm]["perm_of"] = "log ratio"
    return {k: v for k, v in out.items() if not k.startswith("_")}


def mean_test(d, scale=100.0, perm=True, nd=3):
    """Paired bootstrap and sign flip of the mean of per-problem values d (that change sign when the arms swap)."""
    d = np.asarray(d, float)
    n = len(d)
    return test2(d[None] / 2, -d[None] / 2, lambda Sa, Sb: scale * (Sa[..., 0] - Sb[..., 0]) / n, perm, nd)


def ci1(X, f, nd=3):
    """One arm: observed value and 95% bootstrap interval of f(column sums of X) over problems."""
    X = np.atleast_2d(np.asarray(X, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = float(f(X.sum(1)))
        bs = np.sort(np.round(np.asarray(f(boot_counts(X.shape[1]) @ X.T), float), 9))
    bs = bs[np.isfinite(bs)]
    if not len(bs) or not math.isfinite(obs):
        return dict(value=rnd(obs, nd))
    return dict(value=rnd(obs, nd), lo=rnd(bs[int(0.025 * len(bs))], nd), hi=rnd(bs[int(0.975 * len(bs))], nd))


# ============================================================================ single-sample settings (MATH, APPS)
class Single:
    """Records of one single-sample setting, arranged as (seed, problem) arrays on the problems common to every arm and
    seed (paper_stats.single: runs with >= N_MIN problems, five seeds at most)."""

    def __init__(self, key, cfg):
        self.key, self.cfg = key, cfg
        self.cap, self.field, self.cut_kind = cfg["cap"], cfg.get("field"), cfg.get("cut", "length")
        runs = {}
        for a, fmt in cfg["arms"].items():
            rr = [(s, R) for s, R in ((s, load(fmt.format(s=s))) for s in cfg.get("seeds", range(5))) if len(R) >= N_MIN]
            if rr:
                runs[a] = rr
        self.ok = "base" in runs and len(runs) >= 2
        if not self.ok:
            self.reason = f"runs found for arms {sorted(runs)}" if runs else "no records"
            return
        self.runs = runs
        self.arms = list(runs)
        self.ids = sorted(set.intersection(*[set(R) for rr in runs.values() for _, R in rr]))
        self.seeds = {a: [s for s, _ in rr] for a, rr in runs.items()}
        self.tags = {a: [cfg["arms"][a].format(s=s) for s, _ in rr] for a, rr in runs.items()}
        mat = lambda rr, k: np.array([[float(R[i].get(k) or 0) for i in self.ids] for _, R in rr])
        self.acc = {a: mat(rr, "acc") for a, rr in runs.items()}
        self.cut = {a: mat(rr, "cut") for a, rr in runs.items()}
        self.tok = {a: mat(rr, "tokens") for a, rr in runs.items()}
        mt = {a: mat(rr, "mean_tokens") for a, rr in runs.items()}
        ks = Counter(int(round(t / m)) for a in runs for t, m in zip(self.tok[a].ravel(), mt[a].ravel()) if m > 0)
        self.K = ks.most_common(1)[0][0] if ks else 4
        self.K_check = dict(ks)
        first = runs[self.arms[0]][0][1]
        self.labels = np.array([level_label(first[i].get(self.field)) for i in self.ids]) if self.field else None
        # the cap from the data: a record whose K samples were all stopped at the cap has tokens = K x cap
        allcut = [t / self.K for a in runs for t, c in zip(self.tok[a].ravel(), self.cut[a].ravel()) if c == self.K]
        self.cap_config, self.cap_from_data = self.cap, None
        if self.cut_kind == "length" and len(allcut) >= 3 and len(set(allcut)) == 1:
            self.cap_from_data = int(allcut[0])
            self.cap = self.cap_from_data
        K, cap = self.K, self.cap
        self.A = {a: v.mean(0) for a, v in self.acc.items()}
        self.F = {a: ((K - v) / K).mean(0) for a, v in self.cut.items()}
        self.T = {a: (v / K).mean(0) for a, v in self.tok.items()}
        self.FT = {a: ((self.tok[a] - self.cut[a] * cap) / K).mean(0) for a in runs}   # finished tokens per sample slot

    def base_F(self, seed_pos=None):
        c = self.cut["base"] if seed_pos is None else self.cut["base"][seed_pos]
        return ((self.K - c) / self.K).mean(0)

    def cap_check(self):
        """Consistency of the aggregates with 'cut = stopped at the cap' and 'a cut sample is wrong' (A = F Q exactly):
        all-cut records should have tokens = K cap; tokens < cut x cap is impossible for samples stopped at the cap;
        correct samples > finished samples proves that some cut samples were graded correct (math_harvest.boxed()
        falls back to the whole text when there is no </think>). The count is only a lower bound (a record with
        correct <= finished can contain such samples too); if it is not 0, Q = A / F is not P(correct | finished)
        and the per-sample fields of a re-evaluation are needed to separate the two."""
        out = {}
        for a in self.arms:
            c, t = self.cut[a], self.tok[a]
            allcut = c == self.K
            excess = np.maximum(0, np.round(self.acc[a] * self.K) - (self.K - c))
            out[a] = dict(records=int(c.size), all_samples_cut=int(allcut.sum()),
                          all_cut_with_tokens_eq_K_cap=int((allcut & (t == self.K * self.cap)).sum()),
                          tokens_below_cut_times_cap=int((t < c * self.cap).sum()),
                          max_tokens_per_sample=rnd(float((t / self.K).max()), 1),
                          records_correct_exceeds_finished=int((excess > 0).sum()),
                          min_cut_samples_graded_correct=int(excess.sum()),
                          min_share_samples_cut_but_correct=rnd(100 * float(excess.sum()) / (self.K * c.size), 4))
        return out

    def exact_capped(self, a):
        """(seeds, problems) samples stopped at the cap, from per-sample fields when every record has them (records
        written by the updated harvesters: 'cap_cut', or 'lens' with a sample at the cap counted as capped); else None."""
        rr = self.runs[a]
        if all("cap_cut" in R[i] for _, R in rr for i in self.ids):
            return np.array([[float(R[i]["cap_cut"]) for i in self.ids] for _, R in rr])
        if all(isinstance(R[i].get("lens"), list) for _, R in rr for i in self.ids):
            return np.array([[float(sum(x >= self.cap for x in R[i]["lens"])) for i in self.ids] for _, R in rr])
        return None


def arm_summary(D, a):
    n = len(D.ids)
    X = np.vstack([D.A[a], D.F[a], D.T[a], D.FT[a]])
    out = dict(n_seeds=len(D.seeds[a]), seeds=D.seeds[a],
               seed_acc=[round(100 * float(v.mean()), 2) for v in D.acc[a]],
               acc=ci1(X, lambda S: 100 * S[..., 0] / n), finish=ci1(X, lambda S: 100 * S[..., 1] / n),
               correct_given_finished=ci1(X, lambda S: 100 * S[..., 0] / S[..., 1]),
               finished_wrong=ci1(X, lambda S: 100 * (S[..., 1] - S[..., 0]) / n),
               tokens=ci1(X, lambda S: S[..., 2] / n, nd=1))
    if D.cut_kind == "length":
        out["finished_mean_tokens"] = ci1(X, lambda S: S[..., 3] / S[..., 1], nd=1)
    return out


CONTRAST_NAMES = ["acc", "finish", "correct_given_finished", "finishing_term", "quality_term", "tokens", "tokens_rel_pct",
                  "_tokens_log", "finished_mean_tokens", "finished_mean_tokens_rel_pct", "_finished_log", "cut_rate"]


def contrast_rows(D, x):
    return np.vstack([D.A[x], D.F[x], D.T[x], D.FT[x]])


def contrast_stats(Sa, Sb, n):
    """From sums over problems of (A, F, tokens per sample, finished tokens per sample) of two arms: the CONTRAST_NAMES
    statistics. Arm F and Q are means / ratios of means over the (resampled) problems, so the finishing and quality
    terms are the products recomputed in every resample, and they add up to the accuracy difference in every one."""
    Aa, Fa, Ta, Ga = (Sa[..., j] for j in range(4))
    Ab, Fb, Tb, Gb = (Sb[..., j] for j in range(4))
    Qa, Qb, fa, fb = Aa / Fa, Ab / Fb, Fa / n, Fb / n
    ma, mb = Ga / Fa, Gb / Fb
    return np.stack([100 * (Aa - Ab) / n, 100 * (fa - fb), 100 * (Qa - Qb),
                     100 * (fa - fb) * (Qa + Qb) / 2, 100 * (Qa - Qb) * (fa + fb) / 2,
                     (Ta - Tb) / n, 100 * (Ta / Tb - 1), np.log(Ta / Tb),
                     ma - mb, 100 * (ma / mb - 1), np.log(ma / mb), 100 * (fb - fa)], axis=-1)


def contrast_block(D, a, b):
    """Accuracy, finishing, quality (ratio of means), the accounting split with the products bootstrapped directly,
    tokens (difference, relative), mean length of finished samples, cut rate."""
    n = len(D.ids)
    out = tests_multi(contrast_rows(D, a), contrast_rows(D, b), lambda Sa, Sb: contrast_stats(Sa, Sb, n), CONTRAST_NAMES,
                      perm_from={"tokens_rel_pct": "_tokens_log", "finished_mean_tokens_rel_pct": "_finished_log"})
    if D.cut_kind != "length":
        for k in ("finished_mean_tokens", "finished_mean_tokens_rel_pct"):
            out.pop(k)
    out["split_check"] = rnd(out["finishing_term"]["diff"] + out["quality_term"]["diff"] - out["acc"]["diff"], 3)
    return out


def per_problem_split(Aa, Fa, Ab, Fb):
    """Per problem, dA = dF Qbar + dQ Fbar + resid (bars: means of the two arms). Q of an arm that finishes no sample
    of the problem is set to the other arm's Q (any value leaves dA unchanged; this one puts the whole difference in
    the finishing term, since there is no finished sample to compare); when neither finishes, both terms are 0.
    resid is 0 except on problems where an arm finishes no sample yet has accuracy > 0, i.e. capped samples graded
    correct (math_harvest.boxed() falls back to the whole text when there is no </think>): there it is that credit."""
    qa = np.divide(Aa, Fa, out=np.full(Aa.shape, np.nan), where=Fa > 0)
    qb = np.divide(Ab, Fb, out=np.full(Ab.shape, np.nan), where=Fb > 0)
    qa, qb = np.where(np.isnan(qa), qb, qa), np.where(np.isnan(qb), qa, qb)
    qa, qb = np.nan_to_num(qa), np.nan_to_num(qb)
    fin = (Fa - Fb) * (qa + qb) / 2
    qual = (qa - qb) * (Fa + Fb) / 2
    resid = (Aa - Ab) - (fin + qual)
    anomalous = ((Fa == 0) & (Aa > 0)) | ((Fb == 0) & (Ab > 0))
    assert np.allclose(np.where(anomalous, 0, resid), 0, atol=1e-12), "per-problem split is not exact"
    return fin, qual, resid


def per_problem_terms(Aa, Fa, Ab, Fb):
    """per_problem_split for data without capped samples graded correct: the two terms add up to dA exactly."""
    fin, qual, resid = per_problem_split(Aa, Fa, Ab, Fb)
    assert np.allclose(resid, 0, atol=1e-12), "per-problem split is not exact"
    return fin, qual


def per_problem_block(D, a, b):
    """Mean over problems of the per-problem finishing and quality terms (with the credit for capped samples graded
    correct on problems that an arm never finishes, they add up to the accuracy difference), bootstrapped and
    sign-flipped as per-problem values. Only problems on which both arms finish some sample can contribute to the
    quality term."""
    fin, qual, resid = per_problem_split(D.A[a], D.F[a], D.A[b], D.F[b])
    both = (D.F[a] > 0) & (D.F[b] > 0)
    anomalous = ((D.F[a] == 0) & (D.A[a] > 0)) | ((D.F[b] == 0) & (D.A[b] > 0))
    out = dict(finishing_term=mean_test(fin), quality_term=mean_test(qual),
               split_check=rnd(100 * float(np.mean(fin + qual + resid) - np.mean(D.A[a] - D.A[b])), 9),
               n_problems=len(D.ids), n_both_arms_finish_some=int(both.sum()),
               n_neither_finishes=int(((D.F[a] == 0) & (D.F[b] == 0)).sum()),
               n_never_finished_but_credited=int(anomalous.sum()))
    if anomalous.any():
        out["capped_credit_term"] = mean_test(resid)
    return out


def lee_masses(Aa, Fa, Ab, Fb):
    """Lee (2009) trimming bounds within a cell (a problem, a level, or all problems), in masses per problem: under
    monotonicity within the cell (the arm that finishes more often finishes every sample that the other finishes),
    the samples that finish under both arms ('always-finishers') have mass w = min(Fa, Fb). The higher-F arm's
    correct always-finisher mass lies in [max(0, A - (F_hi - F_lo)), min(A, F_lo)]; the lower-F arm's is its A.
    Returns w, lo, hi with dQ(a - b) among always-finishers in [sum lo / sum w, sum hi / sum w]."""
    a_hi = Fa >= Fb
    w = np.minimum(Fa, Fb)
    lo = np.where(a_hi, np.maximum(0, Aa - Fa + Fb) - Ab, Aa - np.minimum(Ab, Fa))
    hi = np.where(a_hi, np.minimum(Aa, Fb) - Ab, Aa - np.maximum(0, Ab - Fb + Fa))
    return w, lo, hi


def lee_block(D, a, b):
    """Bounds on dQ (points of Q among samples that finish under both arms), pooled (one cell), within MATH level /
    APPS difficulty (cells), within problem (500 cells; plug-in cell estimates from 8-20 samples bias the bounds
    inward). CI = [2.5th percentile of the lower bound, 97.5th of the upper] covers the identified set (conservative)."""
    n = len(D.ids)
    Aa, Fa, Ab, Fb = D.A[a], D.F[a], D.A[b], D.F[b]
    out = dict(higher_finishing_arm=a if Fa.sum() >= Fb.sum() else b)

    def bounds(Aa, Fa, Ab, Fb):          # masses (cells on the last axis) -> 100 x [lower, upper]
        w, lo, hi = lee_masses(Aa, Fa, Ab, Fb)
        return np.stack([100 * lo.sum(-1) / w.sum(-1), 100 * hi.sum(-1) / w.sum(-1)], axis=-1)

    def summ(obs, bs, w):
        lo, hi = np.sort(bs[:, 0][np.isfinite(bs[:, 0])]), np.sort(bs[:, 1][np.isfinite(bs[:, 1])])
        ci = [rnd(lo[int(0.025 * len(lo))]), rnd(hi[int(0.975 * len(hi))])] if len(lo) and len(hi) else [None, None]
        return dict(lower=rnd(obs[0]), upper=rnd(obs[1]), ci_identified_set=ci, always_finisher_share=rnd(100 * w / n),
                    excludes_zero=bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0)))

    cells = {"pooled": np.zeros(n, int)}
    if D.labels is not None:
        u = sorted(set(D.labels))
        cells["within_level"] = np.array([u.index(x) for x in D.labels])
    for name, lab in cells.items():
        L = int(lab.max()) + 1
        I = np.stack([(lab == k).astype(float) for k in range(L)])
        Xa, Xb = np.vstack([I * Aa, I * Fa]), np.vstack([I * Ab, I * Fb])
        obs, bs, _ = resample(Xa, Xb, lambda Sa, Sb, L=L: bounds(Sa[..., :L], Sa[..., L:], Sb[..., :L], Sb[..., L:]), perm=False)
        w = lee_masses(Xa[:L].sum(1), Xa[L:].sum(1), Xb[:L].sum(1), Xb[L:].sum(1))[0].sum()
        out[name] = summ(obs, bs, w)
    w, lo, hi = lee_masses(Aa, Fa, Ab, Fb)
    obs, bs, _ = resample(np.vstack([lo, hi, w]), np.zeros((3, n)),
                          lambda Sa, Sb: np.stack([100 * Sa[..., 0] / Sa[..., 2], 100 * Sa[..., 1] / Sa[..., 2]], -1), perm=False)
    out["within_problem"] = summ(obs, bs, w.sum())
    return out


def common_set_block(D):
    """Accuracy on the fixed set of problems that the compared arms finish on every sample of every seed (cut == 0);
    there A = Q. Pairs (each contrast's own set) and all arms of the setting (one set). Bootstrap over that set."""
    fin_all = {a: np.all(D.cut[a] == 0, axis=0) for a in D.arms}

    def on(mask, a, b):
        idx = np.where(mask)[0]
        m = len(idx)
        if m < 5:
            return dict(n_problems=m, note="fewer than 5 problems")
        r = test2(D.A[a][idx][None], D.A[b][idx][None], lambda Sa, Sb: 100 * (Sa[..., 0] - Sb[..., 0]) / m)
        return dict(n_problems=m, acc_a=rnd(100 * D.A[a][idx].mean()), acc_b=rnd(100 * D.A[b][idx].mean()), diff=r)

    allmask = np.all([fin_all[a] for a in D.arms], axis=0)
    out = dict(definition="problems with cut == 0 for every seed of each compared arm", arms_in_all_set=D.arms,
               n_all=int(allmask.sum()),
               acc_all={a: rnd(100 * D.A[a][allmask].mean()) if allmask.any() else None for a in D.arms},
               all_arms={}, pairs={})
    for a, b in CONTRASTS:
        if a in D.arms and b in D.arms:
            out["all_arms"][f"{a}-{b}"] = on(allmask, a, b)
            out["pairs"][f"{a}-{b}"] = on(fin_all[a] & fin_all[b], a, b)
    return out


FBINS = (("F=0", lambda f: f <= 1e-9), ("0<F<=0.5", lambda f: (f > 1e-9) & (f <= 0.5 + 1e-9)),
         ("0.5<F<1", lambda f: (f > 0.5 + 1e-9) & (f < 1 - 1e-9)), ("F=1", lambda f: f >= 1 - 1e-9))


def strata_block(D, a, b, kind):
    """Contrast a - b within strata. kind 'level': MATH level / APPS difficulty (pre-treatment). kind 'base_finish':
    the untouched model's finishing rate on the problem, in bins. For contrasts that involve the untouched model the
    strata are cross-fitted (fold 1: bins from the base seeds at even positions, base outcomes from the seeds at odd
    positions; fold 2 the reverse; per stratum the two folds' estimates are averaged), so that the stratifier and the
    outcome are independent samples (no regression to the mean). Domain estimates over all problems; bootstrap and
    sign flip over problems. Heterogeneity across strata: Cochran's Q with bootstrap SEs, df = strata - 1 (approximate
    when cross-fitted, since a problem can fall in different strata in the two folds)."""
    K = D.K
    folds = []    # (stratum label per problem, base seed positions for the outcomes or None = all, stratifier seeds)
    if kind == "level":
        if D.labels is None:
            return None
        folds = [(D.labels, None, None)]
    else:
        if "base" in (a, b):
            ss = list(range(len(D.seeds["base"])))
            h1, h2 = ss[0::2], ss[1::2]
            if not h2:
                return dict(note="one untouched seed: no cross-fitting possible")
            for hs, ho in ((h1, h2), (h2, h1)):
                lab = np.full(len(D.ids), "", dtype=object)
                fb = D.base_F(hs)
                for nm, g in FBINS:
                    lab[g(fb)] = nm
                folds.append((lab, ho, hs))
        else:
            lab = np.full(len(D.ids), "", dtype=object)
            fb = D.base_F()
            for nm, g in FBINS:
                lab[g(fb)] = nm
            folds.append((lab, None, None))
    names = [nm for nm, _ in FBINS] if kind == "base_finish" else sorted(set(D.labels), key=lambda x: (len(x), x))

    def arm_vals(x, ho):
        if x == "base" and ho is not None:
            return D.acc["base"][ho].mean(0), ((K - D.cut["base"][ho]) / K).mean(0), (D.tok["base"][ho] / K).mean(0)
        return D.A[x], D.F[x], D.T[x]

    out, est, ses = {}, [], []
    for k in names:
        Xa, Xb = [], []
        for lab, ho, _ in folds:
            I = (lab == k).astype(float)
            Aa, Fa, Ta = arm_vals(a, ho)
            Ab, Fb, Tb = arm_vals(b, ho)
            Xa += [I * Aa, I * Fa, I * Ta, I]
            Xb += [I * Ab, I * Fb, I * Tb, I]
        nf = len(folds)
        sizes = [int(((lab == k)).sum()) for lab, _, _ in folds]
        if max(sizes) < 3:
            out[k] = dict(n_problems=sizes, note="fewer than 3 problems")
            continue

        def f(Sa, Sb, nf=nf):
            vals = []
            for j in range(nf):
                Aa, Fa, Ta, m = (Sa[..., 4 * j + t] for t in range(4))
                Ab, Fb, Tb, _ = (Sb[..., 4 * j + t] for t in range(4))
                with np.errstate(divide="ignore", invalid="ignore"):
                    Qa, Qb, fa, fb = Aa / Fa, Ab / Fb, Fa / m, Fb / m
                    vals.append(np.stack([100 * (Aa - Ab) / m, 100 * (fa - fb), 100 * (Qa - Qb),
                                          100 * (fa - fb) * (Qa + Qb) / 2, 100 * (Qa - Qb) * (fa + fb) / 2,
                                          (Ta - Tb) / m, 100 * (fb - fa)], axis=-1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)     # a stratum empty in one fold of a resample
                return np.nanmean(np.stack(vals, 0), axis=0)

        obs, bs, ts = resample(np.vstack(Xa), np.vstack(Xb), f)
        nm_ = ["acc", "finish", "correct_given_finished", "finishing_term", "quality_term", "tokens", "cut_rate"]
        r = {nm: summarize(obs[..., j], bs[:, j], ts[:, j]) for j, nm in enumerate(nm_)}
        r["n_problems"] = sizes if nf > 1 else sizes[0]
        out[k] = r
        sd = np.nanstd(bs[:, 0])
        if math.isfinite(obs[0]) and sd > 0 and min(sizes) >= 5:
            est.append(obs[0])
            ses.append(sd)
    if len(est) >= 2:
        w = 1 / np.square(ses)
        m = float((w * est).sum() / w.sum())
        Q = float((w * np.square(np.array(est) - m)).sum())
        out["heterogeneity_acc"] = dict(Q=rnd(Q), df=len(est) - 1, P=rnd(chi2_sf(Q, len(est) - 1), 4))
    out["cross_fitted"] = bool(len(folds) > 1)
    return out


def analyse_single(key):
    D = Single(key, SINGLE[key])
    if not D.ok:
        return None, D.reason
    out = dict(n_problems=len(D.ids), K=D.K, K_check=D.K_check, cap=D.cap, cap_config=D.cap_config,
               cap_from_data=D.cap_from_data, seeds=D.seeds, tags=D.tags,
               finish_definition=FINISH_DEF[D.cut_kind], cap_check=D.cap_check(),
               arms={a: arm_summary(D, a) for a in D.arms}, contrasts={})
    for a, b in CONTRASTS:
        if a in D.arms and b in D.arms:
            c = contrast_block(D, a, b)
            c["per_problem_decomposition"] = per_problem_block(D, a, b)
            c["lee_bounds"] = lee_block(D, a, b)
            out["contrasts"][f"{a}-{b}"] = c
    out["common_finished_set"] = common_set_block(D)
    if key in MAIN_SINGLE:
        out["strata"] = {kind: {f"{a}-{b}": strata_block(D, a, b, kind) for a, b in STRATA_CONTRASTS
                                if a in D.arms and b in D.arms} for kind in ("base_finish", "level")}
    return out, D


def check_vs_paper_stats(sec, PS):
    """The recomputed contrasts against paper_stats.json: the same resamples, so diff, lo and hi must agree to rounding
    and paper_stats' p must lie in [p - p_tie, p] (its float sums put exact-zero resamples on either side of 0)."""
    out = {}
    for key, v in sec.items():
        ps = (PS or {}).get(key) if isinstance(v, dict) else None
        if not ps or "contrasts" not in v:
            continue
        dev, bad, n = Counter(), [], 0
        for c, stats in ps.get("contrasts", {}).items():
            for m, ref in stats.items():
                mine = v["contrasts"].get(c, {}).get(m)
                if mine and mine.get("diff") is not None:
                    n += 1
                    for k in ("diff", "lo", "hi"):
                        dev[k] = max(dev[k], abs(mine[k] - ref[k]))
                    if not (mine["p"] - mine.get("p_tie", 0) - 1e-4 <= ref["p"] <= mine["p"] + 1e-4):
                        bad.append(f"{c} {m}: paper_stats p {ref['p']}, here {mine['p']} (ties {mine.get('p_tie', 0)})")
        out[key] = dict(compared=n, max_abs_dev={k: round(x, 4) for k, x in dev.items()}, p_outside_tie_range=bad,
                        agree=bool(n and max(dev.values(), default=0) <= 1e-3 + 1e-9 and not bad))
    return out


# ============================================================================ item 10: finishing from token counts, crossing
def apps_finishing(D):
    """Finishing on APPS from the token sums. A sample stopped at the cap generated exactly `cap` tokens, so per record
    (K samples, token sum T, cut field c): capped samples <= floor(T / cap), and >= T - K (cap - 1) if finished samples
    end below the cap; if, as paper_stats.py's notes say, c also counts finished samples without a code block, then
    capped <= c as well. Records with c > floor(T / cap) prove that c counts samples that were not capped."""
    K, cap = D.K, D.cap
    arms, X = {}, {}
    for a in D.arms:
        T, c = D.tok[a], D.cut[a]
        nmax = np.minimum(K, np.floor(T / cap))
        nmin = np.maximum(0, T - K * (cap - 1))
        nmax_c = np.minimum(nmax, c)
        arms[a] = dict(records=int(T.size),
                       finish_cut_field=rnd(100 * float((1 - c / K).mean())),
                       finish_length_bounds_tokens_only=[rnd(100 * float((1 - nmax / K).mean())),
                                                         rnd(100 * float((1 - nmin / K).mean()))],
                       finish_length_bounds_if_cut_counts_capped_and_no_code=[rnd(100 * float((1 - nmax_c / K).mean())),
                                                                               rnd(100 * float((1 - nmin / K).mean()))],
                       records_cut_exceeds_floor_T_over_cap=int((c > nmax).sum()),
                       min_samples_flagged_cut_but_not_capped=int(np.maximum(0, c - nmax).sum()),
                       share_samples_flagged_cut_but_not_capped_min=rnd(100 * float(np.maximum(0, c - nmax).sum() / (K * T.size))),
                       records_cut_below_token_minimum=int((c < nmin).sum()),
                       records_all_samples_capped=int((T == K * cap).sum()),
                       records_no_sample_can_be_capped=int((T < cap).sum()),
                       capped_mean_tokens=rnd(float((T / K).mean()), 1))
        X[a] = np.vstack([(1 - nmax / K).mean(0), (1 - nmax_c / K).mean(0), (T == K * cap).mean(0), (T < cap).mean(0)])
        ex = D.exact_capped(a)
        if ex is not None:
            arms[a]["finish_exact_from_per_sample_fields"] = rnd(100 * float((1 - ex / K).mean()))
    exact = all(D.exact_capped(a) is not None for a in X)
    con = {}
    names = ["finish_length_lower_bound_tokens_only", "finish_length_lower_bound_with_cut_field",
             "share_records_all_capped", "share_records_none_cappable"]
    n = len(D.ids)

    def exact_rows(x):      # (A, F, tokens, finished tokens) per problem with F from the per-sample fields
        cc = D.exact_capped(x)
        return np.vstack([D.A[x], ((K - cc) / K).mean(0), D.T[x], ((D.tok[x] - cc * cap) / K).mean(0)])

    for a, b in CONTRASTS:
        if a in X and b in X:
            f = lambda Sa, Sb: np.stack([100 * (Sa[..., j] - Sb[..., j]) / n for j in range(len(names))], -1)
            con[f"{a}-{b}"] = tests_multi(X[a], X[b], f, names)
            if exact:
                con[f"{a}-{b}"]["exact_decomposition"] = tests_multi(
                    exact_rows(a), exact_rows(b), lambda Sa, Sb: contrast_stats(Sa, Sb, n), CONTRAST_NAMES,
                    perm_from={"tokens_rel_pct": "_tokens_log", "finished_mean_tokens_rel_pct": "_finished_log"})
    return dict(note=apps_finishing.__doc__.split("\n")[0].strip(), cap=cap, K=K, arms=arms, contrasts=con,
                delivery_decomposition="item4_quality.apps_qwen4b.contrasts: finish = program delivered within the cap")


def crossing_block(D, a, b):
    """Do the two arms' length distributions cross? If L_a <=st L_b (a stochastically shorter), both the cut rate and
    the capped mean E[min(L, cap)] are lower (or equal) under a; if L_b <=st L_a, both are higher. A lower capped mean
    together with a higher cut rate (or the reverse) is therefore incompatible with either ordering. Test of that
    pattern: intersection-union, P = max of the two one-sided P (bootstrap and sign flip); also the share of
    resamples that show the pattern. Per-problem means over seeds; pooled over problems. Per problem: the shares of
    problems on which a is shorter yet cuts more, or longer yet cuts less (seed means), and the difference of the two
    shares (bootstrap and sign flip: swapping the arms exchanges the two kinds of problem)."""
    n = len(D.ids)
    dl, dc_ = D.T[a] - D.T[b], D.F[b] - D.F[a]
    q1, q2 = (dl < 0) & (dc_ > 0), (dl > 0) & (dc_ < 0)
    per_problem = dict(shorter_but_more_cut=rnd(100 * q1.mean(), 2), longer_but_less_cut=rnd(100 * q2.mean(), 2),
                       shorter_and_less_cut=rnd(100 * ((dl < 0) & (dc_ < 0)).mean(), 2),
                       longer_and_more_cut=rnd(100 * ((dl > 0) & (dc_ > 0)).mean(), 2),
                       same_cut_rate=rnd(100 * (dc_ == 0).mean(), 2),
                       asymmetry_pct_points=mean_test(q1.astype(float) - q2.astype(float)))
    rows = lambda x: np.vstack([D.T[x], 1 - D.F[x], D.FT[x], D.F[x]])

    def f(Sa, Sb):
        return np.stack([(Sa[..., 0] - Sb[..., 0]) / n, 100 * (Sa[..., 1] - Sb[..., 1]) / n,
                         Sa[..., 2] / Sa[..., 3] - Sb[..., 2] / Sb[..., 3]], -1)

    obs, bs, ts = resample(rows(a), rows(b), f)
    dm, dc = obs[0], obs[1]
    out = dict(capped_mean_tokens=summarize(dm, bs[:, 0], ts[:, 0], 1), cut_rate=summarize(dc, bs[:, 1], ts[:, 1]),
               finished_mean_tokens=summarize(obs[2], bs[:, 2], ts[:, 2], 1), per_problem=per_problem)
    crossing = bool(np.sign(dm) * np.sign(dc) < 0)
    out["crossing_point_estimates"] = crossing
    if crossing:
        sm, sc = np.sign(dm), np.sign(dc)
        p_m = float(np.mean(sm * np.round(bs[:, 0], 9) <= 0))          # share of resamples against dm's sign
        p_c = float(np.mean(sc * np.round(bs[:, 1], 9) <= 0))
        pp_m = (1 + int(np.sum(sm * ts[:, 0] >= abs(dm) - 1e-9))) / (1 + len(ts))
        pp_c = (1 + int(np.sum(sc * ts[:, 1] >= abs(dc) - 1e-9))) / (1 + len(ts))
        out["pattern"] = f"{a} {'lower' if dm < 0 else 'higher'} capped mean, {'higher' if dc > 0 else 'lower'} cut rate than {b}"
        out["intersection_union"] = dict(p_boot=rnd(max(p_m, p_c), 4), p_perm=rnd(max(pp_m, pp_c), 4),
                                         share_resamples_with_pattern=rnd(float(np.mean((sm * bs[:, 0] > 0) & (sc * bs[:, 1] > 0))), 4))
    return out


def tail_strata(sec_strata):
    """Where the extra cuts arise: cut-rate and capped-mean differences by the untouched model's finishing bin."""
    out = {}
    for c, v in (sec_strata or {}).items():
        if not v:
            continue
        out[c] = {k: dict(n_problems=s.get("n_problems"), cut_rate=s.get("cut_rate"), tokens=s.get("tokens"))
                  for k, s in v.items() if isinstance(s, dict) and "cut_rate" in s}
    return out


# ============================================================================ item 5: ARC-AGI-2 agent
class Arc:
    def __init__(self, cfg=ARC):
        self.cfg, self.cap = cfg, cfg["cap"]
        runs = {}
        for a, fmt in cfg["arms"].items():
            rr = [(s, R) for s, R in ((s, load(fmt.format(s=s))) for s in range(5)) if R]
            if rr:
                runs[a] = rr
        self.ok = len(runs) >= 2
        if not self.ok:
            self.reason = "no ARC records"
            return
        self.runs, self.arms = runs, list(runs)
        self.ids = sorted(set.intersection(*[set(R) for rr in runs.values() for _, R in rr]))
        n, cap = len(self.ids), self.cap
        self.seeds = {a: [s for s, _ in rr] for a, rr in runs.items()}
        self.Y, self.TOK, self.SEC, self.LOG, self.M, self.M1, self.threads = {}, {}, {}, {}, {}, {}, {}
        # M[a]: (8, ns, n) per seed x puzzle: messages, cut, code, sum min(tok, cap), tokens, tokens in cut messages,
        #       code within cap/2, code within cap/6;  M1[a]: (4, ns, n) first messages: count, cut, code, min(tok, cap)
        for a, rr in runs.items():
            ns = len(rr)
            self.Y[a] = np.array([[bool(R[i].get("pass2")) for i in self.ids] for _, R in rr], float)
            self.TOK[a] = np.array([[float(R[i].get("tokens") or 0) for i in self.ids] for _, R in rr])
            self.SEC[a] = np.array([[float(R[i].get("seconds") or 0) for i in self.ids] for _, R in rr])
            M = np.zeros((8, ns, n))
            M1 = np.zeros((4, ns, n))
            LOG = np.zeros((ns, n), bool)
            th = []
            for j, (_, R) in enumerate(rr):
                for i, tid in enumerate(self.ids):
                    ml = R[tid].get("msg_log") or []
                    LOG[j, i] = any(len(t) for t in ml)
                    for t in ml:
                        if not t:
                            continue
                        th.append((j, i, [(float(m[0]), bool(m[3]), bool(m[4])) for m in t]))
                        m0 = t[0]
                        M1[:, j, i] += (1, bool(m0[3]), bool(m0[4]), min(float(m0[0]), cap))
                        for m in t:
                            tk, c, k = float(m[0]), bool(m[3]), bool(m[4])
                            M[:, j, i] += (1, c, k, min(tk, cap), tk, tk * c, k and tk <= cap / 2, k and tk <= cap / 6)
            self.M[a], self.M1[a], self.LOG[a], self.threads[a] = M, M1, LOG, th
        # seeds with any message log (paper_stats.py pools every message it finds; seeds without logs add nothing to the
        # pooled shares, but must not count as seeds in the two-way bootstrap); per-seed coverage is in item5_arc.per_seed
        self.log_seeds = {a: [j for j in range(len(self.seeds[a])) if self.LOG[a][j].any()] for a in self.arms}


def seed_counts(rng, ns):
    idx = rng.integers(0, ns, size=(B, ns))
    return np.bincount((idx + ns * np.arange(B)[:, None]).ravel(), minlength=B * ns).reshape(B, ns).astype(float)


def twoway(Ya, Yb, f, paired=False, nd=3, puzzle_seed=BOOT_SEED):
    """Crossed puzzle x seed bootstrap. Ya: (k, ns_a, n) per seed x puzzle quantities. Each resample draws puzzles
    (paper_stats draws, shared by the two arms) and seeds with replacement (independently per arm; paired=True: the
    same seed positions in both arms); arm sums = mean over the drawn seeds of the sums over the drawn puzzles."""
    Ya, Yb = np.asarray(Ya, float), np.asarray(Yb, float)
    k, na, n = Ya.shape
    nb = Yb.shape[1]
    C = boot_counts(n, puzzle_seed)
    rng = np.random.default_rng([TWOWAY_SEED, n, na, nb, int(paired)])
    Da = seed_counts(rng, na)
    Db = Da if paired else seed_counts(rng, nb)
    Sa = np.einsum("bks,bs->bk", np.einsum("ksi,bi->bks", Ya, C), Da) / na
    Sb = np.einsum("bks,bs->bk", np.einsum("ksi,bi->bks", Yb, C), Db) / nb
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = np.asarray(f(Ya.sum((1, 2)) / na, Yb.sum((1, 2)) / nb), float)
        bs = np.asarray(f(Sa, Sb), float)
    return obs, bs


def twoway_summ(Ya, Yb, f, names, paired=False, nd=3):
    obs, bs = twoway(Ya, Yb, f, paired, nd)
    obs = np.atleast_1d(obs)
    bs = bs.reshape(len(bs), -1)
    return {nm: summarize(obs[j], bs[:, j], None, nd) for j, nm in enumerate(names) if not nm.startswith("_")}


def perm_unpaired(xa, xb):
    """Exact two-sided permutation test on per-seed values, all C(na + nb, na) splits (derived_stats.py's test)."""
    pool, na = list(xa) + list(xb), len(xa)
    obs = np.mean(xa) - np.mean(xb)
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), na):
        s = set(idx)
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(len(pool)) if i not in s]
        hits += abs(np.mean(x) - np.mean(y)) >= abs(obs) - 1e-9
        tot += 1
    return dict(diff=rnd(obs), hits=hits, splits=tot, P=rnd(hits / tot, 4))


def perm_paired(xa, xb):
    """Exact two-sided sign-flip test on per-seed-index differences (arm labels swapped within each seed pair)."""
    d = np.asarray(xa, float) - np.asarray(xb, float)
    obs = d.mean()
    hits = sum(abs((d * np.array(sg)).mean()) >= abs(obs) - 1e-9 for sg in itertools.product((1, -1), repeat=len(d)))
    return dict(diff=rnd(obs), hits=int(hits), flips=2 ** len(d), P=rnd(hits / 2 ** len(d), 4))


def cochran_q(Y, nmc=None):
    """Seed heterogeneity of one arm (seeds x puzzles, binary): Cochran's Q = (k - 1) s^2 / E0[s^2], s^2 = variance of
    the seed totals, E0 = its expectation when each puzzle's solves are spread at random over the seeds
    (sum_i R_i (k - R_i) / (k (k - 1))). P from chi-square(k - 1) and from the exact conditional distribution
    (Monte Carlo over random allocations with the puzzle totals fixed)."""
    Y = np.asarray(Y, float)
    k, n = Y.shape
    C, R = Y.sum(1), Y.sum(0)
    den = (R * (k - R)).sum()
    if k < 2 or den <= 0:
        return dict(note="no variation")
    s2 = C.var(ddof=1)
    e0 = den / (k * (k - 1))
    Q = (k - 1) * s2 / e0
    rng = np.random.default_rng([PERM_SEED, k, n])
    nmc = nmc or B_PERM
    hits = 0
    for start in range(0, nmc, 1000):
        m = min(1000, nmc - start)
        ranks = rng.random((m, n, k)).argsort(-1).argsort(-1)
        Cs = (ranks < R[None, :, None]).sum(1)
        hits += int(np.sum(Cs.var(1, ddof=1) >= s2 - 1e-9))
    return dict(seed_totals=C.astype(int).tolist(), variance=rnd(s2), expected_variance_no_seed_effect=rnd(e0),
                dispersion_ratio=rnd(s2 / e0), Q=rnd(Q), df=k - 1, P_chi2=rnd(chi2_sf(Q, k - 1), 4),
                P_exact_mc=rnd((1 + hits) / (1 + nmc), 4),
                binomial_ceiling_variance=rnd(n * (C.mean() / n) * (1 - C.mean() / n)))


def km_curve(times, events, W, grid):
    """Weighted Kaplan-Meier F(t) = 1 - S(t) at the grid; W (R, N) weights, one row per resample."""
    times, events = np.asarray(times, float), np.asarray(events, bool)
    ut = np.unique(times[events])
    if not len(ut):
        return np.zeros((W.shape[0], len(grid)))
    risk = (times[None, :] >= ut[:, None]).astype(float)
    ev = ((times[None, :] == ut[:, None]) & events[None, :]).astype(float)
    nu, du = W @ risk.T, W @ ev.T
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(nu > 0, du / nu, 0.0)
        logS = np.cumsum(np.log(np.clip(1 - h, 0, 1)), axis=1)
    pos = np.searchsorted(ut, np.asarray(grid, float), side="right") - 1
    S = np.where(pos[None, :] >= 0, np.exp(logS[:, np.clip(pos, 0, None)]), 1.0)
    return 1 - S


def arc_scan_tag(tag):
    files = sorted(glob.glob(f"runs_synth/agent_{tag}/shard_*.jsonl"))
    lines, ids, dup, keys = 0, set(), 0, Counter()
    for fn in files:
        for l in open(fn):
            if l.strip():
                r = json.loads(l)
                lines += 1
                dup += r["task_id"] in ids
                ids.add(r["task_id"])
                keys.update(r.keys())
    mt = [os.path.getmtime(fn) for fn in files]
    iso = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(timespec="seconds")
    return dict(shards=len(files), lines=lines, unique_tasks=len(ids), duplicate_lines=dup,
                first_mtime_utc=iso(min(mt)) if mt else None, last_mtime_utc=iso(max(mt)) if mt else None,
                fields={k: v for k, v in sorted(keys.items())})


LOG_PATTERNS = {"traceback": r"Traceback", "llm_chat_failed": r"llm\.chat failed", "time_limit_cancel": r"RuntimeError: cancelled",
                "cuda_oom": r"OutOfMemoryError|CUDA out of memory", "timeout": r"[Tt]ime ?out|timed out",
                "speculative_retry": r"retrying without", "request_rejected": r"request rejected"}


def chain_provenance(tag, kind="agent_dbg"):
    """Best effort, from leonardo/chain_dbg.sh's log (logs/chain_<tag>.log: dates, job ids) and the jobs' agent logs:
    vLLM version strings, speculative decoding, chat kwargs, counts of error-like lines."""
    path = os.path.join("logs", f"chain_{tag}.log")
    if not os.path.exists(path):
        return None
    dates, jobs = [], []
    for line in open(path, errors="replace"):
        m = re.match(r"(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}) iter \d+: (\d+)/(\d+)", line)
        if m:
            dates.append(m.group(1))
        m = re.match(r"\s*submitted:\s*(.*)", line)
        if m:
            jobs += m.group(1).split()
    versions, spec, kw, events, nlogs = Counter(), Counter(), Counter(), Counter(), 0
    for j in jobs:
        for fn in glob.glob(os.path.join("logs", f"{kind}-{j}_s*.out")):
            nlogs += 1
            for line in open(fn, errors="replace"):
                for pat in (r"LLM engine \(v([0-9][^)\s]*)\)", r"vLLM(?: API server)? version:? ?([0-9][^\s,]*)"):
                    m = re.search(pat, line)
                    if m:
                        versions[m.group(1)] += 1
                m = re.search(r"engine up; speculative=(\w+)", line)
                if m:
                    spec[m.group(1)] += 1
                m = re.search(r"chat_template_kwargs: (.*)", line)
                if m:
                    kw[m.group(1).strip()] += 1
                for name, pat in LOG_PATTERNS.items():
                    if re.search(pat, line):
                        events[name] += 1
    return dict(chain_log=path, first_iter=dates[0] if dates else None, last_iter=dates[-1] if dates else None,
                jobs=len(jobs), agent_logs_found=nlogs, vllm_versions=dict(versions), speculative=dict(spec),
                chat_kwargs=dict(kw), log_events=dict(events))


def arc_per_seed(D):
    rows = []
    for a in D.arms:
        for j, (s, R) in enumerate(D.runs[a]):
            tag = D.cfg["arms"][a].format(s=s)
            recs = [R[i] for i in D.ids]
            st = Counter()
            for r in recs:
                st.update(r.get("status") or {})
            M = D.M[a][:, j]
            nmsg = M[0].sum()
            rows.append(dict(arm=a, seed=s, tag=tag, records=len(R), common_puzzles=len(recs), files=arc_scan_tag(tag),
                             solved=int(D.Y[a][j].sum()), pass1=int(sum(bool(r.get("pass1")) for r in recs)),
                             oracle=int(sum(bool(r.get("oracle")) for r in recs)),
                             tokens_per_puzzle_k=rnd(D.TOK[a][j].mean() / 1e3, 1),
                             seconds_per_puzzle=rnd(D.SEC[a][j].mean(), 1), threads=int(sum(r.get("sampled") or 0 for r in recs)),
                             thread_status=dict(st), explores=int(sum(r.get("explores") or 0 for r in recs)),
                             finals=int(sum(r.get("finals") or 0 for r in recs)),
                             puzzles_with_msg_log=int(D.LOG[a][j].sum()), messages=int(nmsg),
                             messages_per_puzzle=rnd(nmsg / len(recs), 2),
                             cut_share=rnd(100 * M[1].sum() / nmsg) if nmsg else None,
                             code_share=rnd(100 * M[2].sum() / nmsg) if nmsg else None,
                             provenance=chain_provenance(tag)))
    return rows


def arc_metric_arrays(D, a, seeds=None):
    """Per seed x puzzle quantities for the Table 1 metrics (k, ns, n)."""
    sel = list(range(len(D.seeds[a]))) if seeds is None else seeds
    return np.stack([D.Y[a][sel], D.TOK[a][sel]])


def arc_contrast(D, a, b, ids_mask=None, puzzle_seed=BOOT_SEED):
    """Table 1 contrast a - b: solved (puzzles), tokens per puzzle and per solved puzzle, relative changes; puzzle
    bootstrap (paper convention) + sign flip over puzzles, two-way bootstrap (independent and paired seeds), exact
    seed-level tests (unpaired permutation of the seed totals, paired sign flips by seed index)."""
    m = np.ones(len(D.ids), bool) if ids_mask is None else ids_mask
    n = int(m.sum())
    Ya, Yb = arc_metric_arrays(D, a)[:, :, m], arc_metric_arrays(D, b)[:, :, m]
    pa, pb = Ya.mean(1), Yb.mean(1)       # per puzzle, seed means of (solved, tokens)

    def f(Sa, Sb):   # Sa: sums over puzzles of the seed means of (solved, tokens)
        return np.stack([Sa[..., 0] - Sb[..., 0], 100 * (Sa[..., 0] / Sb[..., 0] - 1), np.log(Sa[..., 0] / Sb[..., 0]),
                         (Sa[..., 1] - Sb[..., 1]) / n / 1e3, 100 * (Sa[..., 1] / Sb[..., 1] - 1), np.log(Sa[..., 1] / Sb[..., 1]),
                         (Sa[..., 1] / Sa[..., 0] - Sb[..., 1] / Sb[..., 0]) / 1e3,
                         100 * ((Sa[..., 1] / Sa[..., 0]) / (Sb[..., 1] / Sb[..., 0]) - 1),
                         np.log((Sa[..., 1] / Sa[..., 0]) / (Sb[..., 1] / Sb[..., 0]))], -1)

    names = ["solved", "solved_rel_pct", "_solved_log", "tokens_per_puzzle_k", "tokens_per_puzzle_rel_pct", "_tpp_log",
             "tokens_per_solved_k", "tokens_per_solved_rel_pct", "_tps_log"]
    pf = {"solved_rel_pct": "_solved_log", "tokens_per_puzzle_rel_pct": "_tpp_log", "tokens_per_solved_rel_pct": "_tps_log"}
    out = dict(puzzle_bootstrap=tests_multi(pa, pb, f, names, perm_from=pf, seed=puzzle_seed),
               twoway_bootstrap=twoway_summ(Ya, Yb, f, names), n_puzzles=n)
    if D.seeds[a] == D.seeds[b]:
        out["twoway_bootstrap_paired_seeds"] = twoway_summ(Ya, Yb, f, names, paired=True)
    ta, tb = Ya[0].sum(1), Yb[0].sum(1)
    out["seed_level"] = dict(totals_a=ta.astype(int).tolist(), totals_b=tb.astype(int).tolist(),
                             unpaired_permutation=perm_unpaired(ta, tb),
                             paired_by_seed_index=perm_paired(ta, tb) if D.seeds[a] == D.seeds[b] else None,
                             tokens_per_puzzle_unpaired=perm_unpaired(Ya[1].mean(1) / 1e3, Yb[1].mean(1) / 1e3))
    return out


def arc_messages(D, a, b, unit="all", mask=None, puzzle_seed=BOOT_SEED, twoway_too=True):
    """Message statistics a - b on seeds with logs. unit 'all': every message (cut share, code share, restart cost
    sum min(tok, cap) / messages with code, share of tokens in cut messages, code within cap/6 and cap/2 tokens);
    unit 'first': the first message of each puzzle x seed thread (written before any feedback, so independent of how
    the puzzle went): cut share, code share, mean tokens, restart cost. Puzzle bootstrap + sign flip, two-way.
    mask: a subset of the puzzles (development / fresh)."""
    sa, sb = D.log_seeds[a], D.log_seeds[b]
    if not sa or not sb:
        return None
    m = np.ones(len(D.ids), bool) if mask is None else mask
    if unit == "all":
        Ma, Mb = D.M[a][:, sa][:, :, m], D.M[b][:, sb][:, :, m]

        def f(Sa, Sb):
            r = lambda S: np.stack([100 * S[..., 1] / S[..., 0], 100 * S[..., 2] / S[..., 0], S[..., 3] / S[..., 2] / 1e3,
                                    100 * S[..., 5] / S[..., 4], 100 * S[..., 7] / S[..., 0], 100 * S[..., 6] / S[..., 0]], -1)
            return r(Sa) - r(Sb)

        names = ["cut_share", "code_share", "restart_cost_k", "tokens_in_cut_share", "code_within_cap_6", "code_within_cap_2"]
    else:
        Ma, Mb = D.M1[a][:, sa][:, :, m], D.M1[b][:, sb][:, :, m]

        def f(Sa, Sb):
            r = lambda S: np.stack([100 * S[..., 1] / S[..., 0], 100 * S[..., 2] / S[..., 0], S[..., 3] / S[..., 0],
                                    S[..., 3] / S[..., 2] / 1e3], -1)
            return r(Sa) - r(Sb)

        names = ["cut_share", "code_share", "mean_tokens", "restart_cost_k"]
    out = dict(seeds_a=[D.seeds[a][j] for j in sa], seeds_b=[D.seeds[b][j] for j in sb],
               puzzle_bootstrap=tests_multi(Ma.sum(1), Mb.sum(1), f, names, seed=puzzle_seed))
    if twoway_too:
        out["twoway_bootstrap"] = twoway_summ(Ma, Mb, f, names)
    if unit == "all":
        out["seed_level_cut_share_unpaired"] = perm_unpaired(100 * Ma[1].sum(1) / Ma[0].sum(1), 100 * Mb[1].sum(1) / Mb[0].sum(1))
    return out


def arc_message_levels(D, a, unit="all", mask=None):
    sa = D.log_seeds[a]
    if not sa:
        return None
    m = np.ones(len(D.ids), bool) if mask is None else mask
    M = (D.M if unit == "all" else D.M1)[a][:, sa][:, :, m].sum(1)
    if unit == "all":
        return dict(messages=int(M[0].sum()), cut_share=ci1(M, lambda S: 100 * S[..., 1] / S[..., 0]),
                    code_share=ci1(M, lambda S: 100 * S[..., 2] / S[..., 0]),
                    restart_cost_k=ci1(M, lambda S: S[..., 3] / S[..., 2] / 1e3),
                    tokens_in_cut_share=ci1(M, lambda S: 100 * S[..., 5] / S[..., 4]),
                    messages_per_puzzle_seed=rnd(M[0].sum() / (len(sa) * M.shape[1]), 2))
    return dict(threads=int(M[0].sum()), cut_share=ci1(M, lambda S: 100 * S[..., 1] / S[..., 0]),
                code_share=ci1(M, lambda S: 100 * S[..., 2] / S[..., 0]), mean_tokens=ci1(M, lambda S: S[..., 3] / S[..., 0], 1),
                restart_cost_k=ci1(M, lambda S: S[..., 3] / S[..., 2] / 1e3))


def arc_by_index(D, rounds):
    """Messages by position k in the thread (among threads that reached k): cut and code shares per arm, puzzle-
    clustered CIs, and S minus each other arm (puzzle bootstrap + sign flip). Descriptive: reaching k depends on the
    earlier messages."""
    n = len(D.ids)
    arr = {}
    for a in D.arms:
        X = np.zeros((rounds, 3, n))
        for j, i, msgs in D.threads[a]:
            if j not in D.log_seeds[a]:
                continue
            for k, (tk, c, cd) in enumerate(msgs[:rounds]):
                X[k, :, i] += (1, c, cd)
        arr[a] = X
    out = {}
    for k in range(rounds):
        row = {a: dict(messages=int(arr[a][k, 0].sum()), cut_share=ci1(arr[a][k], lambda S: 100 * S[..., 1] / S[..., 0]),
                       code_share=ci1(arr[a][k], lambda S: 100 * S[..., 2] / S[..., 0])) for a in D.arms}
        if "S" in D.arms:
            for b in D.arms:
                if b != "S":
                    row[f"S-{b}"] = tests_multi(arr["S"][k], arr[b][k], lambda Sa, Sb: np.stack(
                        [100 * (Sa[..., 1] / Sa[..., 0] - Sb[..., 1] / Sb[..., 0]),
                         100 * (Sa[..., 2] / Sa[..., 0] - Sb[..., 2] / Sb[..., 0])], -1), ["cut_share", "code_share"])
        out[str(k + 1)] = row
    return out


def arc_time_to_program(D, grid):
    """Kaplan-Meier share of puzzle x seed threads that have delivered a program (any code block) by the time they
    have generated t tokens in total (cut messages count their tokens); threads that end without code are censored at
    their total. CIs and contrasts by the two-way bootstrap (puzzles shared, seeds per arm)."""
    n = len(D.ids)
    C = boot_counts(n)
    rng = np.random.default_rng([TWOWAY_SEED, 99])
    curves, draws = {}, {}
    for a in D.arms:
        th = [(j, i, msgs) for j, i, msgs in D.threads[a] if j in D.log_seeds[a]]
        if not th:
            continue
        times, ev = [], []
        for _, _, msgs in th:
            t, e = 0.0, False
            for tk, _, cd in msgs:
                t += tk
                if cd:
                    e = True
                    break
            times.append(t)
            ev.append(e)
        pos = {j: q for q, j in enumerate(D.log_seeds[a])}
        J = np.array([pos[j] for j, _, _ in th])
        I = np.array([i for _, i, _ in th])
        Ds = seed_counts(rng, len(pos))
        W = C[:, I] * Ds[:, J]
        obs = km_curve(times, ev, np.ones((1, len(th))), grid)[0]
        bs = km_curve(times, ev, W, grid)
        draws[a] = (obs, bs)
        curves[a] = {str(int(t)): dict(value=rnd(100 * o), lo=rnd(100 * np.nanpercentile(bs[:, g], 2.5)),
                                        hi=rnd(100 * np.nanpercentile(bs[:, g], 97.5)))
                     for g, (t, o) in enumerate(zip(grid, obs))}
        curves[a]["threads"] = len(th)
        curves[a]["events"] = int(sum(ev))
    con = {}
    for a, b in ARC_CONTRASTS:
        if a in draws and b in draws:
            con[f"{a}-{b}"] = {str(int(t)): summarize(100 * (draws[a][0][g] - draws[b][0][g]),
                                                      100 * (draws[a][1][:, g] - draws[b][1][:, g]))
                               for g, t in enumerate(grid)}
    return dict(grid_tokens=[int(t) for t in grid], arms=curves, contrasts=con,
                note="two-way bootstrap intervals and P only (no relabelling test for Kaplan-Meier differences here)")


def analyse_arc():
    D = Arc()
    if not D.ok:
        return None, D.reason
    n = len(D.ids)
    out = dict(n_puzzles=n, seeds=D.seeds, log_seeds={a: [D.seeds[a][j] for j in v] for a, v in D.log_seeds.items()},
               seed_notes="synth/solve_agent.py (local copy): one vLLM engine per shard with LLM(seed=--seed); no "
                          "per-request seed in SamplingParams; leonardo scripts pass --seed only for seeds != 0 (default 0)",
               per_seed=arc_per_seed(D), arms={}, contrasts={}, seed_heterogeneity={})
    for a in D.arms:
        Y = D.Y[a]
        out["arms"][a] = dict(seed_totals=Y.sum(1).astype(int).tolist(), solved=rnd(Y.sum(1).mean(), 2),
                              solved_ci=ci1(Y.mean(0)[None], lambda S: S[..., 0]),
                              tokens_per_puzzle_k=ci1(D.TOK[a].mean(0)[None], lambda S: S[..., 0] / n / 1e3, 1),
                              tokens_per_solved_k=ci1(np.vstack([D.TOK[a].sum(0), Y.sum(0)]), lambda S: S[..., 0] / S[..., 1] / 1e3, 1),
                              messages_all=arc_message_levels(D, a, "all"), messages_first=arc_message_levels(D, a, "first"))
        out["seed_heterogeneity"][a] = cochran_q(Y)
    for a, b in ARC_CONTRASTS:
        if a in D.arms and b in D.arms:
            c = arc_contrast(D, a, b)
            c["messages_all"] = arc_messages(D, a, b, "all")
            c["messages_first"] = arc_messages(D, a, b, "first")
            out["contrasts"][f"{a}-{b}"] = c
    dev = read_ids(ARC["dev"])
    excl = read_ids(ARC["excl"])
    if dev:
        mask = np.array([i in set(dev) for i in D.ids])
        sub = dict(n_dev=int(mask.sum()), n_fresh=int((~mask).sum()),
                   ids_check=dict(dev_list=len(dev), excl_list=len(excl) if excl else None,
                                  fresh_in_excl=int(sum(i in set(excl) for i in np.array(D.ids)[~mask])) if excl else None),
                   arms={a: dict(dev=rnd(D.Y[a][:, mask].sum(1).mean(), 2), fresh=rnd(D.Y[a][:, ~mask].sum(1).mean(), 2),
                                 dev_seed_totals=D.Y[a][:, mask].sum(1).astype(int).tolist(),
                                 fresh_seed_totals=D.Y[a][:, ~mask].sum(1).astype(int).tolist(),
                                 dev_messages=arc_message_levels(D, a, "all", mask),
                                 fresh_messages=arc_message_levels(D, a, "all", ~mask)) for a in D.arms},
                   contrasts={})
        if mask.sum() >= 5 and (~mask).sum() >= 5:
            for a, b in ARC_CONTRASTS:
                if a in D.arms and b in D.arms:
                    cd = arc_contrast(D, a, b, mask)
                    cf = arc_contrast(D, a, b, ~mask, puzzle_seed=1)
                    # interaction: dev contrast minus fresh contrast (percentage points of the subset), puzzles
                    # resampled within each subset, with independent streams for the two subsets
                    da = D.Y[a].mean(0) - D.Y[b].mean(0)
                    Cd, Cf = boot_counts(int(mask.sum())), boot_counts(int((~mask).sum()), seed=1)
                    obs = da[mask].sum() / mask.sum() * 100 - da[~mask].sum() / (~mask).sum() * 100
                    bs = (Cd @ da[mask]) / mask.sum() * 100 - (Cf @ da[~mask]) / (~mask).sum() * 100
                    md = arc_messages(D, a, b, "all", mask, twoway_too=False)
                    mf = arc_messages(D, a, b, "all", ~mask, puzzle_seed=1, twoway_too=False)
                    pick = lambda x: None if x is None else {k: x["puzzle_bootstrap"][k] for k in ("cut_share", "code_share", "restart_cost_k")}
                    sub["contrasts"][f"{a}-{b}"] = dict(
                        dev=dict(puzzle_bootstrap=cd["puzzle_bootstrap"]["solved"], twoway=cd["twoway_bootstrap"]["solved"],
                                 seed_level=cd["seed_level"]["unpaired_permutation"], messages=pick(md)),
                        fresh=dict(puzzle_bootstrap=cf["puzzle_bootstrap"]["solved"], twoway=cf["twoway_bootstrap"]["solved"],
                                   seed_level=cf["seed_level"]["unpaired_permutation"], messages=pick(mf)),
                        dev_minus_fresh_pct_points=dict(summarize(obs, bs), note="bootstrap only: an interaction "
                                                        "between two puzzle subsets has no paired relabelling test"))
        out["dev_vs_fresh"] = sub
    out["messages_by_index"] = arc_by_index(D, ARC["rounds"])
    out["time_to_first_program"] = arc_time_to_program(D, [D.cap / 2, D.cap, 2 * D.cap, 3 * D.cap, 4 * D.cap, 6 * D.cap])
    return out, D


def arc_summary_section(PS):
    """Seed-level tests from the seed totals stored in paper_stats.json (no records needed): exact permutation over all
    splits of the ten totals (derived_stats.py), sign flips within seed-index pairs (32), and the dispersion of each
    arm's totals against the largest variance independent puzzles allow (72 pbar (1 - pbar), chi-square, df = 4)."""
    arc = (PS or {}).get("arc")
    if not arc:
        return None
    A, n = arc["arms"], arc["n_tasks"]
    out = dict(source="paper_stats.json arc.arms[*].seeds (seed order s0..s4 as in Table 1)", n_tasks=n, contrasts={}, dispersion={})
    for a, b in ARC_CONTRASTS:
        if a in A and b in A:
            out["contrasts"][f"{a}-{b}"] = dict(unpaired_permutation=perm_unpaired(A[a]["seeds"], A[b]["seeds"]),
                                                paired_by_seed_index=perm_paired(A[a]["seeds"], A[b]["seeds"])
                                                if len(A[a]["seeds"]) == len(A[b]["seeds"]) else None)
    for a, v in A.items():
        x = np.array(v["seeds"], float)
        p = x.mean() / n
        ceil = n * p * (1 - p)
        chi = (len(x) - 1) * x.var(ddof=1) / ceil
        out["dispersion"][a] = dict(variance=rnd(x.var(ddof=1)), binomial_ceiling=rnd(ceil), chi2=rnd(chi), df=len(x) - 1,
                                    P=rnd(chi2_sf(chi, len(x) - 1), 4))
    return out


# ---------------------------------------------------------------- arc_msgs.json (local): message-level statistics
def arc_messages_local(path, PS=None, chunk=500):
    """Message-level statistics of paper_brevity/data/arc_msgs.json (per arm, a flat list of messages [tokens, thinking
    chars, visible chars, cut, has code]; no puzzle or seed labels, so the intervals treat messages as independent;
    the puzzle-clustered intervals are in item5_arc). Per-arm CIs: 10,000 message resamples; contrasts: independent
    resamples of the two arms' messages, and a two-sample permutation P (labels reshuffled, add-one)."""
    if not path or not os.path.exists(path):
        return None
    M = json.load(open(path))
    cap = M["cap"]
    X = {a: np.asarray(v, float) for a, v in M["arms"].items()}
    tgrid = (500, 750, 4000, 8000, 12000, 16000, 20000, 24000)
    cgrid = (8000, 12000, 16000, 20000, 24000)

    def V(x):   # per-message columns summed by the statistics below
        tok, cut, code = x[:, 0], x[:, 3], x[:, 4]
        cols = [np.ones(len(x)), cut, code, np.minimum(tok, cap), tok, tok * cut, (1 - cut), code * (1 - cut)]
        cols += [code * (tok <= t) for t in tgrid] + [np.minimum(tok, t) for t in cgrid] + [code * (tok <= t) for t in cgrid]
        return np.stack(cols, 1)

    nt, nc = len(tgrid), len(cgrid)

    def stats(S):
        base = [100 * S[..., 1] / S[..., 0], 100 * S[..., 2] / S[..., 0], S[..., 3] / S[..., 2] / 1e3,
                100 * S[..., 5] / S[..., 4], 100 * S[..., 7] / S[..., 6]]
        within = [100 * S[..., 8 + g] / S[..., 0] for g in range(nt)]
        cost = [S[..., 8 + nt + g] / S[..., 8 + nt + nc + g] / 1e3 for g in range(nc)]
        return np.stack(base + within + cost, -1)

    names = (["cut_share", "code_share", "restart_cost_k", "tokens_in_cut_share", "finished_with_code_share"]
             + [f"code_within_{t}" for t in tgrid] + [f"restart_cost_k_cap_{t}" for t in cgrid])
    rng = np.random.default_rng([PERM_SEED, 1])
    VV = {a: V(x) for a, x in X.items()}

    def boot(Va):
        N = len(Va)
        out = []
        for s in range(0, B, chunk):
            W = rng.multinomial(N, np.full(N, 1 / N), size=min(chunk, B - s)).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out.append(stats(W @ Va))
        return np.concatenate(out)

    draws = {a: boot(v) for a, v in VV.items()}
    arms = {}
    for a, v in VV.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            obs = stats(v.sum(0))
        arms[a] = dict(messages=len(v), cut_messages=int(v[:, 1].sum()))
        for j, nm in enumerate(names):
            d = np.sort(draws[a][:, j][np.isfinite(draws[a][:, j])])
            arms[a][nm] = dict(value=rnd(obs[j], 4), lo=rnd(d[int(0.025 * len(d))], 4), hi=rnd(d[int(0.975 * len(d))], 4))
        cost = {t: obs[5 + nt + g] for g, t in enumerate(cgrid)}
        arms[a]["restart_cost_min_cap_on_grid"] = int(min(cost, key=cost.get))
    con = {}
    for a, b in ARC_CONTRASTS:
        if a not in VV or b not in VV:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            obs = stats(VV[a].sum(0)) - stats(VV[b].sum(0))
        bs = draws[a] - draws[b]
        pool = np.vstack([VV[a], VV[b]])
        na, N = len(VV[a]), len(VV[a]) + len(VV[b])
        ts = []
        for s in range(0, B_PERM, chunk):
            m = min(chunk, B_PERM - s)
            mask = np.zeros((m, N))
            perm = rng.permuted(np.tile(np.arange(N), (m, 1)), axis=1)[:, :na]
            np.put_along_axis(mask, perm, 1.0, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ts.append(stats(mask @ pool) - stats((1 - mask) @ pool))
        ts = np.concatenate(ts)
        con[f"{a}-{b}"] = {nm: summarize(obs[j], bs[:, j], ts[:, j], 4) for j, nm in enumerate(names)}
    deff = {}
    ref = ((PS or {}).get("arc") or {}).get("contrasts", {})
    for c, v in ref.items():
        if c in con:
            for nm, key in (("cut_share", "cut_msgs"), ("restart_cost_k", "restart_cost_k")):
                w_cl = v[key]["hi"] - v[key]["lo"]
                w_iid = con[c][nm]["hi"] - con[c][nm]["lo"]
                deff.setdefault(c, {})[nm] = dict(width_puzzle_clustered=rnd(w_cl), width_message_level=rnd(w_iid),
                                                  design_effect=rnd((w_cl / w_iid) ** 2, 2) if w_iid else None)
    return dict(note=arc_messages_local.__doc__.split("\n")[0].strip(), source=path,
                sha256=hashlib.sha256(open(path, "rb").read()).hexdigest()[:16], cap=cap, arms=arms, contrasts=con,
                design_effect_vs_paper_stats=deff,
                caveat="message-level intervals ignore that messages cluster within puzzle x seed threads, which "
                       "understates the uncertainty of a single arm's share; for arm differences the paired puzzle "
                       "bootstrap of paper_stats.json gives intervals of similar width (design_effect_vs_paper_stats: "
                       "squared width ratios), because pairing over puzzles offsets the clustering. item5_arc has the "
                       "clustered and outcome-independent (first message) versions")


# ============================================================================ item 9: training sets
def sft_paths(spec):
    """distill/sft_student.py's --data expansion: comma-separated globs, 'glob@K' repeats those files K times."""
    paths = []
    for s in spec.split(","):
        g, _, k = s.rpartition("@") if "@" in s else (s, "", "1")
        paths += sorted(glob.glob(g)) * int(k)
    return paths


def visible(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def traj_summary(tr, msg_log, system_added):
    """Light-weight copy of one verified trajectory: the key sft_student sorts by, a content hash, and per assistant
    turn: generated tokens (from the record's msg_log when a thread matches, else chars / 3.6), whether a cut notice
    follows it, approximate context tokens (chars / 3.6 + 6 per message) of the example built for it."""
    turns = tr.get("turns", [])
    asst = [k for k, t in enumerate(turns) if t["role"] == "assistant"]
    h = hashlib.sha1(json.dumps([turns[k]["content"] for k in asst]).encode()).hexdigest()
    gen = None
    if len(asst) == 1 and tr.get("tokens") is not None:      # single-sample harvests: 'tokens' is that sample's count
        gen = [int(tr["tokens"])]
    for th in (msg_log or []) if gen is None else []:
        if len(th) == len(asst) and sum(m[0] for m in th) == tr.get("tokens"):
            gen = [int(m[0]) for m in th]
            break
    sys_added = system_added and not (turns and turns[0]["role"] == "system")
    ctx_chars, ctx_msgs, info = (len(ARC_SFT_SYSTEM) if sys_added else 0), int(sys_added), []
    for k, t in enumerate(turns):
        if t["role"] == "assistant":
            nxt = turns[k + 1]["content"] if k + 1 < len(turns) and turns[k + 1]["role"] == "user" else ""
            info.append((len(t["content"]), nxt.startswith(CUT_NOTICE), "</think>" not in t["content"],
                         ctx_chars / 3.6 + 6 * (ctx_msgs + 1)))
            ctx_chars += len(visible(t["content"]))
        else:
            ctx_chars += len(t["content"])
        ctx_msgs += 1
    return dict(key=tr.get("tokens", 1e18), tokens=tr.get("tokens"), hash=h, n_turns=len(asst),
                turn_chars=[x[0] for x in info], cut_turns=sum(x[1] for x in info), unclosed_think=sum(x[2] for x in info),
                ctx_tokens_approx=[x[3] for x in info], turn_tokens=gen)


def sft_candidates(spec, task_cap=0, max_traj_tokens=0, max_msg_tokens=0, system_added=False):
    """sft_student.build_samples' candidate pool: every verified trajectory of every record of the --data files, in file
    order, per task; then the --task-cap filter (tasks whose shortest verified trajectory has <= task_cap tokens)."""
    cand, nrec = {}, 0
    for p in sft_paths(spec):
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            nrec += 1
            for tr in r.get("trajectories", []):
                if max_traj_tokens and tr.get("tokens", 0) > max_traj_tokens:
                    continue
                if max_msg_tokens and max(len(t["content"]) for t in tr["turns"] if t["role"] == "assistant") / 3.6 > max_msg_tokens:
                    continue
                cand.setdefault(r["task_id"], []).append(traj_summary(tr, r.get("msg_log"), system_added))
    if task_cap:
        cand = {t: v for t, v in cand.items() if min(tr["key"] for tr in v) <= task_cap}
    return cand, nrec


def sft_select(cand, select, seed, per_task=1):
    """sft_student.build_samples' pick(): sort by tokens (stable); longest = reversed[:per_task]; random =
    rng.sample(sorted, per_task) with one random.Random(seed) for all tasks in sorted task order; shortest = [:per_task]."""
    rng = random.Random(seed)
    out = {}
    for tid, v in sorted(cand.items()):
        v = sorted(v, key=lambda tr: tr["key"])
        if select == "longest":
            out[tid] = v[::-1][:per_task]
        elif select == "random":
            out[tid] = rng.sample(v, min(per_task, len(v)))
        else:
            out[tid] = v[:per_task]
    return out


def excluded(tid, bad):
    return tid in bad or tid.split("_")[0] in bad


def dist(x):
    x = np.asarray([v for v in x if v is not None], float)
    if not len(x):
        return None
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return dict(n=len(x), mean=rnd(x.mean(), 1), sd=rnd(x.std(ddof=1), 1) if len(x) > 1 else None, min=rnd(x.min(), 1),
                p10=rnd(q[0], 1), p25=rnd(q[1], 1), median=rnd(q[2], 1), p75=rnd(q[3], 1), p90=rnd(q[4], 1),
                max=rnd(x.max(), 1), sum=rnd(x.sum(), 0))


def sft_cache_stats(path, bad):
    """sft_student's own cache: (samples, skipped, n_traj), samples = [(context ids, target ids, task id)], saved before
    the --exclude-ids filter, which is applied here as in sft_student."""
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return dict(note=f"torch unavailable: {type(e).__name__}")
    samples, skipped, n_traj = torch.load(path, weights_only=False)
    kept = [x for x in samples if not excluded(x[2], bad)]
    tl = [len(x[1]) for x in kept]
    cl = [len(x[0]) for x in kept]
    per_task = Counter(x[2] for x in kept)
    return dict(path=path, n_traj_before_exclusion=n_traj, samples_before_exclusion=len(samples), skipped_too_long=skipped,
                excluded_samples=len(samples) - len(kept), samples=len(kept), tasks=len(per_task),
                loss_tokens=int(sum(tl)), context_tokens=int(sum(cl)), target_tokens_per_example=dist(tl),
                context_tokens_per_example=dist(cl), examples_per_task=dist(list(per_task.values())))


def sft_log_stats(student):
    """Best effort: sft_chain.sh's log names the jobs; each job log (logs/sft-<job>.out) prints the dataset line and
    'step k/steps'."""
    path = os.path.join("logs", f"sftchain_{student}.log")
    if not os.path.exists(path):
        return None
    jobs = re.findall(r"iter \d+: job (\d+)", open(path, errors="replace").read())
    ds, steps, excl = None, set(), None
    for j in jobs:
        fn = os.path.join("logs", f"sft-{j}.out")
        if not os.path.exists(fn):
            continue
        txt = open(fn, errors="replace").read()
        m = re.search(r"(\d+) trajectories from (\d+) tasks -> (\d+) samples \(skipped (\d+) > (\d+) tokens\) \| target "
                      r"tokens ([\d.]+)M, context tokens ([\d.]+)M", txt)
        if m and ds is None:
            ds = dict(trajectories=int(m.group(1)), tasks=int(m.group(2)), samples=int(m.group(3)), skipped=int(m.group(4)),
                      max_len=int(m.group(5)), target_tokens_M=float(m.group(6)), context_tokens_M=float(m.group(7)))
        steps.update(int(x) for x in re.findall(r"step \d+/(\d+)", txt))
        m = re.search(r"excluded (\d+) samples", txt)
        if m:
            excl = int(m.group(1))
    return dict(chain_log=path, jobs=len(jobs), dataset_line=ds, steps_total=sorted(steps), excluded_samples=excl)


def training_set(key, cfg, students_dir):
    cand, nrec = sft_candidates(cfg["data"], cfg.get("task_cap", 0), system_added=cfg["system_prompt_added"])
    if not cand:
        return None
    bad = set(read_ids(cfg.get("exclude")) or [])
    nver = Counter(len(v) for v in cand.values())
    ndist = sum(len({tr["hash"] for tr in v}) >= 2 for v in cand.values())
    nlen = sum(len({tr["key"] for tr in v}) >= 2 for v in cand.values())
    sel = {a: sft_select(cand, how, cfg["seed"]) for a, (how, _) in cfg["arms"].items()}
    keep = [t for t in sorted(cand) if not excluded(t, bad)]
    out = dict(script=cfg["script"], data=cfg["data"], harvest_records=nrec, tasks_with_verified=len(cand),
               task_cap=cfg.get("task_cap", 0), exclude_list=cfg.get("exclude"),
               exclude_list_found=None if not cfg.get("exclude") else bool(bad), excluded_tasks=len(cand) - len(keep),
               tasks_trained=len(keep),
               verified_per_task={str(k): v for k, v in sorted(nver.items())},
               tasks_ge2_verified=sum(len(v) >= 2 for v in cand.values()), tasks_ge2_distinct=ndist,
               tasks_ge2_distinct_lengths=nlen,
               recipe=dict(seed=cfg["seed"], grad_accum=cfg["accum"], processes=cfg["world"], epochs=1.0,
                           effective_batch=cfg["accum"] * cfg["world"], max_len=cfg["max_len"], lora="r 32, alpha 64",
                           lr="5e-5, AdamW (0.9, 0.99), warm-up 20 steps then cosine", sft_end=repr(cfg["sft_end"]),
                           arc_system_prompt_prepended=cfg["system_prompt_added"],
                           loss_normalisation="per example: mean cross-entropy over its target tokens (chunked_loss "
                                              "returns total / len(target)); examples weighted equally within the "
                                              "accumulation (loss / accum) and gradients averaged over processes; so "
                                              "every example has the same total weight and a token of a long target "
                                              "weighs less than a token of a short one"),
               arms={}, same_pick={})
    arms_kept = {a: {t: sel[a][t] for t in keep} for a in sel}
    if len(sel) > 1:
        for a, b in itertools.combinations(sel, 2):
            same = sum(arms_kept[a][t][0]["hash"] == arms_kept[b][t][0]["hash"] for t in keep)
            out["same_pick"][f"{a}={b}"] = dict(tasks=same, share=rnd(100 * same / max(1, len(keep)), 2))
    for a, (how, student) in cfg["arms"].items():
        trs = [tr for t in keep for tr in arms_kept[a][t]]
        ex = sum(tr["n_turns"] for tr in trs)
        turns = [(c, tt) for tr in trs
                 for c, tt in zip(tr["ctx_tokens_approx"], tr["turn_tokens"] or [ch / 3.6 for ch in tr["turn_chars"]])]
        fits = [c + tt + cfg["end_tokens"] <= cfg["max_len"] for c, tt in turns]
        r = dict(select=how, student=student, trajectories=len(trs), target_tokens_per_trajectory=dist([tr["tokens"] for tr in trs]),
                 examples=ex, turns_per_trajectory=dist([tr["n_turns"] for tr in trs]),
                 cut_turns=int(sum(tr["cut_turns"] for tr in trs)), unclosed_think_turns=int(sum(tr["unclosed_think"] for tr in trs)),
                 tokens_per_turn=dist([tt for _, tt in turns]),
                 trajectories_with_turn_tokens_from_msg_log=sum(tr["turn_tokens"] is not None for tr in trs),
                 approx_examples_too_long=int(len(fits) - sum(fits)),
                 approx_loss_tokens=rnd(sum(tt + cfg["end_tokens"] for (_, tt), ok in zip(turns, fits) if ok), 0),
                 approx_steps=math.ceil(sum(fits) / (cfg["accum"] * cfg["world"])),
                 approx_note="generated tokens per turn (msg_log) or characters / 3.6; context = characters / 3.6 + 6 "
                             "per message; examples longer than max_len are skipped by sft_student")
        cp = os.path.join(students_dir, student, "samples_cache.pt") if students_dir else None
        if cp and os.path.exists(cp):
            ex_stats = sft_cache_stats(cp, bad)
            r["exact_from_cache"] = ex_stats
            if "samples" in ex_stats:
                r["steps"] = math.ceil(ex_stats["samples"] / (cfg["accum"] * cfg["world"]))
        r["logs"] = sft_log_stats(student)
        out["arms"][a] = r
    if "S" in sel and "L" in sel:
        ratio = [arms_kept["L"][t][0]["tokens"] / arms_kept["S"][t][0]["tokens"] for t in keep
                 if arms_kept["S"][t][0]["tokens"] and arms_kept["L"][t][0]["hash"] != arms_kept["S"][t][0]["hash"]]
        out["L_over_S_length_ratio_where_different"] = dist(ratio)
    return out


# ============================================================================ item 13: cap interaction, replicates, registry
def cap_section():
    """paper_stats.cap_interaction recomputed (same problems, same resamples) with sign-flip P (the two caps swapped
    within a problem), plus the logit-scale interaction of the arms' accuracies and the change in the relative token
    saving, with CIs."""
    D8, D16 = Single("math_qwen4b", SINGLE["math_qwen4b"]), Single("math_qwen4b_cap16k", SINGLE["math_qwen4b_cap16k"])
    if not (D8.ok and D16.ok):
        return None
    ids = sorted(set(D8.ids) & set(D16.ids))
    p8, p16 = {i: k for k, i in enumerate(D8.ids)}, {i: k for k, i in enumerate(D16.ids)}
    i8, i16 = [p8[i] for i in ids], [p16[i] for i in ids]
    n = len(ids)
    out = dict(n_problems=n, contrasts={})
    logit = lambda p: np.log(p / (1 - p))
    for a, b in (("S", "L"), ("S", "base"), ("L", "base")):
        if not all(x in D.arms for D in (D8, D16) for x in (a, b)):
            continue
        X8 = np.vstack([D8.A[a][i8], D8.A[b][i8], D8.F[a][i8], D8.F[b][i8], D8.T[a][i8], D8.T[b][i8]])
        X16 = np.vstack([D16.A[a][i16], D16.A[b][i16], D16.F[a][i16], D16.F[b][i16], D16.T[a][i16], D16.T[b][i16]])

        def f(S8, S16):
            d = lambda S, j: (S[..., j] - S[..., j + 1]) / n
            lg = lambda S: logit(S[..., 0] / n) - logit(S[..., 1] / n)
            lr = lambda S: np.log(S[..., 4] / S[..., 5])
            return np.stack([100 * d(S8, 0), 100 * d(S16, 0), 100 * (d(S8, 0) - d(S16, 0)),
                             100 * d(S8, 2), 100 * d(S16, 2), 100 * (d(S8, 2) - d(S16, 2)),
                             lg(S8), lg(S16), lg(S8) - lg(S16),
                             100 * (np.exp(lr(S8)) - 1), 100 * (np.exp(lr(S16)) - 1), lr(S8) - lr(S16)], -1)

        names = ["acc_8k", "acc_16k", "acc_8k_minus_16k", "finish_8k", "finish_16k", "finish_8k_minus_16k",
                 "logit_8k", "logit_16k", "logit_8k_minus_16k", "tokens_rel_pct_8k", "tokens_rel_pct_16k",
                 "tokens_log_ratio_8k_minus_16k"]
        r = tests_multi(X8, X16, f, names)
        for k in names:   # the swap exchanges caps: its P tests only the interactions; the per-cap rows keep the bootstrap
            if "minus" not in k:
                r[k].pop("P_perm", None)
                r[k].pop("p_perm", None)
        out["contrasts"][f"{a}-{b}"] = r
    return out


def replicates_section(fmts=None):
    """paper_stats.replicates reproduced exactly (its random.Random(0) stream, drawing three runs and then the problems
    for each resample, contrasts in the order S-R, S-L, R-L) plus two sign-flip tests: problem-level (one flip per
    problem shared by the three runs; tests the sharp null of no arm difference given these runs) and run-level (exact,
    2^3 flips of the three run differences; the smallest attainable two-sided P is 0.25)."""
    fmts = fmts or {"rep0": "math_eval_{a}_s{s}", "rep1": "math_eval_{a}rep1_s{s}", "rep2": "math_eval_{a}rep2_s{s}"}
    arms = ("S", "R", "L")
    acc = {}
    for rep, fmt in fmts.items():
        for a in arms:
            rr = [R for R in (load(fmt.format(a=a, s=s)) for s in range(5)) if len(R) >= N_MIN]
            if not rr:
                return None
            ids = set.intersection(*[set(R) for R in rr])
            acc[rep, a] = {i: sum(R[i]["acc"] for R in rr) / len(rr) for i in ids}
    reps = list(fmts)
    ids = sorted(set.intersection(*[set(v) for v in acc.values()]))
    n = len(ids)
    rng = random.Random(0)
    out = dict(n_problems=n, contrasts={})
    for a, b in (("S", "R"), ("S", "L"), ("R", "L")):
        d = np.array([[acc[r, a][i] - acc[r, b][i] for i in ids] for r in reps])     # (runs, problems)
        per_rep = 100 * d.mean(1)
        bs = []
        for _ in range(B):
            rs = [rng.choice(reps) for _ in reps]
            I = [rng.choice(range(n)) for _ in range(n)]
            bs.append(float(np.mean([100 * d[reps.index(r), I].mean() for r in rs])))
        bs = np.sort(np.round(bs, 9))
        p1 = float(np.mean(bs <= 0))
        flip = mean_test(d.mean(0))
        sg = [abs(np.mean(per_rep * np.array(s))) >= abs(per_rep.mean()) - 1e-9 for s in itertools.product((1, -1), repeat=len(reps))]
        out["contrasts"][f"{a}-{b}"] = dict(per_replicate=[rnd(x) for x in per_rep], mean=rnd(per_rep.mean()),
                                            hier_boot=dict(lo=rnd(bs[int(0.025 * B)]), hi=rnd(bs[int(0.975 * B)]), p=round(p1, 4),
                                                           p_two_sided=round(two_sided(p1), 4)),
                                            signflip_problems=dict(P_perm=flip.get("P_perm"), p_perm=flip.get("p_perm")),
                                            signflip_runs_exact=dict(P=rnd(sum(sg) / len(sg), 4), flips=len(sg)))
    return out


def registry_section():
    """Every run directory in runs_synth/: files, records, unique tasks, modification times, record fields, and the
    mean accuracy / pass@2 of eval-like records (harvest files larger than 50 MB are only counted)."""
    rows = []
    for d in sorted(glob.glob("runs_synth/agent_*")):
        files = sorted(glob.glob(os.path.join(d, "shard_*.jsonl")))
        if not files:
            continue
        size = sum(os.path.getsize(f) for f in files)
        mt = [os.path.getmtime(f) for f in files]
        iso = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(timespec="seconds")
        row = dict(tag=os.path.basename(d)[len("agent_"):], shards=len(files), bytes=size, first_mtime_utc=iso(min(mt)),
                   last_mtime_utc=iso(max(mt)))
        if size <= 50e6:
            ids, keys, acc, p2, lines = set(), Counter(), [], [], 0
            for f in files:
                for l in open(f):
                    if l.strip():
                        r = json.loads(l)
                        lines += 1
                        if r["task_id"] in ids:
                            continue
                        ids.add(r["task_id"])
                        keys.update(k for k in r.keys() if k not in ("trajectories", "transcripts", "msg_log", "programs", "attempts"))
                        if "acc" in r:
                            acc.append(float(r["acc"]))
                        if "pass2" in r:
                            p2.append(bool(r["pass2"]))
            row.update(lines=lines, unique_tasks=len(ids), fields=sorted(keys),
                       mean_acc=rnd(100 * np.mean(acc), 2) if acc else None, pass2=int(sum(p2)) if p2 else None)
        else:
            row["lines"] = sum(1 for f in files for l in open(f, "rb") if l.strip())
        rows.append(row)
    return dict(n_runs=len(rows), runs=rows) if rows else None


# ============================================================================ derived sections (from the merged output)
def get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def selection_tails(res, PS):
    """One-sided bootstrap tails P(diff <= 0) of the eight selection contrasts: from this script's records-based
    sections when present, else from paper_stats.json."""
    rows = []
    for key, m in SELECTION_SETTINGS:
        for o in "RL":
            c = f"S-{o}"
            if key == "arc":
                mine = get(res, "item5_arc", "contrasts", c, "puzzle_bootstrap", "solved")
            else:
                mine = get(res, "item4_quality", key, "contrasts", c, "acc")
            ref = get(PS, key, "contrasts", c, m)
            src = mine if mine and mine.get("p") is not None else ref
            if src is None:
                continue
            rows.append(dict(setting=key, contrast=c, diff=src["diff"], lo=src.get("lo"), hi=src.get("hi"), p=src["p"],
                             P=round(two_sided(src["p"]), 4), p_perm=(mine or {}).get("p_perm"), P_perm=(mine or {}).get("P_perm"),
                             source="records (this script)" if src is mine else "paper_stats.json"))
    return rows


def confirmatory_section(res, PS):
    rows = selection_tails(res, PS)
    if not rows:
        return None
    out = dict(definition="confirmatory family: the selection rule's effect pooled over the settings, shortest minus "
                          "random and shortest minus longest replay (Stouffer's z on the one-sided bootstrap tails, "
                          "tails clipped to [1/B, 1 - 1/B], two-sided P = 2 min(P1, 1 - P1)); Holm over the two. "
                          "Everything else in this file is exploratory.", settings=rows, tests={})
    zp = lambda tails: dict(zip(("z", "P_one_sided", "P"), (lambda z, p1, P: (rnd(z), sig(p1), sig(P)))(*stouffer(tails))))
    for o in "RL":
        sub = [r for r in rows if r["contrast"] == f"S-{o}"]
        tails = [r["p"] for r in sub]
        t = dict(settings=[r["setting"] for r in sub], **zp(tails))
        t["leave_one_out"] = {r["setting"]: zp([q["p"] for q in sub if q is not r]) for r in sub}
        t["without_arc"] = zp([r["p"] for r in sub if r["setting"] != "arc"]) if any(r["setting"] == "arc" for r in sub) else None
        t["partial_conjunction_one_sided"] = {f"at_least_{u}_of_{len(tails)}": sig(partial_conjunction(tails, u))
                                              for u in range(1, len(tails) + 1)}
        pp = [r["p_perm"] for r in sub]
        if all(x is not None for x in pp) and len(pp) == len(sub):
            t["signflip_stouffer"] = zp(pp)
        tw = get(res, "item5_arc", "contrasts", f"S-{o}", "twoway_bootstrap", "solved")
        if tw and any(r["setting"] == "arc" for r in sub):
            t["with_arc_twoway_tail"] = zp([tw["p"] if r["setting"] == "arc" else r["p"] for r in sub])
        out["tests"][f"S-{o}"] = t
    hp = holm([max(out["tests"][c]["P"], 2 / B) for c in out["tests"]])
    for c, h in zip(out["tests"], hp):
        out["tests"][c]["P_holm"] = sig(h)
    return out


def tokens_section(res):
    """Token and cost claims with CIs, collected from the sections that computed them."""
    out = {}
    for key, v in (res.get("item4_quality") or {}).items():
        if not isinstance(v, dict) or "arms" not in v:
            continue
        out[key] = dict(arms={a: dict(tokens=s["tokens"], finished_mean_tokens=s.get("finished_mean_tokens")) for a, s in v["arms"].items()},
                        contrasts={c: {k: s[k] for k in ("tokens", "tokens_rel_pct", "finished_mean_tokens",
                                                         "finished_mean_tokens_rel_pct") if k in s}
                                   for c, s in v["contrasts"].items()})
    arc = res.get("item5_arc")
    if isinstance(arc, dict) and "contrasts" in arc:
        out["arc"] = dict(arms={a: dict(tokens_per_puzzle_k=s["tokens_per_puzzle_k"], tokens_per_solved_k=s["tokens_per_solved_k"])
                                for a, s in arc["arms"].items()},
                          contrasts={c: {m: {k: s[m].get(k) for k in ("solved_rel_pct", "tokens_per_puzzle_k", "tokens_per_puzzle_rel_pct",
                                                                       "tokens_per_solved_k", "tokens_per_solved_rel_pct")}
                                         for m in ("puzzle_bootstrap", "twoway_bootstrap")}
                                     for c, s in arc["contrasts"].items()})
    cap = res.get("item13_cap_interactions")
    if isinstance(cap, dict) and "contrasts" in cap:
        out["cap_8k_vs_16k"] = {c: {k: s[k] for k in ("tokens_rel_pct_8k", "tokens_rel_pct_16k", "tokens_log_ratio_8k_minus_16k")}
                                for c, s in cap["contrasts"].items()}
    return out or None


def families(res, PS):
    """Holm and Benjamini-Hochberg within each secondary family (bootstrap P floored at 2/B, the resolution, as
    derived_stats.py does; permutation P as computed)."""
    fam = {}

    def add(name, tid, t):
        if t and t.get("P") is not None:
            fam.setdefault(name, []).append(dict(id=tid, diff=t.get("diff"), lo=t.get("lo"), hi=t.get("hi"),
                                                 P=t.get("P_ties", t["P"]), P_perm=t.get("P_perm")))

    for r in selection_tails(res, PS):
        add("selection_contrasts", f"{r['setting']} {r['contrast']}", r)
    q = res.get("item4_quality") or {}
    for key, m in SELECTION_SETTINGS:          # 'never significantly worse than the untouched model' (referee 3)
        for a in ("S", "R", "L"):
            t = (get(res, "item5_arc", "contrasts", f"{a}-base", "puzzle_bootstrap", "solved") if key == "arc"
                 else get(q, key, "contrasts", f"{a}-base", "acc")) or get(PS, key, "contrasts", f"{a}-base", m)
            if t:
                add("versus_untouched", f"{key} {a}-base", dict(t, P=t.get("P", round(two_sided(t["p"]), 4))))
    for key in ("math_qwen4b", "math_llama8b", "math_qwen4b_cap16k", "apps_qwen4b", "math_qwen4b_rep1", "math_qwen4b_rep2"):
        v = q.get(key)
        if not isinstance(v, dict) or "contrasts" not in v:
            continue
        rep = "rep" in key
        for c, s in v["contrasts"].items():
            for term in ("finishing_term", "quality_term"):
                add("decomposition_replicate_runs" if rep else "decomposition_terms", f"{key} {c} {term}", s.get(term))
                add("decomposition_per_problem_replicate_runs" if rep else "decomposition_per_problem",
                    f"{key} {c} {term}", get(s, "per_problem_decomposition", term))
            if not rep:
                add("quality_on_fixed_finished_set", f"{key} {c}", get(v, "common_finished_set", "pairs", c, "diff"))
    cap = res.get("item13_cap_interactions") or {}
    for c, s in (cap.get("contrasts") or {}).items():
        for k in ("acc_8k_minus_16k", "finish_8k_minus_16k"):
            add("cap_interactions", f"{c} {k}", s.get(k))
    arc = res.get("item5_arc") or {}
    for c, s in (arc.get("contrasts") or {}).items():
        tw = get(s, "twoway_bootstrap", "solved")
        sl = get(s, "seed_level", "unpaired_permutation")
        if tw:
            add("arc_accuracy_seed_level", f"{c} solved", dict(tw, P_perm=(sl or {}).get("P")))
        for unit in ("messages_all", "messages_first"):
            for k, t in (get(s, unit, "puzzle_bootstrap") or {}).items():
                if k in ("cut_share", "code_share", "restart_cost_k"):
                    add(f"arc_{unit}", f"{c} {k}", t)
    loc = res.get("item5_arc_messages_local") or {}
    for c, s in (loc.get("contrasts") or {}).items():
        for k in ("cut_share", "code_share", "restart_cost_k"):
            add("arc_messages_local_message_level", f"{c} {k}", s.get(k))
        for k, t in s.items():
            if k.startswith("code_within_"):
                add("arc_messages_local_within_t", f"{c} {k}", t)
    out = {}
    for name, rows in fam.items():
        pb = [max(r["P"], 2 / B) for r in rows]
        for r, h, b in zip(rows, holm(pb), bh(pb)):
            r["P_holm"], r["P_bh"] = rnd(h, 4), rnd(b, 4)
        if all(r["P_perm"] is not None for r in rows):
            pp = [r["P_perm"] for r in rows]
            for r, h, b in zip(rows, holm(pp), bh(pp)):
                r["P_perm_holm"], r["P_perm_bh"] = rnd(h, 4), rnd(b, 4)
        out[name] = dict(m=len(rows), tests=rows)
    return out


# ============================================================================ printing
def fmt_t(t, unit=""):
    if not t or t.get("diff") is None:
        return "n/a"
    s = f"{t['diff']:+.3f}{unit} [{t['lo']:+.3f}, {t['hi']:+.3f}] P={t['P']:.4f}"
    return s + (f" Pperm={t['P_perm']:.4f}" if t.get("P_perm") is not None else "")


def print_summary(res, ran):
    print(f"natcomm_analyses: sections computed this run: {', '.join(ran) or 'none'}")
    q = res.get("item4_quality") or {}
    for key, v in q.items():
        if not isinstance(v, dict) or "contrasts" not in v:
            continue
        print(f"== {key}: {v['n_problems']} problems, cap {v['cap']}")
        for c, s in v["contrasts"].items():
            pp = s["per_problem_decomposition"]
            lee = s["lee_bounds"]["within_problem"]
            print(f"   {c:7s} acc {fmt_t(s['acc'])} | finishing {fmt_t(s['finishing_term'])} | quality {fmt_t(s['quality_term'])}"
                  f" | per-problem quality {fmt_t(pp['quality_term'])} | Lee dQ within problem [{lee['lower']}, {lee['upper']}]")
    arc = res.get("item5_arc")
    if isinstance(arc, dict) and "contrasts" in arc:
        print(f"== ARC: {arc['n_puzzles']} puzzles")
        for c, s in arc["contrasts"].items():
            print(f"   {c:7s} puzzle {fmt_t(s['puzzle_bootstrap']['solved'])} | two-way {fmt_t(s['twoway_bootstrap']['solved'])}"
                  f" | seeds P={s['seed_level']['unpaired_permutation']['P']}")
    loc = res.get("item5_arc_messages_local")
    if isinstance(loc, dict) and "arms" in loc:
        print("== ARC message logs (arc_msgs.json, message-level CIs)")
        for a, s in loc["arms"].items():
            print(f"   {a:4s} cut {s['cut_share']['value']:.2f} [{s['cut_share']['lo']:.2f}, {s['cut_share']['hi']:.2f}]  "
                  f"code {s['code_share']['value']:.2f}  restart cost {s['restart_cost_k']['value']:.2f}k "
                  f"[{s['restart_cost_k']['lo']:.2f}, {s['restart_cost_k']['hi']:.2f}]")
        for c, s in loc["contrasts"].items():
            print(f"   {c:7s} cut {fmt_t(s['cut_share'])} | restart cost {fmt_t(s['restart_cost_k'])}")
    conf = res.get("item13_confirmatory")
    if isinstance(conf, dict) and "tests" in conf:
        for c, t in conf["tests"].items():
            print(f"== confirmatory {c}: z = {t['z']:.2f}, P = {t['P']:.2g} (Holm {t['P_holm']:.2g}); partial conjunction "
                  + ", ".join(f"{k} {v}" for k, v in t["partial_conjunction_one_sided"].items()))


# ============================================================================ main
RECORD_SECTIONS = ("item4_quality", "item5_arc", "item9_training_sets", "item10_finishing", "item13_cap_interactions",
                   "item13_replicates", "item13_registry")
LOCAL_SECTIONS = ("item5_arc_summary", "item5_arc_messages_local")
DERIVED_SECTIONS = ("item13_confirmatory", "item13_tokens", "item13_multiplicity")


def main(argv=None):
    global B, B_PERM, N_MIN
    ap = argparse.ArgumentParser(description=" ".join(__doc__.split("\n")[:2]))
    ap.add_argument("--root", default=os.path.dirname(HERE), help="repository root (runs_synth/, logs/, synth/)")
    ap.add_argument("--out", default=None, help="output JSON (default <root>/paper_brevity/data/natcomm_stats.json)")
    ap.add_argument("--paper-stats", default=None, help="paper_stats.json (default <root>/paper_brevity/data/, then <root>/logs/)")
    ap.add_argument("--arc-msgs", default=None, help="arc_msgs.json (default <root>/paper_brevity/data/arc_msgs.json)")
    ap.add_argument("--only", default="", help="comma-separated sections to compute (default: all): "
                                               + ", ".join(RECORD_SECTIONS + LOCAL_SECTIONS))
    ap.add_argument("--no-merge", action="store_true", help="do not keep the sections already in the output file")
    ap.add_argument("--students-dir", default=None, help="sft_student outputs (default $WORK/ccapone0/students)")
    ap.add_argument("--boot", type=int, default=None, help=f"bootstrap resamples (default paper_stats.B = {B})")
    ap.add_argument("--perm", type=int, default=B_PERM, help="sign-flip / permutation draws")
    ap.add_argument("--n-min", type=int, default=N_MIN, help="records a single-sample run needs (paper_stats: 490)")
    a = ap.parse_args(argv)
    B = a.boot or B
    B_PERM, N_MIN = a.perm, a.n_min
    root = os.path.abspath(a.root)
    out_path = os.path.abspath(a.out) if a.out else os.path.join(root, "paper_brevity", "data", "natcomm_stats.json")
    if os.path.basename(out_path) == "paper_stats.json":
        sys.exit("refusing to write paper_stats.json")
    ps_path = os.path.abspath(a.paper_stats) if a.paper_stats else next(
        (p for p in (os.path.join(root, "paper_brevity", "data", "paper_stats.json"), os.path.join(root, "logs", "paper_stats.json"))
         if os.path.exists(p)), None)
    msgs_path = os.path.abspath(a.arc_msgs) if a.arc_msgs else os.path.join(root, "paper_brevity", "data", "arc_msgs.json")
    students = a.students_dir or (os.path.join(os.environ["WORK"], "ccapone0", "students") if os.environ.get("WORK") else None)
    os.chdir(root)
    PS = json.load(open(ps_path)) if ps_path and os.path.exists(ps_path) else None
    only = {s.strip() for s in a.only.split(",") if s.strip()}
    want = lambda s: not only or s in only
    res = {}
    if os.path.exists(out_path) and not a.no_merge:
        res = json.load(open(out_path))
    meta = res.get("meta") or {}
    meta.setdefault("sections", {})
    stamp = lambda inputs: dict(computed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                host=socket.gethostname(), python=platform.python_version(), B=B, B_perm=B_PERM, inputs=inputs)
    ran, skipped = [], {}

    def put(name, val, inputs, reason=None):
        if val is None:
            skipped[name] = reason or "inputs not found"
            if name not in res:
                res[name] = dict(skipped=skipped[name])
            return
        res[name] = jsonable(val)
        meta["sections"][name] = stamp(inputs)
        ran.append(name)

    if want("item4_quality") or want("item10_finishing"):
        q, objs, reasons = {}, {}, {}
        for key in SINGLE:
            sec, D = analyse_single(key)
            if sec is None:
                reasons[key] = D
            else:
                q[key], objs[key] = sec, D
        if q:
            q["check_vs_paper_stats"] = check_vs_paper_stats(q, PS)
            q["not_found"] = reasons
        if want("item4_quality"):
            put("item4_quality", q or None, {k: v["tags"] for k, v in q.items() if isinstance(v, dict) and "tags" in v},
                f"no single-sample records ({reasons})")
        if want("item10_finishing"):
            fin = None
            if objs:
                fin = dict(apps=apps_finishing(objs["apps_qwen4b"]) if "apps_qwen4b" in objs else None, crossing={})
                for key, D in objs.items():
                    if D.cut_kind != "length":
                        continue
                    fin["crossing"][key] = {f"{x}-{y}": crossing_block(D, x, y) for x, y in CONTRASTS if x in D.arms and y in D.arms}
                    if key in MAIN_SINGLE:
                        fin["crossing"][key]["where_cuts_change_by_base_finishing"] = tail_strata(get(q, key, "strata", "base_finish"))
                fin["crossing_note"] = crossing_block.__doc__.split("\n")[0].strip()
            put("item10_finishing", fin, sorted(objs), "no single-sample records")
    if want("item5_arc"):
        sec, D = analyse_arc()
        put("item5_arc", sec, {"tags": ARC["arms"], "dev_ids": ARC["dev"], "excl_ids": ARC["excl"]}, D if sec is None else None)
    if want("item5_arc_summary"):
        put("item5_arc_summary", arc_summary_section(PS), ps_path, "paper_stats.json not found")
    if want("item5_arc_messages_local"):
        put("item5_arc_messages_local", arc_messages_local(msgs_path, PS), msgs_path, f"{msgs_path} not found")
    if want("item9_training_sets"):
        ts = {}
        for key, cfg in TRAIN.items():
            t = training_set(key, cfg, students)
            if t:
                ts[key] = t
        put("item9_training_sets", ts or None, dict(students_dir=students, data={k: v["data"] for k, v in TRAIN.items()}),
            "no harvest records")
    if want("item13_cap_interactions"):
        put("item13_cap_interactions", cap_section(), ["math_eval_{a}_s{s}", "math16_eval_{a}_s{s}"], "no 8k/16k records")
    if want("item13_replicates"):
        put("item13_replicates", replicates_section(), ["math_eval_{a}_s{s}", "math_eval_{a}rep1_s{s}", "math_eval_{a}rep2_s{s}"],
            "no replicate records")
    if want("item13_registry"):
        put("item13_registry", registry_section(), "runs_synth/agent_*", "no runs_synth/agent_* directories")
    for name, fn in (("item13_confirmatory", lambda: confirmatory_section(res, PS)), ("item13_tokens", lambda: tokens_section(res)),
                     ("item13_multiplicity", lambda: families(res, PS))):
        put(name, fn(), "derived from the other sections of this file" + (" and paper_stats.json" if PS else ""))
    meta.update(script="synth/natcomm_analyses.py", B=B, B_perm=B_PERM, n_min=N_MIN, paper_stats=ps_path,
                last_run_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                last_run_host=socket.gethostname(), last_run_sections=ran, last_run_skipped=skipped,
                conventions=__doc__.split("CONVENTIONS")[1].split("SECTIONS")[0].strip())
    res["meta"] = meta
    res = {k: res[k] for k in ["meta"] + sorted(k for k in res if k != "meta")}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(jsonable(res), fh, indent=1)
    os.replace(tmp, out_path)
    print_summary(res, ran)
    print(f"wrote {out_path}")
    return res


if __name__ == "__main__":
    main()
