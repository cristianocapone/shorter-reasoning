#!/usr/bin/env python3
"""Exploratory (added 2026-11-26 after an independent review): how much of the gain that survives a forced answer is
predicted by faster progress alone? Speed-only prediction for a method M against its reference B at budget t:
forced_B(t / lam) - forced_B(t), with lam = median length ratio (M / B) and forced_B interpolated linearly on log2(budget).
Writes paper_brevity/data/natcomm_speed_only.json."""
import glob, json, os
import numpy as np
E = json.load(open("paper_brevity/data/natcomm_ext.json"))
A = json.load(open("paper_brevity/data/conf4500_forced.json"))
Q = json.load(open("paper_brevity/data/natcomm_32k.json"))

def med_ext(arm):
    x = []
    for f in glob.glob(f"runs_synth_ext/runs_synth/agent_ncx_{arm}_s0/shard_*.jsonl"):
        for l in open(f): x += json.loads(l)["lens"]
    return float(np.median(x))

def interp(curve, t):  # curve: {budget: acc}; linear in log2 budget, flat beyond the ends
    b = sorted(curve); lb = np.log2(b); y = [curve[k] for k in b]
    return float(np.interp(np.log2(t), lb, y))

out = []
for m, b in (("arora_a02", "r1base"), ("thinkprune", "r1base"), ("dler", "r1base"), ("l1max4k", "dsr")):
    lam = med_ext(m) / med_ext(b)
    cb = {int(k): v["forced"] for k, v in E["arms"][b].items()}
    for t in (2048, 4096, 8192, 16384):
        obs = E["arms"][m][str(t)]["forced"] - E["arms"][b][str(t)]["forced"]
        pred = interp(cb, t / lam) - cb[t]
        out.append(dict(method=m, ref=b, budget=t, lam=round(lam, 3), observed=round(obs, 2), speed_only=round(pred, 2)))
# shortest vs random replay on 4,500 problems: lam = ratio of mean tokens (forced answers counted), R's curve from 4k and 8k
r = A["arms"]["R"]; s = A["arms"]["S"]
cR = {4096: r["4096"]["forced"], 8192: r["8192"]["forced"]}
slope = (cR[8192] - cR[4096])  # points per doubling
for t in (4096, 8192):
    lam = s[str(t)]["tokens"] / r[str(t)]["tokens"]
    pred = slope * np.log2(1 / lam)
    obs = s[str(t)]["forced"] - r[str(t)]["forced"]
    out.append(dict(method="S", ref="R (4,500 problems)", budget=t, lam=round(lam, 3), observed=round(obs, 2), speed_only=round(float(pred), 2)))
# shortest replay vs untouched on MATH-500 (32k generations): lam = ratio of mean tokens at 32k
F = Q["forced"]["arms"]; lam = Q["m32"]["arms"]["S"]["tokens"] / Q["m32"]["arms"]["base"]["tokens"]
cb = {int(k): v["forced"] for k, v in F["base"].items()}
for t in (4096, 8192, 16384):
    out.append(dict(method="S", ref="untouched (MATH-500)", budget=t, lam=round(lam, 3),
                    observed=round(F["S"][str(t)]["forced"] - cb[t], 2), speed_only=round(interp(cb, t / lam) - cb[t], 2)))
json.dump(out, open("paper_brevity/data/natcomm_speed_only.json", "w"), indent=1)
for o in out: print(o)
