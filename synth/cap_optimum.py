"""Restart-cutoff analysis from the per-message log (msg_log: tokens, thinking chars, visible chars, cut, has_code).
For each cap t: P(code within t) and E[min(L, t)] over assistant messages -> expected tokens per code-producing message
(Luby–Sinclair–Zuckerman objective). Messages cut at the run's cap are right-censored (no code at any t <= cap).
   python synth/cap_optimum.py TAG [TAG ...]"""
import glob, json, sys
import numpy as np

caps = [4000, 6000, 8000, 10000, 12000, 14000, 16000, 20000, 24000]
for tag in sys.argv[1:]:
    msgs = [m for f in glob.glob(f"runs_synth/agent_{tag}/shard_*.jsonl") for l in open(f) if l.strip()
            for th in json.loads(l).get("msg_log", []) for m in th]
    if not msgs:
        print(f"{tag}: no msg_log"); continue
    tok = np.array([m[0] for m in msgs], float); code = np.array([m[4] for m in msgs], bool); cut = np.array([m[3] for m in msgs], bool)
    print(f"{tag}: {len(msgs)} messages, cut {cut.mean():.2f}, with code {code.mean():.2f}, mean tokens {tok.mean():.0f}")
    row = []
    for t in caps:
        p = np.mean(code & (tok <= t)); spent = np.minimum(tok, t).mean()
        row.append(f"{t//1000}k: P={p:.2f} cost={spent/max(p,1e-9)/1e3:.0f}k")
    print("   " + " | ".join(row))
