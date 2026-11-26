"""(2) Occam: among demo-consistent finals, wrong-on-test rate vs size of the smallest consistent program (AST nodes),
pooled over every agent run with ground truth. (1) LSZ: per cap of the 4B sweep, tokens per executed program and per
solved task = the restart-cost objective E[min(L,t)]/P(L<=t) estimated from runs.
   python synth/occam_lsz.py"""
import ast, collections, glob, json

bins = collections.defaultdict(lambda: [0, 0]); per_run = collections.Counter()
for f in glob.glob("runs_synth/agent_*/shard_*.jsonl"):
    run = f.split("/")[1]
    for l in open(f):
        if not l.strip(): continue
        r = json.loads(l)
        if not r.get("consistent") or "pass2" not in r or not r.get("programs"): continue
        sizes = []
        for p in r["programs"]:
            try: sizes.append(sum(1 for _ in ast.walk(ast.parse(p))))
            except Exception: pass
        if not sizes: continue
        s = min(sizes); b = min(s // 100 * 100, 1000)
        bins[b][0] += not r["pass2"]; bins[b][1] += 1; per_run[run] += 1
print("OCCAM: demo-consistent records, wrong-on-test rate by smallest consistent program size (AST nodes)")
tot = [0, 0]
for b in sorted(bins):
    w, n = bins[b]; tot[0] += w; tot[1] += n
    print(f"  {b:4d}-{b+99 if b < 1000 else '':<4} n={n:5d}  wrong {w/n:.2f}")
print(f"  all n={tot[1]} wrong {tot[0]/tot[1]:.2f} from {len(per_run)} runs")
print("LSZ (4B sweep, 60 ARC-AGI-1 eval tasks): cap, tokens/task, code runs/task, tokens per executed program, tokens per solve")
for cap in (4000, 6000, 8000, 12000, 16000, 24000, 32000):
    recs = [json.loads(l) for f in glob.glob(f"runs_synth/agent_phys_smoke_4b_c{cap}/shard_*.jsonl") for l in open(f) if l.strip()]
    if len(recs) < 50: continue
    tok = sum(r["tokens"] for r in recs); runs = sum(r.get("explores", 0) + r.get("finals", 0) for r in recs); sol = sum(bool(r.get("pass2")) for r in recs)
    print(f"  {cap:6d} {tok/len(recs)/1e3:6.0f}k {runs/len(recs):5.2f} {tok/max(runs,1)/1e3:8.0f}k {tok/max(sol,1)/1e3:8.0f}k  (solved {sol}/{len(recs)})")
