"""Mediation check: is the brevity gain carried by more code executions? Per arm, solve rate stratified by the number of
code runs per task (explores), and solve probability per code run.
   python synth/brev_mediation.py ARM=tag1,tag2 ARM=tag3,...   (tags of runs_synth/agent_<tag>)"""
import collections, glob, json, sys

for spec in sys.argv[1:]:
    arm, tags = spec.split("=")
    recs = []
    for t in tags.split(","):
        recs += [json.loads(l) for f in glob.glob(f"runs_synth/agent_{t}/shard_*.jsonl") for l in open(f) if l.strip()]
    bins = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        k = min(r.get("explores", 0) + r.get("finals", 0), 6)
        bins[k][0] += bool(r.get("pass2")); bins[k][1] += 1
    runs = sum(r.get("explores", 0) + r.get("finals", 0) for r in recs); sol = sum(bool(r.get("pass2")) for r in recs)
    row = "  ".join(f"{k}:{s}/{n}" for k, (s, n) in sorted(bins.items()))
    print(f"{arm:9s} tasks {len(recs):3d} solved {sol:3d} | code executions {runs/len(recs):.2f}/task | solves per execution {sol/max(runs,1):.3f} | by executions {row}")
