"""Mechanism observables of agent runs: per assistant message, cut-off rate (the model was still reasoning when the
per-message cap hit), code presence; per trajectory, tokens and rounds; per task, solve and code runs.
   python synth/agent_observables.py TAG [TAG ...]"""
import glob, json, re, sys
import numpy as np

CUT = "(no answer: the message was cut off)"
for tag in sys.argv[1:]:
    recs = [json.loads(l) for f in glob.glob(f"runs_synth/agent_{tag}/shard_*.jsonl") for l in open(f) if l.strip()]
    msgs = [m for r in recs for t in r.get("transcripts", []) for m in (t.get("turns", []) if isinstance(t, dict) else t)
            if m.get("role") == "assistant"]
    cut = sum(m["content"] == CUT for m in msgs)
    code = sum("```python" in m["content"] for m in msgs)
    tok = [t.get("tokens", 0) for r in recs for t in r.get("transcripts", []) if isinstance(t, dict)]
    rounds = [t.get("rounds", 0) for r in recs for t in r.get("transcripts", []) if isinstance(t, dict)]
    print(f"{tag:28s} tasks {len(recs):3d} solved {sum(bool(r.get('pass2')) for r in recs):3d} | assistant msgs {len(msgs):4d} "
          f"cut {cut/max(len(msgs),1):.2f} with-code {code/max(len(msgs),1):.2f} | code runs/task {np.mean([r.get('explores',0) for r in recs]):.2f} "
          f"finals/task {np.mean([r.get('finals',0) for r in recs]):.2f} | tokens/traj {np.mean(tok) if tok else 0:.0f} rounds/traj {np.mean(rounds) if rounds else 0:.1f}")
