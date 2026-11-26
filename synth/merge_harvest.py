"""Expert iteration, step 2: mix the harvested ARC-2 programs into the BARC training set.

Each harvested record holds a task's demos and up to 8 demo-verified program sources. One training example is
written per (task, program): demos are the task's train pairs, target is the program. Harvested examples are
repeated `--repeat` times so they are not drowned by 156k BARC programs.

    python synth/merge_harvest.py --barc verified_train.jsonl --harvest 'harvest_*.jsonl' --out mixed.jsonl
"""
import argparse, glob, json

ap = argparse.ArgumentParser()
ap.add_argument("--barc", required=True); ap.add_argument("--harvest", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--repeat", type=int, default=50)
ap.add_argument("--max-programs", type=int, default=4)
a = ap.parse_args()

n_barc = n_new = n_tasks = 0
with open(a.out, "w") as out:
    for line in open(a.barc):
        out.write(line); n_barc += 1
    for f in sorted(glob.glob(a.harvest)):
        for line in open(f):
            r = json.loads(line)
            progs = r.get("programs") or []
            if not progs:
                continue
            n_tasks += 1
            pairs = [[i, o] for i, o in zip(r.get("demo_inputs", []), r.get("demo_outputs", []))] if "demo_inputs" in r else None
            if pairs is None:                       # solve.py stores demos implicitly: rebuild from the task file
                import os
                d = json.load(open(os.path.join("ARC-AGI-2/data/training", r["task_id"] + ".json")))
                pairs = [[p["input"], p["output"]] for p in d["train"]]
            for src in progs[: a.max_programs]:
                rec = json.dumps({"pid": f"arc2_{r['task_id']}", "source": src, "pairs": pairs, "seeds": []})
                for _ in range(a.repeat):
                    out.write(rec + "\n"); n_new += 1
print(f"wrote {a.out}: {n_barc} BARC + {n_new} harvested examples from {n_tasks} ARC-2 tasks "
      f"(<= {a.max_programs} programs each, repeated {a.repeat}x) = {n_barc + n_new} lines")
