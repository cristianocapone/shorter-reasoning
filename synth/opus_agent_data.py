"""Turn the Opus generators (synth/opus_gen/*.py: generate_input + solve, verified) into induction-agent data:
  (1) SFT records in the harvest format read by distill/sft_student.py: one-message trajectories
      (system, first user message exactly as solve_agent builds it, assistant = brief reasoning + `# FINAL` program);
  (2) ARC-format task files for GRPO / evaluation (3 demos + 1 test with output).
Generators whose seed id is in --heldout are written only as evaluation tasks.

  python synth/opus_agent_data.py --k 12 --out-jsonl synth/opus_agent_sft.jsonl --task-dir synth/opus_tasks
"""
import argparse, copy, importlib.util, inspect, json, os, random, re, signal, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solve_agent as A

ap = argparse.ArgumentParser()
ap.add_argument("--gen-dir", default="synth/opus_gen"); ap.add_argument("--k", type=int, default=12)
ap.add_argument("--demos", type=int, default=3); ap.add_argument("--heldout", default="synth/opus_heldout16.txt")
ap.add_argument("--out-jsonl", required=True); ap.add_argument("--task-dir", required=True); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
held = set(open(a.heldout).read().split())
os.makedirs(a.task_dir, exist_ok=True)


def alarm(*_): raise TimeoutError()


signal.signal(signal.SIGALRM, alarm)


def rule_text(src):
    src = re.sub(r"#\s*[0-9a-f]{8}:", "#", src); m = re.search(r"#\s*Rule:(.*?)(?:\n(?!#)|\Z)", src, re.S)
    if m:
        r = " ".join(l.strip().lstrip("#").strip() for l in m.group(1).splitlines()).strip()
        return re.sub(r"\b[0-9a-f]{8}(_[fn])?\b:?\s*", "", r).strip()
    m = re.search(r'"""(.*?)"""', src, re.S)
    return " ".join(m.group(1).split())[:400] if m else ""


def program(mod_src):
    """Self-contained FINAL program: the generator file minus generate_input, plus a numpy transform wrapper."""
    body = re.sub(r"\ndef generate_input\(.*?(?=\ndef |\Z)", "\n", mod_src, flags=re.S)
    return ("# FINAL\n" + body.strip() + "\n\n\ndef transform(grid):\n    import numpy as np\n"
            "    return np.array(solve(np.asarray(grid).tolist()))\n")


n_sft = n_tasks = 0; train_ids, held_ids = [], []
with open(a.out_jsonl, "w") as fh:
    for f in sorted(os.listdir(a.gen_dir)):
        if not f.endswith(".py") or f.startswith("_"):
            continue
        name = f[:-3]; seed_id = name.split("_")[0]; src = open(os.path.join(a.gen_dir, f)).read()
        spec = importlib.util.spec_from_file_location(name, os.path.join(a.gen_dir, f)); mod = importlib.util.module_from_spec(spec)
        try:
            signal.alarm(20); spec.loader.exec_module(mod); signal.alarm(0)
        except Exception:
            signal.alarm(0); continue
        rule = re.sub(r"\b[0-9a-f]{8}(_[fn])?\b:?\s*", "", rule_text(src)).strip(); prog = program(src); rng = random.Random(f"{a.seed}-{name}")
        for k in range(a.k):
            pairs = []
            for _ in range(40):
                if len(pairs) == a.demos + 1:
                    break
                try:
                    signal.alarm(20); x = mod.generate_input(rng); y = mod.solve(copy.deepcopy(x)); signal.alarm(0)
                except Exception:
                    signal.alarm(0); continue
                if x != y and all(x != p[0] for p in pairs):
                    pairs.append((x, y))
            if len(pairs) < a.demos + 1:
                continue
            demos, (tx, ty) = pairs[:-1], pairs[-1]
            tid = f"{name}_{k}"
            json.dump({"train": [{"input": x, "output": y} for x, y in demos], "test": [{"input": tx, "output": ty}]},
                      open(os.path.join(a.task_dir, tid + ".json"), "w"))
            n_tasks += 1
            if seed_id in held:
                held_ids.append(tid); continue
            train_ids.append(tid)
            think = (f"Compare each input with its output. {rule} Check this on every train pair, then write it as a program."
                     if rule else "Compare each input with its output, find the rule, check it on every train pair, then write it.")
            ans = f"{think}\n</think>\n\nRULE: {rule or 'see program'}\n```python\n{prog}```"
            turns = [{"role": "system", "content": A.SYSTEM},
                     {"role": "user", "content": A.intro([(x, y) for x, y in demos], [tx], 6)},
                     {"role": "assistant", "content": ans}]
            fh.write(json.dumps({"task_id": tid, "trajectories": [{"rounds": 1, "tokens": len(ans) // 3, "turns": turns}]}) + "\n")
            n_sft += 1
open(os.path.join(a.task_dir, "train_ids.txt"), "w").write("\n".join(train_ids) + "\n")
open(os.path.join(a.task_dir, "heldout_ids.txt"), "w").write("\n".join(held_ids) + "\n")
print(f"SFT records {n_sft} | task files {n_tasks} (train {len(train_ids)}, held-out {len(held_ids)})")
