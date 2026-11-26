"""Validate a puzzle generator written by an agent and render its pairs in the gen_pairs JSON format.

  python synth/opus_harness.py synth/opus_gen/NAME.py [--check TASKID] [--n 12]

The generator module must define  generate_input(rng: random.Random) -> grid  and  solve(grid) -> grid
(grid = list of lists of ints 0-9, 1..30 x 1..30). --check TASKID verifies solve() on the real demos + test of that
eval80 task (faithful re-implementations). On success writes synth/sdg_pairs_opus/NAME.json.
"""
import argparse, copy, importlib.util, json, os, random, signal, sys

ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--check", default=None); ap.add_argument("--n", type=int, default=12)
ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
name = os.path.splitext(os.path.basename(a.path))[0]
dev = set(open("synth/eval40_dev.txt").read().split())
assert a.check not in dev, "dev-40 task ids are off limits"


def fail(msg):
    print(f"FAIL {name}: {msg}"); sys.exit(1)


def ok_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and 1 <= len(r) <= 30 and len(r) == len(g[0])
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for v in r) for r in g))


def alarm(*_): raise TimeoutError("generator/solver too slow (>20 s)")


signal.signal(signal.SIGALRM, alarm)
spec = importlib.util.spec_from_file_location(name, a.path); mod = importlib.util.module_from_spec(spec)
try:
    signal.alarm(20); spec.loader.exec_module(mod); signal.alarm(0)
except Exception as e:
    fail(f"import error: {e!r}")
for fn in ("generate_input", "solve"):
    if not callable(getattr(mod, fn, None)): fail(f"missing {fn}()")

if a.check:
    ch = json.load(open("TRM/kaggle/combined/arc-agi_evaluation2_challenges.json"))
    so = json.load(open("TRM/kaggle/combined/arc-agi_evaluation2_solutions.json"))
    pairs = [(p["input"], p["output"]) for p in ch[a.check]["train"]] + [(p["input"], so[a.check][i]) for i, p in enumerate(ch[a.check]["test"])]
    for i, (x, y) in enumerate(pairs):
        try:
            signal.alarm(20); yh = mod.solve(copy.deepcopy(x)); signal.alarm(0)
        except Exception as e:
            fail(f"solve() crashed on real pair {i+1}: {e!r}")
        if yh != y: fail(f"solve() wrong on real pair {i+1} of {a.check} ({'demo' if i < len(ch[a.check]['train']) else 'test'})")
    print(f"check {a.check}: solve() reproduces all {len(pairs)} real pairs")

rng = random.Random(a.seed); out, seen = [], set(); tries = 0
while len(out) < a.n and tries < a.n * 6:
    tries += 1
    try:
        signal.alarm(20); x = mod.generate_input(rng); y = mod.solve(copy.deepcopy(x)); signal.alarm(0)
    except Exception as e:
        signal.alarm(0); print(f"  warn: exception in generate/solve: {e!r}"); continue
    if not ok_grid(x) or not ok_grid(y): print("  warn: invalid grid"); continue
    key = json.dumps(x)
    if key in seen: continue
    seen.add(key); out.append({"input": x, "output": y})
if len(out) < a.n: fail(f"only {len(out)} valid distinct pairs after {tries} tries")
if sum(p["input"] == p["output"] for p in out) > len(out) // 4: fail("output == input in more than 1/4 of the pairs (near-identity rule)")
if all(len(set(v for r in p["output"] for v in r)) == 1 for p in out): fail("outputs are constant grids")
if len(set(json.dumps(p["output"]) for p in out)) < len(out) // 2: fail("outputs repeat too much (degenerate rule)")
os.makedirs("synth/sdg_pairs_opus", exist_ok=True)
json.dump({"task": name, "pairs": out, "program": open(a.path).read(), "source": "opus55_gen"}, open(f"synth/sdg_pairs_opus/{name}.json", "w"))
sizes = sorted(set((len(p["input"]), len(p["input"][0])) for p in out))
print(f"OK {name}: {len(out)} pairs, input sizes {sizes[:4]}{'...' if len(sizes) > 4 else ''}, checked={a.check or '-'}")
