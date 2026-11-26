"""Write compact text views of the eval80 tasks (public eval minus dev-40) for the Opus puzzle-generator agents.
Each file: demos (input/output) and the test pair as rows of digits. dev-40 ids are never written."""
import json, os
ch = json.load(open("TRM/kaggle/combined/arc-agi_evaluation2_challenges.json"))
so = json.load(open("TRM/kaggle/combined/arc-agi_evaluation2_solutions.json"))
e80 = open("synth/eval80_train.txt").read().split(); dev = set(open("synth/eval40_dev.txt").read().split())
assert not (set(e80) & dev)
def g2s(g): return "\n".join("".join(str(v) for v in r) for r in g)
for t in e80:
    out = [f"# ARC-AGI-2 task {t}: rows of digits = colours 0-9 (0 = black background)"]
    for i, p in enumerate(ch[t]["train"]):
        out += [f"\n## demo {i+1} input ({len(p['input'])}x{len(p['input'][0])})", g2s(p["input"]), f"## demo {i+1} output ({len(p['output'])}x{len(p['output'][0])})", g2s(p["output"])]
    for i, p in enumerate(ch[t]["test"]):
        out += [f"\n## test {i+1} input ({len(p['input'])}x{len(p['input'][0])})", g2s(p["input"]), f"## test {i+1} output ({len(so[t][i])}x{len(so[t][i][0])})", g2s(so[t][i])]
    open(f"synth/eval80_compact/{t}.txt", "w").write("\n".join(out) + "\n")
print(len(e80), "tasks written")
