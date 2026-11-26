"""Confirmatory test set for the NatComm revision (run on a LOGIN node, needs internet).
  conf = MATH *test* split (EleutherAI/hendrycks_math, 7 subjects, all levels) minus the 500 problems of MATH-500,
         i.e. the ~4,500 test problems never evaluated in this project. Disjoint from the train pool (MATH train split).
  The answer is the last \\boxed{} of the reference solution (as for train2000); problems without one are dropped.
Pre-registered in paper_brevity/natcomm/PREREG_MATH4500.md before any evaluation on this set.
Writes synth/math/conf4500.jsonl (id, problem, answer, level, type) and synth/math/conf4500_ids.txt.
"""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset
sys.argv = [sys.argv[0], '--model', '-', '--ids', '-', '--out', '-']   # math_harvest parses its CLI at import
from math_harvest import boxed


def norm(p):
    return re.sub(r"\s+", " ", p).strip()


m500 = {norm(r["problem"]) for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
rows, n_test, n_in500, n_nobox = [], 0, 0, 0
for cfg in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"):
    for k, r in enumerate(load_dataset("EleutherAI/hendrycks_math", cfg, split="test")):
        n_test += 1
        if norm(r["problem"]) in m500:
            n_in500 += 1
            continue
        ans = boxed(r["solution"])
        if not ans:
            n_nobox += 1
            continue
        rows.append(dict(id=f"cf_{cfg}_{k}", problem=r["problem"], answer=ans, level=r.get("level", ""), type=cfg))
os.makedirs("synth/math", exist_ok=True)
with open("synth/math/conf4500.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
open("synth/math/conf4500_ids.txt", "w").write("\n".join(r["id"] for r in rows) + "\n")
h = hashlib.sha256(open("synth/math/conf4500.jsonl", "rb").read()).hexdigest()
print(f"MATH test {n_test}; matched to MATH-500 {n_in500} (expected 500); no boxed answer {n_nobox}; kept {len(rows)}")
print(f"sha256 conf4500.jsonl {h}")
