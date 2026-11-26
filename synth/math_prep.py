"""Data for the math brevity study (run on a LOGIN node, needs internet):
  train pool = MATH train split (EleutherAI/hendrycks_math, 7 subjects), levels 3-5, 2000 problems (seed 0)
  test       = MATH-500 (HuggingFaceH4/MATH-500, drawn from the MATH *test* split -> disjoint from the train pool)
Writes synth/math/{train2000,test500}.jsonl (id, problem, answer, level) and matching id lists.
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset
sys.argv = [sys.argv[0], '--model', '-', '--ids', '-', '--out', '-']   # math_harvest parses its CLI at import
from math_harvest import boxed

os.makedirs("synth/math", exist_ok=True)
pool = []
for cfg in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"):
    for k, r in enumerate(load_dataset("EleutherAI/hendrycks_math", cfg, split="train")):
        lv = r.get("level", "")
        if lv in ("Level 3", "Level 4", "Level 5"):
            ans = boxed(r["solution"])
            if ans:
                pool.append(dict(id=f"tr_{cfg}_{k}", problem=r["problem"], answer=ans, level=lv))
random.Random(0).shuffle(pool); train = pool[:2000]
test = [dict(id=f"te_{r['unique_id'].replace('/', '_').replace('.json', '')}", problem=r["problem"], answer=r["answer"], level=r["level"])
        for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
for name, rows in (("train2000", train), ("test500", test)):
    with open(f"synth/math/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    open(f"synth/math/{name}_ids.txt", "w").write("\n".join(r["id"] for r in rows) + "\n")
print(f"train pool {len(pool)} -> {len(train)}; test {len(test)}")
