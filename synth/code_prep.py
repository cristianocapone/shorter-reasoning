"""Data for the code brevity study (LOGIN node, needs internet): APPS (codeparrot/apps), stdin/stdout problems only.
  train pool = APPS train split, all difficulties, >= 1 test case (only 322 have >= 3) -> 2000 problems (seed 0)
  test       = APPS test split, introductory + interview, >= 3 test cases -> 500 problems (seed 0); the remaining test-split
               problems top the train pool up to 2000 (ids disjoint from the 500).
Writes synth/code/{train2000,test500}.jsonl (id, question, inputs, outputs, difficulty) and id lists.
"""
import json, os, random
from huggingface_hub import hf_hub_download

os.makedirs("synth/code", exist_ok=True)


def load(split, levels, n, min_tests=3):
    path = hf_hub_download("codeparrot/apps", f"{split}.jsonl", repo_type="dataset")
    rows = []
    for l in open(path):
        r = json.loads(l)
        if r.get("difficulty") not in levels or r.get("starter_code"):
            continue
        try:
            io = json.loads(r["input_output"])
        except Exception:
            continue
        ins, outs = io.get("inputs") or [], io.get("outputs") or []
        if io.get("fn_name") or len(ins) < min_tests or len(ins) != len(outs):
            continue
        if not all(isinstance(x, str) and isinstance(y, str) for x, y in zip(ins, outs)):
            continue
        rows.append(dict(id=f"{split}_{r['id']}", question=r["question"], inputs=ins[:20], outputs=outs[:20],
                         difficulty=r["difficulty"]))
    random.Random(0).shuffle(rows)
    return rows[:n], len(rows)


train, ntr = load("train", ("introductory", "interview", "competition"), 2000, min_tests=1)   # APPS train: few problems have >= 3 tests
pool_te, nte = load("test", ("introductory", "interview"), 100000)
test = pool_te[:500]                                               # held-out
train = train + pool_te[500:500 + (2000 - len(train))]             # APPS train has only 532 stdin problems: top up from the rest of the test split (disjoint ids)
for name, rows in (("train2000", train), ("test500", test)):
    with open(f"synth/code/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    open(f"synth/code/{name}_ids.txt", "w").write("\n".join(r["id"] for r in rows) + "\n")
print(f"train pool {ntr} -> {len(train)}; test pool {nte} -> {len(test)}")
