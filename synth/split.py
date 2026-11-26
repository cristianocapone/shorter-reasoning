"""Fixed split of the verified BARC programs: train / held-out (by program), streaming, pairs capped.

Held-out problems are never seen by the proposer. A truncated last line (job killed mid-write) is skipped.
Also marks which held-out ARC-2 training tasks are ARC-1-derived (their rule family seeded BARC programs).

    python synth/split.py --data verified.jsonl --n-heldout 2000 --max-pairs 8
"""
import argparse, glob, json, os, re, zlib
import numpy as np

SEED_RE = re.compile(r"^[0-9a-f]{8}\.py$")  # the seeds field also holds in-context example programs; keep ARC ids

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--n-heldout", type=int, default=2000)
ap.add_argument("--max-pairs", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
base = a.data[: -len(".jsonl")]
n_total = bad = 0
for _ in open(a.data):
    n_total += 1
# held-out chosen by a stable hash of the program id, so the split does not need the file in memory
thresh = a.n_heldout / max(n_total, 1)
seeds, n_tr, n_ho = set(), 0, 0
rng = np.random.default_rng(a.seed)
with open(a.data) as fin, open(base + "_train.jsonl", "w") as tr, open(base + "_heldout.jsonl", "w") as ho:
    for line in fin:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1; continue
        if len(r["pairs"]) > a.max_pairs:
            sel = sorted(rng.choice(len(r["pairs"]), a.max_pairs, replace=False).tolist())
            r["pairs"] = [r["pairs"][i] for i in sel]
        r["seeds"] = [s[:8] for s in r.get("seeds") or [] if isinstance(s, str) and SEED_RE.match(s)]
        is_ho = (zlib.crc32(str(r["pid"]).encode()) % 1_000_000) / 1_000_000 < thresh
        (ho if is_ho else tr).write(json.dumps(r) + "\n")
        if is_ho:
            n_ho += 1
        else:
            n_tr += 1; seeds.update(r["seeds"])
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ids = [l.strip() for l in open(os.path.join(root, "synth", "arc2_train_heldout100.txt")) if l.strip()]
a1 = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(root, "ARC-AGI-1", "data", "training", "*.json"))}
flag = {t: {"arc1_training": t in a1, "barc_seed": t in seeds} for t in ids}
json.dump(flag, open(base + "_arc2heldout_contamination.json", "w"), indent=1)
print(f"lines {n_total} (unparseable {bad}) | train {n_tr} | heldout {n_ho} | pairs capped at {a.max_pairs} | "
      f"distinct seed tasks in train {len(seeds)} | ARC-2 held-out tasks that are ARC-1 training: "
      f"{sum(v['arc1_training'] for v in flag.values())}/100, seeded a training program: {sum(v['barc_seed'] for v in flag.values())}/100")
