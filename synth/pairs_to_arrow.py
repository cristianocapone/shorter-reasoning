"""Generated puzzle pairs (synth/gen_inputs.py output) -> NVARC-format training rows (arrow), like build_datasets.py:
per task, R rows; each row = 5-6 (input, output) pairs sampled from the pool (original demos + generated pairs),
shuffled, with one random dihedral transform + one random colour permutation applied to every grid of the row
(row 0 of each task unaugmented). Columns: puzzle_name, messages[{role, content}] (grid = rows of digits joined by \\n).

python synth/pairs_to_arrow.py --pairs synth/gen_pairs_v1 --arc-dir synth/all_tasks --out $WORK/ccapone0/nvarc_aug/ours_gen_v1 --rows 32
"""
import argparse, glob, json, os, random
import numpy as np


def dihedral(g, k):
    g = np.asarray(g)
    if k >= 4:
        g = g.T
    return np.rot90(g, k % 4)


def grid_str(g):
    return "\n".join("".join(str(int(v)) for v in row) for row in np.asarray(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True); ap.add_argument("--arc-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=32); ap.add_argument("--min-pairs", type=int, default=5); ap.add_argument("--max-pairs", type=int, default=6)
    ap.add_argument("--exclude-ids", default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import pyarrow as pa, pyarrow.ipc as ipc
    bad = set(open(a.exclude_ids).read().split()) if a.exclude_ids else set()
    rng = random.Random(a.seed)
    names, msgs = [], []
    n_tasks = 0
    for f in sorted(glob.glob(os.path.join(a.pairs, "*.json"))):
        d = json.load(open(f)); tid = d["task"]
        if set(tid.split("_")) & bad:
            continue
        pool = [(p["input"], p["output"]) for p in d["pairs"]]
        try:
            demos = json.load(open(os.path.join(a.arc_dir, tid + ".json")))["train"]
            pool += [(p["input"], p["output"]) for p in demos]
        except FileNotFoundError:
            pass
        if len(pool) < a.min_pairs:
            continue
        n_tasks += 1
        for r in range(a.rows):
            k = rng.randint(a.min_pairs, a.max_pairs)
            pairs = rng.sample(pool, min(k, len(pool)))
            if r == 0:
                t, perm = 0, list(range(10))
            else:
                t = rng.randrange(8); perm = list(range(10)); rng.shuffle(perm)
            lut = np.array(perm)
            m = []
            for x, y in pairs:
                gx, gy = lut[dihedral(x, t)], lut[dihedral(y, t)]
                m.append({"role": "user", "content": grid_str(gx)}); m.append({"role": "assistant", "content": grid_str(gy)})
            names.append(tid); msgs.append(m)
    os.makedirs(a.out, exist_ok=True)
    schema = pa.schema([("puzzle_name", pa.string()),
                        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])))])
    table = pa.Table.from_pydict({"puzzle_name": names, "messages": msgs}, schema=schema)
    path = os.path.join(a.out, "data-00000-of-00001.arrow")
    with pa.OSFile(path, "wb") as sink, ipc.new_stream(sink, schema) as w:
        w.write_table(table)
    chars = np.mean([sum(len(x["content"]) for x in m) for m in msgs[:2000]]) if msgs else 0
    print(f"{n_tasks} tasks -> {len(names)} rows -> {path} | mean chars/row {chars:.0f}", flush=True)


if __name__ == "__main__":
    main()
