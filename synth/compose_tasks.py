"""Compositional synthetic tasks at zero LLM cost: for a task A with generated pairs (x, f(x)) (synth/gen_pairs_*/A.json)
and a verified program g of another task B (harvest records), the new task A__B maps x -> g(f(x)). Exact by
construction (no majority voting needed); harder than either parent (two rules). Filters: g runs on >= `min_valid`
of A's generated inputs, outputs are valid grids <= 30x30, g changes f(x) on >= 80% of them, and g(f(x)) != x.
Output: <out>/A__B.json in the gen_pairs format (task, pairs, program) -> synth/pairs_to_arrow.py.

python synth/compose_tasks.py --pairs synth/gen_pairs_v1 --records 'runs_synth/agent_harv*/shard_*.jsonl' \
    --arc-dir ARC-AGI-2/data/training --out synth/comp_pairs_v1 --exclude-ids synth/arc2_eval_120.txt --per-a 25 --cand 200
"""
import argparse, glob, io, json, os, random, signal, sys, time
from contextlib import redirect_stderr, redirect_stdout
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import barc_exec as X                                                  # noqa: E402
from solve_llm import SafePool                                         # noqa: E402


def _run(src, grids, demos):
    """g on each grid; per-call alarm; returns list of grids or None."""
    ns = {"__name__": "comp", "np": np, "train": [(np.array(a), np.array(b)) for a, b in demos], "test_inputs": []}
    out = []
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            signal.setitimer(signal.ITIMER_REAL, X.TIMEOUT)
            exec(compile(src, "<g>", "exec"), ns)
            signal.setitimer(signal.ITIMER_REAL, 0)
            f = ns.get("transform")
            if not callable(f):
                return [None] * len(grids)
            for g in grids:
                try:
                    signal.setitimer(signal.ITIMER_REAL, X.TIMEOUT)
                    y = X._as_grid(np.asarray(f(np.array(g, dtype=np.int64))))
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    out.append(y.tolist() if y is not None else None)
                except BaseException:
                    signal.setitimer(signal.ITIMER_REAL, 0); out.append(None)
    except BaseException:
        signal.setitimer(signal.ITIMER_REAL, 0)
        return [None] * len(grids)
    return out


def _worker(conn):
    X._worker_init()
    conn.send("ready")
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if job is None:
            return
        key, src, grids, demos = job
        try:
            conn.send((key, _run(src, grids, demos)))
        except BaseException:
            conn.send((key, None))


def ok_grid(g):
    return g is not None and 1 <= len(g) <= 30 and 1 <= len(g[0]) <= 30 and all(len(r) == len(g[0]) for r in g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True); ap.add_argument("--records", required=True); ap.add_argument("--arc-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--exclude-ids", default=None)
    ap.add_argument("--per-a", type=int, default=25, help="max accepted compositions per A task")
    ap.add_argument("--cand", type=int, default=200, help="candidate B programs tried per A task")
    ap.add_argument("--probe", type=int, default=12); ap.add_argument("--min-valid", type=int, default=32)
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    bad = set(open(a.exclude_ids).read().split()) if a.exclude_ids else set()
    is_bad = lambda tid: bool(set(tid.replace("__", "_").split("_")) & bad)
    # programs g: shortest verified program per task (oracle = exact on test too)
    progs = {}
    for f in glob.glob(a.records):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("programs") or not r.get("oracle") or is_bad(r["task_id"]):
                continue
            src = min(r["programs"], key=len)
            if r["task_id"] not in progs or len(src) < len(progs[r["task_id"]]):
                progs[r["task_id"]] = src
    demos_of = {}
    for tid in progs:
        try:
            demos_of[tid] = [(p["input"], p["output"]) for p in json.load(open(os.path.join(a.arc_dir, tid + ".json")))["train"]]
        except Exception:
            demos_of[tid] = []
    A = {}
    for f in sorted(glob.glob(os.path.join(a.pairs, "*.json"))):
        d = json.load(open(f))
        if is_bad(d["task"]) or len(d["pairs"]) < a.min_valid:
            continue
        A[d["task"]] = d
    print(f"{len(progs)} programs g, {len(A)} A tasks with generated pairs", flush=True)
    rng = random.Random(a.seed); os.makedirs(a.out, exist_ok=True)
    pool = SafePool(a.workers, target=_worker, fail=lambda job: (job[0], None))
    B_ids = sorted(progs)
    # pass 1: probe each (A, B) on `probe` inputs
    jobs = []
    for tid, d in A.items():
        fx = [p["output"] for p in d["pairs"][: a.probe]]
        for b in rng.sample(B_ids, min(a.cand, len(B_ids))):
            if b == tid or progs[b] == d["program"]:
                continue
            jobs.append(((tid, b), progs[b], fx, demos_of[b]))
    print(f"pass 1: {len(jobs)} probes", flush=True); t0 = time.time()
    accepted = {}
    for i in range(0, len(jobs), 2000):
        chunk = jobs[i: i + 2000]
        for (key, res) in pool.run(chunk):
            if not res:
                continue
            tid, b = key; d = A[tid]; fx = [p["output"] for p in d["pairs"][: a.probe]]; x = [p["input"] for p in d["pairs"][: a.probe]]
            valid = [ok_grid(y) for y in res]
            if sum(valid) < a.probe:
                continue
            changed = sum(y != f for y, f in zip(res, fx)); notx = sum(y != xi for y, xi in zip(res, x))
            if changed < 0.8 * a.probe or notx < 0.8 * a.probe or len({json.dumps(y) for y in res}) < 3:
                continue
            accepted.setdefault(tid, []).append(b)
        print(f"  probed {min(i+2000, len(jobs))}/{len(jobs)} | A tasks with accepted B: {len(accepted)} | {time.time()-t0:.0f}s", flush=True)
    # pass 2: full pairs for up to per_a compositions per A
    jobs = []
    for tid, bs in accepted.items():
        rng.shuffle(bs)
        for b in bs[: a.per_a]:
            jobs.append(((tid, b), progs[b], [p["output"] for p in A[tid]["pairs"]], demos_of[b]))
    print(f"pass 2: {len(jobs)} full runs", flush=True); n_out = 0
    for i in range(0, len(jobs), 2000):
        for (key, res) in pool.run(jobs[i: i + 2000]):
            if not res:
                continue
            tid, b = key; d = A[tid]
            pairs = [dict(input=p["input"], output=y) for p, y in zip(d["pairs"], res) if ok_grid(y) and y != p["output"] and y != p["input"]]
            if len(pairs) < a.min_valid:
                continue
            json.dump(dict(task=f"{tid}__{b}", pairs=pairs, program=d["program"] + "\n\n# ---- composed with ----\n" + progs[b],
                           parents=[tid, b]), open(os.path.join(a.out, f"{tid}__{b}.json"), "w"))
            n_out += 1
    print(f"DONE {n_out} composed tasks -> {a.out} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
