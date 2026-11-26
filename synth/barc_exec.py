"""Execute BARC programs against their example pairs, in worker processes with a per-program timeout.

A BARC record is {"source": <python>, "examples": [[input, output], ...], "seeds": [...]}. The program
defines `main(input_grid) -> np.ndarray` on top of the seeds library (`common.py`, object-level primitives:
connected components, sprites, symmetries). Execution is the ground truth we can verify against, which is
what the neural route never had.

    python synth/barc_exec.py check   --jsonl F [F ...] --out stats.json [--limit N] [--workers 8]
    python synth/barc_exec.py dataset --jsonl F [F ...] --out data.jsonl [--limit N] [--workers 8]

`check` reports executability (how many programs reproduce all their own examples). `dataset` writes the
verified demos→program training set, one JSON object per line:
    {"pid": int, "source": str, "pairs": [[in, out], ...], "seeds": [...]}
Programs that fail, time out or disagree with their own examples are dropped — training on unverified
programs would teach the proposer to write code that does not run.

Safety: programs come from the public BARC corpus (LLM-written, executed by its authors). They run in a
subprocess with a wall-clock timeout, no network use of their own, and their output is only compared to
grids. Do not point this at untrusted sources.
"""
import argparse, json, os, re, sys, signal, time, resource
import multiprocessing as mp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = os.environ.get("BARC_SEEDS", os.path.join(ROOT, "external_rules", "barc_repo", "seeds"))
TIMEOUT = float(os.environ.get("BARC_TIMEOUT", "5"))
MEM_GB = float(os.environ.get("BARC_MEM_GB", "4"))


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _worker_init():
    sys.path.insert(0, SEEDS)
    devnull = open(os.devnull, "w")                  # programs print debug output; it floods the job log
    sys.stdout = devnull; sys.stderr = devnull
    try:
        # Limit relative to the CURRENT address space: a process forked from a CUDA parent already maps tens of
        # GB of virtual memory, and an absolute 4 GB cap below that makes every allocation fail -> the worker
        # dies, the pool respawns it forever and pool.map hangs silently.
        vm = 0
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmSize:"):
                    vm = int(line.split()[1]) * 1024
        cap = vm + int(MEM_GB * 1e9)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except Exception:
        pass
    signal.signal(signal.SIGALRM, _alarm)
    np.random.seed(0)


def _as_grid(x):
    a = np.asarray(x)
    if a.ndim != 2 or a.size == 0 or a.shape[0] > 30 or a.shape[1] > 30:
        return None
    try:
        a = a.astype(np.int64)
    except Exception:
        return None
    return a if (a >= 0).all() and (a <= 9).all() else None


def strip_program(src):
    """Remove `generate_input` (data generation, useless at test time) and any function only it uses;
    keep imports, header comments (concepts / description — the natural-language statement of the rule),
    `main` and its helpers. Returns None if the source does not parse."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "main" not in funcs:
        return None
    needed, stack = set(), ["main"]
    while stack:
        f = stack.pop()
        if f in needed or f not in funcs:
            continue
        needed.add(f)
        stack.extend(n.id for n in ast.walk(funcs[f]) if isinstance(n, ast.Name) and n.id in funcs)
    drop = [funcs[f] for f in funcs if f not in needed]
    if not drop:
        return src
    lines = src.split("\n")
    for node in sorted(drop, key=lambda n: n.lineno, reverse=True):
        start = node.lineno - 1 - len(node.decorator_list)
        del lines[start:node.end_lineno]
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).rstrip() + "\n"


def _unsafe_run_record(rec):
    """-> (status, n_ok, n_pairs, kept_pairs). status: ok | mismatch | error:<Type> | timeout | no_main."""
    src, examples = rec.get("source", ""), rec.get("examples") or []
    if "def main" not in src:
        return "no_main", 0, len(examples), []
    ns = {"__name__": "barc_prog"}
    try:
        signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
        exec(compile(src, "<barc>", "exec"), ns)
        main = ns.get("main")
        if not callable(main):
            signal.setitimer(signal.ITIMER_REAL, 0)
            return "no_main", 0, len(examples), []
        ok, kept = 0, []
        for pair in examples:
            a, b = (pair[0], pair[1]) if isinstance(pair, (list, tuple)) else (pair.get("input"), pair.get("output"))
            gi, go = _as_grid(a), _as_grid(b)
            if gi is None or go is None:
                continue
            got = _as_grid(main(np.copy(gi)))
            if got is not None and got.shape == go.shape and np.array_equal(got, go):
                ok += 1; kept.append([gi.tolist(), go.tolist()])
        signal.setitimer(signal.ITIMER_REAL, 0)
        return ("ok" if ok == len(examples) and ok > 0 else "mismatch"), ok, len(examples), kept
    except _Timeout:
        return "timeout", 0, len(examples), []
    except BaseException as e:                       # programs can raise anything, including SystemExit
        return f"error:{type(e).__name__}", 0, len(examples), []
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


_FAILED_RUN_RECORD = ("timeout", 0, 0, [])


def _safe_run_record(*args, **kw):
    """Guard the SIGALRM race: the timeout can fire while `finally` disarms the timer."""
    try:
        return _unsafe_run_record(*args, **kw)
    except BaseException:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException:
            pass
        return _FAILED_RUN_RECORD

run_record = _safe_run_record


def _job(args):
    pid, line = args
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return pid, "badjson", 0, 0, None, None, []
    st, ok, n, kept = run_record(rec)
    src = rec["source"] if st == "ok" else None
    if st == "ok":
        mini = strip_program(rec["source"])
        if mini is not None and mini != rec["source"]:
            st2, ok2, _, kept2 = run_record(dict(rec, source=mini))
            if st2 == "ok" and ok2 == ok:
                src = mini                              # stripped program verified identical on every example
            else:
                st = "strip_changed_behaviour"; src = None
    return pid, st, ok, n, src, (kept if st == "ok" else None), rec.get("seeds") or []


def stream(jsonls, limit=0):
    pid = 0
    for path in jsonls:
        with open(path) as fh:
            for line in fh:
                yield pid, line
                pid += 1
                if limit and pid >= limit:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "dataset"])
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--min-pairs", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    counts, n_pairs_ok, written = {}, 0, 0
    ctx = mp.get_context("fork")
    out = open(a.out, "w") if a.cmd == "dataset" else None
    with ctx.Pool(a.workers, _worker_init) as pool:
        for i, res in enumerate(pool.imap_unordered(_job, stream(a.jsonl, a.limit), chunksize=16), 1):
            pid, st, ok, n, src, kept, seeds = res
            counts[st] = counts.get(st, 0) + 1
            n_pairs_ok += ok
            if out is not None and st == "ok" and len(kept) >= a.min_pairs:
                out.write(json.dumps({"pid": pid, "source": src, "pairs": kept, "seeds": seeds}) + "\n")
                written += 1
            if i % 20000 == 0:
                el = time.time() - t0
                print(f"  {i} records, {el:.0f}s ({i/el:.0f}/s), ok {counts.get('ok',0)/i:.1%}", flush=True)
    if out is not None:
        out.close()
    total = sum(counts.values())
    el = time.time() - t0
    summary = {"records": total, "status": counts, "ok_frac": counts.get("ok", 0) / max(total, 1),
               "verified_pairs": n_pairs_ok, "written": written, "seconds": el,
               "rate_per_s": total / max(el, 1e-9), "workers": a.workers, "timeout_s": TIMEOUT}
    print(json.dumps(summary, indent=1))
    if a.cmd == "check":
        json.dump(summary, open(a.out, "w"), indent=1)
    else:
        json.dump(summary, open(a.out + ".stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
