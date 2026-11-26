"""Extract execution TRACES from verified programs: the grid after every library call.

A verified BARC program is a proof that a rule is executable; running it with the seeds library instrumented
also yields the intermediate states, which is the dense supervision the emergent-language work could never
produce. Each trace step is (call name, arguments summary, grid before, grid after) whenever the call returns
a grid; calls that return objects/masks are recorded with their type only.

Instrumentation: every public callable of `common` is wrapped, plus numpy's grid-shaped returns are ignored —
we only need library-level steps, which is the level a step-wise reasoner would predict.

    python synth/traces.py --data verified_train.jsonl --out traces.jsonl --limit 20000 --workers 8
"""
import argparse, json, os, signal, sys, time
import multiprocessing as mp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from synth import barc_exec as X  # noqa: E402

_W = {}
MODE = {"line": True}


def _init():
    X._worker_init()
    import common  # noqa
    _W["common"] = common


def _grid(x):
    a = np.asarray(x)
    if a.ndim == 2 and 1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30 and np.issubdtype(a.dtype, np.integer):
        v = a.astype(int)
        if (v >= 0).all() and (v <= 10).all():
            return v.tolist()
    return None


def instrument(common, log):
    """Wrap every public callable of the seeds library so each call appends a step to `log`."""
    import types
    orig = {}
    for name in dir(common):
        if name.startswith("_"):
            continue
        fn = getattr(common, name)
        if not isinstance(fn, types.FunctionType):
            continue
        orig[name] = fn

        def make(name, fn):
            def wrapped(*args, **kw):
                out = fn(*args, **kw)
                if len(log) < 64:
                    g_in = next((g for g in (_grid(a) for a in args) if g is not None), None)
                    log.append({"call": name, "in": g_in, "out": _grid(out),
                                "out_type": type(out).__name__, "n_args": len(args)})
                return out
            return wrapped
        setattr(common, name, make(name, fn))
    return orig


def restore(common, orig):
    for name, fn in orig.items():
        setattr(common, name, fn)


def line_trace(src, gi, limit=64):
    """Snapshot every local variable that holds a grid, whenever it changes, line by line.
    Covers programs that manipulate numpy directly and never call the library."""
    ns = {"__name__": "traced_line"}
    exec(compile(src, "<traced>", "exec"), ns)
    main = ns.get("main")
    if not callable(main):
        return None, []
    steps, seen = [], {}

    def tracer(frame, event, arg):
        if event != "line" or frame.f_code.co_name != "main" or len(steps) >= limit:
            return tracer
        for k, v in list(frame.f_locals.items()):
            g = _grid(v)
            if g is None or seen.get(k) == g:
                continue
            steps.append({"call": f"line:{frame.f_lineno}", "var": k, "in": seen.get(k), "out": g,
                          "out_type": "ndarray", "n_args": 0})
            seen[k] = g
        return tracer

    sys.settrace(tracer)
    try:
        out = main(np.array(gi, dtype=np.int64))
    finally:
        sys.settrace(None)
    return _grid(out), steps


def _job(line):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        return None
    common = _W["common"]
    ns = {"__name__": "traced"}
    out_traces = []
    try:
        signal.setitimer(signal.ITIMER_REAL, X.TIMEOUT)
        exec(compile(r["source"], "<traced>", "exec"), ns)
        main = ns.get("main")
        if not callable(main):
            return None
        for gi, go in r["pairs"][:3]:
            log = []
            orig = instrument(common, log)
            ns2 = dict(ns)
            try:
                exec(compile(r["source"], "<traced>", "exec"), ns2)      # rebind to the wrapped library
                got = ns2["main"](np.array(gi, dtype=np.int64))
            finally:
                restore(common, orig)
            if _grid(got) != [list(map(int, row)) for row in go]:
                continue
            steps = [s for s in log if s["out"] is not None]
            if len(steps) < 2 and MODE["line"]:          # library-free program: fall back to line-level states
                try:
                    got2, lsteps = line_trace(r["source"], gi)
                    if got2 == [list(map(int, row)) for row in go] and lsteps:
                        steps = lsteps
                except BaseException:
                    pass
            if steps:
                out_traces.append({"input": gi, "output": go, "steps": steps})
        signal.setitimer(signal.ITIMER_REAL, 0)
    except BaseException:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException:
            pass
        return None
    if not out_traces:
        return None
    return {"pid": r["pid"], "source": r["source"], "traces": out_traces}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--no-line", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    MODE["line"] = not a.no_line
    t0 = time.time(); n = kept = n_steps = 0
    ctx = mp.get_context("fork")
    with open(a.out, "w") as out, ctx.Pool(a.workers, _init) as pool:
        lines = (l for i, l in enumerate(open(a.data)) if not a.limit or i < a.limit)
        for res in pool.imap_unordered(_job, lines, chunksize=16):
            n += 1
            if res:
                kept += 1; n_steps += sum(len(t["steps"]) for t in res["traces"])
                out.write(json.dumps(res) + "\n")
            if n % 5000 == 0:
                print(f"  {n} programs, {kept} traced, {time.time()-t0:.0f}s", flush=True)
    print(json.dumps({"programs": n, "traced": kept, "trace_steps": n_steps,
                      "steps_per_trace": n_steps / max(kept, 1), "seconds": time.time() - t0}, indent=1))


if __name__ == "__main__":
    main()
