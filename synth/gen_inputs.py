"""Turn verified ARC programs into puzzle generators: for every task the agent solved (record with `programs`, exact on
demos and — with --pass1-only — on the test), ask an LLM for `generate_input(rng)` that samples inputs from the demo
input distribution, then verify it by running the verified `transform` on N fresh inputs (re-arc / BARC style).
Accepted generators yield `synth/gen_pairs/<task>.json` with N new (input, output) pairs — new training rows for the
transduction recipe, in the pure ARC-2 distribution.

python synth/gen_inputs.py --model M --records 'runs_synth/agent_rft1_studentB/shard_*.jsonl' --arc-dir synth/rft_tasks \
    --out synth/gen_pairs --n 64 [--pass1-only] [--shard i --n-shards n] [--tp 2]
"""
import argparse, glob, io, json, os, random, re, signal, sys, time, traceback
from contextlib import redirect_stdout, redirect_stderr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import barc_exec as X                                                  # noqa: E402  (TIMEOUT, _as_grid, _Timeout)
from solve_llm import SafePool                                         # noqa: E402
from solve_agent import grid_txt                                       # noqa: E402

SYSTEM = ("You write ARC-AGI puzzle generators in Python. You are given the demonstration pairs of a task and a program "
          "`transform(grid)` that is verified to reproduce every demonstration output exactly. Write `generate_input(rng)` "
          "that returns ONE new random input grid (numpy int array, values 0-9, at most 30x30) drawn from the same "
          "distribution as the demonstration inputs: same kinds of objects, sizes, colours, background, symmetry and "
          "constraints, so that `transform` is well defined and meaningful on it. Vary sizes, positions, counts and "
          "colours the way the demonstrations vary; never copy a demonstration verbatim. Use only numpy and the "
          "`rng` (numpy.random.Generator) passed in; no other imports, no randomness outside `rng`, no I/O. "
          "Answer with exactly one ```python block defining `generate_input(rng)` (helpers allowed).")
GEN_TIMEOUT = float(os.environ.get("GEN_TIMEOUT", 60))


def prompt(demos, program):
    parts = []
    for k, (a, b) in enumerate(demos, 1):
        parts.append(f"Demonstration {k} input:\n{grid_txt(a)}\nDemonstration {k} output:\n{grid_txt(b)}")
    return ("\n\n".join(parts) + "\n\nVerified program:\n```python\n" + program.strip() + "\n```\n\n"
            "Write `generate_input(rng)`.")


def feedback(res):
    return (f"Your generator was rejected: {res['reason']}. Failures on generated inputs: {res.get('errors', [])[:3]}. "
            "Fix it: every generated input must be a valid grid (2-D, ints 0-9, 1..30 per side) on which `transform` "
            "runs without error and produces a valid grid, and the inputs must vary (not all identical, not copies of "
            "the demonstrations). Answer with the corrected ```python block.")


def extract_code(text):
    text = text.split("</think>")[-1]
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return blocks[-1].strip() if blocks else None


# --------------------------------------------------------------------------------------------- verification worker
def _valid(g):
    if g is None:
        return False
    a = np.asarray(g)
    return a.ndim == 2 and 1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30 and a.dtype.kind in "iu" and a.min() >= 0 and a.max() <= 9


def verify(job):
    """job = (task_id, gen_code, program, demos, n, seed) -> dict(ok, pairs, reason, errors, stats)"""
    tid, gen_code, program, demos, n, seed = job
    res = dict(ok=False, pairs=[], reason=None, errors=[], stats={})
    buf = io.StringIO()
    try:
        signal.setitimer(signal.ITIMER_REAL, GEN_TIMEOUT)
        with redirect_stdout(buf), redirect_stderr(buf):
            ns_g = {"__name__": "gen", "np": np}; exec(compile(gen_code, "<gen>", "exec"), ns_g)
            ns_p = {"__name__": "prog", "np": np,                                  # same globals the agent gave the program
                    "train": [(np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)) for a, b in demos],
                    "test_inputs": [np.array(a, dtype=np.int64) for a, _ in demos]}
            exec(compile(program, "<prog>", "exec"), ns_p)
            gen, tf = ns_g.get("generate_input"), ns_p.get("transform")
            if not callable(gen):
                res["reason"] = "no function named `generate_input`"; return res
            if not callable(tf):
                res["reason"] = "no function named `transform` in the program"; return res
            rng = np.random.default_rng(seed)
            demo_in = {json.dumps(np.asarray(a).tolist()) for a, _ in demos}
            pairs, fails, seen = [], 0, set()
            for i in range(n):
                try:
                    g = np.asarray(gen(rng))
                    if not _valid(g):
                        fails += 1; res["errors"].append(f"invalid input grid (shape {getattr(g, 'shape', None)})"); continue
                    g = g.astype(np.int64)
                    o = X._as_grid(np.asarray(tf(g.copy())))
                    if not _valid(o):
                        fails += 1; res["errors"].append("transform produced an invalid output"); continue
                    key = json.dumps(g.tolist())
                    if key in demo_in or key in seen:
                        fails += 1; res["errors"].append("duplicate / demonstration input"); continue
                    seen.add(key); pairs.append({"input": g.tolist(), "output": np.asarray(o).astype(int).tolist()})
                except X._Timeout:
                    raise
                except BaseException as e:
                    fails += 1; res["errors"].append(f"{type(e).__name__}: {str(e)[:120]}")
            identical = sum(p["input"] == p["output"] for p in pairs)
            outs = {json.dumps(p["output"]) for p in pairs}
            res["stats"] = dict(n=n, ok=len(pairs), fails=fails, identical=identical, distinct_outputs=len(outs))
            if len(pairs) < 0.9 * n:
                res["reason"] = f"only {len(pairs)}/{n} valid generated pairs"
            elif len(outs) <= max(1, len(pairs) // 8):
                res["reason"] = "outputs almost all identical (degenerate generator)"
            elif identical > 0.5 * len(pairs) and not any(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in demos):
                res["reason"] = "transform leaves most generated inputs unchanged (inputs off-distribution)"
            else:
                res["ok"] = True; res["pairs"] = pairs
    except X._Timeout:
        res["reason"] = f"generation + transform took longer than {GEN_TIMEOUT:.0f} s"
    except BaseException as e:
        res["reason"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException:
            pass
    return res


def _worker(conn):
    signal.signal(signal.SIGALRM, X._alarm)
    conn.send("ready")
    while True:
        try:
            job = conn.recv()
        except EOFError:
            return
        conn.send(verify(job))


def _fail(job):
    return dict(ok=False, pairs=[], reason="verification killed (overrun or crash)", errors=[], stats={})


# ----------------------------------------------------------------------------------------------------------- main
def load_programs(paths, pass1_only):
    best = {}
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("programs") or (pass1_only and not r.get("pass1")):
                continue
            if r["task_id"] not in best:
                best[r["task_id"]] = r["programs"][0]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--records", required=True, help="glob(s), comma-separated")
    ap.add_argument("--arc-dir", required=True); ap.add_argument("--out", default="synth/gen_pairs")
    ap.add_argument("--n", type=int, default=64); ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--pass1-only", action="store_true"); ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=16000); ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X.TIMEOUT = GEN_TIMEOUT                                            # SafePool budget = TIMEOUT + grace
    pool = SafePool(a.workers, target=_worker, fail=_fail)
    paths = sorted(f for g in a.records.split(",") for f in glob.glob(g))
    progs = load_programs(paths, a.pass1_only)
    tids = sorted(progs)[a.shard :: a.n_shards]
    tids = [t for t in tids if not os.path.exists(os.path.join(a.out, t + ".json"))]
    print(f"{len(progs)} verified programs | shard {a.shard}/{a.n_shards}: {len(tids)} tasks to do", flush=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, gpu_memory_utilization=0.9,
              enable_prefix_caching=True, seed=a.seed)
    sp = SamplingParams(temperature=0.7, max_tokens=a.max_tokens)

    def n_tok(conv):
        ids = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=True)
        return len(ids["input_ids"]) if hasattr(ids, "keys") else len(ids)
    summary = open(os.path.join(a.out, f"summary_{a.shard}.jsonl"), "a")
    for b in range(0, len(tids), a.batch):
        batch = tids[b : b + a.batch]; t0 = time.time()
        demos = {}
        for t in batch:
            d = json.load(open(os.path.join(a.arc_dir, t + ".json")))
            demos[t] = [(p["input"], p["output"]) for p in d["train"]]
        convs = {t: [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(demos[t], progs[t])}] for t in batch}
        pending = list(batch); accepted = {}; last_fail = {}
        for rnd in range(a.rounds):
            if not pending:
                break
            fit = [t for t in pending if n_tok(convs[t]) + a.max_tokens < a.max_model_len]
            for t in pending:
                if t not in fit:
                    last_fail[t] = dict(reason="conversation exceeds the context window", stats={})
            pending = fit
            if not pending:
                break
            try:
                outs = llm.chat([convs[t] for t in pending], sp, use_tqdm=False)
            except Exception as e:                                      # one bad conversation must not kill the batch
                print(f"    llm.chat failed ({type(e).__name__}: {str(e)[:120]}); retrying one by one", flush=True)
                outs = []
                for t in pending:
                    try:
                        outs.append(llm.chat([convs[t]], sp, use_tqdm=False)[0])
                    except Exception as e2:
                        outs.append(None); last_fail[t] = dict(reason=f"llm error: {str(e2)[:80]}", stats={})
                keep = [(t, o) for t, o in zip(pending, outs) if o is not None]; pending = [t for t, _ in keep]; outs = [o for _, o in keep]
            codes = {t: extract_code(o.outputs[0].text) for t, o in zip(pending, outs)}
            jobs = [(t, codes[t] or "", progs[t], demos[t], a.n, a.seed + rnd) for t in pending]
            results = pool.run(jobs, grace=5.0)
            still = []
            for t, o, r in zip(pending, outs, results):
                convs[t].append({"role": "assistant", "content": o.outputs[0].text.split("</think>")[-1].strip()})
                if r["ok"]:
                    accepted[t] = (codes[t], r)
                else:
                    if codes[t] is None:
                        r["reason"] = "no ```python block in the answer"
                    convs[t].append({"role": "user", "content": feedback(r)}); still.append((t, r))
            pending = [t for t, _ in still]
            last_fail.update({t: r for t, r in still})
        for t in batch:
            if t in accepted:
                code, r = accepted[t]
                json.dump(dict(task=t, pairs=r["pairs"], generator=code, program=progs[t], stats=r["stats"]),
                          open(os.path.join(a.out, t + ".json"), "w"))
                summary.write(json.dumps(dict(task=t, ok=True, stats=r["stats"])) + "\n")
            else:
                r = last_fail.get(t, {})
                summary.write(json.dumps(dict(task=t, ok=False, reason=r.get("reason"), stats=r.get("stats"))) + "\n")
        summary.flush()
        print(f"batch {b//a.batch+1}: {len(accepted)}/{len(batch)} generators accepted ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
