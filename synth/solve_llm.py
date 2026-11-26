"""Induction with a fine-tuned code LLM (BARC's released Llama-3.1-8B induction model) + symbolic execution,
with an optional Berman-style evolutionary revision loop.

Generation 0: sample K programs from the task prompt (BARC's exact training format: grids as colour names,
`transform` function). Every program is executed on the demos in a sandboxed pool and gets a dense fitness =
mean fraction of demo cells reproduced (0 when the shape is wrong), plus the exact-consistency flag.
Generations 1..G (`--generations`): the top-P programs by fitness become parents; the model is shown a parent
program together with, for each demo, the grid it produced vs the expected grid, and asked for a corrected
`transform`. Offspring are scored the same way. (Berman 2024, "evolutionary test-time compute", with a local
fine-tuned model in place of an API model.)

Answer: test outputs of demo-consistent programs, voted; top-2 distinct. Conventions follow BARC's
`eval_code_samples.py`: grids are transposed before calling `transform` and transposed back.

    python synth/solve_llm.py --model $WORK/ccapone0/hf/barc_induction_8b --arc-dir ARC-AGI-2/data/training \
        --ids synth/arc2_train_heldout_clean.txt --k 128 --generations 0 --out res.jsonl
"""
import argparse, json, os, re, signal, sys, time
import multiprocessing as mp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    from synth import barc_exec as X  # noqa: E402
except ImportError:                                                    # flat layout (Kaggle dataset)
    import barc_exec as X  # noqa: E402

COLORS = ["Black", "Blue", "Red", "Green", "Yellow", "Gray", "Pink", "Orange", "Purple", "Brown"]
SYSTEM = ("You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in "
          "Python programming. Your task is to analyze puzzles and provide Python solutions.")
HEAD = ("Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output "
        "grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as "
        "strings, with cells (colors) separated by spaces and rows by newlines.")
TAIL = ("Write a Python function `transform` that can convert any given input grid to its corresponding output grid "
        "based on the pattern observed in the reference examples.")


def grid_str(g):
    return "\n".join(" ".join(COLORS[int(c)] for c in row) for row in g)


def task_prompt(demos, test_input):
    p = HEAD + "\nHere are the input and output grids for the reference examples:\n"
    for i, (a, b) in enumerate(demos):
        p += f"Example {i+1}\nInput:\n{grid_str(a)}\n\nOutput:\n{grid_str(b)}\n\n\n"
    p += "Here is the input grid for the test example:\nInput:\n" + grid_str(test_input) + "\n\n" + TAIL
    return p


def revision_prompt(demos, test_input, parent_src, produced):
    """Parent program + what it produced on each demo vs what was expected."""
    p = task_prompt(demos, test_input)
    p += "\n\nA previous attempt is shown below, followed by what it produced on the reference examples.\n"
    p += f"```python\n{parent_src}\n```\n"
    for i, ((a, b), got) in enumerate(zip(demos, produced)):
        if got is None:
            p += f"Example {i+1}: the code raised an error or returned an invalid grid.\n"
        elif got == b:
            p += f"Example {i+1}: correct.\n"
        else:
            p += f"Example {i+1}: WRONG. It produced:\n{grid_str(got)}\n\nbut the expected output is:\n{grid_str(b)}\n\n"
    p += "Write a corrected Python function `transform`."
    return p


CODE_RE = re.compile(r"```python\n(.*?)```", re.S)


def extract_code(text):
    m = CODE_RE.findall(text)
    if m:
        return m[-1]
    return text if "def transform" in text else None


def _run(args):
    """-> (fitness, consistent, produced demo grids, test outputs)"""
    src, demos, tests = args
    fail = (0.0, False, [None] * len(demos), None)
    if not src or "def transform" not in src:
        return fail
    ns = {"__name__": "cand"}
    try:
        signal.setitimer(signal.ITIMER_REAL, X.TIMEOUT)
        exec(compile("from common import *\nimport numpy as np\nfrom typing import *\n" + src, "<cand>", "exec"), ns)
        f = ns.get("transform")
        if not callable(f):
            return fail
        produced, fit = [], []
        for a, b in demos:
            try:
                got = X._as_grid(np.asarray(f(np.array(a, dtype=np.int64).T)).T)
            except X._Timeout:
                raise                                                  # the timer is armed once: swallowing it here
            except BaseException:                                      # would run the remaining pairs with no limit
                got = None
            g = got.tolist() if got is not None else None
            produced.append(g)
            if g is None or np.shape(g) != np.shape(b):
                fit.append(0.0)
            else:
                fit.append(float((np.array(g) == np.array(b)).mean()))
        ok = all(p == b for p, (_, b) in zip(produced, demos))
        outs = None
        if ok:
            outs = []
            for t in tests:
                try:
                    got = X._as_grid(np.asarray(f(np.array(t, dtype=np.int64).T)).T)
                    outs.append(got.tolist() if got is not None else None)
                except X._Timeout:
                    raise
                except BaseException:
                    outs.append(None)
        return float(np.mean(fit)), ok, produced, outs
    except BaseException:
        return fail
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException:
            pass


def run_safe(args):
    try:
        return _run(args)
    except BaseException:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException:
            pass
        return (0.0, False, [None] * len(args[1]), None)


def _worker_main(conn):
    X._worker_init()
    conn.send("ready")
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if job is None:
            return
        conn.send(run_safe(job))


class SafePool:
    """Process pool that cannot hang the caller and does not lose innocent programs.

    `multiprocessing.Pool` waits forever when a worker dies (segfault, OOM kill, os._exit) or sits in a loop SIGALRM
    cannot break — LLM-written code does all of these — and a batch deadline on top of it throws away every program
    queued behind the stuck workers. Here each worker is its own process on its own pipe: the parent knows which
    program each one is running and since when, kills exactly the worker that overruns (`_run` arms ONE timer for
    the whole program, so the budget is TIMEOUT + grace), scores that program as a failure and starts a fresh
    worker. Nothing else is affected."""

    def __init__(self, workers, recycle=500, target=None, fail=None):
        self.n = workers; self.recycle = recycle; self.rebuilds = 0
        self.target = target or _worker_main                          # other job types: own worker loop + failure value
        self.fail = fail or (lambda job: (0.0, False, [None] * len(job[1]), None))
        self.ctx = mp.get_context("spawn")
        self.w = [self._spawn() for _ in range(workers)]

    def _spawn(self):
        parent, child = self.ctx.Pipe()
        p = self.ctx.Process(target=self.target, args=(child,), daemon=True); p.start(); child.close()
        return dict(proc=p, conn=parent, ready=False, job=None, t0=time.time(), served=0)

    def _kill(self, w):
        try:
            w["conn"].close()
        except BaseException:
            pass
        try:
            w["proc"].kill(); w["proc"].join(2)
        except BaseException:
            pass

    def run(self, jobs, grace=2.0, boot=120.0):
        from multiprocessing.connection import wait
        out = [None] * len(jobs); todo = list(range(len(jobs)))[::-1]; pending = len(jobs); killed = 0
        budget = X.TIMEOUT + grace
        while pending:
            for i, w in enumerate(self.w):
                if w["ready"] and w["job"] is None and todo:
                    if w["served"] >= self.recycle:
                        self._kill(w); self.w[i] = self._spawn(); continue
                    w["job"] = todo.pop(); w["t0"] = time.time(); w["served"] += 1
                    try:
                        w["conn"].send(jobs[w["job"]])
                    except BaseException:
                        pass                                           # dead pipe: the overrun check below replaces it
            for c in wait([w["conn"] for w in self.w], timeout=0.2):
                w = next(x for x in self.w if x["conn"] is c)
                try:
                    msg = c.recv()
                except BaseException:
                    msg = EOFError                                     # worker died
                if msg is EOFError:
                    w["t0"] = -1e18                                    # force replacement below
                elif msg == "ready":
                    w["ready"] = True
                else:
                    out[w["job"]] = msg; w["job"] = None; pending -= 1
            now = time.time()
            for i, w in enumerate(self.w):
                late = (w["job"] is not None and now - w["t0"] > budget) or (not w["ready"] and now - w["t0"] > boot) \
                       or w["t0"] < 0
                if late:
                    if w["job"] is not None:
                        out[w["job"]] = self.fail(jobs[w["job"]]); pending -= 1; killed += 1
                    self._kill(w); self.w[i] = self._spawn()
        if killed:
            self.rebuilds += killed
            print(f"      [pool] {killed}/{len(jobs)} programs killed (overrun or crash); workers replaced", flush=True)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arc-dir", required=True); ap.add_argument("--ids", required=True)
    ap.add_argument("--k", type=int, default=128); ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--generations", type=int, default=0); ap.add_argument("--parents", type=int, default=8)
    ap.add_argument("--offspring", type=int, default=16, help="samples per parent per generation")
    ap.add_argument("--max-tokens", type=int, default=1536); ap.add_argument("--max-model-len", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=7); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--keep-programs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pool = SafePool(a.workers)                                          # before any CUDA context exists
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, max_model_len=a.max_model_len, enable_prefix_caching=True, dtype="bfloat16",
              gpu_memory_utilization=0.92, seed=a.seed)

    def chat(user):
        return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)

    def fits(prompt):
        return len(tok(prompt).input_ids) + a.max_tokens < a.max_model_len

    ids = [l.strip() for l in open(a.ids) if l.strip()][a.shard :: a.n_shards]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
                if r.get("task_id"): done.add(r["task_id"])
            except json.JSONDecodeError:
                pass
    fh = open(a.out, "a"); tot = dict(tasks=0, oracle=0, p1=0, p2=0, any_consistent=0, skipped=0)
    ids = [t for t in ids if t not in done]
    print(f"resume: {len(done)} tasks already done, {len(ids)} to go", flush=True)
    for tid in ids:
        t0 = time.time()
        d = json.load(open(os.path.join(a.arc_dir, tid + ".json")))
        demos = [(p["input"], p["output"]) for p in d["train"]]
        tests = [(q["input"], q.get("output")) for q in d["test"]]
        prompt = chat(task_prompt(demos, tests[0][0]))
        if not fits(prompt):
            tot["skipped"] += 1; tot["tasks"] += 1
            fh.write(json.dumps(dict(task_id=tid, skipped="prompt too long", consistent=0, pass2=False, oracle=False)) + "\n")
            print(f"  {tid}: SKIP prompt too long", flush=True); continue
        pop = []                                                       # (fitness, ok, produced, outs, src, gen)
        gen_stats = []
        # vLLM never schedules a request whose n exceeds max_num_seqs (256): it waits forever. Split into requests.
        ns = [min(256, a.k - c) for c in range(0, a.k, 256)]
        outs0 = llm.generate([prompt] * len(ns), [SamplingParams(n=n, temperature=a.temperature, max_tokens=a.max_tokens) for n in ns],
                             use_tqdm=False)
        srcs = [extract_code(o.text) for req in outs0 for o in req.outputs]
        res = pool.run([(s, demos, [t[0] for t in tests]) for s in srcs])
        pop += [(r[0], r[1], r[2], r[3], s, 0) for r, s in zip(res, srcs) if s]
        gen_stats.append(dict(gen=0, n=len(srcs), consistent=sum(r[1] for r in res), best_fit=max([r[0] for r in res] + [0])))
        for g in range(1, a.generations + 1):
            if any(p[1] for p in pop) and g > 1:
                pass                                                   # keep evolving: more consistent programs = better vote
            parents, seen = [], set()
            for p in sorted(pop, key=lambda p: -p[0]):
                if p[1] or p[4] in seen:
                    continue                                           # only revise programs that are still wrong
                seen.add(p[4]); parents.append(p)
                if len(parents) == a.parents:
                    break
            if not parents:
                break
            prompts = [chat(revision_prompt(demos, tests[0][0], p[4], p[2])) for p in parents]
            prompts = [q for q in prompts if fits(q)]
            if not prompts:
                break
            outs = llm.generate(prompts, SamplingParams(n=a.offspring, temperature=a.temperature, max_tokens=a.max_tokens), use_tqdm=False)
            srcs = [extract_code(o.text) for req in outs for o in req.outputs]
            res = pool.run([(s, demos, [t[0] for t in tests]) for s in srcs])
            pop += [(r[0], r[1], r[2], r[3], s, g) for r, s in zip(res, srcs) if s]
            gen_stats.append(dict(gen=g, n=len(srcs), consistent=sum(r[1] for r in res), best_fit=max([r[0] for r in res] + [0])))
        votes = [{} for _ in tests]
        for p in pop:
            if not p[1] or p[3] is None:
                continue
            for i, o in enumerate(p[3]):
                if o is not None:
                    e = votes[i].setdefault(json.dumps(o), {"grid": o, "votes": 0}); e["votes"] += 1
        ranked = [sorted(v.values(), key=lambda e: -e["votes"]) for v in votes]
        n_cons = sum(p[1] for p in pop)
        rec = dict(task_id=tid, sampled=len(pop), consistent=n_cons, gens=gen_stats, seconds=time.time() - t0,
                   best_fitness=max([p[0] for p in pop] + [0]),
                   consistent_by_gen={str(g): sum(1 for p in pop if p[1] and p[5] == g) for g in range(a.generations + 1)},
                   attempts=[[e["grid"] for e in r[:2]] for r in ranked],
                   programs=[p[4] for p in sorted(pop, key=lambda p: -p[0]) if p[1]][: a.keep_programs])
        if all(t[1] is not None for t in tests):
            rec["pass1"] = all(r and r[0]["grid"] == t[1] for r, t in zip(ranked, tests))
            rec["pass2"] = all(any(e["grid"] == t[1] for e in r[:2]) for r, t in zip(ranked, tests))
            rec["oracle"] = all(any(p[1] and p[3] and p[3][i] == t[1] for p in pop) for i, t in enumerate(tests))
            tot["p1"] += rec["pass1"]; tot["p2"] += rec["pass2"]; tot["oracle"] += rec["oracle"]
        tot["tasks"] += 1; tot["any_consistent"] += n_cons > 0
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(f"  {tid}: sampled {len(pop)} consistent {n_cons} best_fit {rec['best_fitness']:.2f} "
              f"by_gen {rec['consistent_by_gen']} | p@2 {rec.get('pass2')} | {rec['seconds']:.0f}s", flush=True)
    fh.write(json.dumps(dict(summary=True, **tot)) + "\n"); fh.close()
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
