"""Strong open-weight reasoner in a code loop: think -> run Python on the puzzle -> read the output -> submit a
`transform` -> get exact verification on the demos -> revise.  (T1 of the teacher/distillation plan.)

Each task runs `--threads` independent conversations of at most `--rounds` model messages. In a message the model
either explores (one ```python block, stdout is returned; `train`, `test_inputs`, `np` predefined, no state kept) or
submits (block starting with `# FINAL`, defining transform(grid) -> grid). A submission is executed on every demo:
all exact -> the thread is *consistent* and stops; otherwise the model sees what differed and continues.
Answers: vote over the test outputs of consistent threads; remaining attempt slots are filled with the fittest
inconsistent submissions (free under pass@2). Output records are compatible with synth/summarize_llm.py.

All conversations of a batch of tasks advance together, one vLLM call per round (prefix caching makes the growing
history cheap). Model-written code runs in solve_llm.SafePool workers (per-program kill, no collateral loss).
"""
import argparse, io, json, os, re, signal, sys, time, traceback
from contextlib import redirect_stderr, redirect_stdout
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import barc_exec as X
import solve_llm as S

SYSTEM = "You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles by writing Python."
TERSE = (" Think in compressed notes, not prose: no full sentences, no filler, no restating the grids. Use symbols and "
         "short tokens, e.g. `obj(c=3,4x2)`, `in->out: rot90`, `bg=0`, `count(c=5)=3 -> h=3`, `H1: ... X(train2)`. "
         "One idea per line. Test hypotheses with code instead of describing them.")
ACT_EARLY = (" Keep each reasoning phase short: after at most a few hundred words, write a ```python block that tests your "
             "current best hypothesis on the train pairs, and use the execution feedback instead of long deliberation. "
             "Long reasoning is cut off at the per-message limit and wastes the whole message.")
OCCAM = (" Prefer the SIMPLEST program that reproduces every train pair: the fewest rules, no hard-coded coordinates, sizes or "
         "colours unless the demos force them, no special cases for a single example. A rule that fits the demos only "
         "because of such details will fail on the test input. If two programs fit all demos, submit the shorter one.")
MAX_STDOUT = 3000
MAX_VISIBLE = 8000


def grid_txt(g):
    return "\n".join("".join(str(int(c)) for c in row) for row in g)


def objects_txt(g, max_objs=12):
    """Compact object summary of a grid (AGENT_OBJECTS=1): background colour, then per colour its 4-connected
    components with size and bounding box (rows r0-r1, cols c0-c1, 0-based). Pure python (no scipy on the cluster)."""
    a = np.asarray(g); H, W = a.shape; vals, cnt = np.unique(a, return_counts=True); bg = int(vals[cnt.argmax()])
    seen = np.zeros_like(a, dtype=bool); comps = {}
    for r0 in range(H):
        for c0 in range(W):
            c = int(a[r0, c0])
            if c == bg or seen[r0, c0]:
                continue
            stack, cells = [(r0, c0)], []; seen[r0, c0] = True
            while stack:
                r, q = stack.pop(); cells.append((r, q))
                for dr, dq in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, qq = r + dr, q + dq
                    if 0 <= rr < H and 0 <= qq < W and not seen[rr, qq] and a[rr, qq] == c:
                        seen[rr, qq] = True; stack.append((rr, qq))
            rs = [x for x, _ in cells]; qs = [y for _, y in cells]
            comps.setdefault(c, []).append(f"{len(cells)}@r{min(rs)}-{max(rs)}c{min(qs)}-{max(qs)}")
    lines = [f"background {bg}"]
    for c in sorted(comps):
        o = comps[c]
        lines.append(f"colour {c}: {len(o)} object(s) " + " ".join(o[:max_objs]) + (" ..." if len(o) > max_objs else ""))
    return "objects: " + "; ".join(lines)


OBJECTS = os.environ.get("AGENT_OBJECTS") == "1"


def intro(demos, tests, rounds):
    parts = ["Find the transformation that maps each input grid to its output grid. Cells are digits 0-9 (colours), "
             "one text line per grid row.\n" + ("Under each grid, an 'objects' line lists the background colour and, per colour, "
             "its 4-connected components as size@rR0-R1cC0-C1 (bounding box rows/cols, 0-based; at most 12 per colour).\n"
             if OBJECTS else "")]
    for i, (a, b) in enumerate(demos):
        oa = ("\n" + objects_txt(a)) if OBJECTS else ""; ob = ("\n" + objects_txt(b)) if OBJECTS else ""
        parts.append(f"train[{i}] input ({len(a)}x{len(a[0])}):\n{grid_txt(a)}{oa}\ntrain[{i}] output ({len(b)}x{len(b[0])}):\n{grid_txt(b)}{ob}\n")
    for i, t in enumerate(tests):
        ot = ("\n" + objects_txt(t)) if OBJECTS else ""
        parts.append(f"test_inputs[{i}] ({len(t)}x{len(t[0])}):\n{grid_txt(t)}{ot}\n")
    parts.append(
        "How to work:\n"
        "* To inspect the puzzle, reply with exactly one ```python block. I run it and send back what it prints. "
        "Predefined: `np` (numpy), `train` = list of (input, output) int arrays, `test_inputs` = list of int arrays. "
        "scipy is installed. Nothing is kept between runs, so re-define what you need. Keep prints short.\n"
        "* When you know the rule, reply with one ```python block whose first line is `# FINAL` and that defines "
        "`def transform(grid: np.ndarray) -> np.ndarray`. I run it on every train input and compare exactly with "
        "the outputs; if something differs you will see where and can continue.\n"
        f"* You have {rounds} messages in total. A rule that fits all train pairs but is over-fitted to them will fail "
        "on the test input: prefer the general rule.")
    return "\n".join(parts)


# ----------------------------------------------------------------------------- execution (inside SafePool workers)
def _agent_worker(conn):
    X._worker_init()
    conn.send("ready")
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if job is None:
            return
        conn.send(run_job(job))


def _fail(job, err="execution killed: exceeded the time limit or crashed the interpreter"):
    return dict(kind=job[0], err=err, stdout="", fitness=0.0, ok=False, produced=[None] * len(job[2]), outs=None)


def run_job(job):
    try:
        return _run_job(job)
    except BaseException as e:                                         # includes the SIGALRM race in `finally`
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException:
            pass
        return _fail(job, f"{type(e).__name__}: {str(e)[:300]}")


def _call(f, a):
    got = X._as_grid(np.asarray(f(np.array(a, dtype=np.int64))))
    return got.tolist() if got is not None else None


def _run_job(job):
    kind, code, demos, tests = job
    ns = {"__name__": "agent", "np": np, "train": [(np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)) for a, b in demos],
          "test_inputs": [np.array(t, dtype=np.int64) for t in tests]}
    buf = io.StringIO(); res = _fail(job, None)
    try:
        signal.setitimer(signal.ITIMER_REAL, X.TIMEOUT)                # ONE timer for the whole job
        with redirect_stdout(buf), redirect_stderr(buf):
            exec(compile(code, "<agent>", "exec"), ns)
            if kind == "final":
                f = ns.get("transform")
                if not callable(f):
                    res["err"] = "no function named `transform` was defined"
                else:
                    produced, fit, errs = [], [], []
                    for a, b in demos:
                        try:
                            g = _call(f, a)
                        except X._Timeout:
                            raise                                      # never swallow the timer (see llm-code-exec-hangs)
                        except BaseException as e:
                            g = None; errs.append(f"{type(e).__name__}: {str(e)[:200]}")
                        produced.append(g)
                        fit.append(float((np.array(g) == np.array(b)).mean()) if g is not None and np.shape(g) == np.shape(b) else 0.0)
                    outs, terrs = [], []
                    for t in tests:
                        try:
                            outs.append(_call(f, t)); terrs.append(None if outs[-1] is not None else "returned no valid grid")
                        except X._Timeout:
                            raise
                        except BaseException as e:
                            outs.append(None); terrs.append(f"{type(e).__name__}: {str(e)[:200]}")
                    res.update(produced=produced, fitness=float(np.mean(fit)), outs=outs, test_errs=terrs,
                               ok=all(p == b for p, (_, b) in zip(produced, demos)), err=errs[0] if errs else None)
    except X._Timeout:
        res["err"] = f"TimeoutError: the code ran longer than {X.TIMEOUT:.0f} s"
    except BaseException as e:
        tb = traceback.extract_tb(e.__traceback__); line = next((t.lineno for t in reversed(tb) if t.filename == "<agent>"), None)
        res["err"] = f"{type(e).__name__}: {str(e)[:300]}" + (f" (line {line} of your code)" if line else "")
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException:
            pass
    res["stdout"] = buf.getvalue()[:MAX_STDOUT]
    return res


# ----------------------------------------------------------------------------- conversation logic
def extract(text, thinking=True, cut=True):
    """-> (visible answer without the thinking block, last python block or None, truncated?)
    The chat template opens `<think>` inside the prompt, so a completion normally contains only `</think>`:
    when thinking is on and that tag is missing, the message was cut off while still reasoning, and none of it may
    enter the history (it is ~max_tokens long; keeping it is what overflowed the context in job 58333958)."""
    truncated = thinking and cut and "</think>" not in text          # cut = generation stopped on the token limit
    vis = "" if truncated else text.split("</think>")[-1].strip()
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", vis, re.S)
    code = blocks[-1].strip("\n") if blocks else None
    if len(vis) > MAX_VISIBLE:                                         # bound history growth: keep the tail (the code)
        vis = "[...]\n" + vis[-MAX_VISIBLE:]
    return vis, code, truncated


def diff_report(produced, demos, err):
    lines = []
    for i, (p, (_, b)) in enumerate(zip(produced, demos)):
        if p == b:
            lines.append(f"train[{i}]: correct"); continue
        if p is None:
            lines.append(f"train[{i}]: no valid grid returned" + (f" ({err})" if err else "")); continue
        if np.shape(p) != np.shape(b):
            lines.append(f"train[{i}]: WRONG SHAPE, expected {np.shape(b)[0]}x{np.shape(b)[1]}, got {np.shape(p)[0]}x{np.shape(p)[1]}"); continue
        d = np.argwhere(np.array(p) != np.array(b))
        ex = ", ".join(f"({r},{c}) expected {b[r][c]} got {p[r][c]}" for r, c in d[:8])
        lines.append(f"train[{i}]: {len(d)} of {np.size(b)} cells wrong; first: {ex}")
        if np.size(b) <= 225:
            lines.append(f"your output for train[{i}]:\n{grid_txt(p)}")
    return "\n".join(lines)


TESTCHECK = os.environ.get("AGENT_TESTCHECK") == "1"   # demo-consistent program must also run cleanly on the test inputs
SECOND = os.environ.get("AGENT_SECOND") == "1"
SHRINK = os.environ.get("AGENT_SHRINK") == "1"         # near the context limit, shorten the message budget instead of stopping
MIN_MSG_TOKENS = int(os.environ.get("AGENT_MIN_MSG", 4096))         # after a consistent program, ask once for an independent simpler one


def test_problems(r, demos, tests):
    """Label-free checks of a demo-consistent program on the test inputs: crash / no grid, and shape or colour
    relations that hold on EVERY demo but are broken on a test input. Returns a feedback string or None."""
    probs = []
    same = all(np.shape(a) == np.shape(b) for a, b in demos); const = len({np.shape(b) for _, b in demos}) == 1
    pal = set(int(v) for _, b in demos for v in np.unique(b))
    for i, (t, o) in enumerate(zip(tests, r.get("outs") or [])):
        e = (r.get("test_errs") or [None] * len(tests))[i]
        if o is None:
            probs.append(f"test_inputs[{i}]: your transform failed ({e or 'no valid grid'})"); continue
        if same and np.shape(o) != np.shape(t):
            probs.append(f"test_inputs[{i}]: every demo output has its input's shape, but yours is {np.shape(o)[0]}x{np.shape(o)[1]} "
                         f"for a {np.shape(t)[0]}x{np.shape(t)[1]} input")
        elif const and not same and np.shape(o) != np.shape(demos[0][1]):
            probs.append(f"test_inputs[{i}]: every demo output is {np.shape(demos[0][1])[0]}x{np.shape(demos[0][1])[1]}, yours is "
                         f"{np.shape(o)[0]}x{np.shape(o)[1]}")
        new = set(int(v) for v in np.unique(o)) - pal - set(int(v) for v in np.unique(t))
        if new:
            probs.append(f"test_inputs[{i}]: your output uses colour(s) {sorted(new)} that appear in no demo output and not in this input")
    return "\n".join(probs) or None


class Thread:
    def __init__(self, tid, k, demos, tests, rounds):
        self.tid, self.k = tid, k
        self.msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": intro(demos, tests, rounds)}]
        self.round = 0; self.active = True; self.status = "running"; self.finals = []; self.tokens = 0; self.explores = 0
        self.raw = []                                                  # unstripped completions (reasoning included), for distillation
        self.msg_log = []                                              # per message: [tokens, thinking chars, visible chars, cut, has code]
        self.checked = False; self.second = False


def save_state(path, batch, tasks, threads, rnd, t0):
    """Checkpoint of the running batch after each round, so a job killed by its time limit resumes mid-batch."""
    st = dict(batch=batch, rnd=rnd, elapsed=time.time() - t0,
              tasks={tid: dict(agree=T["agree"], written=bool(T.get("written"))) for tid, T in tasks.items()},
              threads=[dict(tid=th.tid, k=th.k, msgs=th.msgs, round=th.round, active=th.active, status=th.status, finals=th.finals,
                            tokens=th.tokens, explores=th.explores, raw=th.raw, checked=th.checked, second=th.second, msg_log=th.msg_log) for th in threads])
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--arc-dir", required=True); ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--tasks-per-batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=12000); ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--no-thinking", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=0, help="unix time: start no new batch/round after it (Kaggle time box)")
    ap.add_argument("--terse", action="store_true", help="ask the model to think in compressed, symbol-like notes")
    ap.add_argument("--share-best", type=int, default=0, help="evolution: from this round on, a thread whose own best "
                    "submission is worse than the task's best (over all threads) is shown that program and its errors")
    ap.add_argument("--task-token-budget", type=int, default=0, help="stop a task once its threads generated this many "
                    "tokens in total (emulates a fixed per-task Kaggle compute budget); 0 = unlimited")
    ap.add_argument("--chat-kwargs", default=None, help='JSON passed as chat_template_kwargs instead of the Qwen defaults, '
                    'e.g. \'{"reasoning_effort": "high"}\' for gpt-oss')
    ap.add_argument("--keep-transcripts", type=int, default=2, help="conversations stored per task (consistent first)")
    ap.add_argument("--harvest", action="store_true", help="distillation data: needs ground-truth test outputs; a task stops at its "
                    "first thread that is exact on demos AND test; such threads are stored whole, reasoning included")
    a = ap.parse_args()
    pool = S.SafePool(a.workers, target=_agent_worker, fail=_fail)     # before any CUDA context exists
    from vllm import LLM, SamplingParams
    # AGENT_PREFIX=0: prefix caching off (on 4xL4 the hybrid-GDN 27B decodes 2x faster per sequence without it);
    # AGENT_NO_CAR=1: no custom all-reduce (L4 / PCIe). Defaults keep the Leonardo behaviour.
    extra = dict(disable_custom_all_reduce=True, distributed_executor_backend="mp") if os.environ.get("AGENT_NO_CAR") == "1" else {}
    if os.environ.get("AGENT_MAXSEQ"):   # caps concurrent sequences AND the CUDA-graph capture sizes (L4: default 256 OOMs at capture)
        extra["max_num_seqs"] = int(os.environ["AGENT_MAXSEQ"])
    if os.environ.get("AGENT_EAGER") == "1":   # skip torch.compile/CUDA graphs (fp8 + GDN ops fail inductor on vLLM 0.29)
        extra["enforce_eager"] = True
    if os.environ.get("AGENT_SPEC"):    # speculative decoding, e.g. {"method":"qwen3_5_mtp","num_speculative_tokens":1}
        extra["speculative_config"] = json.loads(os.environ["AGENT_SPEC"])
    if os.environ.get("AGENT_QUANT"):   # e.g. fp8: vLLM quantises BF16 weights at load (no calibration); ~lossless, half the bytes
        extra["quantization"] = os.environ["AGENT_QUANT"]
    lora_req = None
    if os.environ.get("AGENT_LORA"):     # LoRA adapter (e.g. brevity self-distillation) served on top of a quantised base
        from vllm.lora.request import LoRARequest
        lp = os.environ["AGENT_LORA"]; rank = json.load(open(os.path.join(lp, "adapter_config.json"))).get("r", 64)
        extra.update(enable_lora=True, max_lora_rank=int(rank), max_loras=1); lora_req = LoRARequest("agent", 1, lp)
        print(f"LoRA adapter: {lp} (rank {rank})", flush=True)
    def _llm(ex):
        return LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len,
                   gpu_memory_utilization=float(os.environ.get("AGENT_GMU", 0.90)),
                   enable_prefix_caching=os.environ.get("AGENT_PREFIX", "1") == "1", seed=a.seed,
                   limit_mm_per_prompt={"image": 0, "video": 0}, **ex)
    try:
        llm = _llm(extra)
    except Exception as e:                                             # e.g. speculative decoding unsupported here
        if "speculative_config" not in extra:
            raise
        print(f"engine with speculative decoding failed ({type(e).__name__}: {str(e)[:200]}); retrying without", flush=True)
        extra.pop("speculative_config"); llm = _llm(extra)
    print(f"engine up; speculative={'speculative_config' in extra}", flush=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=1, temperature=a.temperature, top_p=0.95, top_k=20, max_tokens=a.max_tokens)
    global SYSTEM
    if a.terse:
        SYSTEM = SYSTEM + TERSE
    if os.environ.get("AGENT_ACT_EARLY") == "1":                        # prompted "think less, act more" (control for distillation)
        SYSTEM = SYSTEM + ACT_EARLY
    if os.environ.get("AGENT_OCCAM") == "1":                            # prompted Occam prior on the program (precision of consistent programs)
        SYSTEM = SYSTEM + OCCAM
    tkw = json.loads(a.chat_kwargs) if a.chat_kwargs else {"enable_thinking": not a.no_thinking, "preserve_thinking": False}
    print(f"chat_template_kwargs: {tkw}", flush=True)

    def n_tokens(msgs):
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **tkw)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc        # transformers 5: BatchEncoding, not a list
        if len(ids) and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return len(ids)

    def write_task(tid, T, ths, t0, n_batch):
        tests = T["tests"]
        cons = [f for th in ths for f in reversed(th.finals) if f["ok"]]    # every demo-consistent program (latest first)
        rest = sorted([f for th in ths for f in th.finals if not f["ok"] and f["outs"]], key=lambda f: -f["fitness"])
        attempts = []
        for i in range(len(tests)):
            votes = {}
            for f in cons:
                if f["outs"] and f["outs"][i] is not None:
                    e = votes.setdefault(json.dumps(f["outs"][i]), [f["outs"][i], 0, 10**9]); e[1] += 1; e[2] = min(e[2], len(f["src"]))
            ranked = [v[0] for v in sorted(votes.values(), key=lambda v: (-v[1], v[2]))]   # votes, then shorter program (MDL)
            for f in rest:                                             # fill free attempt slots with the fittest near misses
                if len(ranked) >= 2:
                    break
                if f["outs"][i] is not None and f["outs"][i] not in ranked:
                    ranked.append(f["outs"][i])
            attempts.append(ranked[:2])
        rec = dict(task_id=tid, sampled=len(ths), consistent=len(cons), seconds=(time.time() - t0) / n_batch,
                   best_fitness=max([f["fitness"] for th in ths for f in th.finals] + [0.0]),
                   consistent_by_gen={str(r): sum(1 for f in cons if f["round"] == r) for r in range(1, a.rounds + 1)},
                   tokens=sum(th.tokens for th in ths), explores=sum(th.explores for th in ths),
                   finals=sum(len(th.finals) for th in ths), status={st: sum(th.status == st for th in ths) for st in set(th.status for th in ths)},
                   attempts=attempts, programs=[f["src"] for f in cons][:4],
                   transcripts=[th.msgs[1:] for th in sorted(ths, key=lambda th: th.status != "consistent")[: a.keep_transcripts]],
                   msg_log=[th.msg_log for th in ths])
        if all(t[1] is not None for t in tests):                      # ground truth known: keep every verified trajectory
            truth = [t[1] for t in tests]; rec["trajectories"] = []    # (reasoning included) whatever the mode
            for th in ths:
                if th.status == "consistent" and th.finals[-1]["outs"] == truth:
                    turns, k = [], 0
                    for m in th.msgs:
                        if m["role"] == "assistant":
                            turns.append({"role": "assistant", "content": th.raw[k]}); k += 1
                        else:
                            turns.append(m)
                    rec["trajectories"].append(dict(rounds=th.round, tokens=th.tokens, turns=turns))
        if all(t[1] is not None for t in tests):
            rec["pass1"] = all(at and at[0] == t[1] for at, t in zip(attempts, tests))
            rec["pass2"] = all(t[1] in at for at, t in zip(attempts, tests))
            rec["oracle"] = all(any(f["outs"] and f["outs"][i] == t[1] for f in cons) for i, t in enumerate(tests))
            rec["pass2_consistent_only"] = rec["oracle"] and rec["pass2"]
        fh.write(json.dumps(rec) + "\n"); fh.flush(); T["written"] = True
        print(f"  {tid}: consistent {len(cons)}/{len(ths)} best_fit {rec['best_fitness']:.2f} p@2 {rec.get('pass2')} "
              f"tokens {rec['tokens']//1000}k explores {rec['explores']} finals {rec['finals']}", flush=True)

    ids = [l.strip() for l in open(a.ids) if l.strip()][a.shard :: a.n_shards]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
                if r.get("task_id"): done.add(r["task_id"])
            except json.JSONDecodeError:
                pass
    ids = [t for t in ids if t not in done]
    print(f"resume: {len(done)} tasks done, {len(ids)} to go | threads {a.threads} rounds {a.rounds}", flush=True)
    fh = open(a.out, "a")

    state_path = a.out + ".state.json"; resume_state = None
    batches = [ids[i : i + a.tasks_per_batch] for i in range(0, len(ids), a.tasks_per_batch)]
    if os.path.exists(state_path):
        st = json.load(open(state_path)); sb = [t for t in st["batch"] if t not in done]
        if sb:                                                         # continue the interrupted batch first
            resume_state = st; batches = [sb] + [[t for t in b if t not in sb] for b in batches]
            batches = [b for b in batches if b]
    for batch in batches:
        if a.deadline and time.time() > a.deadline:
            print("deadline reached: no new batch", flush=True); break
        t0 = time.time(); tasks = {}; threads = []; start_rnd = 1
        for tid in batch:
            d = json.load(open(os.path.join(a.arc_dir, tid + ".json")))
            demos = [(p["input"], p["output"]) for p in d["train"]]; tests = [(q["input"], q.get("output")) for q in d["test"]]
            tasks[tid] = dict(demos=demos, tests=tests, agree=False)
            threads += [Thread(tid, k, demos, [t[0] for t in tests], a.rounds) for k in range(a.threads)]
        if resume_state is not None:
            for d in resume_state["threads"]:
                th = next((x for x in threads if x.tid == d["tid"] and x.k == d["k"]), None)
                if th is None:
                    continue
                for key in ("msgs", "round", "active", "status", "finals", "tokens", "explores", "raw"):
                    setattr(th, key, d[key])
                th.checked, th.second = d.get("checked", False), d.get("second", False)
                th.msg_log = d.get("msg_log", [])
            for tid, T in tasks.items():
                T["agree"] = resume_state["tasks"].get(tid, {}).get("agree", False)
            start_rnd = resume_state["rnd"] + 1; t0 = time.time() - resume_state["elapsed"]
            print(f"resumed batch {batch} at round {start_rnd}", flush=True); resume_state = None
        for rnd in range(start_rnd, a.rounds + 1):
            if a.deadline and time.time() > a.deadline:
                print(f"deadline reached at round {rnd}: closing the batch", flush=True); break
            live = []
            if a.task_token_budget:                                    # per-task generation budget (Kaggle emulation)
                for tid in tasks:
                    if sum(th.tokens for th in threads if th.tid == tid) >= a.task_token_budget:
                        for th in threads:
                            if th.tid == tid and th.active:
                                th.active = False; th.status = "token budget"
            for th in threads:
                if not th.active or tasks[th.tid]["agree"]:
                    continue
                room = a.max_model_len - n_tokens(th.msgs) - 64
                if room < MIN_MSG_TOKENS or (room < a.max_tokens and not SHRINK):
                    th.active = False; th.status = "context full"; continue
                th.sp = sp if room >= a.max_tokens else SamplingParams(n=1, temperature=a.temperature, top_p=0.95, top_k=20, max_tokens=room)
                live.append(th)
            if not live:
                break
            try:
                outs = llm.chat([th.msgs for th in live], [th.sp for th in live], use_tqdm=False, chat_template_kwargs=tkw, lora_request=lora_req)
            except Exception as e:                                     # one bad request must not kill the shard
                print(f"    round {rnd}: llm.chat failed ({type(e).__name__}: {str(e)[:160]}); retrying thread by thread", flush=True)
                keep, outs = [], []
                for th in live:
                    try:
                        outs += llm.chat([th.msgs], th.sp, use_tqdm=False, chat_template_kwargs=tkw, lora_request=lora_req); keep.append(th)
                    except Exception:
                        th.active = False; th.status = "request rejected"
                live = keep
            jobs, owners = [], []
            for th, o in zip(live, outs):
                text = o.outputs[0].text; th.tokens += len(o.outputs[0].token_ids); th.round = rnd; th.raw.append(text)
                vis, code, truncated = extract(text, thinking=not a.no_thinking, cut=o.outputs[0].finish_reason == "length")
                th.msgs.append({"role": "assistant", "content": vis if vis else "(no answer: the message was cut off)"})
                th.msg_log.append([len(o.outputs[0].token_ids), len(text.split("</think>")[0]) if "</think>" in text else len(text),
                                   len(vis or ""), bool(truncated), code is not None])
                th.pending = None
                if code is None:
                    th.pending = ("Your message was cut off before any code (token limit). Think less, act more: reply with one "
                                  "```python block now." if truncated else "I found no ```python block in your message. Reply with exactly one.")
                    continue
                kind = "final" if code.lstrip().startswith("# FINAL") else "explore"
                jobs.append((kind, code, tasks[th.tid]["demos"], [t[0] for t in tasks[th.tid]["tests"]])); owners.append(th)
            for th, r in zip(owners, pool.run(jobs) if jobs else []):
                left = a.rounds - rnd
                if r["kind"] == "explore":
                    th.explores += 1
                    body = (r["stdout"] or "(nothing was printed)") + (f"\n{r['err']}" if r["err"] else "")
                    th.pending = f"Output:\n{body}"
                else:
                    th.finals.append(dict(fitness=r["fitness"], ok=r["ok"], outs=r["outs"], src=jobs[owners.index(th)][1], round=rnd))
                    if r["ok"]:
                        if th.second:                                  # the extra program was submitted: stop either way
                            th.active = False; th.status = "consistent"; continue
                        prob = test_problems(r, tasks[th.tid]["demos"], [t[0] for t in tasks[th.tid]["tests"]]) if TESTCHECK and not th.checked else None
                        if prob and left >= 1:
                            th.checked = True                          # one test-check message per thread
                            th.pending = ("Every train demo is now correct, but the rule does not carry over to the test inputs:\n" + prob +
                                          "\nThe true rule must produce a sensible output for every test input. Revise it and submit a new `# FINAL`.")
                            continue
                        if SECOND and left >= 1:
                            th.second = True
                            th.pending = ("Every train demo is correct. A program can fit all demos and still be wrong on the test inputs "
                                          "when it relies on an accidental detail of the demos. Write an INDEPENDENT second transform: the "
                                          "simplest, shortest program that explains every demo, with the fewest special cases and no values "
                                          "hard-coded from particular demos. If you are confident the first rule is already the simplest, you "
                                          "may resubmit it. Submit it as a `# FINAL` block.")
                            continue
                        th.active = False; th.status = "consistent"; continue
                    if th.second:                                      # second program failed the demos: the first one stands
                        th.active = False; th.status = "consistent"; continue
                    rep = diff_report(r["produced"], tasks[th.tid]["demos"], r["err"]) if r["produced"] and any(p is not None for p in r["produced"]) \
                        else f"transform failed: {r['err']}"
                    th.pending = f"Not correct yet.\n{rep}"
            for th in live:
                if not th.active:
                    continue
                left = a.rounds - rnd
                if left == 0:
                    th.active = False; th.status = "out of rounds"; continue
                tail = " This is your LAST message: submit your best `# FINAL` transform now." if left == 1 else f" ({left} messages left.)"
                share = ""
                if a.share_best and rnd >= a.share_best:              # Berman-style: seed the thread with the task's fittest program
                    cands = [f for o in threads if o.tid == th.tid and o is not th for f in o.finals if not f["ok"] and f.get("fitness", 0) > 0]
                    mine = max([f["fitness"] for f in th.finals] + [0.0])
                    if cands:
                        best = max(cands, key=lambda f: f["fitness"])
                        if best["fitness"] > mine + 1e-9:
                            share = (f"\n\nFor reference, another attempt at this puzzle reached {100*best['fitness']:.0f}% of the train "
                                     f"cells with this program (not fully correct). Use it only if it helps:\n```python\n{best['src']}\n```")
                th.msgs.append({"role": "user", "content": (th.pending or "Continue.") + share + tail})
            for tid, T in tasks.items():                               # early stop: two consistent threads agree on the test
                cons = [json.dumps(f["outs"]) for th in threads if th.tid == tid and th.status == "consistent" for f in th.finals[-1:]]
                T["agree"] = any(cons.count(c) >= 2 for c in cons)
                if a.harvest:                                          # ground truth known: one verified trajectory is enough
                    truth = json.dumps([t[1] for t in T["tests"]]); T["agree"] = truth in cons
                if os.environ.get("AGENT_NO_EARLYSTOP") == "1":            # mechanism study: every thread runs to its own end
                    T["agree"] = False
            for tid, T in tasks.items():                               # finished tasks are written immediately (crash-safe)
                ths = [th for th in threads if th.tid == tid]
                if not T.get("written") and (T["agree"] or not any(th.active for th in ths) or rnd == a.rounds):
                    write_task(tid, T, ths, t0, len(batch))
            n_act = sum(th.active and not tasks[th.tid]["agree"] for th in threads)
            save_state(state_path, batch, tasks, threads, rnd, t0)
            print(f"    round {rnd}: ran {len(live)} threads, {len(jobs)} code runs, {n_act} still active, "
                  f"{sum(th.status == 'consistent' for th in threads)} consistent | {time.time()-t0:.0f}s", flush=True)

        for tid, T in tasks.items():
            if not T.get("written"):
                write_task(tid, T, [th for th in threads if th.tid == tid], t0, len(batch))
        if os.path.exists(state_path):
            os.remove(state_path)
    fh.close()


if __name__ == "__main__":
    main()
