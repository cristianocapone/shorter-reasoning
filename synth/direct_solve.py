"""Direct (transductive) ARC solving with a thinking LLM, ARC-Prize-harness style: one prompt per test input with the
train pairs as JSON grids, the model thinks (reasoning effort from the chat template, Qwen3.8 default xhigh) and answers
with a grid. N samples per test input = the attempts (pass@2 with N=2). No tools, no code execution.

Measured (dev-40, Qwen3.8-27B INT4, temp 0.6, N=2): 64k budget 17/40 (seed 1: 16), 48k 10, 32k 4, 96k 14; the model's
verified ARC-AGI-2 semi-private score at xhigh is 42.4%. Half of the samples hit the 64k cap, 21% of test inputs got no grid.

Levers in this file (all env-switchable, default off):
  DIRECT_FORCE=1     budget forcing: a sample cut at the cap mid-thought gets "</think>" + a short answer phase (DIRECT_FORCE_TOK)
  DIRECT_FILTER=1    demo-invariant filter (shape / palette rules from the train pairs; offline: rejects 41% of wrong attempts,
                     0 correct ones): a rejected attempt is replaced by a fresh sample, up to DIRECT_EXTRA extra samples per
                     test input, until two distinct passing grids exist
  DIRECT_SLICE_MIN   resumable across 30-min debug slices (in-flight token ids + finished samples go to <out>.partial.json)
  DIRECT_FINAL_MIN   hard deadline: flush every task with >= 1 finished sample per test input, then exit
Request order: sample 0 of every task (smallest first) before any sample 1, extras last -> under a budget every task gets a try.
Accepts solve_agent.py's command line (unknown flags ignored): runs under leonardo/job_agent_dbg.slurm via SOLVER=synth/direct_solve.py.
"""
import argparse, json, os, re, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--arc-dir", required=True); ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
ap.add_argument("--max-tokens", type=int, default=65536); ap.add_argument("--max-model-len", type=int, default=73728)
ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chat-kwargs", default=None)
ap.add_argument("--tasks-per-batch", type=int, default=0, help="work on the first N unfinished tasks only (bounded in-flight state per slice)")
a, _unknown = ap.parse_known_args()
N = int(os.environ.get("DIRECT_N", 2))
EFFORT = os.environ.get("DIRECT_EFFORT", "")
FORCE = os.environ.get("DIRECT_FORCE") == "1"
FORCE_TOK = int(os.environ.get("DIRECT_FORCE_TOK", 3072))
FORCE_TXT = os.environ.get("DIRECT_FORCE_TXT", "\n\nI am out of thinking budget; I commit to my current best hypothesis and write the output grid now.\n</think>\n\n")
FILT = os.environ.get("DIRECT_FILTER") == "1"
EXTRA = int(os.environ.get("DIRECT_EXTRA", 2))
SLICE = float(os.environ.get("DIRECT_SLICE_MIN", 0)) * 60
FINAL = float(os.environ.get("DIRECT_FINAL_MIN", 0)) * 60
HARVEST = os.environ.get("DIRECT_HARVEST") == "1"          # keep the raw text of every correct, unforced sample (SFT trajectories)
REFLECT = os.environ.get("DIRECT_REFLECT") == "1"          # sample 1 sees sample 0's proposed output and is asked to check / correct it
T_START = time.time()

PROMPT = ("You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\n"
          "Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation "
          "in the training examples that maps the input to the output, then apply that pattern to the test input to give a "
          "final output.\n\nRespond in the format of the training output examples\n\n--Training Examples--\n")


def fmt(g):
    return "[" + ",\n".join(json.dumps(r, separators=(",", ":")) for r in g) + "]"


def prompt(task, i, proposal=None):
    s = PROMPT
    for k, p in enumerate(task["train"]):
        s += f"--Example {k}-- \n\n INPUT: \n\n{fmt(p['input'])}\n\nOUTPUT: \n\n{fmt(p['output'])}\n\n"
    s += f"--End of Training Examples--\n\n--Test Input--\n{fmt(task['test'][i]['input'])}\n--End of Test Input--\n\n"
    if proposal is not None:
        s += ("A previous attempt proposed this output for the test input:\n" + fmt(proposal) + "\n\nIt may be wrong. Re-derive the "
              "rule from the training examples, check whether this proposal follows it exactly, and give the correct output grid.\n\n")
    s += "Your response:"
    return s


GRID = re.compile(r"\[\s*\[[\d,\s\[\]]*\]\s*\]")


def parse(text):
    """Last well-formed 2-D integer grid in the answer (after the thinking block if present, else anywhere)."""
    for part in ([text.split("</think>")[-1]] if "</think>" in text else []) + [text]:
        for m in reversed(GRID.findall(part)):
            try:
                g = json.loads(m)
            except Exception:
                continue
            if isinstance(g, list) and g and all(isinstance(r, list) and r and len(r) == len(g[0]) for r in g) \
                    and all(isinstance(v, int) and 0 <= v <= 9 for r in g for v in r) and len(g) <= 30 and len(g[0]) <= 30:
                return g
    return None


# ---- demo-invariant filter (nvarc_port/filter_analysis.py rules) ----
def shape(g): return (len(g), len(g[0]))
def pal(g): return {v for r in g for v in r}


def rules(task):
    tr = task["train"]; R = {}
    if all(shape(p["input"]) == shape(p["output"]) for p in tr): R["same_shape"] = True
    outs = {shape(p["output"]) for p in tr}
    if len(outs) == 1 and not R.get("same_shape"): R["fixed_shape"] = outs.pop()
    ratios = {(shape(p["output"])[0] / shape(p["input"])[0], shape(p["output"])[1] / shape(p["input"])[1]) for p in tr}
    if len(ratios) == 1 and not R.get("same_shape"): R["ratio"] = ratios.pop()
    if all(pal(p["output"]) <= pal(p["input"]) for p in tr): R["pal_sub"] = True
    ops = [pal(p["output"]) for p in tr]
    if all(o == ops[0] for o in ops): R["pal_fixed"] = ops[0]
    return R


def passes(g, tin, R):
    if g is None: return False
    if R.get("same_shape") and shape(g) != shape(tin): return False
    if "fixed_shape" in R and shape(g) != R["fixed_shape"]: return False
    if "ratio" in R:
        r = R["ratio"]
        if shape(g) != (round(shape(tin)[0] * r[0]), round(shape(tin)[1] * r[1])): return False
    if R.get("pal_sub") and not pal(g) <= pal(tin): return False
    if "pal_fixed" in R and pal(g) != R["pal_fixed"]: return False
    return True


def main():
    ids = [l.strip() for l in open(a.ids) if l.strip()][a.shard::a.n_shards]
    done = {json.loads(l)["task_id"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()
    tasks = {t: json.load(open(os.path.join(a.arc_dir, t + ".json"))) for t in ids if t not in done}
    size = lambda t: sum(len(g) * len(g[0]) for p in tasks[t]["train"] for g in (p["input"], p["output"]))
    todo = sorted(tasks, key=size)
    if a.tasks_per_batch > 0:                                      # debug slices: keep the in-flight set small (re-prefill cost on resume)
        todo = todo[:a.tasks_per_batch]; tasks = {t: tasks[t] for t in todo}
    print(f"resume: {len(done)} done, {len(todo)} to go | n={N} cap={a.max_tokens} force={FORCE} filter={FILT}+{EXTRA}", flush=True)
    if not todo:
        return
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    extra = {}
    if os.environ.get("AGENT_MAXSEQ"):
        extra["max_num_seqs"] = int(os.environ["AGENT_MAXSEQ"])
    if os.environ.get("AGENT_SPEC"):
        extra["speculative_config"] = json.loads(os.environ["AGENT_SPEC"])
    if os.environ.get("AGENT_EAGER") == "1":
        extra["enforce_eager"] = True

    def _llm(ex):
        return LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, seed=a.seed,
                   gpu_memory_utilization=float(os.environ.get("AGENT_GMU", 0.90)),
                   enable_prefix_caching=os.environ.get("AGENT_PREFIX", "1") == "1",
                   limit_mm_per_prompt={"image": 0, "video": 0}, **ex)
    try:
        llm = _llm(extra)
    except Exception as e:
        if "speculative_config" not in extra:
            raise
        print(f"spec decoding failed ({type(e).__name__}); retrying without", flush=True)
        extra.pop("speculative_config"); llm = _llm(extra)
    print(f"engine up; speculative={'speculative_config' in extra}", flush=True)
    tok = llm.get_tokenizer(); eng = llm.llm_engine
    ckw = json.loads(a.chat_kwargs) if a.chat_kwargs else {"enable_thinking": True}
    if EFFORT:
        ckw["reasoning_effort"] = EFFORT
    fids = tok.encode(FORCE_TXT, add_special_tokens=False)

    part_path = a.out + ".partial.json"
    _pj = json.load(open(part_path)) if os.path.exists(part_path) else {}
    run_prev, fin_prev = _pj.get("run", {}), _pj.get("fin", {})      # rid -> ids ; rid -> [ids, c]

    pids = {}                                                       # (t, i) -> prompt token ids
    RL = {t: rules(tasks[t]) for t in todo}
    S = {}                                                          # (t, i) -> {spawned, fin: {k: (text, ntok, c, ids)}, inflight}
    for t in todo:
        for i in range(len(tasks[t]["test"])):
            text = tok.apply_chat_template([{"role": "user", "content": prompt(tasks[t], i)}], tokenize=False,
                                           add_generation_prompt=True, **ckw)
            pids[(t, i)] = tok.encode(text, add_special_tokens=False)
            S[(t, i)] = dict(spawned=N, fin={}, inflight=set())
    for rid in list(run_prev) + list(fin_prev):                      # extras from an earlier slice keep their slot
        t, i, k = rid.split("|")[:3]
        if (t, int(i)) in S:
            S[(t, int(i))]["spawned"] = max(S[(t, int(i))]["spawned"], int(k) + 1)
    meta = {}                                                       # rid -> (t, i, k, prior ids)
    running = {}                                                    # rid -> ids so far (for the slice save)
    base_of = {}                                                    # forced rid -> base ids (saved instead of the forced tail)
    flushed = []
    fh = open(a.out, "a")

    def passing_grids(t, i):
        out = []
        for k in sorted(S[(t, i)]["fin"]):
            g = parse(S[(t, i)]["fin"][k][0])
            if passes(g, tasks[t]["test"][i]["input"], RL[t]) and g not in out:
                out.append(g)
        return out

    def prompt_ids(t, i, k):
        if REFLECT and k == 1 and 0 in S[(t, i)]["fin"]:
            g0 = parse(S[(t, i)]["fin"][0][0])
            if g0 is not None:
                key = (t, i, 1)
                if key not in pids:
                    text = tok.apply_chat_template([{"role": "user", "content": prompt(tasks[t], i, g0)}], tokenize=False,
                                                   add_generation_prompt=True, **ckw)
                    pids[key] = tok.encode(text, add_special_tokens=False)
                return pids[key]
        return pids[(t, i)]

    def add(rid, prior, max_new, seed):
        t, i, k = meta[rid][0], meta[rid][1], meta[rid][2]
        pr = prompt_ids(t, i, k); room = a.max_model_len - len(pr) - len(prior) - 1   # never exceed the context window
        if room <= 0:                                                   # (hybrid-direct v1 commit: 69657 > 69632 -> VLLMValidationError)
            S[(t, i)]["inflight"].discard(k); finish(t, i, k, base_of.get(rid, prior), 1); return
        eng.add_request(rid, {"prompt_token_ids": pr + prior},
                        SamplingParams(n=1, temperature=a.temperature, top_p=0.95, top_k=20, max_tokens=min(max_new, room), seed=seed,
                                       output_kind=RequestOutputKind.CUMULATIVE))
        running[rid] = prior

    def spawn_force(t, i, k, ids):
        rid = f"{t}|{i}|{k}|f"; base_of[rid] = ids
        keep = a.max_model_len - len(prompt_ids(t, i, k)) - len(fids) - FORCE_TOK - 1
        if keep < len(ids):                                             # make room for the answer: drop the tail of the thinking
            ids = ids[:max(0, keep)]
        meta[rid] = (t, i, k, ids + fids)
        S[(t, i)]["inflight"].add(k)
        add(rid, ids + fids, FORCE_TOK, a.seed * 1000 + k * 7 + 99)

    def spawn(t, i, k, prior):
        rid = f"{t}|{i}|{k}"; meta[rid] = (t, i, k, prior)
        left = a.max_tokens - len(prior)
        if left <= 0:
            if FORCE and "</think>" not in tok.decode(prior):
                spawn_force(t, i, k, prior)
            else:
                finish(t, i, k, prior, 1)
            return
        S[(t, i)]["inflight"].add(k)
        add(rid, prior, left, a.seed * 1000 + k * 7 + len(prior))

    def complete(t, i):
        st = S[(t, i)]
        return not st["inflight"] and not st.get("deferred") and len(st["fin"]) >= st["spawned"]

    def flush(t, final=False):
        attempts, toks, cut, forced, nsamp, sols = [], 0, 0, 0, 0, []
        for i in range(len(tasks[t]["test"])):
            st = S[(t, i)]; seen = passing_grids(t, i)
            for k in sorted(st["fin"]):
                g = parse(st["fin"][k][0])
                if g is not None and g not in seen:
                    seen.append(g)
            attempts.append(seen[:2]); sols.append(tasks[t]["test"][i].get("output"))
            toks += sum(v[1] for v in st["fin"].values()); cut += sum(1 for v in st["fin"].values() if v[2])
            forced += sum(1 for v in st["fin"].values() if v[2] == 2); nsamp += len(st["fin"])
        ok = all(s is not None for s in sols) and all(s in att for s, att in zip(sols, attempts))
        rec = dict(task_id=t, pass2=bool(ok), oracle=bool(ok), consistent=0, tokens=toks, cut=cut, forced=forced,
                   samples=nsamp, attempts=attempts, seconds=time.time() - T_START, partial=final)
        if HARVEST:
            rec["trajectories"] = [dict(rounds=1, tokens=v[1], test_input=i,
                                        turns=[{"role": "user", "content": prompt(tasks[t], i)}, {"role": "assistant", "content": v[0]}])
                                   for i in range(len(tasks[t]["test"])) for k, v in sorted(S[(t, i)]["fin"].items())
                                   if v[2] == 0 and k < N and sols[i] is not None and parse(v[0]) == sols[i]]
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        flushed.append(t)
        print(f"[{time.time() - T_START:6.0f}s] {t} {'SOLVED' if ok else 'wrong'} tokens {toks} samples {nsamp} cut {cut} "
              f"forced {forced} | done {len(flushed)}", flush=True)

    def maybe_flush(t):
        if t not in flushed and all(complete(t, i) for i in range(len(tasks[t]["test"]))):
            flush(t)

    def finish(t, i, k, ids, c):
        st = S[(t, i)]; st["inflight"].discard(k); st["fin"][k] = (tok.decode(ids), len(ids), c, ids)
        if REFLECT and k == 0 and st.pop("deferred", False):
            spawn(t, i, 1, [])
        if FILT and not st["inflight"] and st["spawned"] < N + EXTRA and len(passing_grids(t, i)) < 2:
            k2 = st["spawned"]; st["spawned"] += 1
            spawn(t, i, k2, [])
        maybe_flush(t)

    # queue: sample 0 of every task, then sample 1, ..., then extras (smallest task first within each level)
    kmax = max(S[x]["spawned"] for x in S)
    for k in range(kmax):
        for t in todo:
            for i in range(len(tasks[t]["test"])):
                if k >= S[(t, i)]["spawned"]:
                    continue
                rid = f"{t}|{i}|{k}"
                if rid in fin_prev:
                    ids_, c_ = fin_prev[rid]
                    S[(t, i)]["fin"][k] = (tok.decode(ids_), len(ids_), c_, ids_)
                elif REFLECT and k == 1 and 0 not in S[(t, i)]["fin"]:
                    S[(t, i)]["deferred"] = True                     # spawned by finish() once sample 0 is in
                else:
                    spawn(t, i, k, run_prev.get(rid, []))
    for t in todo:
        maybe_flush(t)

    last_log = time.time()
    while eng.has_unfinished_requests():
        for ro in eng.step():
            rid = ro.request_id; t, i, k, prior = meta[rid]
            ids = prior + list(ro.outputs[0].token_ids)
            if not ro.finished:
                running[rid] = ids; continue
            running.pop(rid, None)
            if rid.endswith("|f"):
                finish(t, i, k, ids, 2)
            elif FORCE and ro.outputs[0].finish_reason == "length" and "</think>" not in tok.decode(ids):
                S[(t, i)]["inflight"].discard(k); spawn_force(t, i, k, ids)
            else:
                finish(t, i, k, ids, int(ro.outputs[0].finish_reason == "length"))
        if time.time() - last_log > 120:
            print(f"[{time.time() - T_START:6.0f}s] in flight {len(running)} seqs, {sum(len(v) for v in running.values())} tokens",
                  flush=True)
            last_log = time.time()
        if SLICE and time.time() - T_START > SLICE:
            break
        if FINAL and time.time() - T_START > FINAL:
            break
    if FINAL and time.time() - T_START > FINAL:                     # deadline: write every task with >= 1 finished sample per input
        for t in todo:
            if t not in flushed and all(S[(t, i)]["fin"] for i in range(len(tasks[t]["test"]))):
                flush(t, final=True)
        print(f"final flush at {(time.time() - T_START) / 60:.0f} min: {len(flushed)} tasks written", flush=True)
        os._exit(0)
    if SLICE and eng.has_unfinished_requests():                     # slice end: save running + finished samples of open tasks
        run = {}
        for rid, ids in running.items():
            run[rid[:-2] if rid.endswith("|f") else rid] = base_of.get(rid, ids)
        fin = {f"{t}|{i}|{k}": [v[3], v[2]] for (t, i), st in S.items() if t not in flushed for k, v in st["fin"].items()}
        json.dump({"run": run, "fin": fin}, open(part_path, "w"))
        print(f"slice end: saved {len(run)} running + {len(fin)} finished samples", flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
