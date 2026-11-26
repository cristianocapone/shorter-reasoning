"""Third domain for the brevity study: competitive programming with hidden unit tests (APPS, stdin/stdout problems).

Same interface and record format as synth/math_harvest.py (harvest mode -> sft_student trajectories; eval mode -> per-problem
accuracy over K samples), so leonardo/chain_dbg.sh + job_agent_dbg.slurm (SOLVER=synth/code_harvest.py) run it unchanged.
Verifier: last ```python block of the visible answer, run on up to CODE_TESTS test inputs (stdin) with a per-test timeout;
correct = every tested stdout matches (whitespace-normalised). Verification runs in a thread pool (subprocesses) so the
engine loop never blocks on it.
  data: MATH_DATA=<jsonl with id, question, inputs, outputs>;  --ids file;  MATH_K samples;  MATH_EVAL=1 eval mode.

NatComm additions (2026-11-02). Defaults are unchanged; old records and synth/paper_stats.py keep working. The old 'cut'
field is left exactly as it was: paper_stats.py and the manuscript say it also counted samples without a code block, while
this local copy computes it as finish_reason == 'length' (compare with the cluster copy when syncing); cap_cut is the
unambiguous count.
  New fields in every record, lists in the order of the K samples:
    cap_cut  number of samples stopped by the length cap (finish_reason 'length')
    lens     generated tokens of each sample (len(token_ids))
    oks      1 if the sample passed the tests, else 0
    fins     1 if the sample stopped by itself (finish_reason 'stop'), else 0
    ans_tok  index (0-based, among the generated tokens) of the token where the graded program (the last ```python block
             of the visible answer) first appears: the first fenced block in the reasoning with the same code up to
             whitespace, else the graded block itself; null when there is no graded program (no code block)
  --save-texts PATH, --data PATH and max_tokens up to 32768: as in synth/math_harvest.py (same text-record format; the
             reference 'answer' is omitted there because the tests are in the data file).
"""
import argparse, concurrent.futures as cf, json, os, re, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
ap.add_argument("--max-tokens", type=int, default=8192); ap.add_argument("--max-model-len", type=int, default=12288)
ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default=None, help="problem jsonl (id, question, inputs, outputs[, difficulty]); default $MATH_DATA")
ap.add_argument("--save-texts", default=os.environ.get("MATH_SAVE_TEXTS"),
                help="gzip JSONL for every generation ('auto', placeholders {shard} {out_stem} {out_dir}); default $MATH_SAVE_TEXTS")
a = ap.parse_known_args()[0] if __name__ == "__main__" else None      # importable without a CLI
K = int(os.environ.get("MATH_K", 4)); EVAL = os.environ.get("MATH_EVAL") == "1"
SLICE = float(os.environ.get("DIRECT_SLICE_MIN", 0)) * 60
NT = int(os.environ.get("CODE_TESTS", 10)); TO = float(os.environ.get("CODE_TIMEOUT", 4))
PROMPT = ("Solve the following competitive programming problem in Python 3. The program reads from standard input and "
          "writes to standard output.\n\n{q}\n\nReason step by step, then give the complete program in a single ```python code block.")
FENCE = r"```(?:python|py)?[ \t]*\n(.*?)```"


def code_of(text):
    vis = text.split("</think>")[-1] if "</think>" in text else ""
    blocks = re.findall(FENCE, vis, re.S)
    return blocks[-1] if blocks else None


def code_first_char(text):
    """Character offset where the graded program (code_of) first appears: the first fenced block of the reasoning (the
    text before the last </think>) whose code equals it up to whitespace, else the graded block itself; None if no program."""
    if code_of(text) is None:
        return None
    cut = text.rfind("</think>")
    key = re.sub(r"\s+", "", code_of(text))
    for m in re.finditer(FENCE, text[:cut], re.S):
        if re.sub(r"\s+", "", m.group(1)) == key:
            return m.start()
    last = list(re.finditer(FENCE, text[cut:], re.S))[-1]
    return cut + last.start()


def norm(s):
    return "\n".join(l.rstrip() for l in s.strip().splitlines())


def run_tests(code, ins, outs):
    if not code:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        for x, y in list(zip(ins, outs))[:NT]:
            try:
                p = subprocess.run([sys.executable, path], input=x, capture_output=True, text=True, timeout=TO)
            except subprocess.TimeoutExpired:
                return False
            if p.returncode != 0 or norm(p.stdout) != norm(y):
                return False
        return True
    finally:
        os.unlink(path)


def per_sample(tok, outs, ok):
    """cap_cut / lens / oks / fins / ans_tok of one problem's samples, in sample order (see the module docstring)."""
    ans_tok = []
    for o in outs:
        try:                                         # a diagnostic must never stop a long evaluation
            ids = list(o.token_ids)
            ans_tok.append(ec.char_to_token(tok, ids, code_first_char(tok.decode(ids, skip_special_tokens=True))))
        except Exception as e:
            print(f"warning: ans_tok failed: {e!r}", flush=True)
            ans_tok.append(None)
    return dict(cap_cut=sum(o.finish_reason == "length" for o in outs), lens=[len(o.token_ids) for o in outs],
                oks=[int(bool(g)) for g in ok], fins=[int(o.finish_reason == "stop") for o in outs], ans_tok=ans_tok)


def main():
    data = {}
    for l in open(a.data or os.environ["MATH_DATA"]):
        if l.strip():
            d = json.loads(l); data[d["id"]] = d
    ids = [l.strip() for l in open(a.ids) if l.strip()][a.shard::a.n_shards]
    done = {json.loads(l)["task_id"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()
    todo = [i for i in ids if i not in done]
    texts = ec.texts_path(a.save_texts, a.out, a.shard, a.n_shards)
    print(f"resume: {len(done)} done, {len(todo)} to go | k={K} cap={a.max_tokens} eval={EVAL}"
          + (f" | texts -> {texts}" if texts else ""), flush=True)
    if not todo:
        return
    t_start = time.time()
    prompts = {pid: [{"role": "user", "content": PROMPT.format(q=data[pid]["question"])}] for pid in todo}
    mml = ec.resolve_max_model_len(a.model, ec.prompt_lengths(a.model, prompts.values(), enable_thinking=True),
                                   a.max_tokens, a.max_model_len)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=mml, seed=a.seed,
              gpu_memory_utilization=float(os.environ.get("AGENT_GMU", 0.90)), enable_prefix_caching=True)
    tok = llm.get_tokenizer(); eng = llm.llm_engine
    print("engine up", flush=True)
    sp = SamplingParams(n=K, temperature=a.temperature, top_p=0.95, top_k=20, max_tokens=a.max_tokens, seed=a.seed,
                        output_kind=RequestOutputKind.FINAL_ONLY)
    sampling = dict(n=K, temperature=a.temperature, top_p=0.95, top_k=20, max_tokens=a.max_tokens, seed=a.seed)
    for pid in todo:
        eng.add_request(pid, {"prompt": tok.apply_chat_template(prompts[pid], tokenize=False, add_generation_prompt=True, enable_thinking=True)}, sp)
    pool = cf.ThreadPoolExecutor(max_workers=int(os.environ.get("CODE_WORKERS", 24)))
    pend = {}                                                       # future -> (pid, outs, prompt token ids)
    n = 0
    fh = open(a.out, "a")

    def drain(block=False):
        nonlocal n
        ready = [f for f in pend if f.done()] if not block else list(cf.as_completed(list(pend)))
        for fut in ready:
            pid, outs, pti = pend.pop(fut); ok = fut.result()
            toks = [len(o.token_ids) for o in outs]; cut = [o.finish_reason == "length" for o in outs]
            rec = dict(task_id=pid, acc=sum(ok) / len(ok), tokens=sum(toks), mean_tokens=sum(toks) / len(toks), cut=sum(cut),
                       pass2=bool(ok and ok[0]), difficulty=data[pid].get("difficulty"))
            rec.update(per_sample(tok, outs, ok))
            if not EVAL:
                rec["trajectories"] = [dict(rounds=1, tokens=tk, turns=prompts[pid] + [{"role": "assistant", "content": o.text}])
                                       for o, g, tk in zip(outs, ok, toks) if g]
            if texts:
                ec.append_texts(texts, [dict(task_id=pid, k=k, text=o.text, n_tokens=tk, prompt_tokens=len(pti),
                                             finish_reason=o.finish_reason, correct=bool(g), prompt_token_ids=pti,
                                             model=a.model, sampling=sampling)
                                        for k, (o, g, tk) in enumerate(zip(outs, ok, toks))])
            fh.write(json.dumps(rec) + "\n"); fh.flush(); n += 1
            if n % 50 == 0:
                print(f"[{time.time() - t_start:6.0f}s] done {n}", flush=True)

    while eng.has_unfinished_requests():
        for ro in eng.step():
            if ro.finished:
                d = data[ro.request_id]
                fut = pool.submit(lambda outs, d: [run_tests(code_of(o.text), d["inputs"], d["outputs"]) for o in outs], ro.outputs, d)
                pend[fut] = (ro.request_id, ro.outputs, list(ro.prompt_token_ids or []))
        drain()
        if SLICE and time.time() - t_start > SLICE:
            drain(block=True)
            print(f"slice end after {n} problems", flush=True)
            os._exit(0)
    drain(block=True)


if __name__ == "__main__":
    main()
