"""Second domain for the brevity study: competition math with an answer verifier (MATH).

Harvest mode (default): k samples per training problem under a per-answer token cap; every sample whose \\boxed{} answer
matches the reference is a verified trajectory. Records use the agent harvest format read by distill/sft_student.py
(task_id, trajectories=[{rounds, tokens, turns=[user, assistant(raw completion incl. reasoning)]}]), so the same
--shortest-per-task --select {shortest,random,longest} arms apply unchanged.
Eval mode (--eval): k samples per test problem; record = per-problem accuracy (mean over samples), tokens, cut rate.

Accepts solve_agent.py's command line (unknown flags ignored) so it runs under leonardo/job_agent_dbg.slurm with
SOLVER=synth/math_harvest.py and leonardo/chain_dbg.sh (one record per finished problem; resumable).
  data: MATH_DATA=<jsonl with id, problem, answer>;  --ids = file of problem ids;  MATH_K samples;  MATH_EVAL=1 eval mode.

NatComm additions (2026-11-02). Defaults are unchanged; old records and synth/paper_stats.py keep working.
  Per-sample fields in every record (eval and harvest), lists in the order of the K samples:
    lens     generated tokens of each sample (len(token_ids); a sample that stopped by itself counts its end token)
    oks      1 if the sample is correct under correct() below, else 0          (acc = mean(oks))
    fins     1 if the sample stopped by itself before the cap (finish_reason 'stop'), else 0 (cut by the cap: 'length')
    ans_tok  index, among the generated tokens (0-based), of the token where the graded answer (the content of the last
             \\boxed{}) first appears in the generation; null when there is no \\boxed{} answer. Rule: answer_first_char().
    Caveat (unchanged checker): when the visible answer has no \\boxed{}, correct() grades the last \\boxed{} of the
    reasoning, so a sample cut by the cap can still have oks 1 (fins 0); A = F x Q holds only if no such sample exists,
    and these fields make them countable.
  --save-texts PATH  also append every generation to a gzip JSONL, one line per sample:
             {task_id, k, text, n_tokens, prompt_tokens (count), finish_reason, correct} plus prompt_token_ids (the prompt as
             the engine tokenised it), answer (the reference), model and sampling, so synth/forced_answer.py needs nothing
             else. PATH 'auto' = next to the shard's records (shard_0.jsonl -> shard_0.texts.jsonl.gz); placeholders
             {shard}, {out_stem}, {out_dir}; with --n-shards > 1 and no shard-specific part, '.shard<i>' is inserted before
             '.jsonl.gz'. Must end in .gz. Env fallback: MATH_SAVE_TEXTS. One gzip member per problem, written before the
             record; a problem redone after a crash appears twice and readers (eval_common.read_texts) keep the last copy.
  --data PATH        problem file (jsonl: id, problem, answer[, level]); overrides MATH_DATA. Integer answers (AIME, built by
             synth/aime_prep.py) are accepted by the checker.
  --max-tokens up to 32768: max_model_len becomes max(--max-model-len, longest prompt + max_tokens), checked against the
             model's config.json before the engine starts (eval_common.resolve_max_model_len). Qwen3-4B allows 40,960
             positions (its model card: 32,768 output + 8,192 prompt, and YaRN is not advised when a request fits), so
             prompt + 32,768 needs no rope scaling: that is the setting used, being the minimal one. If a request ever does
             not fit, the card's YaRN option is rope_scaling {"rope_type": "yarn", "factor": F,
             "original_max_position_embeddings": 32768} with the smallest F that covers it; static YaRN can hurt short
             texts, so it is not enabled here. DeepSeek-R1-Distill-Llama-8B allows 131,072 positions.
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
ap.add_argument("--max-tokens", type=int, default=8192); ap.add_argument("--max-model-len", type=int, default=10240)
ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default=None, help="problem jsonl (id, problem, answer[, level]); default $MATH_DATA")
ap.add_argument("--think-prefill", action="store_true",
                help="end the generation prompt with '<think>\\n' when the chat template does not already (R1-Distill family)")
ap.add_argument("--strip-bos", action="store_true",
                help="drop the chat template's leading BOS when the tokenizer adds one itself (avoids a double BOS in vLLM)")
ap.add_argument("--top-k", type=int, default=20, help="top-k sampling (-1 disables; Qwen3 default 20)")
ap.add_argument("--save-texts", default=os.environ.get("MATH_SAVE_TEXTS"),
                help="gzip JSONL for every generation ('auto', placeholders {shard} {out_stem} {out_dir}); default $MATH_SAVE_TEXTS")
a = ap.parse_known_args()[0] if __name__ == "__main__" else None      # importable (math_prep.py, forced_answer.py) without a CLI
K = int(os.environ.get("MATH_K", 4)); EVAL = os.environ.get("MATH_EVAL") == "1"
SLICE = float(os.environ.get("DIRECT_SLICE_MIN", 0)) * 60
PROMPT_DEFAULT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
PROMPT = os.environ.get("MATH_PROMPT") or PROMPT_DEFAULT   # user-prompt template override (e.g. a model's own training prompt)


def boxed(text):
    """Content of the last \\boxed{...} (balanced braces) in the visible answer, else in the whole text."""
    for part in ([text.split("</think>")[-1]] if "</think>" in text else []) + [text]:
        i = part.rfind("\\boxed")
        if i < 0:
            continue
        j = part.find("{", i)
        if j < 0:
            continue
        d = 0
        for k in range(j, len(part)):
            d += part[k] == "{"; d -= part[k] == "}"
            if d == 0:
                return part[j + 1:k]
    return None


def norm(s):
    if s is None:
        return None
    s = str(s).strip()                       # str(): integer answers (AIME) arrive as JSON numbers; strings are unchanged
    for x in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "$", " ", "\\displaystyle", "^\\circ", "^{\\circ}", "\\%", "%"):
        s = s.replace(x, "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("\\text{", "{").replace("\\mbox{", "{")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"(\\frac\{[^}]*\})(\d)", r"\1{\2}", s)
    s = s.rstrip(".")
    if s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]
    try:
        f = float(s.replace(",", ""))
        return f"{f:.6g}"
    except ValueError:
        return s


def correct(text, ref):
    p = norm(boxed(text)); r = norm(ref)
    return p is not None and p == r


def boxed_spans(text):
    """(start, content) of every \\boxed{...} in text, in order, with the brace matching of boxed()."""
    out = []
    for m in re.finditer(r"\\boxed", text):
        j = text.find("{", m.end())
        if j < 0:
            break
        d = 0
        for k in range(j, len(text)):
            d += text[k] == "{"; d -= text[k] == "}"
            if d == 0:
                out.append((m.start(), text[j + 1:k]))
                break
    return out


_ARG_OPEN = re.compile(r"(\\(?!boxed\b|text\b|textbf\b|mathrm\b|mathbf\b|mbox\b)[A-Za-z]+|[\^_}])\s*$")


def _standalone(text, s, e, cand):
    """text[s:e] is not part of a longer number, word, command argument or exponent."""
    b = text[s - 1] if s > 0 else " "
    f = text[e] if e < len(text) else " "
    if b.isalnum() or b in "\\./^_" or (b == "-" and not cand.startswith("-")):
        return False
    if f.isalnum() or f in "_/" or (f in ".," and text[e + 1:e + 2].isdigit()):
        return False
    if f == "^" and not (text.startswith("^\\circ", e) or text.startswith("^{\\circ}", e)):
        return False
    if b == "{" and _ARG_OPEN.search(text[max(0, s - 40):s - 1]):      # argument of \frac, \sqrt, ^{...}, the 2nd of \frac
        return False
    if f == "}" and text[e + 1:e + 2] == "{":                            # first argument of \frac{..}{..}
        return False
    return True


def answer_first_char(text, ans):
    """Character offset in text where the graded answer ans (the raw content of the last \\boxed{}) first appears, or None
    when ans is None. The earliest of
      (a) a \\boxed{...} whose content normalises (norm) to norm(ans);
      (b) ans itself, or norm(ans), written out (whitespace-insensitive) as a standalone expression: the character before
          it is not a letter, digit, '\\', '.', '/', '^', '_' or (unless ans starts with '-') '-', the character after it is
          not a letter, digit, '_', '/', '^' (except ^\\circ) or a '.'/',' followed by a digit, and it is not inside a
          command argument such as \\frac{..}{..}, \\sqrt{..} or ^{..}.
    (a) guarantees a hit whenever the answer is boxed. (b) is literal, without algebraic equivalence, and a short integer can
    match an intermediate quantity that happens to equal the answer; the saved texts allow other rules to be applied."""
    if ans is None:
        return None
    ans = str(ans); target = norm(ans)
    best = next((s for s, c in boxed_spans(text) if norm(c) == target), None)
    for cand in {ans.strip(), target}:
        chars = [ch for ch in cand if not ch.isspace()]
        if not chars:
            continue
        pat = re.compile("(?=(" + r"\s*".join(re.escape(ch) for ch in chars) + "))")
        for m in pat.finditer(text):
            if best is not None and m.start() >= best:
                break
            if _standalone(text, m.start(), m.end(1), cand):
                best = m.start()
                break
    return best


def per_sample(tok, outs, ok):
    """lens / oks / fins / ans_tok of one problem's samples, in sample order (see the module docstring)."""
    ans_tok = []
    for o in outs:
        try:                                         # a diagnostic must never stop a long evaluation
            ids = list(o.token_ids)
            full = tok.decode(ids, skip_special_tokens=True)
            ans_tok.append(ec.char_to_token(tok, ids, answer_first_char(full, boxed(o.text))))
        except Exception as e:
            print(f"warning: ans_tok failed: {e!r}", flush=True)
            ans_tok.append(None)
    return dict(lens=[len(o.token_ids) for o in outs], oks=[int(bool(g)) for g in ok],
                fins=[int(o.finish_reason == "stop") for o in outs], ans_tok=ans_tok)


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
    prompts = {pid: [{"role": "user", "content": PROMPT.format(problem=data[pid]["problem"])}] for pid in todo}
    mml = ec.resolve_max_model_len(a.model, ec.prompt_lengths(a.model, prompts.values(), enable_thinking=True),
                                   a.max_tokens, a.max_model_len)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    # VLLM_V100=1 (voltron2, sm70): float16 (no bfloat16 before sm80), and neither prefix caching nor chunked prefill,
    # whose Triton prefix-prefill kernel does not compile for sm70
    v100 = os.environ.get("VLLM_V100") == "1"
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=mml, seed=a.seed,
              gpu_memory_utilization=float(os.environ.get("AGENT_GMU", 0.90)), enable_prefix_caching=not v100,
              **(dict(dtype="half", enable_chunked_prefill=False) if v100 else
                 {"dtype": os.environ["VLLM_DTYPE"]} if os.environ.get("VLLM_DTYPE") else {}))
    tok = llm.get_tokenizer(); eng = llm.llm_engine
    print("engine up", flush=True)
    sp = SamplingParams(n=K, temperature=a.temperature, top_p=0.95, top_k=a.top_k, max_tokens=a.max_tokens, seed=a.seed,
                        output_kind=RequestOutputKind.FINAL_ONLY)
    sampling = dict(n=K, temperature=a.temperature, top_p=0.95, top_k=a.top_k, max_tokens=a.max_tokens, seed=a.seed)
    if PROMPT != PROMPT_DEFAULT or a.think_prefill:
        sampling.update(prompt=PROMPT, think_prefill=a.think_prefill, strip_bos=a.strip_bos)
    for pid in todo:
        text = tok.apply_chat_template(prompts[pid], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        if a.think_prefill and not text.endswith("<think>\n"):
            text += "<think>\n"
        if a.strip_bos and tok.bos_token and text.startswith(tok.bos_token) and tok("x").input_ids[:1] == [tok.bos_token_id]:
            text = text[len(tok.bos_token):]
        eng.add_request(pid, {"prompt": text}, sp)
    n = 0
    with open(a.out, "a") as fh:
        while eng.has_unfinished_requests():
            for ro in eng.step():
                if not ro.finished:
                    continue
                pid = ro.request_id; ref = data[pid]["answer"]
                outs = ro.outputs
                ok = [correct(o.text, ref) for o in outs]
                toks = [len(o.token_ids) for o in outs]; cut = [o.finish_reason == "length" for o in outs]
                rec = dict(task_id=pid, acc=sum(ok) / len(ok), tokens=sum(toks), mean_tokens=sum(toks) / len(toks),
                           cut=sum(cut), pass2=bool(ok and ok[0]), level=data[pid].get("level"))
                rec.update(per_sample(tok, outs, ok))
                if not EVAL:
                    rec["trajectories"] = [dict(rounds=1, tokens=tk, turns=prompts[pid] + [{"role": "assistant", "content": o.text}])
                                           for o, g, tk in zip(outs, ok, toks) if g]
                if texts:
                    pti = list(ro.prompt_token_ids or [])
                    ec.append_texts(texts, [dict(task_id=pid, k=k, text=o.text, n_tokens=tk, prompt_tokens=len(pti),
                                                 finish_reason=o.finish_reason, correct=bool(g), prompt_token_ids=pti,
                                                 answer=ref, model=a.model, sampling=sampling)
                                            for k, (o, g, tk) in enumerate(zip(outs, ok, toks))])
                fh.write(json.dumps(rec) + "\n"); fh.flush(); n += 1
                if n % 50 == 0:
                    print(f"[{time.time() - t_start:6.0f}s] done {n}", flush=True)
            if SLICE and time.time() - t_start > SLICE:
                print(f"slice end after {n} problems", flush=True)
                os._exit(0)


if __name__ == "__main__":
    main()
