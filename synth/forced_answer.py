"""Forced answers at a thinking budget, from generations saved by synth/math_harvest.py --save-texts (NatComm must-fix 6:
evaluate the MATH arms with an early stop at the cap instead of scoring every cut sample as wrong, and report the share of
cut samples whose truncated reasoning already contains the reference answer).

    python synth/forced_answer.py --model M --texts 'runs_synth/agent_TAG/*.texts.jsonl.gz' --budgets 4096,8192,16384 --out F

--model must be the model that wrote the texts. For every saved sample and budget t:
  finished       the sample stopped by itself within t tokens (finish_reason 'stop', n_tokens <= t): it keeps its original
                 correctness and costs no generation;
  forced         it had not stopped by t (n_tokens >= t): the generation is cut to its first t tokens (re-tokenised with the
                 model's tokenizer), the reasoning is closed with --stop-text, and the model writes at most --answer-tokens
                 (1024) new tokens;
  continued      as forced, but the first t tokens already contain the end of the reasoning (--think-end, '</think>'): the
                 model simply continues, without the stop text (as in Qwen's reference implementation);
  not_evaluable  the sample was cut by a cap below t (finish_reason 'length', n_tokens < t): null.
A sample cut exactly at t (an 8k-cap run evaluated at t = 8192) is forced at t, i.e. at its cap. The continuation is scored
with math_harvest.correct, the eval's own checker, on the decoded prefix + stop text + continuation, exactly as the eval
scores a whole generation.

Stop text. Default: Qwen's thinking-budget early-stopping text as implemented in the Qwen documentation (quickstart, section
'Thinking Budget', https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html, checked 2026-11-02):
    "\\n\\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\\n</think>\\n\\n"
The Qwen3 technical report (arXiv:2505.09388v1, sec. 4.3, 'Thinking Budget') prints the instruction as
    "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\\n</think>.\\n\\n"
(no leading blank line and a '.' after </think>); pass that string to --stop-text ('\\n' is unescaped) to use it.
DeepSeek-R1-Distill-Llama-8B has no official phrase; it ends its reasoning with the same '</think>' followed by a blank
line, so the same text closes it. The stop text is tokenised on its own and appended at token level, as Qwen's code does.

Sampling: as in the eval (temperature, top_p, top_k and seed from the texts' 'sampling' field; --temperature and --seed
override) or --greedy. All continuations go to the engine as one batch, ordered by sample and then budget so that the
prefix cache can reuse a sample's shorter budgets; results are written as they finish, so a rerun, or a job ended by
DIRECT_SLICE_MIN like the harvesters, resumes from --out. The tokenizer must round-trip text with spaces (the
R1-Distill-Llama trap under transformers 5: tokenizer_class PreTrainedTokenizerFast); the script stops if it does not.

Output: JSONL, one line per sample x budget, fields task_id, k, budget, status, n_tokens, finish_reason, correct (the
original), and
  correct_t          correctness at budget t with forced answers (null if not evaluable)
  truncated_t        correctness at t without forcing, scored as the eval scores a cut sample (the checker on the first t
                     tokens): what an independent run with cap t would give; (status == 'finished') * correct is the
                     paper's convention that a cut sample is wrong
  ref_in_prefix      the reference answer occurs in the first t tokens (math_harvest.answer_first_char rule); null unless cut
  prefix_checker_ok  the checker already accepts the first t tokens (a \\boxed{} reference answer in the cut reasoning)
  prefix_tokens, retok_tokens (length of the re-tokenised text: n_tokens minus the end token of a finished sample when
  re-tokenisation reproduces the generation; the gap measures any drift), forced_text, forced_tokens,
  forced_finish_reason, cfg (hash of model, stop text, end marker, sampling and answer tokens; a resume never mixes configs).
OUT.meta.json records the configuration.
"""
import argparse, collections, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as ec
from math_harvest import answer_first_char, correct

QWEN_STOP = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="forced answers at thinking budgets (see the module docstring)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--texts", required=True, help="--save-texts file(s): comma-separated paths and/or quoted globs")
    ap.add_argument("--budgets", required=True, help="comma-separated budgets in generated tokens, e.g. 4096,8192,16384")
    ap.add_argument("--out", required=True)
    ap.add_argument("--answer-tokens", type=int, default=1024, help="new tokens allowed after the budget")
    ap.add_argument("--greedy", action="store_true", help="temperature 0 for the forced answer")
    ap.add_argument("--temperature", type=float, default=None, help="default: the eval's (texts' sampling field)")
    ap.add_argument("--seed", type=int, default=None, help="default: the eval's (texts' sampling field)")
    ap.add_argument("--stop-text", default=None, help="early-stop text, '\\n' unescaped; default Qwen's (module docstring)")
    ap.add_argument("--think-end", default="</think>", help="marker that ends the reasoning")
    ap.add_argument("--max-samples", type=int, default=0, help="only the first N samples (smoke tests)")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--tp", type=int, default=1)
    return ap.parse_known_args(argv)[0]


def unescape(s):
    return s.replace("\\n", "\n").replace("\\t", "\t")


def sampling_from(recs, args):
    """Sampling of the forced answers: the eval's (most common 'sampling' field of the texts) unless overridden."""
    seen = collections.Counter(json.dumps(r.get("sampling") or {}, sort_keys=True) for r in recs)
    s = json.loads(seen.most_common(1)[0][0]) if seen else {}
    if len(seen) > 1:
        print(f"warning: the texts mix {len(seen)} sampling settings; using the most common: {s}", flush=True)
    if not s and args.temperature is None and not args.greedy:
        print("warning: texts carry no sampling field; using temperature 0.6, top_p 0.95, top_k 20, seed 0", flush=True)
    if args.greedy:
        return dict(temperature=0.0, seed=0)
    return dict(temperature=args.temperature if args.temperature is not None else s.get("temperature", 0.6),
                top_p=s.get("top_p", 0.95), top_k=s.get("top_k", 20),
                seed=args.seed if args.seed is not None else s.get("seed", 0))


def check_tokenizer(tok, think_end):
    ec.check_roundtrip(tok)
    n = len(tok.encode(think_end, add_special_tokens=False))
    if n != 1:
        print(f"warning: the reasoning end marker {think_end!r} is {n} tokens for this tokenizer", flush=True)


def plan(recs, tok, budgets, stop_text, think_end):
    """Rows that need no generation, and engine requests for the rest, in (sample, budget) order."""
    stop_ids = tok.encode(stop_text, add_special_tokens=False)
    rows, reqs = [], []
    for r in recs:
        n, fr, ok0 = r["n_tokens"], r["finish_reason"], bool(r["correct"])
        gen = None
        for t in budgets:
            base = dict(task_id=r["task_id"], k=r["k"], budget=t, n_tokens=n, finish_reason=fr, correct=ok0)
            if fr == "stop" and n <= t:
                rows.append(dict(base, status="finished", correct_t=int(ok0), truncated_t=int(ok0), ref_in_prefix=None,
                                 prefix_checker_ok=None))
            elif n < t:
                rows.append(dict(base, status="not_evaluable", correct_t=None, truncated_t=None, ref_in_prefix=None,
                                 prefix_checker_ok=None))
            else:
                if gen is None:
                    gen = tok.encode(r["text"], add_special_tokens=False)
                pre = gen[:t]
                pre_text = tok.decode(pre, skip_special_tokens=True)
                closed = think_end in pre_text
                extra = [] if closed else stop_ids
                cut_ok = correct(pre_text, r["answer"])
                reqs.append(dict(key=f"{r['task_id']}#{r['k']}#{t}", prompt_token_ids=list(r["prompt_token_ids"]) + pre + extra,
                                 head=pre + extra, answer=r["answer"],
                                 row=dict(base, status="continued" if closed else "forced", truncated_t=int(cut_ok),
                                          ref_in_prefix=answer_first_char(pre_text, r["answer"]) is not None,
                                          prefix_checker_ok=cut_ok, prefix_tokens=len(pre), retok_tokens=len(gen))))
    return rows, reqs


def score(tok, q, o):
    """The finished row of request q given the engine's completion o."""
    text = tok.decode(q["head"] + list(o.token_ids), skip_special_tokens=True)
    ok = correct(text, q["answer"])
    return dict(q["row"], correct_t=int(ok), forced_text=o.text, forced_tokens=len(o.token_ids),
                forced_finish_reason=o.finish_reason)


def main(argv=None):
    args = parse_args(argv)
    budgets = sorted({int(x) for x in args.budgets.split(",") if x.strip()})
    if not budgets or budgets[0] <= 0:
        raise ValueError(f"--budgets {args.budgets!r}: give positive token counts, e.g. 4096,8192,16384")
    stop_text = unescape(args.stop_text) if args.stop_text is not None else QWEN_STOP
    recs = sorted(ec.read_texts(args.texts), key=lambda r: (str(r["task_id"]), r["k"]))
    if args.max_samples:
        recs = recs[:args.max_samples]
    missing = [k for k in ("prompt_token_ids", "answer", "text", "n_tokens", "finish_reason", "correct") if recs and k not in recs[0]]
    if missing:
        raise ValueError(f"texts lack {missing}: forced_answer.py needs MATH texts written by math_harvest.py --save-texts")
    models = {r.get("model") for r in recs} - {None}
    if models and args.model not in models:
        print(f"warning: --model {args.model} differs from the model that wrote the texts: {sorted(models)}", flush=True)
    sp_kw = sampling_from(recs, args)
    cfg_obj = dict(model=args.model, stop_text=stop_text, think_end=args.think_end, answer_tokens=args.answer_tokens, **sp_kw)
    cfg = hashlib.sha1(json.dumps(cfg_obj, sort_keys=True).encode()).hexdigest()[:8]
    done, other = set(), 0
    if os.path.exists(args.out):
        for l in open(args.out):
            if l.strip():
                x = json.loads(l)
                if x.get("cfg") == cfg:
                    done.add((x["task_id"], x["k"], x["budget"]))
                else:
                    other += 1
    if other:
        raise ValueError(f"{args.out} holds {other} rows of another configuration; write this one to a new --out")
    with open(args.out + ".meta.json", "w") as f:
        json.dump(dict(cfg=cfg, budgets=budgets, texts=ec.expand_paths(args.texts), n_samples=len(recs), **cfg_obj), f, indent=1)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    check_tokenizer(tok, args.think_end)
    rows, reqs = plan(recs, tok, budgets, stop_text, args.think_end)
    rows = [x for x in rows if (x["task_id"], x["k"], x["budget"]) not in done]
    reqs = [q for q in reqs if (q["row"]["task_id"], q["row"]["k"], q["row"]["budget"]) not in done]
    print(f"{len(recs)} samples x {len(budgets)} budgets {budgets}: {len(done)} rows done, {len(rows)} without generation, "
          f"{len(reqs)} to generate | stop text {stop_text!r} | sampling {sp_kw} | cfg {cfg}", flush=True)
    t_start = time.time()
    slice_s = float(os.environ.get("DIRECT_SLICE_MIN", 0)) * 60
    with open(args.out, "a") as fh:
        for x in rows:
            fh.write(json.dumps(dict(x, cfg=cfg)) + "\n")
        fh.flush()
        if reqs:
            mml = ec.resolve_max_model_len(args.model, [len(q["prompt_token_ids"]) for q in reqs], args.answer_tokens,
                                           args.max_model_len)
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import RequestOutputKind
            llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=mml, seed=sp_kw["seed"],
                      gpu_memory_utilization=float(os.environ.get("AGENT_GMU", 0.90)), enable_prefix_caching=True,
                      **({"dtype": os.environ["VLLM_DTYPE"]} if os.environ.get("VLLM_DTYPE") else {}))
            etok = llm.get_tokenizer()
            probe = recs[0]["text"][:4000]
            if etok.encode(probe, add_special_tokens=False) != tok.encode(probe, add_special_tokens=False):
                raise ValueError("the engine's tokenizer encodes the saved text differently from transformers' AutoTokenizer")
            sp = SamplingParams(n=1, max_tokens=args.answer_tokens, output_kind=RequestOutputKind.FINAL_ONLY, **sp_kw)
            eng = llm.llm_engine
            byk = {}
            for q in reqs:                                   # one batch: every continuation is queued before the first step
                eng.add_request(q["key"], {"prompt_token_ids": q.pop("prompt_token_ids")}, sp)
                byk[q["key"]] = q
            print("engine up, all requests queued", flush=True)
            n = 0
            while eng.has_unfinished_requests():
                for ro in eng.step():
                    if ro.finished:
                        fh.write(json.dumps(dict(score(tok, byk.pop(ro.request_id), ro.outputs[0]), cfg=cfg)) + "\n")
                        fh.flush(); n += 1
                        if n % 200 == 0:
                            print(f"[{time.time() - t_start:6.0f}s] forced {n}/{len(reqs)}", flush=True)
                if slice_s and time.time() - t_start > slice_s:
                    print(f"slice end after {n} forced answers", flush=True)
                    os._exit(0)
    summary(args.out, cfg, budgets)


def summary(out, cfg, budgets):
    rows = [x for x in (json.loads(l) for l in open(out) if l.strip()) if x.get("cfg") == cfg]
    for t in budgets:
        R = [x for x in rows if x["budget"] == t and x["correct_t"] is not None]
        F = [x for x in R if x["status"] in ("forced", "continued")]
        if R:
            print(f"budget {t}: {len(R)} samples, {len(F)} forced | acc forced {sum(x['correct_t'] for x in R) / len(R):.4f} "
                  f"truncated {sum(x['truncated_t'] for x in R) / len(R):.4f} | reference in cut prefix "
                  f"{sum(bool(x['ref_in_prefix']) for x in F)}/{len(F)}", flush=True)


if __name__ == "__main__":
    main()
