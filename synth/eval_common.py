"""Shared helpers for the brevity-study evaluators (synth/math_harvest.py, synth/code_harvest.py, synth/forced_answer.py).

  context_limit / resolve_max_model_len   max_model_len = max(--max-model-len, longest prompt + max_tokens), checked against the
                                          positions the model's config.json allows (vLLM checks again when the engine starts)
  prompt_lengths / check_roundtrip        prompt sizes before the engine exists; stop on a tokenizer that drops spaces
  texts_path / append_texts / read_texts  the --save-texts gzip JSONL (one gzip member per problem; tolerant reader)
  char_to_token                           character offset in a generation -> index of the generated token that holds it
No GPU and no vLLM import here, so the tests in synth/tests_natcomm/ run on a laptop.
"""
import glob, gzip, json, os, zlib


def context_limit(model):
    """(positions, source) that the model's config.json allows, or (None, reason) when there is no readable local config.

    Mirrors vLLM's rule: max_position_embeddings, except that a YaRN rope_scaling (or transformers-5 rope_parameters) entry
    allows factor x original_max_position_embeddings. Qwen3-4B: 40,960 (its model card: 32,768 output + 8,192 prompt, and
    no YaRN advised when a request fits); Qwen3-4B-Thinking-2507: 262,144; DeepSeek-R1-Distill-Llama-8B: 131,072 (llama3
    rope scaling is already folded into max_position_embeddings)."""
    path = os.path.join(model, "config.json")
    try:
        with open(path) as f:
            c = json.load(f)
    except (OSError, ValueError):
        return None, f"no readable {path}"
    c = c.get("text_config") or c
    mpe = c.get("max_position_embeddings")
    rs = c.get("rope_scaling") or c.get("rope_parameters") or {}
    if isinstance(rs, dict) and (rs.get("rope_type") or rs.get("type")) == "yarn" and rs.get("factor") \
            and rs.get("original_max_position_embeddings"):
        return int(rs["factor"] * rs["original_max_position_embeddings"]), \
            f"YaRN factor {rs['factor']} x {rs['original_max_position_embeddings']} in {path}"
    if mpe:
        return int(mpe), f"max_position_embeddings in {path}"
    return None, f"no max_position_embeddings in {path}"


def resolve_max_model_len(model, prompt_lens, max_tokens, user_max_len=None):
    """The engine's max_model_len for requests of prompt_lens tokens that may generate max_tokens each.

    Returns max(user_max_len, longest prompt + max_tokens): the cap, never the context window, ends a sample, and the
    usual setting (8k cap, --max-model-len 10240, MATH prompts < 2k tokens) is left exactly as it was. Raises ValueError
    before the engine is built when the longest request does not fit in context_limit(model); the remedy is a smaller
    cap or, for Qwen3, the YaRN rope_scaling of its model card ({"rope_type": "yarn", "factor": F,
    "original_max_position_embeddings": 32768}, smallest F that covers the length; static YaRN can hurt short texts)."""
    need = max(prompt_lens) + max_tokens
    mml = max(user_max_len or 0, need)
    lim, src = context_limit(model)
    print(f"context: longest prompt {max(prompt_lens)} + max_tokens {max_tokens} = {need} tokens; max_model_len {mml}; "
          f"model allows {lim if lim is not None else '?'} ({src})", flush=True)
    if lim is not None and need > lim:
        raise ValueError(f"longest prompt + max_tokens = {need} > {lim} positions allowed by {model} ({src}). Lower "
                         f"--max-tokens, or enable YaRN as the Qwen3 model card describes (rope_scaling rope_type 'yarn', "
                         f"original_max_position_embeddings 32768, smallest factor with factor x 32768 >= {need}).")
    if lim is not None and mml > lim:
        print(f"warning: --max-model-len {mml} exceeds the {lim} positions of the model config; vLLM decides, as before",
              flush=True)
    return mml


PROBE = "What is 1 + 1? Answer: 2.\nSo  x = 2."


def check_roundtrip(tok):
    """Stop if the tokenizer does not give back text with spaces: DeepSeek-R1-Distill-Llama under transformers 5 loads a
    tokenizer that silently drops every space unless tokenizer_config.json says PreTrainedTokenizerFast (the fix applied
    on the cluster). Qwen and the fixed R1 tokenizer pass; the tokenizer itself is used exactly as before."""
    back = tok.decode(tok.encode(PROBE, add_special_tokens=False), skip_special_tokens=True)
    if back != PROBE:
        raise ValueError(f"tokenizer round trip changes text ({PROBE!r} -> {back!r}); for DeepSeek-R1-Distill-Llama under "
                         "transformers 5 set tokenizer_class to PreTrainedTokenizerFast in the model's tokenizer_config.json")


def prompt_lengths(model, convs, **chat_kwargs):
    """Token counts of chat prompts with the model's own tokenizer (transformers' AutoTokenizer, the loader vLLM uses),
    before the engine exists, to size max_model_len. encode() keeps its default special tokens, an upper bound."""
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(model)
    check_roundtrip(t)
    return [len(t.encode(t.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **chat_kwargs))) for m in convs]


def texts_path(spec, out, shard, n_shards):
    """Resolve --save-texts. 'auto' = '{out_stem}.texts.jsonl.gz' (next to the shard's record file); placeholders
    {out_stem} (out without .jsonl), {out_dir}, {shard}. With n_shards > 1 and nothing shard-specific in the path,
    '.shard<i>' goes before '.jsonl.gz', so two processes never append to the same gzip file. The path must end in .gz,
    which also keeps it out of the 'shard_*.jsonl' globs of chain_dbg.sh and paper_stats.py."""
    if not spec:
        return None
    if spec == "auto":
        spec = "{out_stem}.texts.jsonl.gz"
    stem = out[:-len(".jsonl")] if out.endswith(".jsonl") else out
    per_shard = "{shard}" in spec or "{out_stem}" in spec
    path = spec.replace("{out_stem}", stem).replace("{out_dir}", os.path.dirname(out) or ".").replace("{shard}", str(shard))
    if not path.endswith(".gz"):
        raise ValueError(f"--save-texts {spec}: the texts file is gzip JSONL and its name must end in .gz")
    if n_shards > 1 and not per_shard:
        base = path[:-len(".jsonl.gz")] if path.endswith(".jsonl.gz") else path[:-len(".gz")]
        path = f"{base}.shard{shard}" + (".jsonl.gz" if path.endswith(".jsonl.gz") else ".gz")
    return path


def append_texts(path, rows):
    """Append rows (dicts) as one gzip member: open in append mode, write, close, so that a killed job can damage at most
    the member it was writing. Called before the problem's record is written: a problem redone after a crash then
    appears twice, and read_texts keeps the last copy, which is the one the record describes."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def expand_paths(spec):
    """Comma-separated list of paths and/or glob patterns (quote globs on the command line) -> sorted existing files."""
    files = []
    for part in (p.strip() for p in spec.split(",")):
        if part:
            hits = sorted(glob.glob(part)) or ([part] if os.path.exists(part) else [])
            if not hits:
                print(f"warning: no file matches {part!r}", flush=True)
            files += hits
    return files


def read_texts(spec):
    """Records of one or more --save-texts files. The last copy of each (task_id, k) wins; a damaged final gzip member or
    a broken line is reported and skipped, everything read before it is kept."""
    files = expand_paths(spec)
    if not files:
        raise FileNotFoundError(f"no texts file matches {spec!r}")
    recs = {}
    for f in files:
        n_ok = n_bad = 0
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        n_bad += 1
                        continue
                    recs[(r["task_id"], r["k"])] = r
                    n_ok += 1
        except (EOFError, OSError, zlib.error) as e:
            print(f"warning: {f}: unreadable gzip data after {n_ok} lines ({e!r}); the rest of this file is skipped",
                  flush=True)
        if n_bad:
            print(f"warning: {f}: {n_bad} unparsable line(s) skipped", flush=True)
    return list(recs.values())


def char_to_token(tok, token_ids, pos):
    """Index of the generated token that contains character pos of tok.decode(token_ids, skip_special_tokens=True):
    the smallest j with len(decode(token_ids[:j + 1])) > pos (binary search on prefix decodes, about 15 decodes)."""
    if pos is None or not token_ids:
        return None
    lo, hi = 0, len(token_ids) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(token_ids[:mid + 1], skip_special_tokens=True)) > pos:
            hi = mid
        else:
            lo = mid + 1
    return lo
