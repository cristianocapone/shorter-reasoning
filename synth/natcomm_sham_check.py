"""Zero-step (sham) adapter check for the NatComm revision (must_fix 9): is BASE -> LoRA init -> merge -> save the identity?

The sham arm runs the exact training pipeline (sft_chain-style job of distill/sft_student.py with --lr 0 --max-steps 1,
then leonardo/job_merge.slurm = distill/merge_adapter.py). PEFT initialises lora_B to zero, so the merged weights must
equal the base weights bit for bit; anything that differs comes from the pipeline itself: the re-saved tokenizer (the
R1-Llama-8B tokenizer workaround: transformers 5 once dropped every space), config.json, generation_config.json or the
chat template. CPU only, one tensor in memory at a time (lrd_all_serial is enough).

    python synth/natcomm_sham_check.py BASE_DIR MERGED_DIR [--weights] [--json OUT]

Exit 0 when nothing that can change generation differs; 1 otherwise. Benign metadata differences are listed separately.
"""
import argparse, glob, hashlib, json, os, sys

BENIGN_CONFIG = {"_name_or_path", "transformers_version", "use_cache", "torch_dtype", "dtype"}
BENIGN_TOKCFG = {"name_or_path", "tokenizer_file"}
PROBES = ["What is 1+1? Answer with spaces and\nnew lines.",
          "Let $x = \\frac{3}{4}$; then \\boxed{\\dfrac{3}{4}} and 2^{10}=1024.",
          "<think>\nreasoning here</think>\n\nFinal answer: \\boxed{42}",
          "  leading spaces, tabs\tand unicode: ∑ α ≤ β — 'quotes' \"double\"",
          "def f(x):\n    return x * 2  # code"]
MATH_PROMPT = [{"role": "user", "content": "Compute $1+1$.\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]


def jload(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dict_diff(a, b, benign):
    a, b = a or {}, b or {}
    if "torch_dtype" in a and "dtype" in b and a["torch_dtype"] == b["dtype"]:
        benign = benign | {"torch_dtype", "dtype"}
    if "dtype" in a and "torch_dtype" in b and a["dtype"] == b["torch_dtype"]:
        benign = benign | {"torch_dtype", "dtype"}
    out, meta = [], []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            (meta if k in benign else out).append({"key": k, "base": a.get(k), "merged": b.get(k)})
    return out, meta


def chat_template(d):
    p = os.path.join(d, "chat_template.jinja")
    if os.path.exists(p):
        with open(p) as f:
            return f.read()
    return (jload(os.path.join(d, "tokenizer_config.json")) or {}).get("chat_template")


def tokenizer_check(base, merged):
    from transformers import AutoTokenizer
    tb, tm = AutoTokenizer.from_pretrained(base), AutoTokenizer.from_pretrained(merged)
    res = dict(class_base=type(tb).__name__, class_merged=type(tm).__name__, probes=[], problems=[])
    for s in PROBES:
        ib = tb(s, add_special_tokens=False)["input_ids"]; im = tm(s, add_special_tokens=False)["input_ids"]
        rt = tm.decode(im)
        res["probes"].append(dict(text=s, same_ids=ib == im, n_base=len(ib), n_merged=len(im), roundtrip=rt == s))
        if ib != im:
            res["problems"].append("token ids differ for %r" % s[:40])
        if " " in s and " " not in rt:
            res["problems"].append("decoded text lost every space (the R1-Llama transformers-5 trap) for %r" % s[:40])
    long_text = "\n\n".join(PROBES) * 40                       # a longer mixed text: merges across probe boundaries
    if tb(long_text, add_special_tokens=False)["input_ids"] != tm(long_text, add_special_tokens=False)["input_ids"]:
        res["problems"].append("token ids differ on the long mixed probe")
    if tb.get_vocab() != tm.get_vocab():
        res["problems"].append("vocabularies differ (%d vs %d entries)" % (len(tb.get_vocab()), len(tm.get_vocab())))
    kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pb, pm = tb.apply_chat_template(MATH_PROMPT, **kw), tm.apply_chat_template(MATH_PROMPT, **kw)
    res["chat_prompt_same"] = pb == pm
    if pb != pm:
        res["problems"].append("rendered chat prompt differs")
    eb, em = tb(pb)["input_ids"], tm(pm)["input_ids"]          # default add_special_tokens, as vLLM encodes a text prompt
    res["chat_prompt_ids_same"] = eb == em
    if eb != em:
        res["problems"].append("encoded chat prompt differs with default special tokens (BOS handling: add_bos_token?)")
    for name in ("eos_token", "bos_token", "pad_token"):
        if getattr(tb, name, None) != getattr(tm, name, None):
            res["problems"].append("%s differs: %r vs %r" % (name, getattr(tb, name, None), getattr(tm, name, None)))
    return res


def tensor_files(d):
    idx = jload(os.path.join(d, "model.safetensors.index.json"))
    if idx:
        return {k: os.path.join(d, v) for k, v in idx["weight_map"].items()}
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    from safetensors import safe_open
    out = {}
    for f in files:
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                out[k] = f
    return out


def weights_check(base, merged):
    import torch
    from safetensors import safe_open
    fb, fm = tensor_files(base), tensor_files(merged)
    res = dict(n_base=len(fb), n_merged=len(fm), only_base=sorted(set(fb) - set(fm))[:20],
               only_merged=sorted(set(fm) - set(fb))[:20], exact=0, differ=[], dtype_or_shape=[])
    handles = {}

    def get(path, key):
        if path not in handles:
            handles[path] = safe_open(path, framework="pt")
        return handles[path].get_tensor(key)

    for k in sorted(set(fb) & set(fm)):
        a, b = get(fb[k], k), get(fm[k], k)
        if a.dtype != b.dtype or a.shape != b.shape:
            res["dtype_or_shape"].append(dict(key=k, base="%s %s" % (a.dtype, tuple(a.shape)), merged="%s %s" % (b.dtype, tuple(b.shape))))
        elif torch.equal(a, b):
            res["exact"] += 1
        else:
            res["differ"].append(dict(key=k, max_abs=float((a.float() - b.float()).abs().max())))
        del a, b
    res["identical"] = not (res["only_base"] or res["only_merged"] or res["differ"] or res["dtype_or_shape"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base"); ap.add_argument("merged")
    ap.add_argument("--weights", action="store_true", help="compare every tensor (reads both checkpoints once)")
    ap.add_argument("--no-tokenizer", action="store_true", help="skip loading the tokenizers (files are still compared)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    rep = dict(base=a.base, merged=a.merged, problems=[], metadata=[])
    for name, benign in (("config.json", BENIGN_CONFIG), ("generation_config.json", {"transformers_version"})):
        d, m = dict_diff(jload(os.path.join(a.base, name)), jload(os.path.join(a.merged, name)), benign)
        rep["problems"] += [dict(file=name, **x) for x in d]
        rep["metadata"] += [dict(file=name, **x) for x in m]
    # tokenizer FILES: re-saving under another transformers version re-serialises them (keys move, chat_template goes
    # to chat_template.jinja), so a file difference only matters if the behaviour differs or cannot be checked
    tok_files = []
    for name, benign in (("tokenizer_config.json", BENIGN_TOKCFG), ("special_tokens_map.json", set())):
        d, m = dict_diff(jload(os.path.join(a.base, name)), jload(os.path.join(a.merged, name)), benign)
        tok_files += [dict(file=name, **x) for x in d]
        rep["metadata"] += [dict(file=name, **x) for x in m]
    tb, tm = os.path.join(a.base, "tokenizer.json"), os.path.join(a.merged, "tokenizer.json")
    if os.path.exists(tb) and os.path.exists(tm) and jload(tb) != jload(tm):
        tok_files.append(dict(file="tokenizer.json", key="content", base=sha(tb)[:12], merged=sha(tm)[:12]))
    if chat_template(a.base) != chat_template(a.merged):
        rep["problems"].append(dict(file="chat template", key="text", base="differs", merged="differs"))
    if a.no_tokenizer:
        rep["problems"] += tok_files                          # behaviour not checked: every file difference counts
    else:
        t = tokenizer_check(a.base, a.merged)
        rep["tokenizer"] = t
        rep["problems"] += [dict(file="tokenizer behaviour", key=p) for p in t["problems"]]
        rep["metadata"] += [dict(x, key=x["key"] + " (file re-serialised; behaviour identical)" if not t["problems"] else x["key"])
                            for x in tok_files] if not t["problems"] else []
        if t["problems"]:
            rep["problems"] += tok_files
    if a.weights:
        w = weights_check(a.base, a.merged)
        rep["weights"] = w
        if not w["identical"]:
            rep["problems"].append(dict(file="weights", key="%d tensors differ, %d dtype/shape, %d/%d missing"
                                        % (len(w["differ"]), len(w["dtype_or_shape"]), len(w["only_base"]), len(w["only_merged"]))))
    rep["identical"] = not rep["problems"]
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1, default=str)
    print("SHAM CHECK %s: %s vs %s" % ("PASS" if rep["identical"] else "FAIL", a.base, a.merged))
    if "weights" in rep:
        w = rep["weights"]
        print("  weights: %d tensors, %d bit-identical, %d differ, %d dtype/shape" % (w["n_base"], w["exact"], len(w["differ"]), len(w["dtype_or_shape"])))
    for p in rep["problems"]:
        print("  DIFF %s: %s" % (p["file"], p["key"]))
    for m in rep["metadata"]:
        b, g = repr(m.get("base"))[:60], repr(m.get("merged"))[:60]
        print("  metadata %s: %s %s -> %s" % (m["file"], m["key"], b, g))
    sys.exit(0 if rep["identical"] else 1)


if __name__ == "__main__":
    main()
