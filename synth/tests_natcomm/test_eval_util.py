"""Helpers for the NatComm evaluator tests (test_eval_*.py): real tokenizers found offline, throwaway model directories,
and running the synth/ scripts in a subprocess with the fake vllm / datasets of test_eval_fakes/. No tests here.

Run every evaluator test:
    env -u PYTHONPATH .venv_mps/bin/python -m unittest discover -s synth/tests_natcomm -p 'test_eval_*.py' -v
Tokenizers: $TEST_EVAL_TOKENIZER_DIRS (os.pathsep-separated directories holding tokenizer.json + tokenizer_config.json,
e.g. Qwen/Qwen3-4B and a DeepSeek-R1-Distill-Llama-8B copy whose tokenizer_class is PreTrainedTokenizerFast), then any
Qwen3 / Qwen2.5 snapshot in the Hugging Face cache. Tests that need one are skipped when none is found.
"""
import functools, glob, json, os, shutil, subprocess, sys, tempfile, warnings

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH = os.path.dirname(HERE)
FAKES = os.path.join(HERE, "test_eval_fakes")
if SYNTH not in sys.path:
    sys.path.insert(0, SYNTH)
warnings.filterwarnings("ignore")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                   "generation_config.json")
R1_MARK = "<｜Assistant｜>"


@functools.lru_cache(maxsize=None)
def tokenizer_dirs():
    cands = [d for d in os.environ.get("TEST_EVAL_TOKENIZER_DIRS", "").split(os.pathsep) if d]
    hub = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
    for pat in ("models--Qwen--Qwen3-*", "models--Qwen--Qwen2.5-*"):
        cands += sorted(glob.glob(os.path.join(hub, pat, "snapshots", "*")))
    return [d for d in cands if all(os.path.exists(os.path.join(d, f)) for f in ("tokenizer.json", "tokenizer_config.json"))]


def load_tok(d):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(d)


def is_r1(d):
    with open(os.path.join(d, "tokenizer.json")) as f:
        return R1_MARK in f.read()


@functools.lru_cache(maxsize=None)
def qwen_dir():
    """First Qwen-family tokenizer (Qwen3 preferred), or None."""
    ds = [d for d in tokenizer_dirs() if not is_r1(d)]
    q3 = [d for d in ds if "<think>" in load_tok(d).get_vocab()]
    return (q3 or ds or [None])[0]


@functools.lru_cache(maxsize=None)
def r1_dir():
    """A DeepSeek-R1-Distill-Llama tokenizer directory that round-trips spaces (the cluster's fixed config), or None."""
    for d in tokenizer_dirs():
        if is_r1(d):
            t = load_tok(d)
            s = "a b c"
            if t.decode(t.encode(s, add_special_tokens=False)) == s:
                return d
    return None


def make_model_dir(tok_dir, max_pos=40960, model_type="qwen3", tokenizer_class=None, rope=None):
    """Temporary 'model' directory: the tokenizer files of tok_dir plus a minimal config.json."""
    d = tempfile.mkdtemp(prefix="eval_model_")
    for f in TOKENIZER_FILES:
        if os.path.exists(os.path.join(tok_dir, f)):
            shutil.copy(os.path.join(tok_dir, f), d)
    if tokenizer_class:
        p = os.path.join(d, "tokenizer_config.json")
        with open(p) as f:
            c = json.load(f)
        c["tokenizer_class"] = tokenizer_class
        with open(p, "w") as f:
            json.dump(c, f)
    cfg = dict(model_type=model_type, max_position_embeddings=max_pos)
    if rope:
        cfg["rope_scaling"] = rope
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return d


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def run_script(script, args, env=None, cwd=None, fake_spec=None):
    """Run synth/<script> like the cluster does (python synth/X.py ...), with the fakes first on PYTHONPATH."""
    e = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MATH_DATA", "MATH_EVAL", "MATH_K",
                                                          "MATH_SAVE_TEXTS", "DIRECT_SLICE_MIN")}
    e["PYTHONPATH"] = FAKES
    if fake_spec:
        e["FAKE_VLLM_SPEC"] = fake_spec
    e.update(env or {})
    return subprocess.run([sys.executable, os.path.join(SYNTH, script)] + list(args), cwd=cwd, env=e,
                          capture_output=True, text=True, timeout=900)


def fake_log(path):
    """(LLM constructions, requests) logged by the fake vllm."""
    rows = read_jsonl(path) if os.path.exists(path) else []
    return [r["llm"] for r in rows if "llm" in r], [r for r in rows if "request" in r]
