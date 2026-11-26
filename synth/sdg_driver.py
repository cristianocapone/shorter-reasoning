"""NVARC-style synthetic data generation (SDG) driven by an open-weight model with vLLM (no thinking, batched, resumable).
Reproduces the file layouts of $WORK/ccapone0/NVARC/SDG/scripts so that NVARC's own CPU scripts finish the job:
  summarize : ARC-2 training task -> <root>/summaries/<split>/<task>.md   (prompts/summary_v2.md; tagged summary)
  mix       : two summaries        -> <root>/summaries/<mix_split>/<a>_<b>.md (prompts/mix_v2.md; the new puzzle, tagged)
  input     : summary              -> <root>/inputs/<split>/<batch>/completions/<name>.md (generate_puzzle_input code)
  output    : summary + input code -> <root>/solutions/<split>/<batch>/<name>/completions/<i>.md (K samples)
then (CPU): generate_input_grids.py, generate_output_grids.py, make_pairs.py (majority vote over the K output programs).

python synth/sdg_driver.py --stage summarize --root sdg --split train_v1 --ids synth/arc2_train_900.txt --arc-dir ARC-AGI-2/data/training
python synth/sdg_driver.py --stage mix --root sdg --split train_v1 --mix-split mix_v1 --n 20000 --seed 0
python synth/sdg_driver.py --stage input --root sdg --split mix_v1
python synth/sdg_driver.py --stage output --root sdg --split mix_v1 --k 8
Sharding: --shard i --n-shards n (items are split round-robin after sorting). Existing files are skipped (resume).
"""
import argparse, glob, json, os, random, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SDG = os.environ.get("SDG_DIR", os.path.expandvars("$WORK/ccapone0/NVARC/SDG"))
sys.path.insert(0, os.path.join(SDG, "scripts"))
from utils import recognize_summary, summary_to_text                 # noqa: E402  (NVARC's parsers)

COLORS = "0 - BLACK, 1 - BLUE, 2 - RED, 3 - GREEN, 4 - YELLOW, 5 - GRAY, 6 - MAGENTA, 7 - ORANGE, 8 - SKY, 9 - BROWN"


def grid_str(g):
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)


def puzzle_text(task):
    parts = []
    for i, p in enumerate(task["train"], 1):
        parts.append(f"Example {i} input ({len(p['input'])}x{len(p['input'][0])}):\n{grid_str(p['input'])}\n"
                     f"Example {i} output ({len(p['output'])}x{len(p['output'][0])}):\n{grid_str(p['output'])}")
    return "\n\n".join(parts)


def barc_examples(seeds_dir, n=3, rng=None):
    """A few BARC seed descriptions as 'solved puzzles with their descriptions' for the summary prompt."""
    files = sorted(glob.glob(os.path.join(seeds_dir, "*.py")))
    (rng or random).shuffle(files); out = []
    for f in files:
        t = open(f, encoding="utf8").read()
        m = re.search(r"# concepts:(.+?)\n# description:(.+?)\ndef ", t, flags=re.S)
        if not m:
            continue
        desc = re.sub(r"\n# ?", " ", m.group(2)).strip()
        out.append(f"<example>\n<description>{desc}</description>\n<concepts>{m.group(1).strip()}</concepts>\n</example>")
        if len(out) == n:
            break
    return "\n".join(out)


def close_fence(text):
    """The 27B often ends a complete answer without the closing ``` -> NVARC's parser (needs ```python ... ```) rejects it."""
    if text.count("```") % 2 == 1:
        text = text.rstrip() + "\n```\n"
    return text


def load_prompt(name):
    return open(os.path.join(SDG, "prompts", name), encoding="utf8").read()


def read_summaries(folder):
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.md"))):
        s = recognize_summary(open(f, encoding="utf8").read().strip())
        if s:
            out[os.path.basename(f)[:-3]] = s
    return out


def shard(items, i, n):
    return [x for k, x in enumerate(sorted(items)) if k % n == i]


CODE_SYSTEM = ("Keep the analysis short (at most 200 words) and then give the complete code in ONE ```python block, "
               "closed with ``` at the end. Never leave the code block unfinished.")
STAGE_MAX_TOKENS = {"summarize": 8000, "mix": 6000, "input": 10000, "output": 10000}
SUMMARY_SYSTEM = ("Keep the puzzle_analysis short (at most 250 words), then output the five tagged sections exactly in the "
                  "required format. Never stop before the closing </puzzle_concepts> tag.")
MIX_SYSTEM = ("Keep your analysis short (at most 300 words), then give the complete <new_puzzle> block with all five tagged "
              "sections. Be concrete and precise in the input_generation and solution_steps.")
STAGE_BATCH = {"summarize": 128, "mix": 128, "input": 64, "output": 12}     # a batch must finish inside a 30-min debug job


def generate(llm, prompts, max_tokens, n=1, temperature=0.8, seed=0, system=None, think=False):
    from vllm import SamplingParams
    sp = SamplingParams(n=n, temperature=temperature, top_p=0.95, top_k=20, max_tokens=max_tokens, seed=seed)
    msgs = [([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": bool(think)})
    return [[(o.text.split("</think>")[-1].strip() if think else o.text) for o in out.outputs] for out in outs]


def plan(a, stage, split, rng):
    """Per-stage work list and prompt/save functions -> (items, make, save)."""
    R = a.root; batch_of = lambda name: name[:2]                       # <batch> = first two hex chars of the name
    if stage == "summarize":
        ids = open(a.ids).read().split() if a.ids else [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(a.arc_dir, "*.json"))]
        out_dir = os.path.join(R, "summaries", split); os.makedirs(out_dir, exist_ok=True)
        todo = [t for t in shard(ids, a.shard, a.n_shards) if not os.path.exists(os.path.join(out_dir, t + ".md"))]
        tpl = load_prompt("summary_v2.md")
        def make(t):
            task = json.load(open(os.path.join(a.arc_dir, t + ".json")))
            return tpl.replace("{PUZZLE}", puzzle_text(task)).replace("{EXAMPLES}", barc_examples(a.barc_seeds, 3, rng))
        def save(t, texts):
            open(os.path.join(out_dir, t + ".md"), "w", encoding="utf8").write(texts[0])
        return todo, make, save
    if stage == "mix":
        S = read_summaries(os.path.join(R, "summaries", split)); names = sorted(S)
        out_dir = os.path.join(R, "summaries", a.mix_split); os.makedirs(out_dir, exist_ok=True)
        rr = random.Random(a.seed); pairs = []
        base = [n for n in names if n in set(open(a.base_ids).read().split())] if a.base_ids else names   # first puzzle = base
        while len(pairs) < a.n and len(names) >= 2 and base:
            x = rr.choice(base); y = rr.choice(names)
            if x != y: pairs.append(f"{x}_{y}")
        pairs = list(dict.fromkeys(pairs))
        todo = [p for p in shard(pairs, a.shard, a.n_shards) if not os.path.exists(os.path.join(out_dir, p + ".md"))
                and not os.path.exists(os.path.join(out_dir, p + ".bad"))]
        tpl = load_prompt("mix_v2.md")
        def make(p):
            x, y = p.split("_")
            return tpl.replace("{PUZZLE1}", summary_to_text(S[x])).replace("{PUZZLE2}", summary_to_text(S[y]))
        def save(p, texts):
            m = re.search(r"<new_puzzle>(.*?)</new_puzzle>", texts[0], flags=re.S)
            body = m.group(1).strip() if m else texts[0]
            if recognize_summary(body):
                open(os.path.join(out_dir, p + ".md"), "w", encoding="utf8").write(body)
            else:                                                      # unparseable mix: mark it, never retry
                open(os.path.join(out_dir, p + ".bad"), "w").close()
        return todo, make, save
    if stage == "input":
        S = read_summaries(os.path.join(R, "summaries", split))
        def path(n): return os.path.join(R, "inputs", split, batch_of(n), "completions", n + ".md")
        todo = [n for n in shard(list(S), a.shard, a.n_shards) if not os.path.exists(path(n))]
        tpl = load_prompt("generate_puzzle_input.md")
        def make(n): return tpl.replace("{PUZZLE}", summary_to_text(S[n]))
        def save(n, texts):
            os.makedirs(os.path.dirname(path(n)), exist_ok=True); open(path(n), "w", encoding="utf8").write(close_fence(texts[0]))
        return todo, make, save
    # output
    S = read_summaries(os.path.join(R, "summaries", split))
    from parser import parse_python_code                                 # NVARC's code extractor
    def in_path(n): return os.path.join(R, "inputs", split, batch_of(n), "completions", n + ".md")
    def out_dir(n): return os.path.join(R, "solutions", split, batch_of(n), n, "completions")
    cand = [n for n in shard(list(S), a.shard, a.n_shards) if os.path.exists(in_path(n)) and not os.path.exists(os.path.join(out_dir(n), "0.md"))]
    codes = {}
    for n in cand:
        c = parse_python_code(open(in_path(n), encoding="utf8").read())
        if c and "def generate_puzzle_input(" in c:
            codes[n] = c
    tpl = load_prompt("generate_puzzle_output.md")
    def make(n): return tpl.replace("{PUZZLE}", summary_to_text(S[n])).replace("{INPUT_CODE}", codes[n])
    def save(n, texts):
        os.makedirs(out_dir(n), exist_ok=True)
        for i, t in enumerate(texts):
            open(os.path.join(out_dir(n), f"{i}.md"), "w", encoding="utf8").write(close_fence(t))
    return sorted(codes), make, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="comma-separated: summarize,mix,input,output (in sequence, one model load)")
    ap.add_argument("--root", default="sdg"); ap.add_argument("--split", required=True); ap.add_argument("--mix-split", default="mix_v1")
    ap.add_argument("--ids", default=None); ap.add_argument("--arc-dir", default="ARC-AGI-2/data/training")
    ap.add_argument("--barc-seeds", default=os.environ.get("BARC_SEEDS", "external_rules/barc_repo/seeds"))
    ap.add_argument("--n", type=int, default=20000, help="mix: number of new puzzles to define"); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--base-ids", default=None, help="mix: file of summary names allowed as the FIRST (base) puzzle of a pair")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--model", default=os.path.expandvars("$WORK/ccapone0/hf/qwen3.8_27b")); ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--max-tokens", type=int, default=10000); ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0, help="debug: process at most this many items per stage")
    ap.add_argument("--deadline", type=float, default=0, help="unix time: start no new batch after it")
    ap.add_argument("--think", action="store_true", help="summarize stage with thinking on (reasoning hidden, clean tagged output)")
    a = ap.parse_args()
    rng = random.Random(a.seed); t0 = time.time()
    split = a.split; llm = tok = None
    for stage in a.stage.split(","):
        items, make, save = plan(a, stage, split, rng)
        if a.limit:
            items = items[: a.limit]
        print(f"stage {stage} split {split}: {len(items)} items to do (shard {a.shard}/{a.n_shards})", flush=True)
        if items:
            if llm is None:
                from vllm import LLM
                llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, gpu_memory_utilization=0.9,
                          enable_prefix_caching=True, seed=a.seed, limit_mm_per_prompt={"image": 0, "video": 0})
                tok = llm.get_tokenizer()
            done = 0
            bs = min(a.batch, STAGE_BATCH.get(stage, a.batch))
            for i in range(0, len(items), bs):
                if a.deadline and time.time() > a.deadline:
                    print("deadline: stopping before a new batch", flush=True); break
                chunk = items[i: i + bs]; prompts = [make(n) for n in chunk]
                mt = min(a.max_tokens, STAGE_MAX_TOKENS.get(stage, a.max_tokens))
                if a.think and stage == "summarize":
                    mt = min(a.max_model_len - 8192, 24000)
                keep = [(n, p) for n, p in zip(chunk, prompts) if len(tok(p)["input_ids"]) + mt + 64 < a.max_model_len]
                if not keep:
                    continue
                n_samp = a.k if stage == "output" else 1
                outs = generate(llm, [p for _, p in keep], mt, n=n_samp, temperature=0.8 if stage == "output" else 0.7, seed=a.seed + i,
                                system=CODE_SYSTEM if stage in ("input", "output") else (MIX_SYSTEM if stage == "mix" else SUMMARY_SYSTEM),
                                think=(a.think and stage == "summarize"))
                for (n, _), texts in zip(keep, outs):
                    save(n, texts); done += 1
                ntok = sum(len(tok(t)["input_ids"]) for texts in outs for t in texts)
                print(f"  {done}/{len(items)} done | {ntok/1e3:.0f}k tokens this batch | {time.time()-t0:.0f}s", flush=True)
            print(f"DONE stage {stage}: {done} items in {time.time()-t0:.0f}s", flush=True)
        if stage == "mix":
            split = a.mix_split                                        # later stages work on the new puzzles


if __name__ == "__main__":
    main()
