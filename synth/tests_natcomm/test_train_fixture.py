"""Synthetic harvest fixture for the NatComm training-side tests (no GPU, no network).

make_harvest(dir) writes runs-style shard files in the format synth/math_harvest.py writes in harvest mode: one record
per problem with acc / tokens / mean_tokens / cut / pass2 / level, the NatComm per-sample fields (lens, oks, fins,
ans_tok) and `trajectories` = the verified samples ({rounds, tokens, turns=[user, assistant raw completion]}).
Built in, to exercise the selection code paths:
  - problems with 0..4 verified samples (K = 4), so S = R = L on single-solution problems;
  - exact token ties between verified samples (sort stability decides S and L);
  - duplicate solutions, verbatim and differing only in whitespace (keep-all must drop both kinds);
  - one problem whose records appear in both shards (sft_student.py pools trajectories across records);
  - a few multi-turn (ARC-agent-like) trajectories in make_multiturn().
FakeTok mimics the two Hugging Face tokenizer calls used by distill/sft_student.py.
"""
import json, os, random

K = 4
WORDS = ("so", "the", "sum", "is", "then", "we", "check", "x", "=", "2", "+", "3", "therefore", "wait", "let", "me",
         "verify", "again", "case", "hence", "\\frac{1}{2}", "integer", "divides", "prime")


def _text(rng, n_words, answer):
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return f"{body}\n</think>\n\nThe answer is \\boxed{{{answer}}}."


def make_harvest(d, n_problems=80, seed=7, n_shards=2):
    """Write d/shard_<i>.jsonl; return {task_id: [verified trajectories]} pooled as sft_student.py pools them."""
    rng = random.Random(seed)
    os.makedirs(d, exist_ok=True)
    recs = [[] for _ in range(n_shards)]
    for p in range(n_problems):
        tid = f"tr_algebra_{p:03d}"
        n_ok = rng.choice((0, 1, 1, 2, 2, 3, 3, 4, 4, 4))
        lens, texts = [], []
        for k in range(K):
            if k < n_ok:
                n_words = rng.randint(40, 1500)
                texts.append(_text(rng, n_words, p))
            else:
                texts.append(_text(rng, rng.randint(200, 2000), p + 1000))   # wrong answer
            lens.append(len(texts[-1]) // 4 + 1)
        if n_ok >= 2 and p % 5 == 0:          # exact token tie between two verified samples (different texts)
            lens[1] = lens[0]
        if n_ok >= 3 and p % 7 == 0:          # verbatim duplicate
            texts[2] = texts[0]; lens[2] = lens[0]
        if n_ok >= 4 and p % 4 == 1:          # whitespace-only duplicate
            texts[3] = texts[1].replace(" ", "  ", 3) + "\n"; lens[3] = lens[1] + 1
        oks = [1] * n_ok + [0] * (K - n_ok)
        order = list(range(K)); rng.shuffle(order)                          # verified samples are not always first
        lens = [lens[i] for i in order]; texts = [texts[i] for i in order]; oks = [oks[i] for i in order]
        prompt = [{"role": "user", "content": f"Problem {p}: compute something.\n\nPlease reason step by step, "
                                              f"and put your final answer within \\boxed{{}}."}]
        trajs = [dict(rounds=1, tokens=tk, turns=prompt + [{"role": "assistant", "content": tx}])
                 for tx, tk, g in zip(texts, lens, oks) if g]
        fins = [1] * K
        rec = dict(task_id=tid, acc=sum(oks) / K, tokens=sum(lens), mean_tokens=sum(lens) / K, cut=K - sum(fins),
                   pass2=bool(oks[0]), level="Level 4", lens=lens, oks=oks, fins=fins, ans_tok=[None] * K,
                   trajectories=trajs)
        recs[p % n_shards].append(rec)
        if p == 3 and trajs:                   # the same problem harvested again in the other shard (pooled)
            extra = dict(rounds=1, tokens=max(1, lens[0] // 2),
                         turns=prompt + [{"role": "assistant", "content": _text(rng, 30, p)}])
            rec2 = dict(rec, trajectories=[extra])
            recs[(p + 1) % n_shards].append(rec2)
    pooled = {}                                # file order: shard_0's records first, then shard_1's

    for s in range(n_shards):
        with open(os.path.join(d, f"shard_{s}.jsonl"), "w") as f:
            for r in recs[s]:
                f.write(json.dumps(r) + "\n")
                for tr in r["trajectories"]:
                    pooled.setdefault(r["task_id"], []).append(tr)
    return pooled


def make_multiturn(d, name="shard_0.jsonl"):
    """Three ARC-agent-like records with system / user / assistant / user / assistant turns."""
    os.makedirs(d, exist_ok=True)
    recs = []
    for i in range(3):
        turns = [{"role": "system", "content": "You are an ARC solver."},
                 {"role": "user", "content": f"puzzle {i} grids ..."},
                 {"role": "assistant", "content": "thinking about it </think> try ```python\nprint(1)\n```"},
                 {"role": "user", "content": "output: wrong cells 3"},
                 {"role": "assistant", "content": "long reasoning " * (5 + i) + "</think> ```python\n# FINAL\n```"}]
        recs.append(dict(task_id=f"arc_{i}", trajectories=[dict(rounds=2, tokens=100 + 10 * i, turns=turns),
                                                           dict(rounds=1, tokens=50 + i, turns=turns[:3])]))
    with open(os.path.join(d, name), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


class FakeTok:
    """Deterministic stand-in for the HF tokenizer API that sft_student.py and the builder call:
    tok(text, add_special_tokens=False)["input_ids"] and tok.apply_chat_template(msgs, tokenize=True, ...)."""

    def __init__(self, dict_ctx=True):
        self.dict_ctx = dict_ctx          # transformers 5 returns a BatchEncoding (dict-like) from apply_chat_template

    def __call__(self, text, add_special_tokens=False):
        ids = [sum(map(ord, text[i:i + 4])) % 997 for i in range(0, len(text), 4)]
        return {"input_ids": ids}

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=True, **kw):
        s = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in msgs) + ("<assistant>" if add_generation_prompt else "")
        ids = self(s)["input_ids"]
        return {"input_ids": ids} if self.dict_ctx else ids
