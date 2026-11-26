"""NatComm revision (2026-11): new SFT training sets for the brevity study, built from the EXISTING harvests (no GPU).

Writes harvest-format JSONL files ({task_id, trajectories}) that distill/sft_student.py trains on WITHOUT
--shortest-per-task (then every trajectory of every record becomes one training example, in file order before the
seeded shuffle), plus <prefix>_stats.json with the per-arm training-set statistics asked for by the review
(must_fix 6, 9, 14).

Sets, over the training problems with >= 1 verified trajectory (the candidate pool of sft_student.py):
  S, R, L   one trajectory per problem: shortest / random / longest. Reproduces sft_student.py
            --shortest-per-task --select {shortest,random,longest} --seed SEED exactly (stable sort on `tokens`,
            same tie-breaking, same random.Random(SEED).sample calls in sorted task order), so with SEED=0 the files
            hold exactly what math4b_S/R/L (and mathl8_S/R/L) trained on; --verify checks this against
            sft_student.build_samples itself.
  KA        keep-all (rejection-sampling fine-tuning): every DISTINCT verified trajectory of every problem;
            duplicates = same assistant text after whitespace normalisation (" ".join(text.split())).
  Rtm, Ltm  token-matched controls: R and L restricted to the problems of one fixed random permutation
            (random.Random(TM_SEED)), taken in that order until their loss tokens reach S's total (the next problem
            is added only if that brings the total closer). The permutation is shared, so Ltm's problems are a
            prefix of Rtm's. They match S in loss tokens, not in examples or optimiser steps (they have fewer).
  Sk2       selection-strength dose-response: per problem, 2 verified trajectories drawn without replacement
            (all, if fewer), keep the shorter. Same problems as S/R/L; k=1 would be R, k=all is S, so Sk2 differs
            from S only on problems with >= 3 verified trajectories.
Loss tokens of a trajectory = sft_student.py's target tokens, len(tok(content + SFT_END)) summed over assistant turns,
when --tokenizer is given (use it on the cluster); otherwise the harvest's generated-token count + --end-tokens per
assistant turn (an approximation, flagged in the stats).
Loss normalisation in sft_student.py (for the Methods): each example's loss is the mean cross-entropy over its target
tokens, the `accum` examples of a rank are averaged and the gradients are averaged over ranks, so every example has
the same weight regardless of its length; S, R and L therefore get the same examples count and optimiser steps.

    python synth/natcomm_sft_sets.py --harvest 'runs_synth/agent_math_harv_4b/shard_*.jsonl' --prefix q4 \
        --out-dir runs_synth/natcomm_sets --tokenizer $WORK/ccapone0/hf/qwen3_4b_thinking --max-len 12288 --verify
    SFT_NO_SYS=1 SFT_END='<｜end▁of▁sentence｜>' python synth/natcomm_sft_sets.py --harvest '...math_harv_l8...' --prefix l8 ...
    python3 synth/natcomm_sft_sets.py --show runs_synth/natcomm_sets/q4_stats.json     (print the table again)
"""
import argparse, glob, hashlib, json, math, os, random, statistics, sys

DISTILL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "distill")
ARC_SYS = "You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles by writing Python."  # = sft_student.py
TKW = {"enable_thinking": True, "preserve_thinking": False}                                             # = sft_student.py main()
ALL_SETS = ("S", "R", "L", "KA", "Rtm", "Ltm", "Sk2")


# ---------------------------------------------------------------------------------------------------------- loading
def harvest_paths(spec):
    """sft_student.py's --data semantics: comma-separated globs, each sorted; 'glob@K' repeats those files K times."""
    paths = []
    for s in spec.split(","):
        g, _, k = s.rpartition("@") if "@" in s else (s, "", "1")
        paths += sorted(glob.glob(g)) * int(k)
    return paths


def load_candidates(paths):
    """task_id -> verified trajectories in file order (all records of a task pooled, as in sft_student.py)."""
    cand, n_rec = {}, 0
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line); n_rec += 1
                    for tr in r.get("trajectories", []):
                        cand.setdefault(r["task_id"], []).append(tr)
    return cand, n_rec


def tok_key(tr):
    return tr.get("tokens", 1e18)   # sft_student.py's sort key


def assistant_text(tr):
    return "\n".join(m["content"] for m in tr["turns"] if m["role"] == "assistant")


def norm_text(t):
    return " ".join(t.split())


# -------------------------------------------------------------------------------------------------------- selection
def select_one(cand, select, seed):
    """Exactly sft_student.build_samples(shortest=True, per_task=1, select=..., seed=...) without length filters."""
    rng = random.Random(seed)
    out = {}
    for tid, v in sorted(cand.items()):
        v = sorted(v, key=tok_key)
        if select == "longest":
            out[tid] = v[::-1][:1]
        elif select == "random":
            out[tid] = rng.sample(v, min(1, len(v)))
        else:
            out[tid] = v[:1]
    return out


def keep_all(cand):
    out, dups = {}, 0
    for tid, v in sorted(cand.items()):
        seen, keep = set(), []
        for tr in v:
            key = norm_text(assistant_text(tr))
            if key in seen:
                dups += 1
                continue
            seen.add(key); keep.append(tr)
        out[tid] = keep
    return out, dups


def shortest_of_k(cand, k, seed):
    rng = random.Random(seed)
    out = {}
    for tid, v in sorted(cand.items()):
        draw = rng.sample(v, min(k, len(v)))
        out[tid] = sorted(draw, key=tok_key)[:1]
    return out


def token_matched(sel, loss, target, order):
    """Prefix of `order` (problem ids) whose loss tokens are closest to `target` (stops at the first overflow)."""
    chosen, total = [], 0
    for tid in order:
        t = sum(loss(tr) for tr in sel[tid])
        if total + t <= target:
            chosen.append(tid); total += t
            continue
        if abs(total + t - target) < abs(total - target):
            chosen.append(tid); total += t
        break
    return {tid: sel[tid] for tid in sorted(chosen)}, total


# ------------------------------------------------------------------------------------------------------ token counts
class Counter:
    """Target (loss) and context tokens per trajectory, mirroring sft_student.build_samples."""

    def __init__(self, tok=None, sft_end="<|im_end|>\n", end_tokens=2, no_sys=False, max_len=0):
        self.tok, self.sft_end, self.end_tokens, self.no_sys, self.max_len = tok, sft_end, end_tokens, no_sys, max_len
        self.exact = tok is not None
        self._cache = {}

    def _encode(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _ctx(self, hist):
        ids = self.tok.apply_chat_template(hist, tokenize=True, add_generation_prompt=True, **TKW)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        return len(ids)

    def examples(self, tr):
        """[(ctx_tokens or None, target_tokens)] for each assistant turn of the trajectory."""
        key = id(tr)
        if key in self._cache:
            return self._cache[key]
        turns = tr["turns"]
        n_asst = sum(m["role"] == "assistant" for m in turns)
        if not self.exact:
            if "tokens" not in tr:
                raise ValueError("trajectory without a 'tokens' field: pass --tokenizer")
            res = [(None, tr["tokens"] + self.end_tokens * n_asst)]   # all turns lumped: only the sum is used
        else:
            res = []
            hist = [] if (turns and turns[0]["role"] == "system") or self.no_sys else [{"role": "system", "content": ARC_SYS}]
            for m in turns:
                if m["role"] == "assistant":
                    res.append((self._ctx(hist), self._encode(m["content"] + self.sft_end)))
                    vis = m["content"].split("</think>")[-1].strip() if "</think>" in m["content"] else m["content"].strip()
                    hist.append({"role": "assistant", "content": vis})
                else:
                    hist.append(m)
        self._cache[key] = res
        return res

    def loss(self, tr):
        """Loss tokens that training will actually see (examples longer than max_len are skipped by sft_student)."""
        return sum(t for c, t in self.examples(tr) if not self._skipped(c, t))

    def _skipped(self, c, t):
        return self.exact and self.max_len and c + t > self.max_len


# ------------------------------------------------------------------------------------------------------------ stats
def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))] if xs else None


def set_stats(name, sel, cnt, cand, accum_world=8, epochs=1.0):
    ex = [(c, t) for v in sel.values() for tr in v for c, t in cnt.examples(tr)]
    kept = [(c, t) for c, t in ex if not cnt._skipped(c, t)]
    tgt = [t for _, t in kept]
    n_traj = sum(len(v) for v in sel.values())
    # length tilt: per problem, selected target length relative to the mean over that problem's verified trajectories
    tilt = []
    for tid, v in sel.items():
        if v:
            mu = statistics.mean(cnt.loss(tr) for tr in cand[tid])
            if mu > 0:
                tilt.append(statistics.mean(cnt.loss(tr) for tr in v) / mu - 1)
    st = dict(set=name, problems=sum(1 for v in sel.values() if v), trajectories=n_traj,
              examples=len(kept), skipped_over_max_len=len(ex) - len(kept) if cnt.exact else None,
              loss_tokens=sum(tgt), context_tokens=sum(c for c, _ in kept) if cnt.exact else None,
              target_mean=round(statistics.mean(tgt), 1) if tgt else None,
              target_median=quantile(tgt, 0.5), target_p10=quantile(tgt, 0.1), target_p90=quantile(tgt, 0.9),
              target_max=max(tgt) if tgt else None,
              optimizer_steps=math.ceil(len(kept) * epochs / accum_world),
              length_tilt_mean=round(statistics.mean(tilt), 4) if tilt else None)
    return st


def harvest_stats(cand, ka, dups, cnt, S, L):
    nver = [len(v) for v in cand.values()]
    ndis = [len(v) for v in ka.values()]
    hist = lambda xs: {str(k): sum(1 for x in xs if (x == k if k < 5 else x >= 5)) for k in (1, 2, 3, 4, 5)}
    s_ne_l = sum(1 for t in cand if cnt.loss(S[t][0]) != cnt.loss(L[t][0]))
    return dict(problems_with_verified=len(cand), verified_trajectories=sum(nver), duplicates_removed=dups,
                distinct_trajectories=sum(ndis), problems_by_n_verified=hist(nver), problems_by_n_distinct=hist(ndis),
                problems_with_ge2_distinct=sum(1 for x in ndis if x >= 2), problems_where_S_and_L_lengths_differ=s_ne_l,
                problems_with_ge3_verified=sum(1 for x in nver if x >= 3))


# ------------------------------------------------------------------------------------------------------------ output
def write_set(path, sel, name, force=False):
    lines = [json.dumps(dict(task_id=tid, trajectories=v, nc_set=name)) + "\n" for tid, v in sorted(sel.items()) if v]
    data = "".join(lines).encode()
    sha = hashlib.sha256(data).hexdigest()
    if os.path.exists(path):
        with open(path, "rb") as f:
            old = hashlib.sha256(f.read()).hexdigest()
        if old == sha:
            return sha, "unchanged"
        if not force:
            raise SystemExit(f"{path} exists with different content (a training may have read it): use a new --prefix or --force")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return sha, "written"


def verify(paths, files, cnt, stats, seed):
    """Cross-check with distill/sft_student.build_samples (imports torch): S/R/L files == --shortest-per-task selection,
    and every file's examples / loss tokens / skips == what this script reports."""
    sys.path.insert(0, DISTILL)
    import sft_student as ss
    os.environ["SFT_END"] = cnt.sft_end
    if cnt.no_sys:
        os.environ["SFT_NO_SYS"] = "1"
    else:
        os.environ.pop("SFT_NO_SYS", None)
    out = {}
    for name, sel in (("S", "shortest"), ("R", "random"), ("L", "longest")):
        if name in files:
            a = ss.build_samples(paths, cnt.tok, cnt.max_len or 10 ** 9, TKW, shortest=True, select=sel, seed=seed)
            b = ss.build_samples([files[name]], cnt.tok, cnt.max_len or 10 ** 9, TKW)
            if a != b:
                raise SystemExit(f"VERIFY FAILED: {name} file differs from sft_student --shortest-per-task --select {sel} --seed {seed}")
            out[f"{name}_equals_sft_student_selection"] = True
    for name, f in files.items():
        samples, skipped, n_traj = ss.build_samples([f], cnt.tok, cnt.max_len or 10 ** 9, TKW)
        got = dict(examples=len(samples), loss_tokens=sum(len(s[1]) for s in samples), skipped=skipped)
        want = dict(examples=stats[name]["examples"], loss_tokens=stats[name]["loss_tokens"],
                    skipped=stats[name]["skipped_over_max_len"] or 0)
        if got != want:
            raise SystemExit(f"VERIFY FAILED: {name}: sft_student sees {got}, builder reported {want}")
        out[f"{name}_counts_match_sft_student"] = True
    return out


# -------------------------------------------------------------------------------------------------------------- main
def build(args, tok=None):
    paths = harvest_paths(args.harvest)
    if not paths:
        raise SystemExit(f"no harvest files match {args.harvest}")
    cand, n_rec = load_candidates(paths)
    if not cand:
        raise SystemExit("no verified trajectories in the harvest")
    cnt = Counter(tok, args.sft_end, args.end_tokens, args.no_sys, args.max_len)
    S = select_one(cand, "shortest", args.seed)
    R = select_one(cand, "random", args.seed)
    L = select_one(cand, "longest", args.seed)
    KA, dups = keep_all(cand)
    sets = dict(S=S, R=R, L=L, KA=KA)
    loss_S = sum(cnt.loss(tr) for v in S.values() for tr in v)
    order = sorted(cand)
    random.Random(args.tm_seed).shuffle(order)
    tm = {}
    for name, base in (("Rtm", R), ("Ltm", L)):
        sets[name], tm[name] = token_matched(base, cnt.loss, loss_S, order)
    sets["Sk2"] = shortest_of_k(cand, 2, args.k2_seed)
    want = [s for s in args.sets.split(",") if s]
    bad = [s for s in want if s not in ALL_SETS]
    if bad:
        raise SystemExit(f"unknown sets {bad}; choose from {ALL_SETS}")
    stats = {name: set_stats(name, sets[name], cnt, cand) for name in want}
    ref = {k: stats[k]["loss_tokens"] for k in ("S", "R", "L") if k in stats}
    for st in stats.values():
        for k, v in ref.items():
            st[f"loss_tokens_vs_{k}"] = round(st["loss_tokens"] / v, 4) if v else None
    for name in ("Rtm", "Ltm"):
        if name in stats:
            stats[name]["token_target_S"] = loss_S
            stats[name]["token_match_rel_error"] = round(tm[name] / loss_S - 1, 5)
    if "Sk2" in stats:
        same = lambda X: sum(1 for t in cand if X[t][0] is sets["Sk2"][t][0])
        stats["Sk2"]["problems_equal_to_S"] = same(S)
        stats["Sk2"]["problems_equal_to_R"] = same(R)
    out = dict(params=dict(harvest=args.harvest, files=paths, records=n_rec, prefix=args.prefix, seed=args.seed,
                           tm_seed=args.tm_seed, k2_seed=args.k2_seed, tokenizer=args.tokenizer, exact_tokens=cnt.exact,
                           sft_end=args.sft_end, end_tokens_fallback=None if cnt.exact else args.end_tokens,
                           no_sys=args.no_sys, max_len=args.max_len, batch=8,
                           loss_normalisation="per-example mean over target tokens; examples weighted equally"),
               harvest=harvest_stats(cand, KA, dups, cnt, S, L), sets=stats, files={})
    os.makedirs(args.out_dir, exist_ok=True)
    files = {}
    for name in want:
        path = os.path.join(args.out_dir, f"{args.prefix}_{name}.jsonl")
        sha, what = write_set(path, sets[name], name, args.force)
        files[name] = path
        out["files"][name] = dict(path=path, sha256=sha, status=what)
    if args.verify:
        if not cnt.exact:
            raise SystemExit("--verify needs --tokenizer (the check compares exact token counts)")
        out["verify"] = verify(paths, files, cnt, stats, args.seed)
    spath = os.path.join(args.out_dir, f"{args.prefix}_stats.json")
    with open(spath + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(spath + ".tmp", spath)
    return out


def table(out):
    h = out["harvest"]
    lines = [f"harvest: {h['problems_with_verified']} problems with >= 1 verified, {h['verified_trajectories']} verified "
             f"({h['distinct_trajectories']} distinct, {h['duplicates_removed']} duplicates); by n verified "
             f"{h['problems_by_n_verified']}; >= 2 distinct: {h['problems_with_ge2_distinct']}; S != L length: "
             f"{h['problems_where_S_and_L_lengths_differ']}",
             f"{'set':5} {'problems':>8} {'examples':>8} {'steps':>6} {'loss tok':>11} {'vs S':>6} {'mean':>7} "
             f"{'median':>7} {'p90':>6} {'tilt':>7} {'skip':>5}"]
    for name, s in out["sets"].items():
        lines.append(f"{name:5} {s['problems']:8d} {s['examples']:8d} {s['optimizer_steps']:6d} {s['loss_tokens']:11d} "
                     f"{s.get('loss_tokens_vs_S') or 0:6.3f} {s['target_mean'] or 0:7.0f} {s['target_median'] or 0:7d} "
                     f"{s['target_p90'] or 0:6d} {s['length_tilt_mean'] if s['length_tilt_mean'] is not None else float('nan'):7.3f} "
                     f"{'-' if s['skipped_over_max_len'] is None else s['skipped_over_max_len']:>5}")
    if not out["params"]["exact_tokens"]:
        lines.append("NOTE: approximate loss tokens (harvest token counts + end tokens); rerun with --tokenizer for exact counts")
    return "\n".join(lines)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harvest", required=True, help="harvest shard glob(s), sft_student.py --data syntax")
    ap.add_argument("--prefix", required=True, help="output name prefix, e.g. q4 (Qwen3-4B MATH), l8 (R1-Llama-8B MATH)")
    ap.add_argument("--out-dir", default="runs_synth/natcomm_sets")
    ap.add_argument("--sets", default=",".join(ALL_SETS))
    ap.add_argument("--seed", type=int, default=0, help="training seed of the original arms: draws R as sft_student.py does")
    ap.add_argument("--tm-seed", type=int, default=20261102, help="permutation of problems for Rtm/Ltm")
    ap.add_argument("--k2-seed", type=int, default=20261102, help="draws of Sk2")
    ap.add_argument("--tokenizer", default=None, help="model dir: exact loss/context tokens (AutoTokenizer)")
    ap.add_argument("--sft-end", default=os.environ.get("SFT_END", "<|im_end|>\n"), help="= SFT_END of the training job")
    ap.add_argument("--end-tokens", type=int, default=2, help="fallback: tokens of SFT_END when no tokenizer")
    ap.add_argument("--no-sys", action="store_true", default=bool(os.environ.get("SFT_NO_SYS")), help="= SFT_NO_SYS")
    ap.add_argument("--max-len", type=int, default=12288, help="= sft_student --max-len of the arms (counts skips)")
    ap.add_argument("--verify", action="store_true", help="cross-check against distill/sft_student.build_samples")
    ap.add_argument("--force", action="store_true", help="overwrite set files whose content changed")
    return ap.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--show"]:                      # print the table of an existing <prefix>_stats.json (login node)
        for path in argv[1:]:
            with open(path) as f:
                print(path + "\n" + table(json.load(f)))
        return
    a = parse(argv)
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = build(a, tok)
    print(table(out))
    print(f"stats: {os.path.join(a.out_dir, a.prefix + '_stats.json')}")
    for name, f in out["files"].items():
        print(f"  {name:4} {f['path']}  sha256 {f['sha256'][:12]}  {f['status']}")
    if "verify" in out:
        print("verify:", ", ".join(k for k, v in out["verify"].items() if v))


if __name__ == "__main__":
    main()
