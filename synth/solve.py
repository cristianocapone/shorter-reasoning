"""Sample-execute-verify solver on top of the proposer.

Per task: build the demo context, sample K programs, execute each on every demo in a sandboxed worker
pool, keep the programs that reproduce ALL demos exactly, run those on the test inputs, and rank the
distinct test outputs by how many consistent programs produced them (ties: mean program log-likelihood).
Attempts = top-2 distinct outputs.

Diagnostics per task: sampled, parsed (compiles + defines main), consistent (reproduces all demos),
oracle (some consistent program is right on test), pass@1, pass@2. `oracle` vs `pass@2` separates the two
failure modes the project cares about: *no correct rule proposed* vs *correct rule proposed but not chosen*.

    python synth/solve.py --ckpt run/latest.pt --barc-heldout verified.jsonl --n 200 --k 64 --out res.jsonl
    python synth/solve.py --ckpt run/latest.pt --arc-dir ARC-AGI-2/data/training --ids ids.txt --k 64 --out res.jsonl
"""
import argparse, json, os, sys, time, signal
import multiprocessing as mp
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from synth.proposer import Proposer, Vocab, collate, PAD, EOS  # noqa: E402
from synth import barc_exec as X  # noqa: E402


def _unsafe_run_program(args):
    """Execute one candidate on demos and tests. -> (consistent: bool, test_outputs or None)."""
    src, demos, tests = args
    ns = {"__name__": "cand"}
    try:
        signal.setitimer(signal.ITIMER_REAL, X.TIMEOUT)
        exec(compile(src, "<cand>", "exec"), ns)
        main = ns.get("main")
        if not callable(main):
            return False, None
        for a, b in demos:
            got = X._as_grid(main(np.array(a, dtype=np.int64)))
            if got is None or got.shape != np.shape(b) or not np.array_equal(got, np.array(b)):
                return False, None
        outs = []
        for a in tests:
            got = X._as_grid(main(np.array(a, dtype=np.int64)))
            outs.append(got.tolist() if got is not None else None)
        return True, outs
    except BaseException:
        return False, None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


_FAILED_RUN_PROGRAM = (False, None)


def _safe_run_program(*args, **kw):
    """Guard the SIGALRM race: the timeout can fire while `finally` disarms the timer."""
    try:
        return _unsafe_run_program(*args, **kw)
    except BaseException:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException:
            pass
        return _FAILED_RUN_PROGRAM

run_program = _safe_run_program


def parses(src):
    try:
        compile(src, "<cand>", "exec")
        return "def main" in src
    except BaseException:
        return False


@torch.no_grad()
def propose(model, vocab, demos, k, batch, max_len, temperature, top_p, dev):
    din, dout, dmask, _ = collate([(demos, "")], vocab, 8)
    ctx = model.context(din.to(dev), dout.to(dev), dmask.to(dev))
    progs, lls = [], []
    for i in range(0, k, batch):
        n = min(batch, k - i)
        ids = model.sample(ctx, n, max_len, temperature, top_p, device=dev)
        logits = model(ctx.expand(n, -1, -1), ids)
        lp = F.log_softmax(logits.float(), -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        mask = ids.ne(PAD)
        for j in range(n):
            row = ids[j].tolist()
            cut = row.index(EOS) if EOS in row else len(row)
            progs.append(vocab.decode(row[1:cut]))
            lls.append(float((lp[j] * mask[j]).sum() / mask[j].sum().clamp(min=1)))
    return progs, lls


def rank_outputs(results, lls, n_tests):
    """-> per test: list of distinct outputs ranked by (votes, mean ll)."""
    ranked = []
    for t in range(n_tests):
        pool = {}
        for (ok, outs), ll in zip(results, lls):
            if not ok or outs is None or outs[t] is None:
                continue
            key = json.dumps(outs[t])
            e = pool.setdefault(key, {"grid": outs[t], "votes": 0, "ll": 0.0})
            e["votes"] += 1; e["ll"] += ll
        ranked.append(sorted(pool.values(), key=lambda e: (e["votes"], e["ll"] / e["votes"]), reverse=True))
    return ranked


def tasks_barc(path, n, seed=0, shard=0, n_shards=1):
    rows = [json.loads(l) for l in open(path)]
    rng = np.random.default_rng(seed)
    rows = [rows[i] for i in rng.permutation(len(rows))[:n]][shard::n_shards]
    for r in rows:
        pairs = r["pairs"]
        n_demo = min(4, len(pairs) - 1)                  # last used pair is the test; >= 3 demos when >= 4 pairs
        yield f"barc_{r['pid']}", pairs[:n_demo], [pairs[n_demo]], r.get("source")


def tasks_arc(arc_dir, ids, shard=0, n_shards=1):
    for tid in ids[shard::n_shards]:
        d = json.load(open(os.path.join(arc_dir, tid + ".json")))
        demos = [(p["input"], p["output"]) for p in d["train"]]
        tests = [(q["input"], q.get("output")) for q in d["test"]]
        yield tid, demos, tests, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--barc-heldout"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--arc-dir"); ap.add_argument("--ids")
    ap.add_argument("--k", type=int, default=64); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8); ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--keep-programs", type=int, default=0, help="store up to N demo-consistent program sources per task")
    a = ap.parse_args()
    # Worker pool FIRST and with spawn: never fork a process that already holds a CUDA context.
    pool = mp.get_context("spawn").Pool(a.workers, X._worker_init)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    targs = ck["args"]
    vocab = Vocab.load(os.path.join(os.path.dirname(a.ckpt), "vocab.json"))
    model = Proposer(vocab, targs["d"], targs["n_layer"], targs["n_head"], targs["n_ctx"], max_len=targs["max_len"]).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    max_len = a.max_len or targs["max_len"]
    if a.barc_heldout:
        src = tasks_barc(a.barc_heldout, a.n, shard=a.shard, n_shards=a.n_shards)
    else:
        ids = [l.strip() for l in open(a.ids) if l.strip()]
        src = tasks_arc(a.arc_dir, ids, shard=a.shard, n_shards=a.n_shards)
    tot = dict(tasks=0, oracle=0, p1=0, p2=0, any_consistent=0)
    fh = open(a.out, "w"); t0 = time.time()
    for tid, demos, tests, ref in src:
        ts = time.time()
        progs, lls = propose(model, vocab, demos, a.k, a.batch, max_len, a.temperature, a.top_p, dev)
        t_samp = time.time() - ts
        n_parse = sum(parses(p) for p in progs)
        results = pool.map(run_program, [(p, demos, [t[0] for t in tests]) for p in progs], chunksize=4)
        if tot["tasks"] == 0:
            print(f"  [timing first task] sampling {t_samp:.1f}s, execution {time.time()-ts-t_samp:.1f}s", flush=True)
        n_cons = sum(ok for ok, _ in results)
        ranked = rank_outputs(results, lls, len(tests))
        has_gt = all(t[1] is not None for t in tests)
        rec = dict(task_id=tid, k=a.k, parsed=n_parse, consistent=n_cons, seconds=time.time() - ts,
                   attempts=[[e["grid"] for e in r[:2]] for r in ranked])
        if a.keep_programs:                       # expert iteration: the demo-verified sources for retraining
            rec["programs"] = [p for p, (ok, _) in zip(progs, results) if ok][: a.keep_programs]
        if has_gt:
            p1 = all(r and r[0]["grid"] == t[1] for r, t in zip(ranked, tests))
            p2 = all(any(e["grid"] == t[1] for e in r[:2]) for r, t in zip(ranked, tests))
            orc = all(any(ok and outs and outs[i] == t[1] for ok, outs in results) for i, t in enumerate(tests))
            rec.update(pass1=p1, pass2=p2, oracle=orc)
            tot["p1"] += p1; tot["p2"] += p2; tot["oracle"] += orc
        tot["tasks"] += 1; tot["any_consistent"] += n_cons > 0
        if tot["tasks"] <= 3:
            ok_prog = next((p for p, (ok, _) in zip(progs, results) if ok), progs[0])
            rec["example_program"] = ok_prog[:1500]
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(f"  {tid}: parsed {n_parse}/{a.k} consistent {n_cons}" +
              (f" | oracle {rec['oracle']} p@2 {rec['pass2']}" if has_gt else "") + f" | {rec['seconds']:.0f}s", flush=True)
    n = max(tot["tasks"], 1)
    summary = dict(summary=True, **tot, oracle_rate=tot["oracle"] / n, pass2_rate=tot["p2"] / n,
                   pass1_rate=tot["p1"] / n, any_consistent_rate=tot["any_consistent"] / n,
                   seconds=time.time() - t0, k=a.k, temperature=a.temperature)
    fh.write(json.dumps(summary) + "\n"); fh.close()
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
