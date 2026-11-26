"""Driver-side checks for the NatComm runs (Leonardo login node; stdlib + synth/eval_common.py; Python >= 3.6).

  status  TAG --ids IDS --nsh N [--k K] [--fields]
          shell assignments for leonardo/natcomm_lib.sh: total n_done complete incomplete (colon-separated shards or
          'none') n_records n_bad_fields (records without lens/oks/fins of length K, counted with --fields)
  summary TAG [--k K]
          one line: problems | acc | tokens/sample | cut/sample | finish | Q = acc/finish (records of runs_synth/agent_TAG)
  texts   TAG [--k K]
          every record has its K saved generations (shard_*.texts.jsonl.gz, last copy wins as in eval_common.read_texts)
          with n_tokens == lens[k] and finish_reason consistent with fins[k]; exit 1 otherwise
  forced  TAG [--out runs_synth/agent_TAG/forced.jsonl]
          shell assignments f_rows f_expected f_complete: rows of the current configuration (OUT.meta.json cfg) of
          synth/forced_answer.py, distinct (task_id, k, budget), against n_samples x budgets (it resumes and exits
          cleanly at DIRECT_SLICE_MIN, so 'job COMPLETED' alone does not mean 'all rows written')
  cost    [--file logs/natcomm_cost.tsv] [--tier T]
          CPUh and A100-h per tag and in total, from the rows natcomm_lib.sh appends after every job

Records are read as synth/paper_stats.py reads them: runs_synth/agent_TAG/shard_*.jsonl, first record per task_id.
"""
import argparse, glob, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def records(tag, root="runs_synth"):
    R = {}
    for f in sorted(glob.glob(os.path.join(root, "agent_" + tag, "shard_*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    try:
                        r = json.loads(line)
                    except ValueError:          # a line cut by a killed job: the problem is redone later
                        continue
                    R.setdefault(r["task_id"], r)
    return R


def fields_ok(r, k):
    return all(isinstance(r.get(x), list) and len(r[x]) == k for x in ("lens", "oks", "fins"))


def status(a):
    with open(a.ids) as f:
        ids = [l.strip() for l in f if l.strip()]
    R = records(a.tag, a.root)
    inc = [str(s) for s in range(a.nsh) if any(i not in R for i in ids[s::a.nsh])]
    done = sum(1 for i in ids if i in R)
    bad = sum(1 for i in ids if i in R and not fields_ok(R[i], a.k)) if a.fields else 0
    out = dict(total=len(ids), n_done=done, complete=int(done == len(ids)), incomplete=":".join(inc) or "none",
               n_records=len(R), n_bad_fields=bad)
    print("\n".join("%s=%s" % (k, out[k]) for k in ("total", "n_done", "complete", "incomplete", "n_records", "n_bad_fields")))


def summary(a):
    R = records(a.tag, a.root)
    v = list(R.values())
    if not v:
        print("0 problems"); return
    n = len(v)
    acc = sum(r["acc"] for r in v) / n
    tok = sum(r["mean_tokens"] for r in v) / n
    cut_field = "cap_cut" if all("cap_cut" in r for r in v) else "cut"      # APPS: cap_cut is the length-cap count
    cut = sum(r[cut_field] for r in v) / (a.k * n)
    if all(fields_ok(r, a.k) for r in v):
        fin = sum(sum(r["fins"]) / a.k for r in v) / n
        src = "fins"
    else:
        fin = 1 - cut
        src = cut_field
    q = acc / fin if fin else float("nan")
    print("%d problems | acc %.4f | tokens/sample %d | cut/sample %.3f | finish %.4f (%s) | Q %.4f"
          % (n, acc, tok, cut, fin, src, q))


def texts(a):
    import eval_common as ec
    R = records(a.tag, a.root)
    pat = os.path.join(a.root, "agent_" + a.tag, "shard_*.texts.jsonl.gz")
    try:
        T = {(t["task_id"], t["k"]): t for t in ec.read_texts(pat)}
    except FileNotFoundError:
        print("texts: no file matches %s" % pat); sys.exit(1)
    missing = mismatch = 0
    for tid, r in R.items():
        for k in range(a.k):
            t = T.get((tid, k))
            if t is None:
                missing += 1
                continue
            if fields_ok(r, a.k):
                fin = 1 if t.get("finish_reason") == "stop" else 0
                if t.get("n_tokens") != r["lens"][k] or fin != r["fins"][k]:
                    mismatch += 1
    extra = len({tid for tid, _ in T} - set(R))
    print("texts: %d records, %d generations saved, %d missing, %d inconsistent with the records, "
          "%d problems with texts but no record (redone or unfinished)" % (len(R), len(T), missing, mismatch, extra))
    sys.exit(1 if missing or mismatch else 0)


def forced(a):
    out = a.out or os.path.join(a.root, "agent_" + a.tag, "forced.jsonl")
    meta_path = out + ".meta.json"
    if not os.path.exists(meta_path) or not os.path.exists(out):
        print("f_rows=0\nf_expected=-1\nf_complete=0"); return
    with open(meta_path) as f:
        meta = json.load(f)
    keys = set()
    with open(out) as f:
        for line in f:
            if line.strip():
                try:
                    x = json.loads(line)
                except ValueError:
                    continue
                if x.get("cfg") == meta["cfg"]:
                    keys.add((x["task_id"], x["k"], x["budget"]))
    exp = meta["n_samples"] * len(meta["budgets"])
    print("f_rows=%d\nf_expected=%d\nf_complete=%d" % (len(keys), exp, int(len(keys) >= exp > 0)))


def cost(a):
    rows = []
    if os.path.exists(a.file):
        with open(a.file) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 9 and (not a.tier or p[1] == a.tier):
                    rows.append(p)
    per = {}
    for p in rows:
        tier, tag, kind = p[1], p[2], p[3]
        c = per.setdefault((tier, tag), [0.0, 0.0, set()])
        c[0] += float(p[7]); c[1] += float(p[8]); c[2].add(kind)
    tot_c = sum(c[0] for c in per.values()); tot_g = sum(c[1] for c in per.values())
    for (tier, tag), c in sorted(per.items()):
        print("%-5s %-28s %-18s %9.1f CPUh %8.1f A100-h" % (tier, tag, ",".join(sorted(c[2])), c[0], c[1]))
    print("total %9.1f CPUh = %.1f A100-h over %d jobs" % (tot_c, tot_g, len(rows)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "summary", "texts", "forced", "cost"])
    ap.add_argument("tag", nargs="?")
    ap.add_argument("--ids"); ap.add_argument("--nsh", type=int, default=4); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--fields", action="store_true"); ap.add_argument("--root", default="runs_synth")
    ap.add_argument("--file", default="logs/natcomm_cost.tsv"); ap.add_argument("--tier", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd != "cost" and not a.tag:
        ap.error("TAG required")
    if a.cmd == "status" and not a.ids:
        ap.error("--ids required")
    dict(status=status, summary=summary, texts=texts, forced=forced, cost=cost)[a.cmd](a)


if __name__ == "__main__":
    main()
