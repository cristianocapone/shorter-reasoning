"""Summarise solve_llm.py shard outputs.   python synth/summarize_llm.py runs_synth/llm_g0 runs_synth/llm_g1 ..."""
import glob, json, sys


def load(d):
    recs = {}
    for f in sorted(glob.glob(d + "/shard_*.jsonl")):
        for l in open(f):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("task_id"):
                recs[r["task_id"]] = r
    return recs


def main():
    runs = {d: load(d) for d in sys.argv[1:]}
    for d, recs in runs.items():
        rs = list(recs.values())
        solved = sorted(t for t, r in recs.items() if r.get("pass2"))
        wrong = sorted(t for t, r in recs.items() if r.get("consistent", 0) > 0 and not r.get("pass2"))
        by_gen = {}
        for r in rs:
            for g, n in (r.get("consistent_by_gen") or {}).items():
                by_gen[g] = by_gen.get(g, 0) + n
        only_rev = sorted(t for t, r in recs.items() if r.get("consistent", 0) > 0
                          and (r.get("consistent_by_gen") or {}).get("0", 0) == 0)
        fit = [r.get("best_fitness", 0) for r in rs if not r.get("skipped")]
        near = sorted(t for t, r in recs.items() if not r.get("consistent") and r.get("best_fitness", 0) >= 0.95)
        print(f"{d}: tasks {len(rs)} | any consistent {sum(r.get('consistent', 0) > 0 for r in rs)} | "
              f"oracle {sum(bool(r.get('oracle')) for r in rs)} | pass@2 {len(solved)} | "
              f"pass@1 {sum(bool(r.get('pass1')) for r in rs)} | skipped {sum(bool(r.get('skipped')) for r in rs)} | "
              f"mean best fitness {sum(fit) / max(len(fit), 1):.3f} | "
              f"mean s/task {sum(r.get('seconds', 0) for r in rs) / max(len(rs), 1):.0f}")
        print(f"   solved: {solved}")
        print(f"   consistent but wrong on test: {wrong}")
        print(f"   consistent programs by generation: {dict(sorted(by_gen.items(), key=lambda kv: int(kv[0])))}")
        print(f"   consistent only through revision: {only_rev}")
        print(f"   near misses (fitness >= 0.95, none consistent): {near}")
    if len(runs) > 1:
        names = list(runs)
        sets = [set(t for t, r in runs[n].items() if r.get("pass2")) for n in names]
        print(f"union solved: {sorted(set().union(*sets))}")
        common = set.intersection(*[set(runs[n]) for n in names])
        for n, s in zip(names, sets):
            print(f"   {n}: {len(s & common)} solved on the {len(common)} tasks all runs finished")


if __name__ == "__main__":
    main()
