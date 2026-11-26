"""Stub of sbatch / squeue / sacct for the driver tests (test_train_drivers.py). Not a test module (no TestCase).

Installed as three executables in a temp dir that wrap `python test_train_slurm_stub.py <command> args...`. State lives in
the JSON file $NC_STUB_STATE: the test puts a plan of job behaviours per job name in state["plans"][name] (consumed one
per sbatch call) and reads back state["submitted"].
A behaviour: {"queued": n   squeue calls that list the job,
              "glitch": n   then calls where squeue lists nothing although sacct still says RUNNING,
              "state": "COMPLETED" | "FAILED" | "TIMEOUT" ...  (sacct state once finished; per array element: "states"),
              "log": text   written to logs/<name>-<id>.out when the job finishes (an adapter "files" map too),
              "eval": {...} simulate synth/math_harvest.py: records (and texts) for the job's shards,
              "forced": {...} simulate synth/forced_answer.py: rows of forced.jsonl + meta}
"""
import gzip, json, os, sys

def state_path():
    return os.environ["NC_STUB_STATE"]          # read at call time: test discovery imports this module without it


def load():
    with open(state_path()) as f:
        return json.load(f)


def save(s):
    with open(state_path() + ".tmp", "w") as f:
        json.dump(s, f)
    os.replace(state_path() + ".tmp", state_path())


def arg(args, name, default=None):
    for i, a in enumerate(args):
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
        if a == name and i + 1 < len(args):
            return args[i + 1]
    return default


def write(path, text, mode="w"):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, mode) as f:
        f.write(text)


def finish(s, jid):
    job = s["jobs"][jid]
    if job["finished"]:
        return
    job["finished"] = True
    b, env = job["behaviour"], job["env"]
    if "log" in b:
        write(f"logs/{job['name']}-{jid}.out", b["log"])
    for path, content in b.get("files", {}).items():
        write(path.replace("$ST", env.get("NC_ST", "")), content)
    if "eval" in b:
        e, tag = b["eval"], env["NC_TAG"]
        with open(env["NC_IDS"]) as f:
            ids = [l.strip() for l in f if l.strip()]
        nsh, k = int(env["NC_NSH"]), int(env["NC_K"])
        shards = job["array"] if job["array"] is not None else [int(x) for x in env["NC_SHARDS"].split(":")]
        for sh in shards:
            mine = ids[sh::nsh][: e.get("per_shard", 10 ** 9)]
            out = f"runs_synth/agent_{tag}/shard_{sh}.jsonl"
            done = set()
            if os.path.exists(out):
                with open(out) as f:
                    done = {json.loads(l)["task_id"] for l in f if l.strip()}
            texts = []
            for tid in mine:
                if tid in done:
                    continue
                lens = [100 + 10 * j for j in range(k)]
                rec = dict(task_id=tid, acc=0.5, tokens=sum(lens), mean_tokens=sum(lens) / k, cut=0, pass2=True)
                if e.get("fields", True):
                    rec.update(lens=lens, oks=[j % 2 for j in range(k)], fins=[1] * k, ans_tok=[None] * k)
                write(out, json.dumps(rec) + "\n", "a")
                texts += [dict(task_id=tid, k=j, text="t", n_tokens=lens[j], finish_reason="stop", correct=bool(j % 2))
                          for j in range(k)]
            if env.get("NC_TEXTS") and texts:
                p = f"runs_synth/agent_{tag}/shard_{sh}.texts.jsonl.gz"
                with gzip.open(p, "at") as f:
                    for t in texts:
                        f.write(json.dumps(t) + "\n")
            write(f"logs/nc_ev_{tag}-{jid}_s{sh}.out", e.get("solver_log", "engine up\nslice end\n"))
    if "forced" in b:
        fo, out = b["forced"], env["NC_OUT"]
        meta = dict(cfg="abc", budgets=fo["budgets"], n_samples=fo["n_samples"])
        write(out + ".meta.json", json.dumps(meta))
        have = set()
        if os.path.exists(out):
            with open(out) as f:
                have = {(x["task_id"], x["k"], x["budget"]) for x in map(json.loads, f) if x}
        rows = [(f"p{i}", 0, bud) for i in range(fo["n_samples"]) for bud in fo["budgets"]]
        rows = [r for r in rows if r not in have][: fo.get("rows_per_job", 10 ** 9)]
        for t, k, bud in rows:
            write(out, json.dumps(dict(task_id=t, k=k, budget=bud, cfg="abc")) + "\n", "a")


def sbatch(args):
    s = load()
    name = arg(args, "--job-name")
    plan = s["plans"].get(name, [])
    if not plan:
        print("sbatch: error: no plan for " + str(name), file=sys.stderr)
        sys.exit(1)
    b = plan.pop(0)
    jid = str(s["next_id"]); s["next_id"] += 1
    arr = arg(args, "--array")
    env = {k: v for k, v in os.environ.items() if k.startswith(("NC_", "TAG", "MODEL", "DATA", "EXTRA", "BASE", "ADAPTER", "OUT"))}
    s["jobs"][jid] = dict(name=name, args=args, behaviour=b, env=env, finished=False,
                          array=[int(x) for x in arr.split(",")] if arr else None,
                          queued=b.get("queued", 1), glitch=b.get("glitch", 0))
    s["submitted"].append(dict(id=jid, name=name, args=args, env=env))
    save(s)
    print(jid)


def squeue(args):
    s = load()
    if "-n" in args:                                   # squeue -h -u USER -n NAME -o %A: adoption check
        name = arg(args, "-n")
        for jid, job in s["jobs"].items():
            if job["name"] == name and not job["finished"] and job.get("adoptable"):
                print(jid)
        return
    jid = arg(args, "-j")
    job = s["jobs"].get(jid)
    if job is None or job["finished"]:
        return
    if job["queued"] > 0:
        job["queued"] -= 1
        print(f"{jid} running")
    elif job["glitch"] > 0:                           # squeue says nothing, the job is still running
        job["glitch"] -= 1
        s["glitches_served"] = s.get("glitches_served", 0) + 1
    else:
        finish(s, jid)
    save(s)


def sacct(args):
    s = load()
    jid = arg(args, "-j")
    job = s["jobs"].get(jid)
    if job is None:
        return
    fmt = arg(args, "-o", "State")
    states = job["behaviour"].get("states") or [job["behaviour"].get("state", "COMPLETED")] * len(job["array"] or [0])
    if not job["finished"]:
        states = ["RUNNING"] * len(states)
    for i, st in enumerate(states):
        if fmt == "State":
            print(st)
        else:
            elem = f"{jid}_{job['array'][i]}" if job["array"] else jid
            print(f"{elem}|{st}|1800|billing=32,cpu=32,gres/gpu=4,mem=480G,node=1")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    dict(sbatch=sbatch, squeue=squeue, sacct=sacct)[cmd](args)
