"""Compute budget of the NatComm runs (leonardo/NATCOMM_RUNS.md), from paper_stats.json and stated assumptions.

    env -u PYTHONPATH .venv_mps/bin/python leonardo/natcomm_cost.py [--stats paper_brevity/data/paper_stats.json] [--md]

Nothing here is a measurement of the new runs. Inputs, in order of reliability:
  1. paper_stats.json: per-arm mean generated tokens per sample (capped) and finishing rates at 8k (5 arms) and 16k
     (S, L, base) for Qwen3-4B on MATH-500; R1-Llama-8B and APPS at 8k.
  2. Documented harvest facts (PAPER_BREVITY_DRAFT.md): Qwen3-4B MATH harvest 2,000 x 4 samples, 59.5% correct,
     1,345 problems with >= 1 verified; APPS 2,000 x 4, 30.7% correct, 765 problems.
  3. Local throughput evidence: LoRA SFT of Qwen3-4B on 4 A100 at 6.3-6.5k tokens/s (HANDOFF_HYBRID.md, 2026-09-28,
     r=256, gradient checkpointing, batch 1; EXPERIMENT_SYNTHESIS.md); vLLM decoding of the 27B agent at ~900 tokens/s on
     2 A100 (EXPERIMENT_SYNTHESIS.md). There is NO local measurement of vLLM decoding for the 4B/8B models, so decode
     throughput comes from a KV-cache-bound roofline (below) in three scenarios; calibrate it on the first 32k run
     (logs/natcomm_cost.tsv) before trusting the totals.
  4. Assumptions, labelled where used: the length distribution below 8k (log-normal, CV 0.5), AIME lengths (no AIME run of
     this model exists), mean target lengths of the training sets (the builder prints the real ones).
Decode roofline per A100 (bf16 weights W = 2P bytes; vLLM gpu_memory_utilization 0.90 of 64 GB, 3 GB reserve):
  KV capacity = (57.6 GB - W - 3 GB) / kv_bytes_per_token; n = min(256, capacity / C) concurrent sequences, C = token-
  weighted mean context = prompt + E[L^2] / (2 E[L]); step time = (W + n C kv) / BW + n 2P / F + t0; tokens/s = n / step.
32k tail: for x >= 8,192 the survival is S(x) = q + a exp(-lambda (x - 8192)), fitted per arm to the 8k and 16k finishing
  rates and to the 8k-16k difference of capped means (q = share of samples that never stop); arms without 16k data use
  the untouched model's lambda and q scaled by their 8k cut rate. 1 A100-hour = 8 CPUh (Leonardo Booster billing).
"""
import argparse, json, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GB = 1e9
MODELS = {  # params, layers, kv heads, head dim, attention width (heads x head dim); public config.json values
    "q4": dict(P=4.02e9, L=36, kvh=8, hd=128, dattn=32 * 128, name="Qwen3-4B"),
    "l8": dict(P=8.03e9, L=32, kvh=8, hd=128, dattn=32 * 128, name="R1-Distill-Llama-8B"),
}
SCEN = {  # effective HBM bandwidth during decode, job efficiency shift
    "low": dict(BW=1.3e12, d_eff=+0.05), "central": dict(BW=1.0e12, d_eff=0.0), "high": dict(BW=0.65e12, d_eff=-0.10)}
F_DEC, F_PF, T0 = 150e12, 125e12, 0.006      # effective FLOP/s in decode and prefill, fixed per-step overhead (s)
CV_FIN = 0.5                                   # assumption: CV of finished lengths below 8k
CPUH_PER_A100H = 8
REMAIN_CPUH, DAYS_LEFT = 6700, 219             # 2026-11-02 -> 2027-06-09
LEAN_ONE_SEED = ("Rtm", "Ltm", "sham", "baseT04")   # CORE-lean: these controls at sampling seed 0 only, AIME without R
TRAIN_TPS = {"q4": 5.5e3, "l8": 3.0e3}         # SFT tokens/s on 4 A100 (4B: measured 6.3-6.5k minus ~15% for the full-
                                               # vocabulary loss head; 8B: scaled by parameter count; assumption)


def kv_bytes(m):
    return 2 * m["L"] * m["kvh"] * m["hd"] * 2


def decode_tps(m, C, BW):
    W = 2 * m["P"]
    cap = (0.9 * 64 * GB - W - 3 * GB) / kv_bytes(m)
    n = min(256.0, cap / C)
    step = (W + n * C * kv_bytes(m)) / BW + n * 2 * m["P"] / F_DEC + T0
    return n / step, n, cap


def lognorm_sf(x, mean, cv):
    s2 = math.log(1 + cv * cv); mu = math.log(mean) - s2 / 2
    return 0.5 * math.erfc((math.log(x) - mu) / math.sqrt(2 * s2))


class Lengths:
    """Capped length distribution of one arm: finished part below 8k (log-normal, mean from paper_stats) + fitted tail."""

    def __init__(self, F8, m8, F16=None, m16=None, tail=None, prompt=150):
        self.S8, self.F8, self.m8, self.prompt = 1 - F8, F8, m8, prompt
        self.mf = (m8 - self.S8 * 8192) / F8
        if tail is None:
            self.q, self.lam = self._fit(1 - F16, m16 - m8)
        else:
            q_ref, lam_ref, S8_ref = tail
            self.q, self.lam = q_ref * self.S8 / S8_ref, lam_ref
        self.a = self.S8 - self.q

    def _fit(self, S16, target):
        best = None
        for i in range(0, 2000):
            q = S16 * i / 2000
            lam = -math.log((S16 - q) / (self.S8 - q)) / 8192
            I = 8192 * q + (self.S8 - S16) / lam
            if best is None or abs(I - target) < best[0]:
                best = (abs(I - target), q, lam)
        return best[1], best[2]

    def sf(self, x):
        if x <= 8192:
            return self.S8 + self.F8 * (lognorm_sf(x, self.mf, CV_FIN) - lognorm_sf(8192, self.mf, CV_FIN)) / (1 - lognorm_sf(8192, self.mf, CV_FIN))
        return self.q + self.a * math.exp(-self.lam * (x - 8192))

    def mean(self, c):
        if c <= 8192:
            return self.m8 if c == 8192 else None
        Y = c - 8192
        return self.m8 + self.q * Y + self.a * (1 - math.exp(-self.lam * Y)) / self.lam

    def second(self, c):
        fin = self.F8 * self.mf ** 2 * (1 + CV_FIN ** 2)
        if c == 8192:
            return fin + self.S8 * 8192 ** 2
        B, Y, l = 8192.0, c - 8192.0, self.lam
        g = lambda y: (B + y) ** 2 + 2 * (B + y) / l + 2 / l ** 2
        tail = self.a * (g(0) - math.exp(-l * Y) * g(Y))
        return fin + tail + self.sf(c) * c ** 2

    def ctx(self, c):
        return self.prompt + self.second(c) / (2 * self.mean(c))


class Fixed:
    """Length distribution given directly by a capped mean and a cut share (AIME, APPS, R1-Llama at 8k)."""

    def __init__(self, cap, mean, cut, prompt=200):
        self.cap, self.m, self.pc, self.prompt = cap, mean, cut, prompt
        self.mf = (mean - cut * cap) / (1 - cut)

    def mean(self, c):
        return self.m

    def sf(self, x):
        if x >= self.cap:
            return self.pc
        return self.pc + (1 - self.pc) * (lognorm_sf(x, self.mf, CV_FIN) - lognorm_sf(self.cap, self.mf, CV_FIN)) / (1 - lognorm_sf(self.cap, self.mf, CV_FIN))

    def ctx(self, c):
        second = (1 - self.pc) * self.mf ** 2 * (1 + CV_FIN ** 2) + self.pc * self.cap ** 2
        return self.prompt + second / (2 * self.m)


def eval_cost(model, dist, cap, n_prob, k, mode, sc):
    m = MODELS[model]
    tokens = n_prob * k * dist.mean(cap)
    C = dist.ctx(cap)
    tps, n, kvcap = decode_tps(m, C, SCEN[sc]["BW"])
    run_h = tokens / tps / 3600                                       # GPU-hours of pure decoding
    if mode == "array":                                               # normal QOS, one shard per GPU, 4 shards
        per_shard = run_h / 4
        eff = per_shard / (per_shard + 5 / 60) * 0.88                 # 5 min engine start, 12% tail with a thin batch
    else:                                                             # debug chain: 24-min slices on 30-min jobs
        waste = n * (C - dist.prompt) / (tps * 24 * 60)               # in-flight work lost at each slice end
        eff = (24 / 27.5) * (1 - waste)
    eff = max(0.3, eff + SCEN[sc]["d_eff"])
    return dict(tokens=tokens, ctx=C, tps=tps, conc=n, a100h=run_h / eff, eff=eff)


def forced_cost(model, dist, n_prob, k, budgets, answer_tokens, sc):
    m = MODELS[model]
    pf = dec = 0.0; n_tot = 0
    for b in budgets:
        nf = n_prob * k * dist.sf(b)
        n_tot += nf
        flops_tok = 2 * m["P"] + 2 * m["L"] * b * m["dattn"]
        pf += nf * (dist.prompt + b) * flops_tok / F_PF
        tps, _, _ = decode_tps(m, b + answer_tokens / 2, SCEN[sc]["BW"])
        dec += nf * answer_tokens / tps
    hours = (pf + dec) / 3600
    eff = hours / (hours + 5 / 60) * (0.85 + SCEN[sc]["d_eff"])
    return dict(n_forced=n_tot, a100h=hours / eff)


def train_cost(model, examples, mean_target, ctx=250, dl=18, sc="central"):
    tokens = examples * (mean_target + ctx)
    minutes = tokens / TRAIN_TPS[model] / 60 * {"low": 0.8, "central": 1.0, "high": 1.4}[sc]
    jobs = max(1, math.ceil(minutes / dl))
    billed_min = minutes + jobs * (5 if model == "q4" else 7)          # load (+ tokenisation) + checkpoint save per job
    return dict(tokens=tokens, jobs=jobs, a100h=4 * billed_min / 60)


def merge_cost(model):
    return (12 if model == "q4" else 18) / 60                         # 1 GPU, ~12-18 min (load, merge, save)


def build_items(st):
    mq, ml, ma = st["math_qwen4b"]["arms"], st["math_llama8b"]["arms"], st["apps_qwen4b"]["arms"]
    m16 = st["math_qwen4b_cap16k"]["arms"]
    D = {}
    for a in ("S", "L", "base"):
        D[a] = Lengths(mq[a]["finish"] / 100, mq[a]["tokens"], m16[a]["finish"] / 100, m16[a]["tokens"])
    ref = (D["base"].q, D["base"].lam, D["base"].S8)
    D["R"] = Lengths(mq["R"]["finish"] / 100, mq["R"]["tokens"], tail=ref)
    # arms that do not exist yet: keep-all / token-matched / Sk2 assumed like R (Ltm like L), sham and low-T like base
    q32 = [("S", D["S"]), ("R", D["R"]), ("L", D["L"]), ("base", D["base"]), ("KA", D["R"]), ("Rtm", D["R"]),
           ("Ltm", D["L"]), ("sham", D["base"]), ("baseT04", D["base"])]
    l8 = {a: Fixed(8192, ml[a]["tokens"], 1 - ml[a]["finish"] / 100) for a in ("S", "R", "L", "base")}
    apps = {a: Fixed(8192, ma[a]["tokens"], 0.60) for a in ("S", "R", "L", "base")}   # assumption: ~60% cut (see notes)
    aime = {sc: Fixed(32768, mean, cut, prompt=250) for sc, (mean, cut) in
            {"low": (14000, 0.15), "central": (18000, 0.25), "high": (24000, 0.40)}.items()}   # assumption, no data
    items = []

    def add(tier, item, n, fn, note=""):
        items.append(dict(tier=tier, item=item, n=n, fn=fn, note=note))

    # ---------------- CORE
    for a, d in q32:
        add("CORE", f"MATH-500 32k eval, Qwen3-4B {a} (K=4)", 2,
            lambda sc, d=d: eval_cost("q4", d, 32768, 500, 4, "array", sc)["a100h"],
            f"tag nc32_q4_{a}_s{{0,1}}" + ("; lean: seed 0 only" if a in LEAN_ONE_SEED else ""))
    for a, d in q32:
        add("CORE", f"forced answers 4k/8k/16k/32k on {a}", 2,
            lambda sc, d=d: forced_cost("q4", d, 500, 4, (4096, 8192, 16384, 32768), 300, sc)["a100h"],
            "lean: 1 run" if a in LEAN_ONE_SEED else "")
    add("CORE", "SFT keep-all Qwen3-4B (+merge)", 1, lambda sc: train_cost("q4", 4760, 3400, sc=sc)["a100h"] + merge_cost("q4"),
        "4,760 verified samples (59.5% of 8,000), assumed mean 3.4k tokens")
    for a, tgt in (("Rtm", 3400), ("Ltm", 4600)):     # S's loss tokens (1,345 x ~2.4k assumed) in R- or L-length examples
        add("CORE", f"SFT {a} (+merge)", 1, lambda sc, tgt=tgt: train_cost("q4", 1345 * 2400 / tgt, tgt, sc=sc)["a100h"] + merge_cost("q4"),
            "loss tokens = S's (~1,345 x 2.4k assumed), fewer examples")
    add("CORE", "SFT sham Qwen3-4B (1 step, lr 0) + merge", 1, lambda sc: 4 * 10 / 60 + merge_cost("q4"))
    for a in ("base", "S", "R", "L"):
        add("CORE", f"AIME 24+25 32k eval, Qwen3-4B {a} (K=8)", 1,
            lambda sc: eval_cost("q4", aime[sc], 32768, 60, 8, "array", sc)["a100h"],
            "assumed mean 18k tokens (14-24k)" + ("; lean: dropped" if a == "R" else ""))
        add("CORE", f"AIME forced answers 4k/8k/16k/32k on {a}", 1,
            lambda sc: forced_cost("q4", aime[sc], 60, 8, (4096, 8192, 16384, 32768), 300, sc)["a100h"],
            "lean: dropped" if a == "R" else "")
    add("CORE", "R1-Llama-8B sham (train+merge) + 8k eval s0-2", 1,
        lambda sc: 4 * 12 / 60 + merge_cost("l8") + 3 * eval_cost("l8", l8["base"], 8192, 500, 4, "dbg", sc)["a100h"])
    # ---------------- EXTENDED
    add("EXT", "R1-Llama-8B keep-all SFT (+merge)", 1, lambda sc: train_cost("l8", 5600, 2800, sc=sc)["a100h"] + merge_cost("l8"),
        "assumed 5,600 verified, mean 2.8k (no local harvest stats)")
    add("EXT", "R1-Llama-8B keep-all 8k eval", 3, lambda sc: eval_cost("l8", l8["R"], 8192, 500, 4, "dbg", sc)["a100h"], "seeds 0-2")
    for a, tgt in (("S", 2000), ("R", 2800), ("L", 3700)):
        add("EXT", f"R1-Llama-8B {a} replicates (SFT+merge)", 2, lambda sc, tgt=tgt: train_cost("l8", 1500, tgt, sc=sc)["a100h"] + merge_cost("l8"),
            "runs 2-3 (training seeds 1, 2), assumed 1,500 problems")
        add("EXT", f"R1-Llama-8B {a} replicate 8k evals", 4, lambda sc, a=a: eval_cost("l8", l8[a], 8192, 500, 4, "dbg", sc)["a100h"],
            "2 runs x seeds 0-1")
    add("EXT", "R1-Llama-8B run-1 base/S/R/L re-eval with texts (s0)", 4,
        lambda sc: eval_cost("l8", l8["R"], 8192, 500, 4, "dbg", sc)["a100h"], "for forced answers at the 8k cap")
    add("EXT", "R1-Llama-8B forced answers 4k/8k", 4 + 3 + 12,
        lambda sc: forced_cost("l8", l8["R"], 500, 4, (4096, 8192), 300, sc)["a100h"])
    for a, tgt in (("S", 3500), ("R", 4500), ("L", 5500)):
        add("EXT", f"APPS {a} replicates (SFT+merge)", 2, lambda sc, tgt=tgt: train_cost("q4", 765, tgt, sc=sc)["a100h"] + merge_cost("q4"),
            "765 problems; assumed target lengths")
        add("EXT", f"APPS {a} replicate 8k evals", 4, lambda sc, a=a: eval_cost("q4", apps[a], 8192, 500, 4, "dbg", sc)["a100h"],
            "2 runs x seeds 0-1")
    add("EXT", "Sk2 dose-response SFT (+merge)", 1, lambda sc: train_cost("q4", 1345, 2800, sc=sc)["a100h"] + merge_cost("q4"))
    add("EXT", "Sk2 8k evals (per-sample records)", 3, lambda sc: eval_cost("q4", D["R"], 8192, 500, 4, "dbg", sc)["a100h"], "seeds 0-2")
    add("OPT", "AIME second seed (8 more samples), S/R/L/base", 4,
        lambda sc: eval_cost("q4", aime[sc], 32768, 60, 8, "array", sc)["a100h"] + forced_cost("q4", aime[sc], 60, 8, (4096, 8192, 16384, 32768), 300, sc)["a100h"])
    add("OPT", "S upweighted (S for L's loss tokens): SFT + 32k evals s0-1 + forced", 1,
        lambda sc: train_cost("q4", 1345 * 4600 / 2400, 2400, sc=sc)["a100h"] + merge_cost("q4")
        + 2 * eval_cost("q4", D["S"], 32768, 500, 4, "array", sc)["a100h"] + 2 * forced_cost("q4", D["S"], 500, 4, (4096, 8192, 16384, 32768), 300, sc)["a100h"])
    return items, D, aime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=os.path.join(ROOT, "paper_brevity", "data", "paper_stats.json"))
    ap.add_argument("--md", action="store_true", help="print the markdown tables used in NATCOMM_RUNS.md")
    a = ap.parse_args()
    with open(a.stats) as f:
        st = json.load(f)
    items, D, aime = build_items(st)
    print("Qwen3-4B MATH-500 length model (tokens per sample, capped):")
    for k in ("S", "R", "L", "base"):
        d = D[k]
        print(f"  {k:5} 8k mean {d.mean(8192):6.0f}  32k mean {d.mean(32768):6.0f}  cut@16k {d.sf(16384):.3f}  cut@32k {d.sf(32768):.4f}  "
              f"never-stop q {d.q:.4f}  tail scale {1 / d.lam:5.0f}  ctx@8k {d.ctx(8192):5.0f}  ctx@32k {d.ctx(32768):5.0f}")
    for k, m in MODELS.items():
        for C in (3000, 7000, 12000):
            tps, n, cap = decode_tps(m, C, SCEN["central"]["BW"])
            print(f"  roofline {m['name']:20} ctx {C:6d}: {tps:6.0f} tok/s/A100, {n:4.0f} seqs (KV cap {cap / 1e3:.0f}k tokens)")
    tiers = {}
    rows = []
    for it in items:
        v = {sc: it["n"] * it["fn"](sc) for sc in ("low", "central", "high")}
        tiers.setdefault(it["tier"], {sc: 0.0 for sc in v})
        for sc in v:
            tiers[it["tier"]][sc] += v[sc]
        rows.append((it, v))
    lean = {sc: 0.0 for sc in ("low", "central", "high")}
    for it, v in rows:
        if it["tier"] != "CORE":
            continue
        f = 1.0
        if "lean: seed 0 only" in it["note"] or "lean: 1 run" in it["note"]:
            f = 0.5
        if "lean: dropped" in it["note"]:
            f = 0.0
        for sc in lean:
            lean[sc] += f * v[sc]
    per_month = REMAIN_CPUH / DAYS_LEFT * 30.44
    if a.md:
        print("\n| tier | item | runs | A100-h (central) | range | CPUh (central) | note |\n|---|---|---|---|---|---|---|")
        for it, v in rows:
            print(f"| {it['tier']} | {it['item']} | {it['n']} | {v['central']:.1f} | {v['low']:.1f}-{v['high']:.1f} | "
                  f"{v['central'] * CPUH_PER_A100H:.0f} | {it['note']} |")
    groups = [("CORE", "Qwen3-4B MATH-500 32k evals (9 arms x 2 seeds)", "MATH-500 32k eval"),
              ("CORE", "forced answers on those texts (4k/8k/16k/32k)", "^forced answers 4k/8k/16k/32k on"),
              ("CORE", "Qwen3-4B SFT: keep-all, Rtm, Ltm, sham (+merges)", "SFT "),
              ("CORE", "AIME 24+25 32k evals (base/S/L/R, K=8) + forced", "AIME"),
              ("CORE", "R1-Llama-8B sham: SFT + merge + 8k evals s0-2", "R1-Llama-8B sham"),
              ("EXT", "R1-Llama-8B keep-all: SFT + 8k evals s0-2", "R1-Llama-8B keep-all"),
              ("EXT", "R1-Llama-8B replicates: 6 SFT + 12 evals", "R1-Llama-8B S rep|R1-Llama-8B R rep|R1-Llama-8B L rep"),
              ("EXT", "R1-Llama-8B run-1 re-evals with texts + forced (19 tags)", "re-eval with texts|R1-Llama-8B forced"),
              ("EXT", "APPS replicates: 6 SFT + 12 evals", "APPS"),
              ("EXT", "Sk2 dose-response: SFT + 8k evals s0-2", "Sk2"),
              ("OPT", "AIME second sampling seed", "AIME second seed"),
              ("OPT", "S upweighted to L's loss tokens (not scripted)", "S upweighted")]
    if a.md:
        import re as _re
        print("\n| tier | group | A100-h central | range | CPUh central | % of 6,700 |\n|---|---|---|---|---|---|")
        for tier, label, pat in groups:
            v = {sc: sum(x[sc] for it, x in rows if it["tier"] == tier and _re.search(pat, it["item"])) for sc in ("low", "central", "high")}
            print(f"| {tier} | {label} | {v['central']:.1f} | {v['low']:.1f}-{v['high']:.1f} | {v['central'] * 8:.0f} | "
                  f"{100 * v['central'] * 8 / REMAIN_CPUH:.1f} |")
    print("\nTier totals (A100-h low / central / high -> CPUh central; share of the 6,700 CPUh left; months of linear pace "
          f"= {per_month:.0f} CPUh/month):")
    cum = 0.0
    for t in ("CORE", "EXT", "OPT"):
        v = tiers[t]; c = v["central"] * CPUH_PER_A100H; cum += c
        print(f"  {t:5} {v['low']:6.1f} / {v['central']:6.1f} / {v['high']:6.1f} A100-h -> {c:5.0f} CPUh "
              f"({v['low'] * 8:.0f}-{v['high'] * 8:.0f}) = {100 * c / REMAIN_CPUH:4.1f}% of 6,700 "
              f"({100 * v['low'] * 8 / REMAIN_CPUH:.0f}-{100 * v['high'] * 8 / REMAIN_CPUH:.0f}%), "
              f"{c / per_month:.2f} months linear; cumulative {cum:5.0f} CPUh = {100 * cum / REMAIN_CPUH:.1f}%")
    c = lean["central"] * CPUH_PER_A100H
    print(f"  CORE-lean (Rtm/Ltm/sham/low-T at seed 0 only, AIME without R): {lean['low']:.1f} / {lean['central']:.1f} / "
          f"{lean['high']:.1f} A100-h -> {c:.0f} CPUh ({lean['low'] * 8:.0f}-{lean['high'] * 8:.0f}) = {100 * c / REMAIN_CPUH:.1f}% of 6,700, "
          f"{c / per_month:.2f} months linear")
    for label, days in (("to 2026-11-30", 28), ("to 2026-12-15", 43), ("to 2026-12-31", 59)):
        print(f"  linear allowance {label}: {REMAIN_CPUH / DAYS_LEFT * days:5.0f} CPUh")


if __name__ == "__main__":
    main()
