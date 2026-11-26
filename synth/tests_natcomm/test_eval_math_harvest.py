"""synth/math_harvest.py end to end through the fake vLLM (test_eval_fakes/): record fields old and new, alignment of the
per-sample arrays with the engine's samples, --save-texts, --data, the env fallbacks, resume, max_model_len and the context
check, harvest mode, and synth/paper_stats.py's own functions on the new records. Needs a cached tokenizer."""
import gzip, json, os, re, tempfile, unittest, warnings

import test_eval_util as U
import eval_common as ec

TOKDIR = U.qwen_dir()
K = 3
CAP = 60
LONG = "Let me think again about this carefully. "
DATA = [dict(id="p1", problem="Compute 3*4.", answer="12", level="Level 1"),
        dict(id="p2", problem="Find m+n.", answer=25, level="AIME 2024 I"),           # integer answer, as in synth/aime/
        dict(id="p3", problem="What is half of one?", answer="\\frac{1}{2}", level="Level 2")]
OUTS = {
    "p1": [dict(text="<think>\nWe get x = 12. Check: 3*4 = 12.\n</think>\n\nThe answer is \\boxed{12}.", finish_reason="stop"),
           dict(text="<think>\nMaybe 13.\n</think>\n\n\\boxed{13}", finish_reason="stop"),
           dict(text="<think>\n" + LONG * 40, finish_reason="stop")],                    # longer than CAP: cut
    "p2": [dict(text="<think>\nm+n = 25.\n</think>\n\n\\boxed{025}", finish_reason="stop"),
           dict(text="<think>\nno idea\n</think>\n\nI cannot box it.", finish_reason="stop"),
           dict(text="<think>\nSo it is \\boxed{25} probably, but wait " + "hmm " * 80, finish_reason="stop")],  # cut, boxed early
    "p3": [dict(text="<think>\nhalf is \\frac{1}{2}\n</think>\n\n\\boxed{\\dfrac{1}{2}}", finish_reason="stop")]}


@unittest.skipIf(TOKDIR is None, "no cached tokenizer (set TEST_EVAL_TOKENIZER_DIRS)")
class TestMathHarvest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = U.load_tok(TOKDIR)
        cls.d = tempfile.mkdtemp(prefix="eval_math_")
        cls.model = U.make_model_dir(TOKDIR, max_pos=40960)
        cls.data = os.path.join(cls.d, "test.jsonl"); U.write_jsonl(cls.data, DATA)
        cls.ids = os.path.join(cls.d, "ids.txt")
        with open(cls.ids, "w") as f:
            f.write("\n".join(r["id"] for r in DATA) + "\n")
        cls.log = os.path.join(cls.d, "fake.log")
        cls.spec = os.path.join(cls.d, "spec.json")
        with open(cls.spec, "w") as f:
            json.dump(dict(outputs=OUTS, default=OUTS["p3"], log=cls.log), f)
        cls.out = os.path.join(cls.d, "runs_synth", "agent_t", "shard_0.jsonl")
        os.makedirs(os.path.dirname(cls.out))
        cls.p = cls.harvest(["--max-tokens", str(CAP), "--data", cls.data, "--save-texts", "auto", "--seed", "3"])
        cls.recs = {r["task_id"]: r for r in U.read_jsonl(cls.out)}

    @classmethod
    def harvest(cls, extra, out=None, env=None, model=None):
        e = dict(MATH_EVAL="1", MATH_K=str(K))
        e.update(env or {})
        return U.run_script("math_harvest.py", ["--model", model or cls.model, "--ids", cls.ids, "--out", out or cls.out,
                                                "--tp", "1", "--threads", "8"] + extra,   # unknown solve_agent flags are ignored
                            env=e, fake_spec=cls.spec)

    def expected(self, pid, k):
        """Token ids the engine produced for sample k of pid (fake engine rule = vLLM's: text + end token, cut at CAP)."""
        o = OUTS[pid][k % len(OUTS[pid])]
        ids = self.tok.encode(o["text"], add_special_tokens=False) + [self.tok.eos_token_id]
        return (ids[:CAP], "length") if len(ids) > CAP else (ids, "stop")

    def test_run_ok(self):
        self.assertEqual(self.p.returncode, 0, self.p.stderr[-3000:])
        self.assertEqual(set(self.recs), {"p1", "p2", "p3"})

    def test_old_fields_unchanged(self):
        for pid, r in self.recs.items():
            for key in ("task_id", "acc", "tokens", "mean_tokens", "cut", "pass2", "level"):
                self.assertIn(key, r)
            self.assertNotIn("trajectories", r)                         # eval mode
            self.assertAlmostEqual(r["acc"], sum(r["oks"]) / K)
            self.assertEqual(r["tokens"], sum(r["lens"]))
            self.assertAlmostEqual(r["mean_tokens"], sum(r["lens"]) / K)
            self.assertEqual(r["cut"], K - sum(r["fins"]))
            self.assertEqual(r["pass2"], bool(r["oks"][0]))
        self.assertEqual(self.recs["p2"]["level"], "AIME 2024 I")

    def test_per_sample_arrays(self):
        for pid, r in self.recs.items():
            for key in ("lens", "oks", "fins", "ans_tok"):
                self.assertEqual(len(r[key]), K, (pid, key))
            for k in range(K):
                ids, fr = self.expected(pid, k)
                self.assertEqual(r["lens"][k], len(ids), (pid, k))
                self.assertEqual(r["fins"][k], int(fr == "stop"), (pid, k))
        self.assertEqual(self.recs["p1"]["oks"], [1, 0, 0])
        self.assertEqual(self.recs["p1"]["lens"][2], CAP)
        self.assertEqual(self.recs["p2"]["oks"][:2], [1, 0])              # '025' == 25; no \boxed{} -> wrong
        self.assertEqual(self.recs["p3"]["oks"], [1, 1, 1])

    def test_cut_sample_with_boxed_reasoning_scores_correct(self):
        """Documents the checker's fallback: a sample cut by the cap whose reasoning holds a matching \\boxed{} is scored
        correct (oks 1, fins 0), so A = F * Q is not exact; the per-sample fields make such samples countable."""
        r = self.recs["p2"]
        self.assertEqual((r["oks"][2], r["fins"][2]), (1, 0))

    def test_ans_tok(self):
        ids, _ = self.expected("p1", 0)
        full = self.tok.decode(ids, skip_special_tokens=True)
        pos = full.index("x = 12") + 4
        j = self.recs["p1"]["ans_tok"][0]
        self.assertLessEqual(len(self.tok.decode(ids[:j], skip_special_tokens=True)), pos)
        self.assertGreater(len(self.tok.decode(ids[:j + 1], skip_special_tokens=True)), pos)
        self.assertIsNone(self.recs["p2"]["ans_tok"][1])                  # no \boxed{} answer
        self.assertIsNone(self.recs["p1"]["ans_tok"][2])                  # cut before any answer
        for pid, r in self.recs.items():
            for k, t in enumerate(r["ans_tok"]):
                if t is not None:
                    self.assertLess(t, r["lens"][k])

    def test_texts_file(self):
        path = self.out[:-len(".jsonl")] + ".texts.jsonl.gz"
        with gzip.open(path, "rt") as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(len(rows), 3 * K)
        _, reqs = U.fake_log(self.log)
        sent = {q["request"]: q["prompt_token_ids"] for q in reqs}
        for t in rows:
            for key in ("task_id", "k", "text", "n_tokens", "prompt_tokens", "finish_reason", "correct",
                        "prompt_token_ids", "answer", "model", "sampling"):
                self.assertIn(key, t)
            r = self.recs[t["task_id"]]
            self.assertEqual(t["n_tokens"], r["lens"][t["k"]])
            self.assertEqual(t["correct"], bool(r["oks"][t["k"]]))
            self.assertEqual(t["finish_reason"] == "stop", bool(r["fins"][t["k"]]))
            self.assertEqual(t["prompt_token_ids"], sent[t["task_id"]])
            self.assertEqual(t["prompt_tokens"], len(t["prompt_token_ids"]))
            ids, _ = self.expected(t["task_id"], t["k"])
            self.assertEqual(t["text"], self.tok.decode([x for x in ids if x != self.tok.eos_token_id], skip_special_tokens=True))
            self.assertEqual(t["sampling"], dict(n=K, temperature=0.6, top_p=0.95, top_k=20, max_tokens=CAP, seed=3))
            self.assertEqual(t["model"], self.model)
        self.assertEqual([t["answer"] for t in rows if t["task_id"] == "p2"], [25] * K)   # integer kept as integer
        self.assertEqual([t["k"] for t in rows if t["task_id"] == "p1"], list(range(K)))
        self.assertEqual(len(ec.read_texts(path)), 3 * K)

    def test_engine_settings(self):
        llms, reqs = U.fake_log(self.log)
        kw = llms[0]
        self.assertEqual(kw["max_model_len"], 10240)                      # default unchanged for the 8k-style runs
        self.assertEqual((kw["seed"], kw["enable_prefix_caching"], kw["tensor_parallel_size"]), (3, True, 1))
        self.assertEqual(reqs[0]["sampling"]["max_tokens"], CAP)
        self.assertEqual((reqs[0]["sampling"]["n"], reqs[0]["sampling"]["temperature"]), (K, 0.6))

    def test_resume(self):
        n_llm = len(U.fake_log(self.log)[0])
        p = self.harvest(["--max-tokens", str(CAP), "--data", self.data, "--save-texts", "auto"])
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        self.assertIn("0 to go", p.stdout)
        self.assertEqual(len(U.read_jsonl(self.out)), 3)
        self.assertEqual(len(U.fake_log(self.log)[0]), n_llm)            # no engine built

    def test_cap_32k_sets_max_model_len(self):
        out = os.path.join(self.d, "c32", "shard_0.jsonl"); os.makedirs(os.path.dirname(out))
        log = os.path.join(self.d, "c32.log"); spec = os.path.join(self.d, "c32.json")
        with open(spec, "w") as f:
            json.dump(dict(outputs=OUTS, default=OUTS["p3"], log=log), f)
        p = U.run_script("math_harvest.py", ["--model", self.model, "--ids", self.ids, "--out", out, "--max-tokens", "32768",
                                             "--data", self.data], env=dict(MATH_EVAL="1", MATH_K=str(K)), fake_spec=spec)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        llms, reqs = U.fake_log(log)
        self.assertEqual(llms[0]["max_model_len"], max(len(q["prompt_token_ids"]) for q in reqs) + 32768)
        self.assertLessEqual(llms[0]["max_model_len"], 40960)
        self.assertIn("context: longest prompt", p.stdout)

    def test_context_check_stops_before_engine(self):
        small = U.make_model_dir(TOKDIR, max_pos=32768)
        out = os.path.join(self.d, "small", "shard_0.jsonl"); os.makedirs(os.path.dirname(out))
        log = os.path.join(self.d, "small.log"); spec = os.path.join(self.d, "small.json")
        with open(spec, "w") as f:
            json.dump(dict(outputs=OUTS, default=OUTS["p3"], log=log), f)
        p = U.run_script("math_harvest.py", ["--model", small, "--ids", self.ids, "--out", out, "--max-tokens", "32768",
                                             "--data", self.data], env=dict(MATH_EVAL="1"), fake_spec=spec)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("YaRN", p.stderr)
        self.assertIn("Traceback", p.stderr)                              # chain_dbg.sh stops on a Traceback
        self.assertEqual(U.fake_log(log)[0], [])

    def test_env_fallbacks_shards_and_harvest_mode(self):
        out = os.path.join(self.d, "harv", "shard_1.jsonl"); os.makedirs(os.path.dirname(out))
        tpl = os.path.join(self.d, "harv", "texts.jsonl.gz")
        p = U.run_script("math_harvest.py", ["--model", self.model, "--ids", self.ids, "--out", out, "--max-tokens", str(CAP),
                                             "--shard", "1", "--n-shards", "2"],
                         env=dict(MATH_K=str(K), MATH_DATA=self.data, MATH_SAVE_TEXTS=tpl), fake_spec=self.spec)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        recs = U.read_jsonl(out)
        self.assertEqual([r["task_id"] for r in recs], ["p2"])           # ids[1::2]
        r = recs[0]
        self.assertEqual(len(r["trajectories"]), sum(r["oks"]))          # harvest mode keeps the verified samples
        self.assertEqual(r["trajectories"][0]["turns"][-1]["role"], "assistant")
        texts = os.path.join(self.d, "harv", "texts.shard1.jsonl.gz")
        self.assertTrue(os.path.exists(texts), os.listdir(os.path.dirname(texts)))
        self.assertEqual(len(ec.read_texts(texts)), K)


R1 = U.r1_dir()


@unittest.skipIf(R1 is None, "no DeepSeek-R1-Distill-Llama tokenizer (add one to TEST_EVAL_TOKENIZER_DIRS)")
class TestMathHarvestR1(unittest.TestCase):
    """The second model family: the fixed tokenizer runs (one BOS, records as for Qwen); the original transformers-5
    config, which drops every space, stops the run before the engine is built."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="eval_r1_")
        self.data = os.path.join(self.d, "test.jsonl"); U.write_jsonl(self.data, DATA)
        self.ids = os.path.join(self.d, "ids.txt")
        with open(self.ids, "w") as f:
            f.write("p1\np2\n")
        self.log = os.path.join(self.d, "fake.log"); self.spec = os.path.join(self.d, "spec.json")
        with open(self.spec, "w") as f:
            json.dump(dict(outputs=OUTS, default=OUTS["p3"], log=self.log), f)

    def harvest(self, model):
        out = os.path.join(self.d, "agent", "shard_0.jsonl"); os.makedirs(os.path.dirname(out), exist_ok=True)
        p = U.run_script("math_harvest.py", ["--model", model, "--ids", self.ids, "--out", out, "--max-tokens", str(CAP),
                                             "--data", self.data, "--max-model-len", "10240"],
                         env=dict(MATH_EVAL="1", MATH_K=str(K)), fake_spec=self.spec)
        return p, out

    def test_fixed_tokenizer(self):
        p, out = self.harvest(U.make_model_dir(R1, max_pos=131072, model_type="llama"))
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        recs = {r["task_id"]: r for r in U.read_jsonl(out)}
        self.assertEqual(recs["p1"]["oks"], [1, 0, 0])
        self.assertEqual(recs["p2"]["oks"][:2], [1, 0])
        _, reqs = U.fake_log(self.log)
        tok = U.load_tok(R1)
        for q in reqs:
            self.assertEqual(q["prompt_token_ids"].count(tok.bos_token_id), 1)
            self.assertTrue(tok.decode(q["prompt_token_ids"]).endswith("<｜Assistant｜><think>\n"))

    def test_original_config_is_refused(self):
        broken = U.make_model_dir(R1, max_pos=131072, model_type="llama", tokenizer_class="LlamaTokenizerFast")
        p, out = self.harvest(broken)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("PreTrainedTokenizerFast", p.stderr)
        self.assertEqual(U.fake_log(self.log)[0], [])
        self.assertFalse(os.path.exists(out) and U.read_jsonl(out))


@unittest.skipIf(TOKDIR is None, "no cached tokenizer (set TEST_EVAL_TOKENIZER_DIRS)")
class TestPaperStatsReadsNewRecords(unittest.TestCase):
    """synth/paper_stats.py's single() on records written by the new math_harvest.py (K = 4, as paper_stats assumes),
    with the texts files next to the shard files."""

    def test_single(self):
        d = tempfile.mkdtemp(prefix="eval_ps_")
        model = U.make_model_dir(TOKDIR)
        data = os.path.join(d, "test.jsonl"); U.write_jsonl(data, DATA)
        ids = os.path.join(d, "ids.txt")
        with open(ids, "w") as f:
            f.write("\n".join(r["id"] for r in DATA) + "\n")
        good = [dict(text="<think>\nok\n</think>\n\n\\boxed{%s}" % a, finish_reason="stop") for a in ("12", "25", "\\frac{1}{2}")]
        specs = {"S": dict(outputs={r["id"]: [g] for r, g in zip(DATA, good)}, default=[good[0]]),
                 "base": dict(outputs=OUTS, default=OUTS["p3"])}
        for arm, sp in specs.items():
            path = os.path.join(d, f"spec_{arm}.json")
            with open(path, "w") as f:
                json.dump(sp, f)
            for s in range(2):
                out = os.path.join(d, "runs_synth", f"agent_math_eval_{arm}_s{s}", "shard_0.jsonl")
                os.makedirs(os.path.dirname(out))
                p = U.run_script("math_harvest.py", ["--model", model, "--ids", ids, "--out", out, "--max-tokens", str(CAP),
                                                     "--seed", str(s), "--save-texts", "auto"],
                                 env=dict(MATH_EVAL="1", MATH_K="4", MATH_DATA=data), fake_spec=path)
                self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        with open(os.path.join(U.SYNTH, "paper_stats.py")) as f:
            src = f.read()
        if "\nC = (" not in src:
            self.skipTest("paper_stats.py layout changed")
        ns = {}
        cwd = os.getcwd()
        warnings.simplefilter("ignore", ResourceWarning)                     # paper_stats.load leaves its files open
        try:
            os.chdir(d)
            exec(compile(src.split("\nC = (")[0], "paper_stats.py", "exec"), ns)
            res = ns["single"]("math_eval_{a}_s{s}", ["S", "base"], 1, (("S", "base"),))
        finally:
            os.chdir(cwd)
        self.assertEqual(res["n_problems"], 3)
        recs = [r for s in range(2) for r in U.read_jsonl(os.path.join(d, "runs_synth", f"agent_math_eval_base_s{s}", "shard_0.jsonl"))]
        fin_from_fields = 100 * sum(sum(r["fins"]) / 4 for r in recs) / len(recs)
        self.assertAlmostEqual(res["arms"]["base"]["finish"], round(fin_from_fields, 2))
        self.assertEqual(res["arms"]["S"]["acc"], 100.0)
        self.assertGreater(res["contrasts"]["S-base"]["finish"]["diff"], 0)


if __name__ == "__main__":
    unittest.main()
