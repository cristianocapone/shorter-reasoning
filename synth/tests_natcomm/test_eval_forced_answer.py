"""synth/forced_answer.py: the truncate-at-t token arithmetic and the status rules on real tokenizers (Qwen; R1-Distill-
Llama when a copy is available, including the transformers-5 space-dropping trap), scoring with the eval's checker, and an
end-to-end run (math_harvest.py --save-texts -> forced_answer.py) through the fake vLLM with resume and config guard."""
import json, os, tempfile, types, unittest

import test_eval_util as U
import eval_common as ec
import forced_answer as FA

QWEN = U.qwen_dir()
R1 = U.r1_dir()
DOCS_STOP = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
REPORT_STOP = "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n"


class TestStopText(unittest.TestCase):
    def test_default_is_qwen_docs_text(self):
        self.assertEqual(FA.QWEN_STOP, DOCS_STOP)

    def test_report_variant_via_cli_escape(self):
        self.assertEqual(FA.unescape(REPORT_STOP.replace("\n", "\\n")), REPORT_STOP)


def records(tok, prompt_ids):
    """Synthetic saved samples: (name, text, finish_reason, answer)."""
    spec = [("A", "<think>\n" + "step one. " * 12 + "so the answer is 12.\n</think>\n\nThus \\boxed{12}.", "stop", "12"),
            ("B", "<think>\n" + "keep going. " * 30, "length", "7"),
            ("C", "<think>\nshort.\n</think>\n\nThe final answer, after a long explanation: " + "more words " * 20 + "\\boxed{3}",
             "stop", "3"),
            ("D", "<think>\nwe find \\boxed{7} early, but then " + "doubt " * 40, "length", "7")]
    out = []
    for name, text, fr, ans in spec:
        ids = tok.encode(text, add_special_tokens=False)
        n = len(ids) + (1 if fr == "stop" else 0)                     # vLLM counts the end token of a finished sample
        out.append(dict(task_id=name, k=0, text=text, n_tokens=n, finish_reason=fr, correct=FA.correct(text, ans),
                        prompt_token_ids=list(prompt_ids), prompt_tokens=len(prompt_ids), answer=ans))
    return out


class PlanMixin:
    TOKDIR = None

    @classmethod
    def setUpClass(cls):
        cls.tok = U.load_tok(cls.TOKDIR)
        msgs = [{"role": "user", "content": "Problem?\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]
        cls.prompt = cls.tok.encode(cls.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True))
        cls.recs = records(cls.tok, cls.prompt)
        cls.stop_ids = cls.tok.encode(FA.QWEN_STOP, add_special_tokens=False)
        cls.gen = {r["task_id"]: cls.tok.encode(r["text"], add_special_tokens=False) for r in cls.recs}
        lens = sorted({r["n_tokens"] for r in cls.recs})
        cls.budgets = sorted({8, 16, 24, 40} | set(lens) | {x - 1 for x in lens} | {max(lens) + 50})
        cls.rows, cls.reqs = FA.plan(cls.recs, cls.tok, cls.budgets, FA.QWEN_STOP, "</think>")

    def by(self):
        out = {}
        for x in self.rows:
            out[x["task_id"], x["budget"]] = (x, None)
        for q in self.reqs:
            out[q["row"]["task_id"], q["row"]["budget"]] = (q["row"], q)
        return out

    def test_every_sample_budget_once(self):
        self.assertEqual(len(self.rows) + len(self.reqs), len(self.recs) * len(self.budgets))
        self.assertEqual(len(self.by()), len(self.recs) * len(self.budgets))

    def test_status_rules(self):
        for r in self.recs:
            for t in self.budgets:
                row, q = self.by()[r["task_id"], t]
                n, fr = r["n_tokens"], r["finish_reason"]
                if fr == "stop" and n <= t:
                    self.assertEqual(row["status"], "finished")
                    self.assertEqual(row["correct_t"], int(r["correct"]))
                elif n < t:
                    self.assertEqual(row["status"], "not_evaluable")
                    self.assertIsNone(row["correct_t"])
                else:
                    self.assertIn(row["status"], ("forced", "continued"))
                    self.assertIsNotNone(q)
        self.assertEqual(self.by()["B", self.recs[1]["n_tokens"]][0]["status"], "forced")   # cut exactly at t: forced at its cap

    def test_token_arithmetic(self):
        for q in self.reqs:
            row = q["row"]; t = row["budget"]; g = self.gen[row["task_id"]]
            pre = g[:t]
            self.assertEqual(row["prefix_tokens"], min(t, len(g)))
            self.assertEqual(row["retok_tokens"], len(g))
            closed = "</think>" in self.tok.decode(pre, skip_special_tokens=True)
            self.assertEqual(row["status"], "continued" if closed else "forced")
            self.assertEqual(q["prompt_token_ids"], self.prompt + pre + ([] if closed else self.stop_ids))
            self.assertEqual(len(q["prompt_token_ids"]), len(self.prompt) + min(t, len(g)) + (0 if closed else len(self.stop_ids)))
            self.assertEqual(q["head"], q["prompt_token_ids"][len(self.prompt):])
            text = next(r["text"] for r in self.recs if r["task_id"] == row["task_id"])
            self.assertTrue(text.startswith(self.tok.decode(pre, skip_special_tokens=True)))
        self.assertEqual(self.tok.decode(self.stop_ids, skip_special_tokens=True), FA.QWEN_STOP)

    def test_retokenisation_matches_generation(self):
        for r in self.recs:
            self.assertEqual(len(self.gen[r["task_id"]]) + (r["finish_reason"] == "stop"), r["n_tokens"])

    def test_prefix_diagnostics(self):
        A = [self.by()["A", t][0] for t in self.budgets if self.by()["A", t][0]["status"] in ("forced", "continued")]
        pos12 = self.recs[0]["text"].index("is 12") + 3
        for row in A:
            pre = self.tok.decode(self.gen["A"][:row["budget"]], skip_special_tokens=True)
            self.assertEqual(row["ref_in_prefix"], len(pre) > pos12 + 1, row["budget"])
            boxed_in = "\\boxed{12}" in pre                             # only once the final answer itself is in the prefix
            self.assertEqual(row["prefix_checker_ok"], boxed_in, row["budget"])
            self.assertEqual(row["truncated_t"], int(boxed_in))
        D = self.by()["D", 40][0]                                       # '\boxed{7}' lies within the first 40 tokens
        self.assertTrue(D["ref_in_prefix"] and D["prefix_checker_ok"])
        self.assertEqual(D["truncated_t"], 1)                           # the eval's checker would accept the cut text

    def test_score_uses_the_eval_checker(self):
        q = next(q for q in self.reqs if q["row"]["task_id"] == "A" and q["row"]["status"] == "forced")
        good = types.SimpleNamespace(text="So \\boxed{12}.", token_ids=self.tok.encode("So \\boxed{12}.", add_special_tokens=False),
                                     finish_reason="stop")
        bad = types.SimpleNamespace(text="\\boxed{13}", token_ids=self.tok.encode("\\boxed{13}", add_special_tokens=False),
                                    finish_reason="stop")
        self.assertEqual(FA.score(self.tok, q, good)["correct_t"], 1)
        self.assertEqual(FA.score(self.tok, q, bad)["correct_t"], 0)
        row = FA.score(self.tok, q, good)
        self.assertEqual((row["forced_tokens"], row["forced_finish_reason"]), (len(good.token_ids), "stop"))
        qc = next((q for q in self.reqs if q["row"]["task_id"] == "C" and q["row"]["status"] == "continued"), None)
        if qc is not None:                                              # reasoning closed before t: plain continuation
            fin = types.SimpleNamespace(text=" \\boxed{3}", token_ids=self.tok.encode(" \\boxed{3}", add_special_tokens=False),
                                        finish_reason="stop")
            self.assertEqual(FA.score(self.tok, qc, fin)["correct_t"], 1)


@unittest.skipIf(QWEN is None, "no cached Qwen tokenizer")
class TestPlanQwen(PlanMixin, unittest.TestCase):
    TOKDIR = QWEN

    def test_tokenizer_check_passes(self):
        FA.check_tokenizer(self.tok, "</think>")


@unittest.skipIf(R1 is None, "no DeepSeek-R1-Distill-Llama tokenizer (add one to TEST_EVAL_TOKENIZER_DIRS)")
class TestPlanR1(PlanMixin, unittest.TestCase):
    TOKDIR = R1

    def test_prompt_has_single_bos(self):
        self.assertEqual(self.prompt.count(self.tok.bos_token_id), 1)

    def test_tokenizer_trap_is_caught(self):
        FA.check_tokenizer(self.tok, "</think>")
        broken = U.make_model_dir(R1, model_type="llama", tokenizer_class="LlamaTokenizerFast")   # the original config
        with self.assertRaises(ValueError) as cm:
            FA.check_tokenizer(U.load_tok(broken), "</think>")
        self.assertIn("PreTrainedTokenizerFast", str(cm.exception))


@unittest.skipIf(QWEN is None, "no cached Qwen tokenizer")
class TestEndToEnd(unittest.TestCase):
    CAP = 48

    @classmethod
    def setUpClass(cls):
        cls.tok = U.load_tok(QWEN)
        cls.d = d = tempfile.mkdtemp(prefix="eval_forced_")
        cls.model = U.make_model_dir(QWEN)
        data = [dict(id="p1", problem="Compute 3*4.", answer="12"), dict(id="p2", problem="Find m+n.", answer=25)]
        cls.data = os.path.join(d, "test.jsonl"); U.write_jsonl(cls.data, data)
        ids = os.path.join(d, "ids.txt")
        with open(ids, "w") as f:
            f.write("p1\np2\n")
        harvest_outs = {"p1": [dict(text="<think>\nx = 12\n</think>\n\n\\boxed{12}"),
                               dict(text="<think>\n" + "a bit longer reasoning here. " * 4 + "\n</think>\n\n\\boxed{12}")],
                        "p2": [dict(text="<think>\n" + "going on and on. " * 30),
                               dict(text="<think>\nperhaps 25, " + "verify verify " * 30)]}
        spec = os.path.join(d, "harvest.json")
        with open(spec, "w") as f:
            json.dump(dict(outputs=harvest_outs, default=harvest_outs["p1"]), f)
        cls.out = os.path.join(d, "runs_synth", "agent_x", "shard_0.jsonl"); os.makedirs(os.path.dirname(cls.out))
        p = U.run_script("math_harvest.py", ["--model", cls.model, "--ids", ids, "--out", cls.out, "--max-tokens", str(cls.CAP),
                                             "--seed", "5", "--data", cls.data, "--save-texts", "auto"],
                         env=dict(MATH_EVAL="1", MATH_K="2"), fake_spec=spec)
        assert p.returncode == 0, p.stderr[-3000:]
        cls.texts_glob = os.path.join(d, "runs_synth", "agent_x", "*.texts.jsonl.gz")
        cls.texts = {(t["task_id"], t["k"]): t for t in ec.read_texts(cls.texts_glob)}
        cls.log = os.path.join(d, "forced.log")
        cls.fspec = os.path.join(d, "forced.json")
        with open(cls.fspec, "w") as f:
            json.dump(dict(outputs={"p1#1#20": [dict(text="So \\boxed{12}.")], "p2#0#20": [dict(text="\\boxed{25}")],
                                    "p2#0#48": [dict(text="\\boxed{24}")]},
                           default=[dict(text="I give up.")], log=cls.log), f)
        cls.fout = os.path.join(d, "forced.jsonl")
        cls.p = cls.force(["--budgets", "20,48,64"])
        cls.rows = {(x["task_id"], x["k"], x["budget"]): x for x in U.read_jsonl(cls.fout)}

    @classmethod
    def force(cls, extra, out=None, spec=None):
        return U.run_script("forced_answer.py", ["--model", cls.model, "--texts", cls.texts_glob, "--out", out or cls.fout] + extra,
                            fake_spec=spec or cls.fspec)

    def test_fixture(self):
        n = {k: t["n_tokens"] for k, t in self.texts.items()}
        self.assertLessEqual(n["p1", 0], 20)
        self.assertTrue(20 < n["p1", 1] <= self.CAP)
        self.assertEqual((n["p2", 0], n["p2", 1]), (self.CAP, self.CAP))

    def test_rows(self):
        self.assertEqual(self.p.returncode, 0, self.p.stderr[-3000:])
        self.assertEqual(len(self.rows), 4 * 3)
        st = {k: x["status"] for k, x in self.rows.items()}
        self.assertEqual([st["p1", 0, t] for t in (20, 48, 64)], ["finished"] * 3)
        self.assertEqual([st["p1", 1, t] for t in (20, 48, 64)], ["forced", "finished", "finished"])
        self.assertEqual([st["p2", 0, t] for t in (20, 48, 64)], ["forced", "forced", "not_evaluable"])
        ok = {k: x["correct_t"] for k, x in self.rows.items()}
        self.assertEqual((ok["p1", 0, 20], ok["p1", 1, 20], ok["p1", 1, 48]), (1, 1, 1))
        self.assertEqual((ok["p2", 0, 20], ok["p2", 0, 48], ok["p2", 0, 64]), (1, 0, None))
        self.assertEqual((ok["p2", 1, 20], ok["p2", 1, 48]), (0, 0))      # 'I give up.' has no \boxed{}
        self.assertTrue(self.rows["p2", 1, 20]["ref_in_prefix"])          # but the reasoning already said 25
        self.assertFalse(self.rows["p2", 0, 20]["ref_in_prefix"])
        self.assertEqual(self.rows["p2", 0, 48]["truncated_t"], 0)
        self.assertEqual(self.rows["p2", 0, 20]["forced_text"], "\\boxed{25}")
        self.assertTrue(os.path.exists(self.fout + ".meta.json"))
        self.assertEqual(len({x["cfg"] for x in self.rows.values()}), 1)

    def test_requests_and_sampling(self):
        llms, reqs = U.fake_log(self.log)
        self.assertEqual(len(llms), 1)
        stop = self.tok.encode(FA.QWEN_STOP, add_special_tokens=False)
        forced = {k for k, x in self.rows.items() if x["status"] in ("forced", "continued")}
        self.assertEqual({tuple(q["request"].rsplit("#", 2)) for q in reqs}, {(a, str(k), str(t)) for a, k, t in forced})
        for q in reqs:
            pid, k, t = q["request"].rsplit("#", 2)
            tx = self.texts[pid, int(k)]
            gen = self.tok.encode(tx["text"], add_special_tokens=False)[:int(t)]
            self.assertEqual(q["prompt_token_ids"], tx["prompt_token_ids"] + gen + stop)
            self.assertEqual(q["sampling"], dict(n=1, max_tokens=1024, output_kind=2, temperature=0.6, top_p=0.95, top_k=20, seed=5))
        self.assertEqual(llms[0]["max_model_len"], max(len(q["prompt_token_ids"]) for q in reqs) + 1024)
        self.assertEqual(llms[0]["seed"], 5)

    def test_resume_and_config_guard(self):
        n_llm = len(U.fake_log(self.log)[0])
        p = self.force(["--budgets", "20,48,64"])
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        self.assertEqual(len(U.read_jsonl(self.fout)), 12)
        self.assertEqual(len(U.fake_log(self.log)[0]), n_llm)            # nothing left: no engine
        p = self.force(["--budgets", "20,48,64", "--greedy"])
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("another configuration", p.stderr)

    def test_greedy_and_report_stop_text(self):
        out = os.path.join(self.d, "greedy.jsonl")
        log = os.path.join(self.d, "greedy.log"); spec = os.path.join(self.d, "greedy.json")
        with open(self.fspec) as f:
            sp = json.load(f)
        with open(spec, "w") as f:
            json.dump(dict(sp, log=log), f)
        p = self.force(["--budgets", "20", "--greedy", "--stop-text", REPORT_STOP.replace("\n", "\\n")], out=out, spec=spec)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        _, reqs = U.fake_log(log)
        last = reqs[-1]
        self.assertEqual(last["sampling"]["temperature"], 0.0)
        pid, k, t = last["request"].rsplit("#", 2)
        report_ids = self.tok.encode(REPORT_STOP, add_special_tokens=False)
        self.assertEqual(last["prompt_token_ids"][-len(report_ids):], report_ids)

    def test_texts_without_answer_are_rejected(self):
        bad = os.path.join(self.d, "code.texts.jsonl.gz")
        ec.append_texts(bad, [dict(task_id="c1", k=0, text="x", n_tokens=2, prompt_tokens=1, finish_reason="stop", correct=True,
                                   prompt_token_ids=[1])])
        p = U.run_script("forced_answer.py", ["--model", self.model, "--texts", bad, "--budgets", "8",
                                              "--out", os.path.join(self.d, "bad.jsonl")], fake_spec=self.fspec)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("texts lack", p.stderr)


if __name__ == "__main__":
    unittest.main()
