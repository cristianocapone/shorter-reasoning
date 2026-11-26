"""synth/code_harvest.py (APPS) end to end through the fake vLLM: the old 'cut' is computed as before, the new cap_cut /
lens / oks / fins / ans_tok are aligned with the samples, and --save-texts works. Programs really run (subprocess)."""
import json, os, tempfile, unittest

import test_eval_util as U
import eval_common as ec

TOKDIR = U.qwen_dir()
K = 4
CAP = 80
DATA = [dict(id="c1", question="Read an integer n and print 2*n.", inputs=["3\n", "10\n", "0\n"], outputs=["6\n", "20\n", "0\n"],
             difficulty="introductory"),
        dict(id="c2", question="Read two integers and print their sum.", inputs=["1 2\n", "5 7\n"], outputs=["3\n", "12\n"],
             difficulty="interview")]
DRAFT = "<think>\nPlan:\n```python\nn = int(input())\nprint(2 * n)\n```\nLooks right.\n</think>\n\n"
OUTS = {"c1": [dict(text=DRAFT + "```python\nn = int(input())\nprint(2*n)\n```", finish_reason="stop"),        # draft == final
               dict(text="<think>\nadd one?\n</think>\n\n```python\nprint(int(input()) + 1)\n```", finish_reason="stop"),
               dict(text="<think>\nhmm\n</think>\n\nJust double the number.", finish_reason="stop"),           # no code block
               dict(text="<think>\n" + "still thinking " * 60, finish_reason="stop")],                        # cut by CAP
        "c2": [dict(text="<think>\nsum\n</think>\n\n```python\na, b = map(int, input().split())\nprint(a + b)\n```",
                    finish_reason="stop")]}


@unittest.skipIf(TOKDIR is None, "no cached tokenizer (set TEST_EVAL_TOKENIZER_DIRS)")
class TestCodeHarvest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = U.load_tok(TOKDIR)
        d = tempfile.mkdtemp(prefix="eval_code_")
        model = U.make_model_dir(TOKDIR)
        data = os.path.join(d, "test.jsonl"); U.write_jsonl(data, DATA)
        ids = os.path.join(d, "ids.txt")
        with open(ids, "w") as f:
            f.write("c1\nc2\n")
        cls.log = os.path.join(d, "fake.log"); spec = os.path.join(d, "spec.json")
        with open(spec, "w") as f:
            json.dump(dict(outputs=OUTS, default=OUTS["c2"], log=cls.log), f)
        cls.out = os.path.join(d, "agent_code", "shard_0.jsonl"); os.makedirs(os.path.dirname(cls.out))
        cls.p = U.run_script("code_harvest.py", ["--model", model, "--ids", ids, "--out", cls.out, "--max-tokens", str(CAP),
                                                 "--data", data, "--save-texts", "auto"],
                             env=dict(MATH_EVAL="1", MATH_K=str(K), CODE_WORKERS="2"), fake_spec=spec)
        cls.recs = {r["task_id"]: r for r in U.read_jsonl(cls.out)} if os.path.exists(cls.out) else {}

    def ids_of(self, pid, k):
        o = OUTS[pid][k % len(OUTS[pid])]
        ids = self.tok.encode(o["text"], add_special_tokens=False) + [self.tok.eos_token_id]
        return ids[:CAP] if len(ids) > CAP else ids

    def test_records(self):
        self.assertEqual(self.p.returncode, 0, self.p.stderr[-3000:])
        self.assertEqual(set(self.recs), {"c1", "c2"})
        for pid, r in self.recs.items():
            for key in ("acc", "tokens", "mean_tokens", "cut", "pass2", "difficulty", "cap_cut", "lens", "oks", "fins", "ans_tok"):
                self.assertIn(key, r)
            for key in ("lens", "oks", "fins", "ans_tok"):
                self.assertEqual(len(r[key]), K)
            self.assertEqual(r["cap_cut"], K - sum(r["fins"]))
            self.assertEqual(r["cut"], r["cap_cut"])        # this copy's 'cut' = finish_reason 'length', computed as before
            self.assertEqual(r["tokens"], sum(r["lens"]))
            self.assertAlmostEqual(r["acc"], sum(r["oks"]) / K)
            self.assertEqual(r["lens"], [len(self.ids_of(pid, k)) for k in range(K)])
        self.assertEqual(self.recs["c1"]["oks"], [1, 0, 0, 0])
        self.assertEqual(self.recs["c1"]["fins"], [1, 1, 1, 0])
        self.assertEqual(self.recs["c2"]["oks"], [1] * K)

    def test_ans_tok(self):
        r = self.recs["c1"]
        ids = self.ids_of("c1", 0)
        pos = self.tok.decode(ids, skip_special_tokens=True).index("```python")       # the draft in the reasoning
        j = r["ans_tok"][0]
        self.assertLessEqual(len(self.tok.decode(ids[:j], skip_special_tokens=True)), pos)
        self.assertGreater(len(self.tok.decode(ids[:j + 1], skip_special_tokens=True)), pos)
        self.assertIsNotNone(r["ans_tok"][1])
        self.assertIsNone(r["ans_tok"][2])                  # no code block
        self.assertIsNone(r["ans_tok"][3])                  # cut inside the reasoning
        ids = self.ids_of("c2", 0)
        full = self.tok.decode(ids, skip_special_tokens=True)
        j = self.recs["c2"]["ans_tok"][0]                    # no draft: the graded block itself
        self.assertGreater(len(self.tok.decode(ids[:j + 1], skip_special_tokens=True)), full.rindex("```python"))

    def test_texts(self):
        rows = ec.read_texts(self.out[:-len(".jsonl")] + ".texts.jsonl.gz")
        self.assertEqual(len(rows), 2 * K)
        for t in rows:
            self.assertNotIn("answer", t)
            self.assertEqual(t["n_tokens"], self.recs[t["task_id"]]["lens"][t["k"]])
            self.assertEqual(t["correct"], bool(self.recs[t["task_id"]]["oks"][t["k"]]))

    def test_engine_settings(self):
        llms, _ = U.fake_log(self.log)
        self.assertEqual(llms[0]["max_model_len"], 12288)   # the script's default, unchanged for short prompts


if __name__ == "__main__":
    unittest.main()
