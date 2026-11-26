"""synth/aime_prep.py on a synthetic fixture through a fake datasets.load_dataset (test_eval_fakes/datasets): 60 problems
with integer answers in the MATH file format, pinned revisions, the repair of decoded escapes, and the cross-check
(mismatch, missing problem, --no-crosscheck). The real sources were checked separately on 2026-11-02."""
import json, os, re, tempfile, unittest

import test_eval_util as U


def fixture():
    mj, ma, h4, yl = [], [], [], []
    for year in (2024, 2025):
        for part in ("I", "II"):
            for i in range(1, 16):
                ans = (37 * i + (0 if part == "I" else 500) + year) % 1000
                text = f"AIME {year} {part} problem {i}: find $m+n$ where $\\frac{{m}}{{n}} = {i}$."
                url = f"https://artofproblemsolving.com/wiki/index.php/{year}_AIME_{part}_Problems/Problem_{i}"
                if year == 2024:
                    mj.append(dict(ID=f"2024-{part}-{i}", Problem=text, Solution="...", Answer=ans))
                    h4.append(dict(id=60 + i, problem=text, solution="...", answer=f"{ans:03d}", url=url, year="2024"))
                else:
                    ma.append(dict(problem_idx=i + (0 if part == "I" else 15), problem=text, answer=ans, problem_type=["Algebra"]))
                    yl.append(dict(id=i, problem=text, answer=str(ans), solution="...", url=url, year=2025))
    mj[0]["Problem"] = mj[0]["Problem"].replace("\\frac", "\x0crac")          # '\f' of \frac decoded to a form feed
    mj[22]["Problem"] = mj[22]["Problem"].replace("\\frac", "\\t" "frac").replace("\\tfrac", "\tfrac")   # '\t' of \tfrac -> tab
    return {"Maxwell-Jia/AIME_2024": mj, "MathArena/aime_2025": ma, "HuggingFaceH4/aime_2024": h4, "yentinglin/aime_2025": yl}


class TestAimePrep(unittest.TestCase):
    def prep(self, fx, *extra):
        d = tempfile.mkdtemp(prefix="eval_aime_")
        with open(os.path.join(d, "fx.json"), "w") as f:
            json.dump(fx, f)
        p = U.run_script("aime_prep.py", ["--out-dir", os.path.join(d, "aime")] + list(extra), cwd=d,
                         env=dict(FAKE_DATASETS_FIXTURE=os.path.join(d, "fx.json"), FAKE_DATASETS_LOG=os.path.join(d, "ds.log")))
        rows = U.read_jsonl(os.path.join(d, "aime", "aime24_25.jsonl")) if p.returncode == 0 else None
        return p, rows, d

    def test_build(self):
        fx = fixture()
        p, rows, d = self.prep(fx)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        self.assertEqual(len(rows), 60)
        self.assertEqual(rows[0]["id"], "aime2024_I_01"); self.assertEqual(rows[15]["id"], "aime2024_II_01")
        self.assertEqual(rows[30]["id"], "aime2025_I_01"); self.assertEqual(rows[-1]["id"], "aime2025_II_15")
        with open(os.path.join(d, "aime", "aime24_25_ids.txt")) as f:
            self.assertEqual(f.read().split(), [r["id"] for r in rows])
        for r in rows:
            self.assertEqual(set(r), {"id", "problem", "answer", "level", "year", "source"})
            self.assertIsInstance(r["answer"], int)
            self.assertTrue(0 <= r["answer"] <= 999)
            self.assertFalse(any(ord(c) < 32 and c != "\n" for c in r["problem"]), r["id"])
        self.assertEqual(rows[0]["answer"], fx["Maxwell-Jia/AIME_2024"][0]["Answer"])
        self.assertIn("$\\frac{m}{n}", rows[0]["problem"])              # form feed restored
        self.assertIn("\\tfrac{m}{n}", rows[22]["problem"])             # tab restored
        self.assertEqual(rows[45]["level"], "AIME 2025 II")
        self.assertIn("2 decoded escape(s) restored", p.stdout)
        calls = U.read_jsonl(os.path.join(d, "ds.log"))
        self.assertEqual([c["name"] for c in calls], ["Maxwell-Jia/AIME_2024", "HuggingFaceH4/aime_2024",
                                                      "MathArena/aime_2025", "yentinglin/aime_2025"])
        for c in calls:
            self.assertRegex(c["revision"], r"^[0-9a-f]{40}$")            # pinned commits
            self.assertEqual(c["split"], "train")

    def test_rows_feed_the_checker(self):
        import math_harvest as M
        _, rows, _ = self.prep(fixture())
        for r in rows[:3]:
            self.assertTrue(M.correct("<think>\n...\n</think>\n\n\\boxed{%03d}" % r["answer"], r["answer"]))

    def test_answer_mismatch_aborts(self):
        fx = fixture()
        fx["HuggingFaceH4/aime_2024"][5]["answer"] = "999"
        p, _, _ = self.prep(fx)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("answers differ", p.stderr)
        p, rows, d = self.prep(fx, "--no-crosscheck")
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        self.assertEqual(len(rows), 60)
        self.assertEqual(len(U.read_jsonl(os.path.join(d, "ds.log"))), 2)

    def test_missing_and_duplicate_problems_abort(self):
        fx = fixture()
        del fx["MathArena/aime_2025"][7]
        p, _, _ = self.prep(fx)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("missing", p.stderr)
        fx = fixture()
        fx["Maxwell-Jia/AIME_2024"].append(dict(fx["Maxwell-Jia/AIME_2024"][3]))
        p, _, _ = self.prep(fx)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("duplicate", p.stderr)

    def test_non_aime_answer_aborts(self):
        fx = fixture()
        fx["MathArena/aime_2025"][0]["answer"] = 1000
        p, _, _ = self.prep(fx, "--no-crosscheck")
        self.assertNotEqual(p.returncode, 0)
        self.assertTrue(re.search(r"not an AIME answer", p.stderr))


if __name__ == "__main__":
    unittest.main()
