"""ARC-AGI-2 analyses of synth/natcomm_analyses.py (item 5) on synthetic agent records, plus the two parts that run on
real local files: the message-level statistics of paper_brevity/data/arc_msgs.json (checked against the values that
paper_brevity/derived_stats.py prints and the paper quotes) and the seed-level tests from paper_stats.json.

    env -u PYTHONPATH .venv_mps/bin/python -m unittest discover -s synth/tests_natcomm -p 'test_analyses_*.py'
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH = os.path.dirname(HERE)
REPO = os.path.dirname(SYNTH)
sys.path.insert(0, SYNTH)
sys.path.insert(0, HERE)
import natcomm_analyses as NA  # noqa: E402
import test_analyses_fixtures as FX  # noqa: E402

ARC_MSGS = os.path.join(REPO, "paper_brevity", "data", "arc_msgs.json")
PAPER_STATS = os.path.join(REPO, "paper_brevity", "data", "paper_stats.json")


class ArcFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", ResourceWarning)
        cls.cwd = os.getcwd()
        cls.root = tempfile.mkdtemp(prefix="natcomm_arc_")
        FX.build_arc(cls.root, n=24)
        os.chdir(cls.root)
        NA.B, NA.B_PERM = 1000, 1000
        NA._COUNTS.clear()
        NA._FLIPS.clear()
        cls.D = NA.Arc()
        cls.sec, _ = NA.analyse_arc()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_loading_and_logs(self):
        D = self.D
        self.assertEqual(len(D.ids), 24)
        self.assertEqual(D.seeds["base"], [0, 1, 2, 3, 4])
        self.assertEqual(self.sec["log_seeds"]["base"], [2, 3, 4])
        self.assertEqual(self.sec["log_seeds"]["S"], [0, 1, 2, 3, 4])

    def test_per_seed_table(self):
        rows = self.sec["per_seed"]
        self.assertEqual(len(rows), 20)
        for r in rows:
            R = NA.load(r["tag"])
            self.assertEqual(r["solved"], sum(bool(R[i]["pass2"]) for i in self.D.ids))
            self.assertEqual(r["files"]["unique_tasks"], 24)
            self.assertEqual(r["files"]["shards"], 4)
            self.assertEqual(sum(r["thread_status"].values()), 24)
            if r["arm"] == "base" and r["seed"] < 2:
                self.assertEqual(r["puzzles_with_msg_log"], 0)
                self.assertEqual(r["messages"], 0)
        prov = [r["provenance"] for r in rows if r["tag"] == "brevcl_S_c72_s0"][0]
        self.assertEqual(prov["vllm_versions"], {"0.11.2": 1})
        self.assertEqual(prov["first_iter"], "2026-10-19_10:00:00")
        self.assertEqual(prov["log_events"].get("llm_chat_failed"), 1)

    def test_first_messages(self):
        D = self.D
        R = NA.load("brevcl_L_c72_s3")
        j = D.seeds["L"].index(3)
        for i, tid in enumerate(D.ids):
            m0 = R[tid]["msg_log"][0][0]
            self.assertEqual(D.M1["L"][:, j, i].tolist(), [1, float(m0[3]), float(m0[4]), float(min(m0[0], 24000))])
            self.assertEqual(D.M["L"][0, j, i], len(R[tid]["msg_log"][0]))

    def test_table1_contrast_and_seed_tests(self):
        c = self.sec["contrasts"]["S-base"]
        ta = self.D.Y["S"].sum(1).astype(int).tolist()
        self.assertEqual(c["seed_level"]["totals_a"], ta)
        self.assertEqual(c["seed_level"]["unpaired_permutation"]["splits"], 252)
        self.assertEqual(c["seed_level"]["paired_by_seed_index"]["flips"], 32)
        self.assertAlmostEqual(c["puzzle_bootstrap"]["solved"]["diff"], np.mean(ta) - self.D.Y["base"].sum(1).mean(), places=3)
        self.assertIn("twoway_bootstrap_paired_seeds", c)
        ma = c["messages_all"]
        self.assertEqual(ma["seeds_b"], [2, 3, 4])

    def test_twoway_reduces_to_puzzle_bootstrap(self):
        y = (np.arange(30) % 4 == 0).astype(float)
        z = (np.arange(30) % 3 == 0).astype(float)
        Ya, Yb = np.tile(y, (5, 1))[None], np.tile(z, (4, 1))[None]      # every seed identical
        f = lambda Sa, Sb: Sa[..., 0] - Sb[..., 0]
        o1, b1 = NA.twoway(Ya, Yb, f)
        o2, b2, _ = NA.resample(y[None], z[None], f, perm=False)
        self.assertAlmostEqual(float(o1), float(o2))
        self.assertTrue(np.allclose(b1, b2))

    def test_heterogeneity_section(self):
        h = self.sec["seed_heterogeneity"]["base"]
        self.assertEqual(h["df"], 4)
        self.assertEqual(h["seed_totals"], self.D.Y["base"].sum(1).astype(int).tolist())

    def test_dev_fresh(self):
        s = self.sec["dev_vs_fresh"]
        self.assertEqual((s["n_dev"], s["n_fresh"]), (12, 12))
        self.assertEqual(s["ids_check"]["fresh_in_excl"], 12)
        a = s["arms"]["S"]
        self.assertAlmostEqual(a["dev"] + a["fresh"], self.D.Y["S"].sum(1).mean(), places=6)
        self.assertIn("messages", s["contrasts"]["S-R"]["dev"])

    def test_time_to_program(self):
        t = self.sec["time_to_first_program"]
        for a, v in t["arms"].items():
            vals = [v[str(g)]["value"] for g in t["grid_tokens"]]
            self.assertTrue(all(x <= y + 1e-9 for x, y in zip(vals, vals[1:])), a)
            self.assertTrue(0 <= vals[0] and vals[-1] <= 100)
        self.assertIn("S-base", t["contrasts"])

    def test_by_index(self):
        b = self.sec["messages_by_index"]
        self.assertEqual(sorted(b, key=int), [str(k) for k in range(1, 7)])
        self.assertGreaterEqual(b["1"]["S"]["messages"], b["2"]["S"]["messages"])
        self.assertEqual(b["1"]["S"]["messages"], 24 * 5)


class LocalFiles(unittest.TestCase):
    """The parts that run here, on the real local files."""

    @classmethod
    def setUpClass(cls):
        NA.B, NA.B_PERM = 2000, 2000
        with open(PAPER_STATS) as fh:
            cls.PS = json.load(fh)
        cls.loc = NA.arc_messages_local(ARC_MSGS, cls.PS)

    def test_message_shares_match_derived_stats(self):
        A = self.loc["arms"]
        for a, cut, rc, w12 in (("S", 58.832, 49.99, 15.2), ("R", 63.247, 57.03, 13.9), ("L", 63.392, 57.61, 12.9),
                                ("base", 61.275, 54.90, 12.2)):
            self.assertAlmostEqual(A[a]["cut_share"]["value"], cut, delta=0.0006)
            self.assertAlmostEqual(A[a]["restart_cost_k"]["value"], rc, delta=0.006)
            self.assertAlmostEqual(A[a]["code_within_12000"]["value"], w12, delta=0.051)
            self.assertAlmostEqual(A[a]["cut_share"]["value"], self.PS["arc"]["arms"][a]["cut_msgs"], delta=0.006)
            self.assertLess(A[a]["cut_share"]["lo"], A[a]["cut_share"]["value"])
            self.assertEqual(A[a]["restart_cost_min_cap_on_grid"], 24000)
        self.assertEqual(A["base"]["messages"], 1255)
        C = self.loc["contrasts"]
        for c, d in (("S-R", -4.4145), ("S-L", -4.5603), ("S-base", -2.4428)):
            self.assertAlmostEqual(C[c]["cut_share"]["diff"], d, places=3)
        self.assertLess(C["S-R"]["cut_share"]["hi"], 0)
        self.assertIn("design_effect", self.loc["design_effect_vs_paper_stats"]["S-R"]["cut_share"])

    def test_summary_seed_tests(self):
        s = NA.arc_summary_section(self.PS)
        self.assertEqual(s["contrasts"]["S-base"]["unpaired_permutation"]["hits"], 38)
        self.assertEqual(s["contrasts"]["S-base"]["paired_by_seed_index"]["hits"], 6)
        self.assertAlmostEqual(s["dispersion"]["base"]["P"], 0.06, delta=0.005)     # referee: P ~ 0.06 vs the ceiling
        self.assertAlmostEqual(s["dispersion"]["base"]["variance"], 27.8)


if __name__ == "__main__":
    unittest.main()
