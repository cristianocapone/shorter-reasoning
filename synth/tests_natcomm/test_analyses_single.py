"""Single-sample analyses of synth/natcomm_analyses.py on synthetic records (item 4: fixed finished set, strata with
cross-fitting, exact accounting split, Lee bounds; item 10: APPS finishing from token counts, length crossing), and
agreement with paper_stats.py's own functions run on the same records.

    env -u PYTHONPATH .venv_mps/bin/python -m unittest discover -s synth/tests_natcomm -p 'test_analyses_*.py'
"""
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import natcomm_analyses as NA  # noqa: E402
import test_analyses_fixtures as FX  # noqa: E402


class SingleFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", ResourceWarning)           # paper_stats.load leaves its files to the GC
        cls.cwd = os.getcwd()
        cls.root = tempfile.mkdtemp(prefix="natcomm_single_")
        FX.build_single(cls.root, n=60)
        same = {"base": (0.0, 0.0, 0.0), "S": (0.0, 0.0, 0.0)}           # identical arms, many problems: no effect
        for a in same:
            for s in range(5):
                FX.write_jsonl(os.path.join(cls.root, "runs_synth", f"agent_null_{a}_s{s}"),
                               FX.single_records(400, 5, a, s, arms=same))
        os.chdir(cls.root)
        NA.B, NA.B_PERM, NA.N_MIN = 1000, 1000, 50
        NA._COUNTS.clear()
        NA._FLIPS.clear()
        cls.D = NA.Single("math_qwen4b", NA.SINGLE["math_qwen4b"])

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        shutil.rmtree(cls.root, ignore_errors=True)

    # ---------------------------------------------------------------------------------------------- loading
    def test_loading(self):
        D = self.D
        self.assertTrue(D.ok)
        self.assertEqual(len(D.ids), 60)
        self.assertEqual(D.K, 4)
        self.assertEqual(D.seeds["S2"], [0, 1, 2])
        self.assertEqual(sorted(set(D.labels)), ["1", "2", "3", "4", "5"])
        cc = D.cap_check()
        for a, v in cc.items():
            self.assertEqual(v["all_samples_cut"], v["all_cut_with_tokens_eq_K_cap"])
            self.assertEqual(v["tokens_below_cut_times_cap"], 0)
        self.assertEqual(D.cap_from_data, 8192)
        for a in ("S", "base"):                    # long outputs at a 16k cap, with a wrongly configured cap of 16000
            for s in range(2):
                FX.write_jsonl(os.path.join(self.root, "runs_synth", f"agent_c16_{a}_s{s}"),
                               FX.single_records(60, 3, a, s, cap=16384, log_mu=9.3))
        D16 = NA.Single("c16", dict(arms={"S": "c16_S_s{s}", "base": "c16_base_s{s}"}, cap=16000, field="level"))
        self.assertEqual((D16.cap_config, D16.cap), (16000, 16384))

    def test_agrees_with_paper_stats_single(self):
        ns = NA.compile_from_paper_stats(("B", "load", "boot", "single"))
        ns["B"] = NA.B
        ref = ns["single"]("math_eval_{a}_s{s}", ["S", "R", "L", "base", "S2"], 50, NA.CONTRASTS)
        sec, _ = NA.analyse_single("math_qwen4b")
        chk = NA.check_vs_paper_stats({"math_qwen4b": sec}, {"math_qwen4b": ref})["math_qwen4b"]
        self.assertTrue(chk["agree"], chk)
        self.assertEqual(chk["compared"], 3 * len(ref["contrasts"]))
        for a, v in ref["arms"].items():
            self.assertAlmostEqual(sec["arms"][a]["acc"]["value"], v["acc"], delta=0.0051)       # paper_stats rounds to 0.01
            self.assertAlmostEqual(sec["arms"][a]["correct_given_finished"]["value"], v["correct_given_finished"], delta=0.0051)

    # ---------------------------------------------------------------------------------------------- (c) accounting split
    def test_split_exact_in_every_resample(self):
        D, n = self.D, len(self.D.ids)
        obs, bs, ts = NA.resample(NA.contrast_rows(D, "L"), NA.contrast_rows(D, "base"), lambda Sa, Sb: NA.contrast_stats(Sa, Sb, n))
        for arr in (obs[None], bs, ts):
            self.assertTrue(np.allclose(arr[:, 0], arr[:, 3] + arr[:, 4], atol=1e-9))

    def test_per_problem_terms(self):
        rng = np.random.default_rng(4)
        Fa, Fb = rng.choice([0, 0.25, 0.5, 1.0], 200), rng.choice([0, 0.5, 0.75, 1.0], 200)
        Aa, Ab = Fa * rng.random(200), Fb * rng.random(200)
        fin, qual = NA.per_problem_terms(Aa, Fa, Ab, Fb)
        self.assertTrue(np.allclose(fin + qual, Aa - Ab))
        one_zero = (Fa == 0) ^ (Fb == 0)
        self.assertTrue(np.all(qual[one_zero] == 0))               # no finished sample to compare: all finishing
        self.assertTrue(np.all(fin[(Fa == 0) & (Fb == 0)] == 0))

    def test_quality_term_p_equals_dq_p(self):
        """Bootstrapping the product directly changes the interval, not the sign: the quality term's P is dQ's P."""
        c = NA.contrast_block(self.D, "L", "base")
        self.assertEqual(c["quality_term"]["p"], c["correct_given_finished"]["p"])
        self.assertEqual(c["split_check"], 0.0)

    # ---------------------------------------------------------------------------------------------- (a) fixed set
    def test_common_set(self):
        D = self.D
        cs = NA.common_set_block(D)
        both = np.all(D.cut["S"] == 0, 0) & np.all(D.cut["L"] == 0, 0)
        self.assertEqual(cs["pairs"]["S-L"]["n_problems"], int(both.sum()))
        allm = np.all([np.all(D.cut[a] == 0, 0) for a in D.arms], 0)
        self.assertEqual(cs["n_all"], int(allm.sum()))
        self.assertAlmostEqual(cs["pairs"]["S-L"]["acc_a"], round(100 * D.A["S"][both].mean(), 3))

    # ---------------------------------------------------------------------------------------------- (b) strata
    def test_cross_fitting_removes_regression_to_the_mean(self):
        cfg = dict(arms={"S": "null_S_s{s}", "base": "null_base_s{s}"}, cap=8192, field="level")
        D = NA.Single("null", cfg)
        st = NA.strata_block(D, "S", "base", "base_finish")
        self.assertTrue(st["cross_fitted"])
        top = st["F=1"]["finish"]
        self.assertLessEqual(top["lo"], 0)
        self.assertGreaterEqual(top["hi"], 0)
        top_bin = D.base_F() >= 1 - 1e-9                            # naive: bins and outcomes from the same base seeds
        naive = 100 * (D.F["S"][top_bin].mean() - D.F["base"][top_bin].mean())
        self.assertLess(naive, -0.5)
        self.assertLess(abs(top["diff"]), abs(naive))

    def test_level_strata_cover_all_problems(self):
        st = NA.strata_block(self.D, "S", "L", "level")
        self.assertEqual(sum(v["n_problems"] for k, v in st.items() if isinstance(v, dict) and "n_problems" in v), 60)
        self.assertFalse(st["cross_fitted"])

    # ---------------------------------------------------------------------------------------------- (d) Lee bounds
    def test_lee_bounds_contain_truth(self):
        rng = np.random.default_rng(9)
        n = 300
        w = rng.uniform(0.2, 0.9, n)                       # finishes under both arms
        m = rng.uniform(0, 0.1, n)                         # finishes under a only (monotonicity: a finishes more)
        qa, qb, qm = rng.uniform(0.7, 1, n), rng.uniform(0.7, 1, n), rng.uniform(0, 1, n)
        D = SimpleNamespace(ids=list(range(n)), labels=np.array([str(i % 3) for i in range(n)]),
                            A={"a": w * qa + m * qm, "b": w * qb}, F={"a": w + m, "b": w})
        truth = 100 * (w * (qa - qb)).sum() / w.sum()
        lb = NA.lee_block(D, "a", "b")
        for cell in ("within_problem", "within_level", "pooled"):
            self.assertLessEqual(lb[cell]["lower"] - 1e-6, truth, cell)
            self.assertGreaterEqual(lb[cell]["upper"] + 1e-6, truth, cell)
        self.assertLessEqual(lb["within_problem"]["upper"] - lb["within_problem"]["lower"],
                             lb["pooled"]["upper"] - lb["pooled"]["lower"] + 1e-9)
        self.assertEqual(lb["higher_finishing_arm"], "a")
        D.A, D.F = {"a": D.A["b"], "b": D.A["a"]}, {"a": D.F["b"], "b": D.F["a"]}     # swap: bounds mirror
        lb2 = NA.lee_block(D, "a", "b")
        self.assertAlmostEqual(lb2["within_problem"]["lower"], -lb["within_problem"]["upper"], places=6)

    def test_lee_identical_arms_give_zero(self):
        A, F = np.array([0.3, 0.5]), np.array([0.4, 0.5])
        w, lo, hi = NA.lee_masses(A, F, A, F)
        self.assertTrue(np.allclose(lo, 0) and np.allclose(hi, 0))

    # ---------------------------------------------------------------------------------------------- item 10
    def test_apps_finishing_bounds(self):
        D = NA.Single("apps_qwen4b", NA.SINGLE["apps_qwen4b"])
        r = NA.apps_finishing(D)
        T, c = D.tok["base"], D.cut["base"]
        self.assertEqual(r["arms"]["base"]["records_cut_exceeds_floor_T_over_cap"], int((c > np.floor(T / 8192)).sum()))
        self.assertGreater(r["arms"]["base"]["records_cut_exceeds_floor_T_over_cap"], 0)
        lo, hi = r["arms"]["base"]["finish_length_bounds_tokens_only"]
        self.assertLessEqual(lo, hi)
        lo2, _ = r["arms"]["base"]["finish_length_bounds_if_cut_counts_capped_and_no_code"]
        self.assertGreaterEqual(lo2, lo)
        self.assertEqual(r["arms"]["base"]["records_cut_below_token_minimum"], 0)

    def test_cut_but_correct_detected(self):
        """acc K > K - cut is impossible if every cut sample is graded wrong: the check counts such records."""
        for a in ("S", "base"):
            for s in range(2):
                rows = [dict(task_id=f"p{i}", acc=0.75 if (i == 3 and a == "S") else 0.5, tokens=2 * 8192 + 2000,
                             mean_tokens=(2 * 8192 + 2000) / 4, cut=2, pass2=True, level=1) for i in range(60)]
                FX.write_jsonl(os.path.join(self.root, "runs_synth", f"agent_cc_{a}_s{s}"), rows)
        D = NA.Single("cc", dict(arms={"S": "cc_S_s{s}", "base": "cc_base_s{s}"}, cap=8192, field="level"))
        cc = D.cap_check()
        self.assertEqual(cc["S"]["records_correct_exceeds_finished"], 2)       # problem p3, both seeds
        self.assertEqual(cc["S"]["min_cut_samples_graded_correct"], 2)
        self.assertEqual(cc["base"]["records_correct_exceeds_finished"], 0)

    def test_apps_exact_from_per_sample_lengths(self):
        rng = np.random.default_rng(12)
        for a in ("S", "base"):
            for s in range(3):
                rows = []
                for i in range(60):
                    lens = [int(x) for x in np.minimum(rng.integers(1000, 12000, 4), 8192)]
                    capped = [x >= 8192 for x in lens]
                    nocode = [(not c) and rng.random() < 0.1 for c in capped]
                    ok = [(not c) and (not nc) and rng.random() < 0.9 for c, nc in zip(capped, nocode)]
                    rows.append(dict(task_id=f"q{i}", acc=float(np.mean(ok)), tokens=sum(lens), mean_tokens=sum(lens) / 4,
                                     cut=int(sum(c or nc for c, nc in zip(capped, nocode))), pass2=bool(ok[0]),
                                     difficulty="interview", lens=lens))
                FX.write_jsonl(os.path.join(self.root, "runs_synth", f"agent_al_{a}_s{s}"), rows)
        D = NA.Single("al", dict(arms={"S": "al_S_s{s}", "base": "al_base_s{s}"}, cap=8192, field="difficulty",
                                 cut="length_or_no_code"))
        r = NA.apps_finishing(D)
        ex = D.exact_capped("S")
        self.assertAlmostEqual(r["arms"]["S"]["finish_exact_from_per_sample_fields"], round(100 * float((1 - ex / 4).mean()), 3))
        lo, hi = r["arms"]["S"]["finish_length_bounds_if_cut_counts_capped_and_no_code"]
        self.assertTrue(lo - 1e-6 <= r["arms"]["S"]["finish_exact_from_per_sample_fields"] <= hi + 1e-6)
        dec = r["contrasts"]["S-base"]["exact_decomposition"]
        self.assertAlmostEqual(dec["finishing_term"]["diff"] + dec["quality_term"]["diff"], dec["acc"]["diff"], places=2)

    def test_crossing(self):
        n = 400
        rng = np.random.default_rng(2)
        Fb = rng.uniform(0.6, 1, n)
        Tb = 3000 + 4000 * (1 - Fb) + rng.normal(0, 50, n)
        D = SimpleNamespace(ids=list(range(n)), T={"r": Tb - 300, "b": Tb}, F={"r": Fb - 0.03, "b": Fb},
                            FT={"r": Fb * 2000 - 400, "b": Fb * 2400})
        r = NA.crossing_block(D, "r", "b")                 # shorter on average, yet more cut: the distributions cross
        self.assertTrue(r["crossing_point_estimates"])
        self.assertLess(r["intersection_union"]["p_boot"], 0.01)
        self.assertLess(r["intersection_union"]["p_perm"], 0.01)
        pp = r["per_problem"]                              # here every problem is shorter yet cut more often
        self.assertEqual(pp["shorter_but_more_cut"], 100.0)
        self.assertEqual(pp["asymmetry_pct_points"]["diff"], 100.0)
        self.assertLess(pp["asymmetry_pct_points"]["P_perm"], 0.01)
        D.F = {"r": Fb + 0.03, "b": Fb}                    # shorter and fewer cuts: consistent with a shift
        self.assertFalse(NA.crossing_block(D, "r", "b")["crossing_point_estimates"])


if __name__ == "__main__":
    unittest.main()
