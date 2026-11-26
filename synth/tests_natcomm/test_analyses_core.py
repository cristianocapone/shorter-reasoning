"""Core statistics of synth/natcomm_analyses.py: the bootstrap reproduces paper_stats.boot draw for draw, sign flips,
ties, Holm / Benjamini-Hochberg, Stouffer and partial conjunction (checked against the real paper_stats.json and the
values the referees and derived_stats.py report), exact seed permutations, chi-square tail, Cochran's Q.

    env -u PYTHONPATH .venv_mps/bin/python -m unittest discover -s synth/tests_natcomm -p 'test_analyses_*.py'
"""
import contextlib
import io
import json
import os
import random
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH = os.path.dirname(HERE)
REPO = os.path.dirname(SYNTH)
sys.path.insert(0, SYNTH)
import natcomm_analyses as NA  # noqa: E402

PAPER_STATS = os.path.join(REPO, "paper_brevity", "data", "paper_stats.json")


def small_b(b=2000, bp=2000):
    NA.B, NA.B_PERM = b, bp
    NA._COUNTS.clear()
    NA._FLIPS.clear()


class Reuse(unittest.TestCase):
    def test_compiled_definitions_have_no_side_effects(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ns = NA.compile_from_paper_stats(("B", "load", "boot", "single", "arc"))
        self.assertEqual(buf.getvalue(), "")
        self.assertEqual(ns["B"], 10000)
        self.assertTrue(callable(ns["load"]))

    def test_load_is_paper_stats_load(self):
        self.assertEqual(NA.load.__code__.co_filename, os.path.join(SYNTH, "paper_stats.py"))


class Bootstrap(unittest.TestCase):
    def setUp(self):
        small_b()

    def test_indices_are_paper_stats_draws(self):
        n = 23
        rng = random.Random(0)
        ref = np.array([[rng.choice(list(range(n))) for _ in range(n)] for _ in range(NA.B)])
        self.assertTrue(np.array_equal(NA.boot_indices(n), ref))

    def test_same_ci_and_p_as_paper_stats_boot(self):
        ns = NA.compile_from_paper_stats(("B", "boot"))
        ns["B"] = NA.B
        w = np.random.default_rng(1).normal(0.15, 1.0, 41)
        ref = ns["boot"](list(range(41)), lambda I: 100 * sum(w[i] for i in I) / len(I))
        mine = NA.mean_test(w, perm=False)
        for k in ("diff", "lo", "hi", "p"):
            self.assertAlmostEqual(mine[k], ref[k], places=9, msg=k)

    def test_signflip_extremes(self):
        d = np.ones(12)
        r = NA.mean_test(d)
        self.assertLess(r["P_perm"], 0.01)
        d = np.array([1.0, -1.0] * 6)
        r = NA.mean_test(d)
        self.assertEqual(r["diff"], 0.0)
        self.assertEqual(r["P_perm"], 1.0)

    def test_ties_and_degenerate(self):
        r = NA.summarize(0.0, np.zeros(1000), np.zeros(1000))
        self.assertEqual(r["P"], 1.0)
        self.assertEqual(r["P_perm"], 1.0)
        bs = np.r_[np.zeros(100), np.ones(900)]
        r = NA.summarize(1.0, bs)
        self.assertAlmostEqual(r["p"], 0.1)
        self.assertAlmostEqual(r["p_tie"], 0.1)
        self.assertAlmostEqual(r["P"], 0.2)
        self.assertAlmostEqual(r["P_ties"], 0.2)
        r = NA.summarize(-1.0, -bs)          # mirror: the paper's 2 min(p, 1 - p) misses the atom on this side
        self.assertAlmostEqual(r["P"], 0.0)
        self.assertAlmostEqual(r["P_ties"], 0.2)

    def test_ratio_statistic_swap(self):
        """A ratio statistic's permutation swaps whole problems: arms with identical rows give P_perm = 1."""
        X = np.vstack([np.arange(1, 11, dtype=float), np.full(10, 2.0)])
        r = NA.test2(X, X.copy(), lambda Sa, Sb: Sa[..., 0] / Sa[..., 1] - Sb[..., 0] / Sb[..., 1])
        self.assertEqual(r["diff"], 0.0)
        self.assertEqual(r["P_perm"], 1.0)


class Multiplicity(unittest.TestCase):
    def test_holm_bh_hand_values(self):
        p = [0.01, 0.04, 0.03, 0.005]
        self.assertEqual([round(x, 6) for x in NA.holm(p)], [0.03, 0.06, 0.06, 0.02])
        self.assertEqual([round(x, 6) for x in NA.bh(p)], [0.02, 0.04, 0.04, 0.02])

    def test_bh_matches_scipy(self):
        try:
            from scipy.stats import false_discovery_control
        except Exception:  # noqa: BLE001
            self.skipTest("scipy unavailable")
        p = list(np.random.default_rng(3).uniform(0, 0.2, 17))
        self.assertTrue(np.allclose(NA.bh(p), false_discovery_control(p, method="bh")))

    def test_stouffer_and_partial_conjunction_on_paper_stats(self):
        with open(PAPER_STATS) as fh:
            S = json.load(fh)
        NA.B = 10000
        tails = {o: [S[k]["contrasts"][f"S-{o}"][m]["p"] for k, m in NA.SELECTION_SETTINGS] for o in "RL"}
        z, p1, P = NA.stouffer(tails["R"])
        self.assertAlmostEqual(z, 3.16, places=2)              # derived_stats.py / the paper
        self.assertAlmostEqual(P, 1.57e-3, delta=1e-5)
        self.assertAlmostEqual(NA.stouffer(tails["L"])[0], 5.73, places=2)
        pc = [NA.partial_conjunction(tails["R"], u) for u in (2, 3, 4)]
        for got, want in zip(pc, (0.014, 0.098, 0.25)):      # referee stats-design's values (two significant digits)
            self.assertEqual(float(f"{got:.2g}"), want)
        loo = NA.stouffer([t for t, (k, _) in zip(tails["R"], NA.SELECTION_SETTINGS) if k != "math_qwen4b"])[2]
        self.assertAlmostEqual(loo, 0.012, delta=0.001)       # editor: 0.012 without Qwen-MATH
        loo = NA.stouffer([t for t, (k, _) in zip(tails["R"], NA.SELECTION_SETTINGS) if k != "math_llama8b"])[2]
        self.assertAlmostEqual(loo, 0.027, delta=0.001)       # editor: 0.027 without R1-Llama
        loo = NA.stouffer([t for t, (k, _) in zip(tails["R"], NA.SELECTION_SETTINGS) if k != "arc"])[2]
        self.assertAlmostEqual(loo, 0.003, delta=0.0005)      # editor: 0.003 without ARC


class ExactTests(unittest.TestCase):
    def test_seed_permutation_reproduces_derived_stats(self):
        with open(PAPER_STATS) as fh:
            A = json.load(fh)["arc"]["arms"]
        for (a, b), hits in ((("S", "base"), 38), (("R", "base"), 106), (("L", "base"), 96), (("S", "R"), 64), (("S", "L"), 44)):
            r = NA.perm_unpaired(A[a]["seeds"], A[b]["seeds"])
            self.assertEqual((r["hits"], r["splits"]), (hits, 252), (a, b))

    def test_paired_signflip_bruteforce(self):
        xa, xb = [21, 20, 21, 17, 19], [22, 13, 20, 12, 10]
        d = np.subtract(xa, xb)
        brute = sum(abs(np.mean(d * np.array(s))) >= abs(d.mean()) - 1e-9 for s in __import__("itertools").product((1, -1), repeat=5))
        r = NA.perm_paired(xa, xb)
        self.assertEqual(r["hits"], brute)
        self.assertEqual(r["flips"], 32)

    def test_chi2_sf(self):
        try:
            from scipy.stats import chi2
        except Exception:  # noqa: BLE001
            self.skipTest("scipy unavailable")
        for x, df in ((9.185, 4), (0.5, 1), (30.0, 7), (3.0, 2), (120.0, 100), (0.01, 3)):
            self.assertAlmostEqual(NA.chi2_sf(x, df), chi2.sf(x, df), places=9)

    def test_cochran_q(self):
        NA.B_PERM = 2000
        self.assertIn("note", NA.cochran_q(np.tile((np.arange(40) % 3 == 0).astype(float), (5, 1))))   # nothing varies
        Y = np.zeros((5, 40))
        Y[np.arange(40) % 5, np.arange(40)] = 1          # every puzzle solved by one seed, every seed solves 8
        r = NA.cochran_q(Y)
        self.assertEqual(r["Q"], 0.0)
        self.assertEqual(r["P_exact_mc"], 1.0)
        rng = np.random.default_rng(0)
        Y = (rng.random((5, 72)) < np.array([0.6, 0.1, 0.5, 0.1, 0.1])[:, None]).astype(float)   # a seed effect
        r = NA.cochran_q(Y)
        k, C, R = 5, Y.sum(1), Y.sum(0)
        Q = (k - 1) * (k * (C ** 2).sum() - C.sum() ** 2) / (k * C.sum() - (R ** 2).sum())        # textbook form
        self.assertAlmostEqual(r["Q"], round(Q, 3), places=3)
        self.assertLess(r["P_exact_mc"], 0.01)
        self.assertLess(r["P_chi2"], 0.01)

    def test_km_without_censoring_is_ecdf(self):
        t = np.array([5.0, 1.0, 3.0, 3.0, 8.0])
        F = NA.km_curve(t, np.ones(5, bool), np.ones((1, 5)), [0.5, 1, 3, 4, 8, 9])[0]
        self.assertTrue(np.allclose(F, [0, 0.2, 0.6, 0.6, 1.0, 1.0]))
        ev = np.array([True, True, False, True, True])            # 3.0 censored once: hand Kaplan-Meier
        F = NA.km_curve(t, ev, np.ones((1, 5)), [1, 3, 5, 8])[0]
        S1 = 1 - 1 / 5
        S3 = S1 * (1 - 1 / 4)
        S5 = S3 * (1 - 1 / 2)
        self.assertTrue(np.allclose(F, [1 - S1, 1 - S3, 1 - S5, 1.0]))


if __name__ == "__main__":
    unittest.main()
