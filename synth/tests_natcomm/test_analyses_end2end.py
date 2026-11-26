"""End to end: synth/natcomm_analyses.py run as a script on a synthetic repository root (records of every setting,
ARC agent, harvests, a paper_stats.json made by paper_stats.py's own functions on the same records, an arc_msgs.json):
every section is written, the recomputed contrasts agree with paper_stats, paper_stats.json is untouched, sections
from earlier runs are kept (merge), and the script refuses to write paper_stats.json.

    env -u PYTHONPATH .venv_mps/bin/python -m unittest discover -s synth/tests_natcomm -p 'test_analyses_*.py'
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH = os.path.dirname(HERE)
sys.path.insert(0, SYNTH)
sys.path.insert(0, HERE)
import natcomm_analyses as NA  # noqa: E402
import test_analyses_fixtures as FX  # noqa: E402

SCRIPT = os.path.join(SYNTH, "natcomm_analyses.py")
RECORD_SECTIONS = ("item4_quality", "item5_arc", "item9_training_sets", "item10_finishing", "item13_cap_interactions",
                   "item13_replicates", "item13_registry")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", ResourceWarning)
        cls.root = tempfile.mkdtemp(prefix="natcomm_e2e_")
        FX.build_root(cls.root, n=60, n_arc=24)
        cwd = os.getcwd()
        os.chdir(cls.root)
        try:      # paper_stats.json from paper_stats.py's own functions on these records (B = 500)
            ns = NA.compile_from_paper_stats(("B", "load", "boot", "single", "arc", "cap_interaction", "replicates"))
            ns["B"] = 500
            C = NA.CONTRASTS
            ps = dict(arc=ns["arc"](),
                      math_qwen4b=ns["single"]("math_eval_{a}_s{s}", ["S", "R", "L", "base", "S2"], 50, C),
                      math_llama8b=ns["single"]("math_l8_eval_{a}_s{s}", ["S", "R", "L", "base"], 50, C),
                      apps_qwen4b=ns["single"]("code_eval_{a}_s{s}", ["S", "R", "L", "base"], 50, C),
                      math_qwen4b_cap16k=ns["single"]("math16_eval_{a}_s{s}", ["S", "L", "base"], 50, C),
                      cap_interaction_qwen4b=ns["cap_interaction"]("math_eval_{a}_s{s}", "math16_eval_{a}_s{s}", n_min=50),
                      replicates_qwen4b=ns["replicates"]({"rep0": "math_eval_{a}_s{s}", "rep1": "math_eval_{a}rep1_s{s}",
                                                          "rep2": "math_eval_{a}rep2_s{s}"}, n_min=50))
            arms = {}
            for a in ("S", "R", "L", "base"):
                fmt = NA.ARC["arms"][a]
                arms[a] = [m for s in range(5) for r in NA.load(fmt.format(s=s)).values() for th in r.get("msg_log", []) for m in th]
        finally:
            os.chdir(cwd)
        cls.data = os.path.join(cls.root, "paper_brevity", "data")
        os.makedirs(cls.data, exist_ok=True)
        cls.ps_path = os.path.join(cls.data, "paper_stats.json")
        json.dump(ps, open(cls.ps_path, "w"), indent=1)
        json.dump(dict(fields=["tokens", "thinking_chars", "visible_chars", "cut", "has_code"], cap=24000, arms=arms),
                  open(os.path.join(cls.data, "arc_msgs.json"), "w"))
        cls.ps_sha = sha(cls.ps_path)
        cls.out = os.path.join(cls.data, "natcomm_stats.json")
        cls.run1 = cls.run_script([])
        cls.res = json.load(open(cls.out))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    @classmethod
    def run_script(cls, extra):
        cmd = [sys.executable, SCRIPT, "--root", cls.root, "--boot", "500", "--perm", "500", "--n-min", "50"] + extra
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("WORK", None)
        return subprocess.run(cmd, capture_output=True, text=True, cwd=tempfile.gettempdir(), env=env)

    def test_runs_and_writes_every_section(self):
        self.assertEqual(self.run1.returncode, 0, self.run1.stderr[-2000:])
        for s in RECORD_SECTIONS + NA.LOCAL_SECTIONS + NA.DERIVED_SECTIONS:
            self.assertIn(s, self.res)
            self.assertNotIn("skipped", self.res[s], s)
            self.assertIn(s, self.res["meta"]["sections"])
        self.assertEqual(self.res["meta"]["B"], 500)

    def test_agrees_with_paper_stats(self):
        chk = self.res["item4_quality"]["check_vs_paper_stats"]
        self.assertEqual(sorted(chk), sorted(NA.MAIN_SINGLE))
        for key, v in chk.items():
            self.assertTrue(v["agree"], (key, v))
        ps = json.load(open(self.ps_path))
        for c, st in ps["arc"]["contrasts"].items():
            mine = self.res["item5_arc"]["contrasts"][c]
            for k in ("diff", "lo", "hi"):
                self.assertAlmostEqual(mine["puzzle_bootstrap"]["solved"][k], st["solved"][k], places=6)
                self.assertAlmostEqual(mine["messages_all"]["puzzle_bootstrap"]["cut_share"][k], st["cut_msgs"][k], places=6)
                self.assertAlmostEqual(mine["messages_all"]["puzzle_bootstrap"]["restart_cost_k"][k], st["restart_cost_k"][k], places=6)
        for c, st in ps["replicates_qwen4b"]["contrasts"].items():
            mine = self.res["item13_replicates"]["contrasts"][c]
            self.assertEqual(mine["per_replicate"], st["per_replicate"])
            self.assertEqual(mine["hier_boot"]["lo"], st["hier_boot"]["lo"])
            self.assertEqual(mine["hier_boot"]["hi"], st["hier_boot"]["hi"])
            self.assertEqual(mine["hier_boot"]["p"], st["hier_boot"]["p"])
        for c, st in ps["cap_interaction_qwen4b"]["contrasts"].items():
            mine = self.res["item13_cap_interactions"]["contrasts"][c]
            for m, ref in st.items():
                for k in ("diff", "lo", "hi"):
                    self.assertAlmostEqual(mine[m][k], ref[k], places=6, msg=(c, m, k))

    def test_paper_stats_untouched(self):
        self.assertEqual(sha(self.ps_path), self.ps_sha)

    def test_confirmatory_and_families(self):
        conf = self.res["item13_confirmatory"]
        self.assertTrue(all(r["source"] == "records (this script)" for r in conf["settings"]))
        self.assertEqual(len(conf["settings"]), 8)
        for c in ("S-R", "S-L"):
            t = conf["tests"][c]
            self.assertEqual(len(t["leave_one_out"]), 4)
            self.assertIn("signflip_stouffer", t)
            self.assertIn("with_arc_twoway_tail", t)
            self.assertGreaterEqual(t["P_holm"], t["P"])
        fam = self.res["item13_multiplicity"]
        for name in ("selection_contrasts", "decomposition_terms", "decomposition_per_problem", "cap_interactions",
                     "arc_messages_all", "arc_messages_first", "arc_messages_local_message_level", "arc_accuracy_seed_level"):
            self.assertIn(name, fam)
            for t in fam[name]["tests"]:
                self.assertGreaterEqual(t["P_holm"] + 1e-12, t["P_bh"])
                self.assertGreaterEqual(t["P_bh"] + 1e-12, min(t["P"], 1.0))
        self.assertEqual(fam["selection_contrasts"]["m"], 8)
        self.assertEqual(fam["cap_interactions"]["m"], 6)
        self.assertEqual(fam["versus_untouched"]["m"], 12)      # S, R, L vs untouched in the four settings
        self.assertIn("P_perm_holm", fam["versus_untouched"]["tests"][0])

    def test_tokens_section(self):
        tok = self.res["item13_tokens"]
        self.assertIn("tokens_rel_pct", tok["math_qwen4b"]["contrasts"]["S-base"])
        self.assertIn("tokens_per_solved_rel_pct", tok["arc"]["contrasts"]["S-base"]["twoway_bootstrap"])
        self.assertIn("tokens_log_ratio_8k_minus_16k", tok["cap_8k_vs_16k"]["S-L"])

    def test_merge_keeps_other_sections(self):
        before = json.load(open(self.out))
        r = self.run_script(["--only", "item5_arc_summary"])
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        after = json.load(open(self.out))
        self.assertEqual(after["item4_quality"], before["item4_quality"])
        self.assertEqual(after["meta"]["sections"]["item4_quality"], before["meta"]["sections"]["item4_quality"])
        self.assertEqual(after["meta"]["last_run_sections"][0], "item5_arc_summary")
        self.assertEqual(after["item13_confirmatory"], before["item13_confirmatory"])

    def test_no_merge_and_refusal(self):
        alt = os.path.join(self.root, "alt.json")
        r = self.run_script(["--only", "item5_arc_summary", "--no-merge", "--out", alt])
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        res = json.load(open(alt))
        self.assertNotIn("item4_quality", res)
        self.assertIn("item5_arc_summary", res)
        r = self.run_script(["--out", self.ps_path, "--only", "item5_arc_summary"])
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(sha(self.ps_path), self.ps_sha)


if __name__ == "__main__":
    unittest.main()
