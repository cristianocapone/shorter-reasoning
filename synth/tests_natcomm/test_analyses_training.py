"""Training-set statistics of synth/natcomm_analyses.py (item 9) on synthetic harvests: the reconstruction picks the
same verified solution per task as distill/sft_student.build_samples (run here with a character-level fake tokenizer),
applies --task-cap and --exclude-ids the same way, counts distinct solutions, and reads sft_student's cache and logs.

    env -u PYTHONPATH .venv_mps/bin/python -m unittest discover -s synth/tests_natcomm -p 'test_analyses_*.py'
"""
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH = os.path.dirname(HERE)
REPO = os.path.dirname(SYNTH)
sys.path.insert(0, SYNTH)
sys.path.insert(0, HERE)
import natcomm_analyses as NA  # noqa: E402
import test_analyses_fixtures as FX  # noqa: E402


def sft_student():
    try:
        spec = importlib.util.spec_from_file_location("sft_student_under_test", os.path.join(REPO, "distill", "sft_student.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:  # noqa: BLE001  (torch missing)
        raise unittest.SkipTest(f"distill/sft_student.py not importable here: {type(e).__name__}")


class CharTok:
    """Tokenises to characters, so sft_student's targets give back the exact assistant text."""

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=True, **kw):
        return list("".join(m["content"] for m in msgs))

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text)}


def picks_from_samples(samples, end="<|im_end|>\n"):
    by = defaultdict(list)
    for _, tgt, tid in samples:
        t = "".join(tgt)
        by[tid].append(t[: -len(end)] if t.endswith(end) else t)
    return {tid: hashlib.sha1(json.dumps(v).encode()).hexdigest() for tid, v in by.items()}


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", ResourceWarning)
        cls.cwd = os.getcwd()
        cls.root = tempfile.mkdtemp(prefix="natcomm_train_")
        FX.build_arc(cls.root, n=24)              # writes synth/brev_excl72.txt (arc000-arc023, trainX)
        FX.build_harvests(cls.root)
        os.chdir(cls.root)
        os.environ.pop("SFT_NO_SYS", None)
        os.environ.pop("SFT_END", None)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_same_picks_as_sft_student(self):
        mod = sft_student()
        for key in ("math_qwen4b", "math_qwen4b_rep2", "arc"):
            cfg = NA.TRAIN[key]
            cand, _ = NA.sft_candidates(cfg["data"], cfg.get("task_cap", 0), system_added=cfg["system_prompt_added"])
            for a, (how, _) in cfg["arms"].items():
                samples, skipped, n_traj = mod.build_samples(NA.sft_paths(cfg["data"]), CharTok(), 10 ** 9, {}, True, 0, 0, 1,
                                                             how, cfg.get("task_cap", 0), cfg["seed"])
                ref = picks_from_samples(samples)
                mine = {t: v[0]["hash"] for t, v in NA.sft_select(cand, how, cfg["seed"]).items()}
                self.assertEqual(mine, ref, (key, a))
                self.assertEqual(skipped, 0)

    def test_exclusion_and_counts(self):
        cfg = NA.TRAIN["arc"]
        out = NA.training_set("arc", cfg, None)
        cand, _ = NA.sft_candidates(cfg["data"], cfg["task_cap"])
        bad = set(NA.read_ids(cfg["exclude"]))
        kept = [t for t in cand if not (t in bad or t.split("_")[0] in bad)]
        self.assertEqual(out["tasks_trained"], len(kept))
        self.assertGreater(out["excluded_tasks"], 0)
        self.assertTrue(out["exclude_list_found"])
        self.assertTrue(all(not t.startswith("arc") for t in kept))
        for t, v in cand.items():
            self.assertLessEqual(min(tr["key"] for tr in v), 15000)
        L = out["arms"]["L"]
        self.assertEqual(L["trajectories"], len(kept))
        self.assertEqual(L["trajectories_with_turn_tokens_from_msg_log"], len(kept))   # thread matched by tokens
        self.assertGreaterEqual(L["cut_turns"], out["arms"]["S"]["cut_turns"])
        self.assertEqual(out["recipe"]["effective_batch"], 8)
        self.assertEqual((out["recipe"]["grad_accum"], out["recipe"]["processes"]), (8, 1))

    def test_distinct_and_histogram(self):
        cfg = NA.TRAIN["math_qwen4b"]
        out = NA.training_set("math_qwen4b", cfg, None)
        rows = [json.loads(l) for f in sorted(NA.sft_paths(cfg["data"])) for l in open(f) if l.strip()]
        nv = Counter(len(r["trajectories"]) for r in rows if r["trajectories"])
        self.assertEqual(out["verified_per_task"], {str(k): v for k, v in sorted(nv.items())})
        distinct = sum(len({t["turns"][-1]["content"] for t in r["trajectories"]}) >= 2 for r in rows)
        self.assertEqual(out["tasks_ge2_distinct"], distinct)
        self.assertLess(out["tasks_ge2_distinct"], out["tasks_ge2_verified"])     # the fixture has duplicates
        S = out["arms"]["S"]
        self.assertEqual(S["examples"], out["tasks_trained"])                       # one example per problem
        self.assertEqual(S["approx_steps"], math.ceil(S["examples"] / 8))
        self.assertEqual(S["tokens_per_turn"]["sum"], S["target_tokens_per_trajectory"]["sum"])   # single turn: exact
        self.assertTrue(out["recipe"]["arc_system_prompt_prepended"])
        same = 0
        for r in rows:                                   # brute force: stable sort by tokens, first vs last
            v = sorted(r["trajectories"], key=lambda t: t.get("tokens", 1e18))
            same += bool(v) and v[0]["turns"][-1]["content"] == v[-1]["turns"][-1]["content"]
        self.assertEqual(out["same_pick"]["S=L"]["tasks"], same)

    def test_cache_and_logs(self):
        try:
            import torch
        except Exception:  # noqa: BLE001
            self.skipTest("torch unavailable")
        mod = sft_student()
        cfg = NA.TRAIN["arc"]
        samples, skipped, n_traj = mod.build_samples(NA.sft_paths(cfg["data"]), CharTok(), 10 ** 9, {}, True, 0, 0, 1,
                                                     "shortest", cfg["task_cap"], 0)
        st = os.path.join(self.root, "students")
        os.makedirs(os.path.join(st, "brevSc"), exist_ok=True)
        torch.save((samples, skipped, n_traj), os.path.join(st, "brevSc", "samples_cache.pt"))
        os.makedirs("logs", exist_ok=True)
        open("logs/sftchain_brevSc.log", "w").write("2026-10-19_11:00:00 iter 1: job 777\n")
        open("logs/sft-777.out", "w").write("12 trajectories from 9 tasks -> 30 samples (skipped 2 > 24576 tokens) | "
                                            "target tokens 0.3M, context tokens 1.2M\nexcluded 5 samples whose task\n"
                                            "step 1/4 | loss 1.0\nstep 4/4 | loss 0.9\n")
        out = NA.training_set("arc", cfg, st)
        ex = out["arms"]["S"]["exact_from_cache"]
        bad = set(NA.read_ids(cfg["exclude"]))
        kept = [s for s in samples if not (s[2] in bad or s[2].split("_")[0] in bad)]
        self.assertEqual(ex["samples"], len(kept))
        self.assertEqual(ex["loss_tokens"], sum(len(s[1]) for s in kept))
        self.assertEqual(out["arms"]["S"]["steps"], math.ceil(len(kept) / 8))
        lg = out["arms"]["S"]["logs"]
        self.assertEqual(lg["dataset_line"]["samples"], 30)
        self.assertEqual(lg["steps_total"], [4])
        self.assertEqual(lg["excluded_samples"], 5)


if __name__ == "__main__":
    unittest.main()
