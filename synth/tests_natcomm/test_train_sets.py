"""Tests for synth/natcomm_sft_sets.py on the synthetic harvest of test_train_fixture.py (no GPU, no network).

    env -u PYTHONPATH .venv_mps/bin/python -m unittest discover -s synth/tests_natcomm -p 'test_train_*.py' -v

The S/R/L reproduction and --verify tests import distill/sft_student.py (needs torch, CPU is enough). The real-tokenizer
test uses the locally cached Qwen/Qwen2.5-0.5B-Instruct tokenizer (same BPE family as Qwen3) and is skipped without it.
"""
import argparse, glob, io, json, math, os, random, statistics, sys, tempfile, unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "synth"))
sys.path.insert(0, os.path.join(ROOT, "distill"))
import natcomm_sft_sets as B                       # noqa: E402
from test_train_fixture import FakeTok, make_harvest, make_multiturn   # noqa: E402

try:
    import sft_student as SS                       # noqa: E402
except Exception as e:                             # torch missing
    SS = None
    SS_ERR = repr(e)

QWEN25 = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots")
QWEN25 = (glob.glob(os.path.join(QWEN25, "*")) or [None])[0]


def args_for(harvest, out, **kw):
    a = B.parse(["--harvest", harvest, "--prefix", kw.pop("prefix", "fx"), "--out-dir", out])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.hdir = os.path.join(self.d, "agent_math_harv_fx")
        self.pooled = make_harvest(self.hdir)
        self.glob = os.path.join(self.hdir, "shard_*.jsonl")
        self.out = os.path.join(self.d, "sets")
        os.environ.pop("SFT_NO_SYS", None); os.environ.pop("SFT_END", None)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, tok=None, **kw):
        a = args_for(self.glob, self.out, **kw)
        with redirect_stdout(io.StringIO()):
            return a, B.build(a, tok)

    def read(self, name, prefix="fx"):
        with open(os.path.join(self.out, f"{prefix}_{name}.jsonl")) as f:
            recs = [json.loads(l) for l in f]
        return {r["task_id"]: r["trajectories"] for r in recs}


class TestHarvestPaths(Base):
    def test_globs_sorted_and_repeat(self):
        p = B.harvest_paths(self.glob)
        self.assertEqual([os.path.basename(x) for x in p], ["shard_0.jsonl", "shard_1.jsonl"])
        p2 = B.harvest_paths(self.glob + "@2," + os.path.join(self.hdir, "shard_0.jsonl"))
        self.assertEqual(len(p2), 5)

    def test_pooled_candidates_follow_file_order(self):
        cand, n_rec = B.load_candidates(B.harvest_paths(self.glob))
        self.assertEqual(cand, self.pooled)
        self.assertGreater(len(self.pooled["tr_algebra_003"]), 0)


@unittest.skipIf(SS is None, "distill/sft_student.py not importable (torch?)")
class TestReproducesSftStudent(Base):
    """The S/R/L files must give sft_student.py exactly the samples that --shortest-per-task --select X --seed s gives."""

    def test_selection_identical_for_seeds(self):
        tok = FakeTok()
        for seed in (0, 1, 2):
            out_dir = os.path.join(self.d, f"sets_seed{seed}")
            a = args_for(self.glob, out_dir, seed=seed, sets="S,R,L")
            with redirect_stdout(io.StringIO()):
                B.build(a, None)
            paths = B.harvest_paths(self.glob)
            for name, sel in (("S", "shortest"), ("R", "random"), ("L", "longest")):
                want = SS.build_samples(paths, tok, 10 ** 9, B.TKW, shortest=True, select=sel, seed=seed)
                got = SS.build_samples([os.path.join(out_dir, f"fx_{name}.jsonl")], tok, 10 ** 9, B.TKW)
                self.assertEqual(want, got, f"{name} seed {seed}")

    def test_random_arm_redraws_with_seed(self):
        cand, _ = B.load_candidates(B.harvest_paths(self.glob))
        r0, r1 = B.select_one(cand, "random", 0), B.select_one(cand, "random", 1)
        multi = [t for t, v in cand.items() if len(v) >= 2]
        self.assertTrue(any(r0[t][0] is not r1[t][0] for t in multi))
        s0, s1 = B.select_one(cand, "shortest", 0), B.select_one(cand, "shortest", 1)
        self.assertTrue(all(s0[t][0] is s1[t][0] for t in cand))      # S and L do not depend on the seed

    def test_verify_with_fake_tokenizer_and_skips(self):
        tok = FakeTok(dict_ctx=False)
        a, out = self.build(tok, verify=True, max_len=300)            # small max_len: long examples are skipped
        self.assertTrue(all(out["verify"].values()))
        self.assertGreater(out["sets"]["L"]["skipped_over_max_len"], 0)
        self.assertTrue(out["params"]["exact_tokens"])

    def test_multiturn_counts_match_build_samples(self):
        tok = FakeTok()
        mdir = os.path.join(self.d, "arc")
        make_multiturn(mdir)
        cnt = B.Counter(tok, "<|im_end|>\n", 2, False, 0)
        cand, _ = B.load_candidates([os.path.join(mdir, "shard_0.jsonl")])
        samples, skipped, n_traj = SS.build_samples([os.path.join(mdir, "shard_0.jsonl")], tok, 10 ** 9, B.TKW)
        mine = [(c, t) for tid in cand for tr in cand[tid] for c, t in cnt.examples(tr)]
        self.assertEqual(mine, [(len(c), len(t)) for c, t, _ in samples])


class TestKeepAll(Base):
    def test_distinct_solutions_only(self):
        a, out = self.build()
        ka = self.read("KA")
        for tid, v in self.pooled.items():
            norm = [B.norm_text(B.assistant_text(tr)) for tr in v]
            self.assertEqual(len(ka.get(tid, [])), len(set(norm)), tid)
            self.assertEqual(len({B.norm_text(B.assistant_text(tr)) for tr in ka.get(tid, [])}), len(ka.get(tid, [])))
        h = out["harvest"]
        self.assertGreater(h["duplicates_removed"], 0)
        self.assertEqual(h["verified_trajectories"] - h["duplicates_removed"], h["distinct_trajectories"])
        self.assertEqual(out["sets"]["KA"]["examples"], h["distinct_trajectories"])
        self.assertEqual(set(ka), {t for t, v in self.pooled.items() if v})

    def test_whitespace_variant_is_a_duplicate(self):
        tr = self.pooled[next(t for t, v in self.pooled.items() if v)][0]
        twin = json.loads(json.dumps(tr))
        twin["turns"][-1]["content"] = "  " + tr["turns"][-1]["content"].replace(" ", "\n ") + "\t"
        out, dups = B.keep_all({"t": [tr, twin]})
        self.assertEqual((len(out["t"]), dups), (1, 1))


class TestTokenMatched(Base):
    def test_matches_S_tokens_and_nests(self):
        a, out = self.build()
        st = out["sets"]
        cand, _ = B.load_candidates(B.harvest_paths(self.glob))
        cnt = B.Counter(None, "<|im_end|>\n", 2)
        S, R, L = (B.select_one(cand, x, 0) for x in ("shortest", "random", "longest"))
        biggest = max(cnt.loss(v[0]) for X in (R, L) for v in X.values())
        for name in ("Rtm", "Ltm"):
            self.assertLessEqual(abs(st[name]["loss_tokens"] - st["S"]["loss_tokens"]), biggest / 2 + 1, name)
            self.assertLess(st[name]["problems"], st["S"]["problems"])
            self.assertLess(st[name]["optimizer_steps"], st["S"]["optimizer_steps"] + 1)
        rtm, ltm = self.read("Rtm"), self.read("Ltm")
        self.assertTrue(set(ltm) <= set(rtm) <= set(S))
        self.assertTrue(all(rtm[t] == [json.loads(json.dumps(R[t][0]))] for t in rtm))   # R's own draw, not a new one
        self.assertTrue(all(ltm[t] == [json.loads(json.dumps(L[t][0]))] for t in ltm))

    def test_permutation_seed_changes_subset(self):
        cand, _ = B.load_candidates(B.harvest_paths(self.glob))
        cnt = B.Counter(None)
        R = B.select_one(cand, "random", 0)
        target = sum(cnt.loss(v[0]) for v in B.select_one(cand, "shortest", 0).values())
        subsets = []
        for s in (1, 2):
            order = sorted(cand); random.Random(s).shuffle(order)
            subsets.append(set(B.token_matched(R, cnt.loss, target, order)[0]))
        self.assertNotEqual(*subsets)

    def test_target_above_total_takes_everything(self):
        cand, _ = B.load_candidates(B.harvest_paths(self.glob))
        cnt = B.Counter(None)
        R = B.select_one(cand, "random", 0)
        sel, tot = B.token_matched(R, cnt.loss, 10 ** 12, sorted(cand))
        self.assertEqual(set(sel), set(cand))


class TestSk2(Base):
    def test_dose_between_R_and_S(self):
        a, out = self.build()
        st = out["sets"]
        self.assertEqual(st["Sk2"]["problems"], st["S"]["problems"])
        self.assertLess(st["S"]["target_mean"], st["Sk2"]["target_mean"])
        self.assertLess(st["Sk2"]["target_mean"], st["R"]["target_mean"])
        self.assertLess(st["S"]["length_tilt_mean"], st["Sk2"]["length_tilt_mean"])
        self.assertLess(st["Sk2"]["length_tilt_mean"], st["L"]["length_tilt_mean"])

    def test_per_problem_rules(self):
        cand, _ = B.load_candidates(B.harvest_paths(self.glob))
        S = B.select_one(cand, "shortest", 0)
        K2 = B.shortest_of_k(cand, 2, 5)
        for t, v in cand.items():
            if len(v) == 1:
                self.assertIs(K2[t][0], v[0])
            if len(v) <= 2:                     # shortest of all == shortest of 2 (ties: same length)
                self.assertEqual(K2[t][0]["tokens"], S[t][0]["tokens"])
            self.assertGreaterEqual(K2[t][0]["tokens"], S[t][0]["tokens"])
        self.assertEqual(B.shortest_of_k(cand, 2, 5), K2)          # deterministic


class TestStatsAndFiles(Base):
    def test_fallback_counts_and_steps(self):
        a, out = self.build()
        S = self.read("S")
        loss = sum(tr["tokens"] + 2 for v in S.values() for tr in v)
        st = out["sets"]["S"]
        self.assertEqual(st["loss_tokens"], loss)
        self.assertEqual(st["examples"], len(S))
        self.assertEqual(st["optimizer_steps"], math.ceil(len(S) / 8))
        self.assertEqual(st["loss_tokens_vs_S"], 1.0)
        self.assertGreater(out["sets"]["L"]["loss_tokens_vs_S"], 1.0)
        self.assertGreater(out["sets"]["KA"]["examples"], st["examples"])
        h = out["harvest"]
        self.assertEqual(sum(h["problems_by_n_verified"].values()), h["problems_with_verified"])
        self.assertFalse(out["params"]["exact_tokens"])
        self.assertIsNone(st["skipped_over_max_len"])
        lens = sorted(tr["tokens"] + 2 for v in S.values() for tr in v)
        self.assertAlmostEqual(st["target_mean"], round(statistics.mean(lens), 1))

    def test_rewrite_is_idempotent_and_guarded(self):
        self.build()
        a, out = self.build()
        self.assertTrue(all(f["status"] == "unchanged" for f in out["files"].values()))
        with self.assertRaises(SystemExit):
            self.build(seed=3)                    # R changes -> refuse to overwrite a set a training may have read
        a, out = self.build(seed=3, force=True)
        self.assertEqual(out["files"]["R"]["status"], "written")

    def test_cli_prints_table(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            B.main(["--harvest", self.glob, "--prefix", "cli", "--out-dir", self.out, "--sets", "S,R,L,KA"])
        txt = buf.getvalue()
        self.assertIn("KA", txt)
        self.assertIn("approximate loss tokens", txt)
        with open(os.path.join(self.out, "cli_stats.json")) as f:
            st = json.load(f)
        self.assertEqual(sorted(st["sets"]), ["KA", "L", "R", "S"])

    def test_empty_harvest_fails(self):
        with self.assertRaises(SystemExit):
            B.build(args_for(os.path.join(self.d, "nothing_*.jsonl"), self.out))

    def test_unknown_set_name_fails(self):
        with self.assertRaises(SystemExit):
            self.build(sets="S,KEEPALL")

    def test_llama_end_token_and_no_sys(self):
        a, out = self.build(end_tokens=1, no_sys=True, prefix="l8")
        S = self.read("S", "l8")
        self.assertEqual(out["sets"]["S"]["loss_tokens"], sum(tr["tokens"] + 1 for v in S.values() for tr in v))
        self.assertTrue(out["params"]["no_sys"])


@unittest.skipIf(SS is None or QWEN25 is None, "needs torch + the cached Qwen2.5-0.5B-Instruct tokenizer")
class TestRealTokenizer(Base):
    def test_exact_counts_and_verify(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        buf = io.StringIO()
        with redirect_stdout(buf):
            B.main(["--harvest", self.glob, "--prefix", "qw", "--out-dir", self.out, "--tokenizer", QWEN25,
                    "--max-len", "12288", "--verify"])
        with open(os.path.join(self.out, "qw_stats.json")) as f:
            st = json.load(f)
        self.assertTrue(st["params"]["exact_tokens"])
        self.assertTrue(all(st["verify"].values()), st["verify"])
        self.assertIn("verify:", buf.getvalue())
        # exact loss tokens differ from the harvest's synthetic counts but keep the ordering S < R < L
        s = st["sets"]
        self.assertLess(s["S"]["loss_tokens"], s["R"]["loss_tokens"])
        self.assertLess(s["R"]["loss_tokens"], s["L"]["loss_tokens"])


if __name__ == "__main__":
    unittest.main()
