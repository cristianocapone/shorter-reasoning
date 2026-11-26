"""Tests for the NatComm Leonardo drivers (leonardo/natcomm_lib.sh, run_natcomm_*.sh, job_natcomm_*.slurm) on a laptop.

The library runs in bash against a stub Slurm (test_train_slurm_stub.py) that simulates queued jobs, the squeue glitch of
2026-10-30, deadline checkpoints, crashes after a periodic adapter save, evaluation jobs writing records, and resumable
forced-answer jobs. What must hold: TRAIN_DONE only after real completion (DONE in the log, sacct COMPLETED, last
checkpoint at the final step); merges only of the TRAIN_DONE adapter; evaluations marked only when complete with the
per-sample fields; forced answers resubmitted until every row is written; no double submission; budget cap respected.

    env -u PYTHONPATH .venv_mps/bin/python -m unittest discover -s synth/tests_natcomm -p 'test_train_*.py' -v
"""
import glob, json, os, shutil, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LEO = os.path.join(ROOT, "leonardo")
STUB = os.path.join(HERE, "test_train_slurm_stub.py")
BASH_FILES = sorted(glob.glob(os.path.join(LEO, "run_natcomm_*.sh")) + [os.path.join(LEO, "natcomm_lib.sh")]
                    + glob.glob(os.path.join(LEO, "job_natcomm_*.slurm")))

def rd(path):
    with open(path) as f:
        return f.read()


DONE_LOG = "resumed at step 100/169\nstep 169/169 | loss 0.31 | lr 0.00e+00 | 600s\ncheckpoint saved at step 169\nDONE (no merge)\n"
MID_LOG = "step 100/169 | loss 0.40 | lr 4e-05 | 900s\ncheckpoint saved at step 100\n"


class Env(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.t = self.tmp.name
        os.makedirs(os.path.join(t, "leonardo"))
        for f in glob.glob(os.path.join(LEO, "*natcomm*")) + [os.path.join(LEO, "job_sft.slurm"), os.path.join(LEO, "job_merge.slurm")]:
            shutil.copy(f, os.path.join(t, "leonardo"))
        os.symlink(os.path.join(ROOT, "synth"), os.path.join(t, "synth"))
        os.symlink(os.path.join(ROOT, "distill"), os.path.join(t, "distill"))
        os.makedirs(os.path.join(t, "bin"))
        for cmd in ("sbatch", "squeue", "sacct"):
            p = os.path.join(t, "bin", cmd)
            with open(p, "w") as f:
                f.write(f'#!/bin/sh\nexec "{sys.executable}" "{STUB}" {cmd} "$@"\n')
            os.chmod(p, 0o755)
        self.state = os.path.join(t, "stub_state.json")
        self.set_state(dict(next_id=5000, jobs={}, plans={}, submitted=[]))
        self.st = os.path.join(t, "students")
        os.makedirs(os.path.join(t, "base"))
        with open(os.path.join(t, "base", "config.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(t, "data.jsonl"), "w") as f:
            f.write('{"task_id": "a", "trajectories": []}\n')
        self.env = dict(PATH=os.path.join(t, "bin") + ":/usr/bin:/bin:/usr/sbin:/sbin", HOME=t, USER="tester",
                        WORK=os.path.join(t, "work"), NC_STUB_STATE=self.state, NC_POLL="0", NC_PAUSE="0",
                        NC_ST=self.st, NC_COST=os.path.join(t, "logs", "natcomm_cost.tsv"), NC_PY3=sys.executable,
                        NC_SUBMIT_TRIES="1", NC_WAIT_MAXMISS="5", LC_ALL="C")

    def tearDown(self):
        self.tmp.cleanup()

    # ------------------------------------------------------------------------------------------------ helpers
    def set_state(self, s):
        with open(self.state, "w") as f:
            json.dump(s, f)

    def get_state(self):
        with open(self.state) as f:
            return json.load(f)

    def plan(self, name, *behaviours):
        s = self.get_state(); s["plans"][name] = list(behaviours); self.set_state(s)

    def bash(self, snippet, extra_env=None):
        env = dict(self.env, **(extra_env or {}))
        return subprocess.run(["bash", "-c", "source leonardo/natcomm_lib.sh\n" + snippet], cwd=self.t, env=env,
                              capture_output=True, text=True, timeout=300)

    def submitted(self, name=None):
        return [x for x in self.get_state()["submitted"] if name is None or x["name"] == name]

    def path(self, *p):
        return os.path.join(self.t, *p)

    def adapter_files(self, tag, content):
        return {f"$ST/{tag}/adapter/adapter_model.safetensors": content}


class TestSyntax(unittest.TestCase):
    def test_bash_n(self):
        self.assertGreaterEqual(len(BASH_FILES), 9)
        for f in BASH_FILES:
            r = subprocess.run(["bash", "-n", f], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{f}: {r.stderr}")

    def test_tags_are_new(self):
        """New tags must not collide with the paper's: every agent tag, student and set is prefixed nc."""
        import re
        text = "".join(rd(f) for f in BASH_FILES)
        for tag in re.findall(r'nc_eval "?([A-Za-z0-9_${}#%:.-]+)', text) + re.findall(r'nc_student "?([A-Za-z0-9_${}#%:.-]+)', text):
            if tag.startswith("$"):
                continue
            self.assertTrue(tag.startswith(("nc32_", "nc8_", "nc_")), tag)
        with open(os.path.join(ROOT, "synth", "paper_stats.py")) as f:
            ps = f.read()
        self.assertNotIn("nc32_", ps); self.assertNotIn("nc8_", ps)


class TestTrain(Env):
    def test_two_jobs_then_done(self):
        self.plan("nc_sft_tq", dict(state="COMPLETED", log=MID_LOG, files=self.adapter_files("tq", "mid")),
                  dict(state="COMPLETED", log=DONE_LOG, files=self.adapter_files("tq", "final")))
        r = self.bash(f'nc_train tq {self.path("base")} {self.path("data.jsonl")} "--lr 5e-5 --no-merge"')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        done = rd(os.path.join(self.st, "tq", "TRAIN_DONE"))
        self.assertIn("steps=169", done)
        import hashlib
        self.assertIn("sha256=" + hashlib.sha256(b"final").hexdigest(), done)
        subs = self.submitted("nc_sft_tq")
        self.assertEqual(len(subs), 2)
        self.assertIn("--deadline-min 18", subs[0]["env"]["EXTRA"])
        self.assertIn("--qos=boost_qos_dbg", subs[0]["args"])
        rows = rd(self.env["NC_COST"]).strip().splitlines()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].split("\t")[1:4], ["CORE", "tq", "sft"])
        self.assertAlmostEqual(float(rows[0].split("\t")[7]), 16.0)          # billing 32 x 1800 s
        r2 = self.bash(f'nc_train tq {self.path("base")} {self.path("data.jsonl")} "--lr 5e-5"')
        self.assertIn("trained earlier", r2.stdout)                          # idempotent: no new job
        self.assertEqual(len(self.submitted("nc_sft_tq")), 2)

    def test_squeue_glitch_does_not_end_the_wait(self):
        """squeue shows nothing for 3 polls while the job still runs: waiting on squeue alone would read a log without
        DONE (the 2026-10-30 failure). nc_wait must keep waiting until sacct reports a terminal state."""
        self.plan("nc_sft_tg", dict(queued=1, glitch=3, state="COMPLETED", log=DONE_LOG, files=self.adapter_files("tg", "final")))
        r = self.bash(f'nc_train tg {self.path("base")} {self.path("data.jsonl")} "--lr 5e-5"')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(os.path.exists(os.path.join(self.st, "tg", "TRAIN_DONE")))
        self.assertEqual(self.get_state().get("glitches_served"), 3)

    def test_mid_training_adapter_is_not_done(self):
        """A crash after the periodic adapter save leaves adapter/ on disk: no TRAIN_DONE, and the merge refuses."""
        self.plan("nc_sft_tm", dict(state="FAILED", log="step 100/169 | loss 0.4\nTraceback (most recent call last):\n",
                                    files=self.adapter_files("tm", "mid")))
        r = self.bash(f'nc_train tm {self.path("base")} {self.path("data.jsonl")} "--lr 5e-5"; echo rc=$?; '
                      f'nc_merge tm {self.path("base")}; echo merge_rc=$?')
        self.assertIn("rc=1", r.stdout)
        self.assertIn("merge_rc=1", r.stdout)
        self.assertIn("refusing to merge", r.stdout)
        self.assertTrue(os.path.exists(os.path.join(self.st, "tm", "adapter", "adapter_model.safetensors")))
        self.assertFalse(os.path.exists(os.path.join(self.st, "tm", "TRAIN_DONE")))
        self.assertEqual(self.submitted("nc_merge_tm"), [])

    def test_done_in_log_but_job_failed(self):
        self.plan("nc_sft_tf", dict(state="FAILED", log=DONE_LOG, files=self.adapter_files("tf", "final")))
        r = self.bash(f'nc_train tf {self.path("base")} {self.path("data.jsonl")} "--lr 5e-5"; echo rc=$?')
        self.assertIn("rc=1", r.stdout)
        self.assertIn("no TRAIN_DONE", r.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.st, "tf", "TRAIN_DONE")))

    def test_done_without_final_checkpoint(self):
        log = "step 169/169 | loss 0.3\ncheckpoint saved at step 150\nDONE (no merge)\n"
        self.plan("nc_sft_tc", dict(state="COMPLETED", log=log, files=self.adapter_files("tc", "x")))
        r = self.bash(f'nc_train tc {self.path("base")} {self.path("data.jsonl")} "--lr 5e-5"; echo rc=$?')
        self.assertIn("rc=1", r.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.st, "tc", "TRAIN_DONE")))

    def test_sham_one_step_log_is_done(self):
        """The real log of sft_student.py --lr 0 --max-steps 1 (run locally) satisfies the DONE test."""
        log = ("trainable params: 65,536\nstep 1/1 | loss 11.8906 | lr 0.00e+00 | 21s\ncheckpoint saved at step 1\n"
               "DONE (no merge)\n")
        self.plan("nc_sft_ts", dict(state="COMPLETED", log=log, files=self.adapter_files("ts", "init")))
        r = self.bash(f'nc_train ts {self.path("base")} {self.path("data.jsonl")} "--lr 0 --max-steps 1"')
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("steps=1", rd(os.path.join(self.st, "ts", "TRAIN_DONE")))

    def test_guard_blocks_submission_over_cap(self):
        os.makedirs(self.path("logs"), exist_ok=True)
        with open(self.env["NC_COST"], "w") as f:
            f.write("d\tCORE\tx\teval\t1\tCOMPLETED\t1\t2000.0\t250.0\n")
        r = self.bash(f'nc_train tb {self.path("base")} {self.path("data.jsonl")} "--lr 5e-5"; echo rc=$?',
                      dict(NC_CAP_CORE="1500"))
        self.assertIn("budget guard", r.stdout)
        self.assertIn("rc=1", r.stdout)
        self.assertEqual(self.submitted(), [])
        r = self.bash('nc_guard; echo rc=$?', dict(NC_CAP_CORE="1500", NC_TIER="EXT"))
        self.assertIn("rc=0", r.stdout)                                     # other tier: not blocked


class TestMerge(Env):
    def trained(self, tag):
        self.plan(f"nc_sft_{tag}", dict(state="COMPLETED", log=DONE_LOG, files=self.adapter_files(tag, "final")))
        r = self.bash(f'nc_train {tag} {self.path("base")} {self.path("data.jsonl")} "--lr 5e-5"')
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_merge_of_the_trained_adapter(self):
        self.trained("mq")
        self.plan("nc_merge_mq", dict(state="COMPLETED", log="MERGED /x\n",
                                      files={"$ST/mq/merged/config.json": "{}", "$ST/mq/merged/model.safetensors": "w"}))
        r = self.bash(f'nc_merge mq {self.path("base")}')
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertTrue(os.path.exists(os.path.join(self.st, "mq", "MERGE_DONE")))
        env = self.submitted("nc_merge_mq")[0]["env"]
        self.assertEqual(env["ADAPTER"], os.path.join(self.st, "mq", "adapter"))
        r = self.bash(f'nc_student mq {self.path("base")} {self.path("data.jsonl")} "--lr 5e-5"')
        self.assertIn("merged earlier", r.stdout)

    def test_adapter_changed_after_train_done(self):
        self.trained("mc")
        with open(os.path.join(self.st, "mc", "adapter", "adapter_model.safetensors"), "w") as f:
            f.write("overwritten by a stray job")
        r = self.bash(f'nc_merge mc {self.path("base")}; echo rc=$?')
        self.assertIn("adapter changed after TRAIN_DONE", r.stdout)
        self.assertEqual(self.submitted("nc_merge_mc"), [])

    def test_merge_without_MERGED_line(self):
        self.trained("mf")
        self.plan("nc_merge_mf", dict(state="COMPLETED", log="Traceback: out of memory\n"))
        r = self.bash(f'nc_merge mf {self.path("base")}; echo rc=$?')
        self.assertIn("rc=1", r.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.st, "mf", "MERGE_DONE")))


class TestEval(Env):
    def setUp(self):
        super().setUp()
        os.makedirs(self.path("model"))
        with open(self.path("model", "config.json"), "w") as f:
            f.write("{}")
        with open(self.path("ids.txt"), "w") as f:
            f.write("".join(f"p{i:02d}\n" for i in range(10)))       # shards of 3, 3, 2, 2 problems
        self.e = ('E_SOLVER=synth/math_harvest.py E_DATA=d.jsonl E_IDS=ids.txt E_K=4 E_MAXTOK=32768 E_MAXLEN=34816 '
                  'E_TEMP=0.6 E_NSH=4 E_TIME=03:00:00 E_SLICE=165 E_TEXTS=1 ')

    def test_array_resubmits_only_unfinished_shards(self):
        self.plan("nc_ev_te", dict(eval=dict(per_shard=2)), dict(eval=dict()))
        r = self.bash(self.e + "E_MODE=array; nc_eval te model 1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        subs = self.submitted("nc_ev_te")
        self.assertEqual(len(subs), 2)
        self.assertIn("--array=0,1,2,3", subs[0]["args"])
        self.assertIn("--array=0,1", subs[1]["args"])                        # shards 2 and 3 were complete
        self.assertIn("--gres=gpu:1", subs[0]["args"])
        self.assertEqual(subs[0]["env"]["NC_SEED"], "1")
        self.assertEqual(subs[0]["env"]["NC_TEXTS"], "1")
        done = rd(self.path("runs_synth", "agent_te", "EVAL_DONE"))
        self.assertIn("10 problems", done)
        self.assertIn("finish 1.0000 (fins)", done)
        r = self.bash(self.e + "E_MODE=array; nc_eval te model 1")
        self.assertIn("evaluated earlier", r.stdout)

    def test_dbg_mode_one_gpu_per_unfinished_shard(self):
        self.plan("nc_ev_td", dict(eval=dict(per_shard=2)), dict(eval=dict()))
        r = self.bash(self.e + "E_MODE=dbg E_SLICE=24; nc_eval td model 0")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        subs = self.submitted("nc_ev_td")
        self.assertIn("--gres=gpu:4", subs[0]["args"])
        self.assertIn("--qos=boost_qos_dbg", subs[0]["args"])
        self.assertIn("--gres=gpu:2", subs[1]["args"])
        self.assertIn("--mem=240G", subs[1]["args"])
        self.assertEqual(subs[1]["env"]["NC_SHARDS"], "0:1")

    def test_records_without_per_sample_fields_stop(self):
        self.plan("nc_ev_tn", dict(eval=dict(fields=False)), dict(eval=dict()))
        r = self.bash(self.e + "E_MODE=array; nc_eval tn model 0; echo rc=$?")
        self.assertIn("rc=1", r.stdout)
        self.assertIn("predates the NatComm interface", r.stdout)
        self.assertEqual(len(self.submitted("nc_ev_tn")), 1)
        self.assertFalse(os.path.exists(self.path("runs_synth", "agent_tn", "EVAL_DONE")))

    def test_no_progress_stops(self):
        self.plan("nc_ev_ts", dict(eval=dict(per_shard=0)), dict(eval=dict(per_shard=0)), dict(eval=dict()))
        r = self.bash(self.e + "E_MODE=array; nc_eval ts model 0; echo rc=$?")
        self.assertIn("no progress", r.stdout)
        self.assertEqual(len(self.submitted("nc_ev_ts")), 2)

    def test_crash_stops(self):
        self.plan("nc_ev_tc", dict(eval=dict(per_shard=1, solver_log="Traceback (most recent call last):\nValueError: x\n")))
        r = self.bash(self.e + "E_MODE=array; nc_eval tc model 0; echo rc=$?")
        self.assertIn("crash in", r.stdout)
        self.assertIn("rc=1", r.stdout)

    def test_cancelled_shutdown_is_not_a_crash(self):
        log = "Traceback (most recent call last):\nRuntimeError: cancelled\n"
        self.plan("nc_ev_tk", dict(eval=dict(per_shard=2, solver_log=log)), dict(eval=dict()))
        r = self.bash(self.e + "E_MODE=array; nc_eval tk model 0")
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_missing_model(self):
        r = self.bash(self.e + "E_MODE=array; nc_eval tx nowhere 0; echo rc=$?")
        self.assertIn("not found", r.stdout)
        self.assertEqual(self.submitted(), [])


class TestForced(Env):
    def setUp(self):
        super().setUp()
        os.makedirs(self.path("runs_synth", "agent_tf"))
        with open(self.path("runs_synth", "agent_tf", "EVAL_DONE"), "w") as f:
            f.write("done\n")

    def test_resubmits_until_all_rows(self):
        fo = dict(n_samples=4, budgets=[4096, 8192])
        self.plan("nc_fa_tf", dict(forced=dict(fo, rows_per_job=5)), dict(forced=dict(fo)))
        r = self.bash('nc_forced tf model 4096,8192')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(self.submitted("nc_fa_tf")), 2)
        self.assertIn("rows=8", rd(self.path("runs_synth", "agent_tf", "FORCED_DONE")))
        env = self.submitted("nc_fa_tf")[0]["env"]
        self.assertEqual(env["NC_TEXTS_GLOB"], "runs_synth/agent_tf/shard_*.texts.jsonl.gz")
        self.assertEqual(env["NC_SLICE"], "105")

    def test_needs_finished_eval(self):
        r = self.bash('nc_forced other model 4096; echo rc=$?')
        self.assertIn("no EVAL_DONE", r.stdout)
        self.assertEqual(self.submitted(), [])

    def test_no_progress_stops(self):
        fo = dict(n_samples=4, budgets=[4096], rows_per_job=0)
        self.plan("nc_fa_tf", dict(forced=fo), dict(forced=fo))
        r = self.bash('nc_forced tf model 4096; echo rc=$?')
        self.assertIn("rc=1", r.stdout)
        self.assertEqual(len(self.submitted("nc_fa_tf")), 1)


class TestQueueAndLane(Env):
    def test_submit_adopts_running_job_of_same_name(self):
        s = self.get_state()
        s["jobs"]["4242"] = dict(name="nc_sft_ta", args=[], behaviour=dict(state="COMPLETED"), env={}, finished=False,
                                 array=None, queued=0, glitch=0, adoptable=True)
        self.set_state(s)
        r = self.bash("nc_submit nc_sft_ta --qos=normal x.slurm")
        self.assertEqual(r.stdout.strip(), "4242")
        self.assertEqual(self.submitted(), [])

    def test_lock(self):
        r = self.bash("nc_lock lane1 && echo first; ( nc_lock lane1 && echo second || echo refused )")
        self.assertIn("first", r.stdout)
        self.assertIn("refused", r.stdout)
        os.makedirs(self.path("logs", "locks", "lane2"))
        host = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
        with open(self.path("logs", "locks", "lane2", "owner"), "w") as f:
            f.write(f"{host} 999999 2026-11-02\n")                          # dead pid on this node
        r = self.bash("nc_lock lane2 && echo taken")
        self.assertIn("stale lock", r.stdout)
        self.assertIn("taken", r.stdout)
        self.assertFalse(os.path.exists(self.path("logs", "locks", "lane2")))   # released at exit

    def test_try_stops_after_two_failures(self):
        r = self.bash("nc_try false; echo after_one; nc_try true; nc_try false; echo after_reset; nc_try false; echo unreachable")
        self.assertIn("after_one", r.stdout)
        self.assertIn("after_reset", r.stdout)
        self.assertNotIn("unreachable", r.stdout)
        self.assertEqual(r.returncode, 1)


class TestDrivers(Env):
    def driver(self, name, *args):
        return subprocess.run(["bash", os.path.join(self.t, "leonardo", name), *args], cwd=self.t, env=self.env,
                              capture_output=True, text=True, timeout=120)

    def test_q4_train_needs_the_sets(self):
        r = self.driver("run_natcomm_q4.sh", "train")
        self.assertEqual(r.returncode, 1)
        self.assertIn("run leonardo/run_natcomm_sets.sh first", r.stdout)
        self.assertEqual(self.submitted(), [])

    def test_unknown_lane(self):
        r = self.driver("run_natcomm_q4.sh", "bogus")
        self.assertEqual(r.returncode, 1)
        self.assertIn("unknown lane", r.stdout)

    def test_l8_refuses_without_tokenizer_workaround(self):
        os.makedirs(os.path.join(self.env["WORK"], "ccapone0", "hf", "r1_llama_8b"))
        with open(os.path.join(self.env["WORK"], "ccapone0", "hf", "r1_llama_8b", "tokenizer_config.json"), "w") as f:
            json.dump({"tokenizer_class": "LlamaTokenizerFast"}, f)
        r = self.driver("run_natcomm_l8.sh", "sham")
        self.assertEqual(r.returncode, 1)
        self.assertIn("PreTrainedTokenizerFast", r.stdout)
        self.assertEqual(self.submitted(), [])

    def test_q4_eval_lane_order_and_settings(self):
        """e32s0 with every eval already marked: walks the arms in order, waits for nothing, submits nothing."""
        st = self.st
        for arm, model in (("base", None), ("S", "math4b_S"), ("R", "math4b_R"), ("L", "math4b_L"), ("KA", "nc_q4_KA"),
                           ("sham", "nc_q4_sham"), ("Rtm", "nc_q4_Rtm"), ("Ltm", "nc_q4_Ltm"), ("baseT04", None)):
            d = self.path("runs_synth", f"agent_nc32_q4_{arm}_s0")
            os.makedirs(d)
            for m in ("EVAL_DONE", "FORCED_DONE"):
                open(os.path.join(d, m), "w").close()
            if model:
                os.makedirs(os.path.join(st, model, "merged"))
                open(os.path.join(st, model, "MERGE_DONE"), "w").close()
        r = self.driver("run_natcomm_q4.sh", "e32s0")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        order = [l.split()[1].rstrip(":") for l in r.stdout.splitlines() if "evaluated earlier" in l]
        self.assertEqual(order, [f"nc32_q4_{a}_s0" for a in ("base", "S", "R", "L", "KA", "sham", "Rtm", "Ltm", "baseT04")])
        self.assertEqual(self.submitted(), [])


QWEN25 = (glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*")) or [None])[0]


@unittest.skipIf(QWEN25 is None, "needs the cached Qwen2.5-0.5B-Instruct tokenizer")
class TestSetsDriver(Env):
    """run_natcomm_sets.sh end to end on the login-node path (NC_SERIAL=0): the three builder command lines as written in
    the driver (quoted globs, SFT_NO_SYS / SFT_END for R1-Llama, --verify against sft_student.py), on fixture harvests,
    with the cached Qwen2.5 tokenizer standing in for both models (renamed to PreTrainedTokenizerFast for the Llama slot)."""

    def test_builds_and_verifies_all_three(self):
        sys.path.insert(0, HERE)
        from test_train_fixture import make_harvest
        for tag in ("math_harv_4b", "math_harv_l8", "code_harv_4b"):
            make_harvest(self.path("runs_synth", f"agent_{tag}"), n_problems=30)
        hf = os.path.join(self.env["WORK"], "ccapone0", "hf")
        for name in ("qwen3_4b_thinking", "r1_llama_8b"):
            os.makedirs(os.path.join(hf, name))
            for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
                shutil.copy(os.path.realpath(os.path.join(QWEN25, f)), os.path.join(hf, name, f))
        cfg = os.path.join(hf, "r1_llama_8b", "tokenizer_config.json")
        c = json.loads(rd(cfg)); c["tokenizer_class"] = "PreTrainedTokenizerFast"
        with open(cfg, "w") as f:
            json.dump(c, f)
        env = dict(self.env, NC_SERIAL="0", NC_PY=sys.executable, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        r = subprocess.run(["bash", os.path.join(self.t, "leonardo", "run_natcomm_sets.sh")], cwd=self.t, env=env,
                           capture_output=True, text=True, timeout=900)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        out = self.path("runs_synth", "natcomm_sets")
        for prefix, sets in (("q4", "S R L KA Rtm Ltm Sk2"), ("l8", "S R L KA Rtm Ltm Sk2"), ("c4", "S R L KA")):
            st = json.loads(rd(os.path.join(out, f"{prefix}_stats.json")))
            self.assertEqual(sorted(st["sets"]), sorted(sets.split()), prefix)
            self.assertTrue(st["params"]["exact_tokens"])
            self.assertTrue(all(st["verify"].values()), prefix)
            self.assertEqual(st["params"]["no_sys"], prefix == "l8")
            self.assertEqual(st["params"]["sft_end"], "<｜end▁of▁sentence｜>" if prefix == "l8" else "<|im_end|>\n")
        self.assertIn("sets built: 18 files", r.stdout)
        self.assertEqual(self.submitted(), [])                               # nothing went to Slurm


if __name__ == "__main__":
    unittest.main()
