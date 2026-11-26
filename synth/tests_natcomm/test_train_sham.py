"""End-to-end test of the zero-step (sham) control through the real pipeline, on CPU in about a minute.

A tiny random Qwen2-architecture model (2 layers, width 64) with the real Qwen2.5 tokenizer files (cached locally) goes
through distill/sft_student.py --lr 0 --max-steps 1 --no-merge (the sham arm's arguments), distill/merge_adapter.py and
synth/natcomm_sham_check.py --weights. The merged weights must equal the base bit for bit and the tokenizer must behave
identically; with --lr 1e-2 (one real step) the same check must fail, so it is not vacuous. The training log must also pass
the DONE test of leonardo/natcomm_lib.sh (last checkpoint at the final step).
Skipped without torch / peft / transformers or the cached Qwen/Qwen2.5-0.5B-Instruct tokenizer.
"""
import glob, json, os, shutil, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from test_train_fixture import make_harvest   # noqa: E402

SNAP = (glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*")) or [None])[0]
try:
    import torch, peft, transformers           # noqa: F401
    HAVE = SNAP is not None
except Exception:
    HAVE = False


@unittest.skipUnless(HAVE, "needs torch, peft, transformers and the cached Qwen2.5-0.5B-Instruct tokenizer")
class TestShamPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = cls.t = cls.tmp.name
        cls.env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        cls.env.pop("PYTHONPATH", None)
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32768, tie_word_embeddings=True)
        Qwen2ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(os.path.join(t, "base"))
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
            shutil.copy(os.path.realpath(os.path.join(SNAP, f)), os.path.join(t, "base", f))
        make_harvest(os.path.join(t, "harv"), n_problems=10)
        subprocess.run([sys.executable, os.path.join(ROOT, "synth", "natcomm_sft_sets.py"), "--harvest",
                        os.path.join(t, "harv", "shard_*.jsonl"), "--prefix", "sh", "--out-dir", os.path.join(t, "sets"),
                        "--sets", "S"], check=True, capture_output=True, env=cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pipeline(self, name, lr):
        t = self.t
        out, merged = os.path.join(t, name), os.path.join(t, name + "_merged")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "distill", "sft_student.py"), "--model", os.path.join(t, "base"),
                            "--data", os.path.join(t, "sets", "sh_S.jsonl"), "--out", out, "--max-len", "12288",
                            "--lora-r", "32", "--lora-alpha", "64", "--lr", lr, "--accum", "2", "--max-steps", "1", "--no-merge"],
                           capture_output=True, text=True, env=self.env, timeout=900)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        log = r.stdout + r.stderr
        subprocess.run([sys.executable, os.path.join(ROOT, "distill", "merge_adapter.py"), os.path.join(t, "base"),
                        os.path.join(out, "adapter"), merged], check=True, capture_output=True, env=self.env, timeout=900)
        c = subprocess.run([sys.executable, os.path.join(ROOT, "synth", "natcomm_sham_check.py"), os.path.join(t, "base"),
                            merged, "--weights", "--json", merged + ".json"], capture_output=True, text=True, env=self.env,
                           timeout=900)
        with open(merged + ".json") as f:
            return log, c, json.load(f)

    def test_zero_step_adapter_is_the_identity(self):
        log, c, rep = self.pipeline("sham", "0")
        self.assertEqual(c.returncode, 0, c.stdout)
        self.assertIn("SHAM CHECK PASS", c.stdout)
        w = rep["weights"]
        self.assertTrue(w["identical"])
        self.assertEqual(w["exact"], w["n_base"])
        self.assertTrue(rep["tokenizer"]["chat_prompt_same"] and rep["tokenizer"]["chat_prompt_ids_same"])
        # the DONE test of natcomm_lib.sh nc_train on the real log
        self.assertRegex(log, r"(?m)^DONE")
        self.assertIn("step 1/1 |", log)
        self.assertIn("checkpoint saved at step 1", log)
        self.assertIn("lr 0.00e+00", log)

    def test_one_real_step_fails_the_check(self):
        log, c, rep = self.pipeline("real", "1e-2")
        self.assertEqual(c.returncode, 1, c.stdout)
        self.assertIn("SHAM CHECK FAIL", c.stdout)
        self.assertGreater(len(rep["weights"]["differ"]), 0)


if __name__ == "__main__":
    unittest.main()
