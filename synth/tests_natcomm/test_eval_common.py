"""synth/eval_common.py: --save-texts paths, the gzip texts file (append, damaged member, duplicates), the context check
behind max_model_len, and char -> token mapping (exact on a character tokenizer, checked on a real one if cached)."""
import fnmatch, gzip, io, json, os, random, tempfile, unittest
from contextlib import redirect_stdout

import test_eval_util as U
import eval_common as ec


class CharTok:
    """One token per character; enough for char_to_token."""
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class TestTextsPath(unittest.TestCase):
    def test_resolution(self):
        out = "runs_synth/agent_X/shard_1.jsonl"
        self.assertIsNone(ec.texts_path(None, out, 1, 2))
        self.assertIsNone(ec.texts_path("", out, 1, 2))
        self.assertEqual(ec.texts_path("auto", out, 1, 2), "runs_synth/agent_X/shard_1.texts.jsonl.gz")
        self.assertEqual(ec.texts_path("t/{shard}.jsonl.gz", out, 1, 2), "t/1.jsonl.gz")
        self.assertEqual(ec.texts_path("{out_dir}/texts.jsonl.gz", out, 1, 1), "runs_synth/agent_X/texts.jsonl.gz")
        self.assertEqual(ec.texts_path("{out_dir}/texts.jsonl.gz", out, 1, 2), "runs_synth/agent_X/texts.shard1.jsonl.gz")
        self.assertEqual(ec.texts_path("texts.gz", out, 0, 3), "texts.shard0.gz")
        with self.assertRaises(ValueError):
            ec.texts_path("texts.jsonl", out, 0, 1)

    def test_texts_stay_out_of_record_globs(self):
        """chain_dbg.sh and paper_stats.py read runs_synth/agent_TAG/shard_*.jsonl."""
        for spec in ("auto", "{out_dir}/texts_{shard}.jsonl.gz", "{out_stem}.jsonl.gz"):
            p = ec.texts_path(spec, "runs_synth/agent_X/shard_0.jsonl", 0, 2)
            self.assertFalse(fnmatch.fnmatch(os.path.basename(p), "shard_*.jsonl"), p)


class TestTextsFile(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def rows(self, pid, n=2, tag=""):
        return [dict(task_id=pid, k=k, text=f"{pid}-{k}{tag}", n_tokens=10 + k, prompt_tokens=3, finish_reason="stop",
                     correct=True) for k in range(n)]

    def test_append_members_and_keep_last_copy(self):
        p = os.path.join(self.d, "sub", "t.jsonl.gz")
        ec.append_texts(p, self.rows("a"))
        ec.append_texts(p, self.rows("b"))
        ec.append_texts(p, self.rows("a", tag="-redo"))           # problem redone after a crash
        with gzip.open(p, "rt") as f:
            self.assertEqual(len(f.readlines()), 6)               # three complete gzip members
        recs = {(r["task_id"], r["k"]): r for r in ec.read_texts(p)}
        self.assertEqual(len(recs), 4)
        self.assertEqual(recs["a", 0]["text"], "a-0-redo")
        self.assertEqual(recs["b", 1]["text"], "b-1")

    def test_damaged_final_member(self):
        p = os.path.join(self.d, "t.jsonl.gz")
        ec.append_texts(p, self.rows("a"))
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as g:
            g.write((json.dumps(self.rows("b")[0]) + "\n").encode() * 50)
        with open(p, "ab") as f:
            f.write(buf.getvalue()[:len(buf.getvalue()) // 2])      # job killed while writing
        with redirect_stdout(io.StringIO()) as log:
            recs = ec.read_texts(p)
        self.assertEqual({r["task_id"] for r in recs} - {"b"}, {"a"})
        self.assertIn("warning", log.getvalue())

    def test_several_files_and_globs(self):
        for s in range(2):
            ec.append_texts(os.path.join(self.d, f"shard_{s}.texts.jsonl.gz"), self.rows(f"p{s}"))
        self.assertEqual(len(ec.read_texts(os.path.join(self.d, "shard_*.texts.jsonl.gz"))), 4)
        files = ",".join(os.path.join(self.d, f"shard_{s}.texts.jsonl.gz") for s in range(2))
        self.assertEqual(len(ec.read_texts(files)), 4)
        with self.assertRaises(FileNotFoundError):
            ec.read_texts(os.path.join(self.d, "nothing_*.gz"))


class TestContext(unittest.TestCase):
    def model(self, cfg):
        d = tempfile.mkdtemp()
        if cfg is not None:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f)
        return d

    def test_context_limit(self):
        self.assertEqual(ec.context_limit(self.model(dict(max_position_embeddings=40960)))[0], 40960)          # Qwen3-4B
        yarn = dict(rope_type="yarn", factor=4.0, original_max_position_embeddings=32768)
        self.assertEqual(ec.context_limit(self.model(dict(max_position_embeddings=40960, rope_scaling=yarn)))[0], 131072)
        self.assertEqual(ec.context_limit(self.model(dict(max_position_embeddings=40960, rope_parameters=yarn)))[0], 131072)
        llama3 = dict(rope_type="llama3", factor=8.0, original_max_position_embeddings=8192)
        self.assertEqual(ec.context_limit(self.model(dict(max_position_embeddings=131072, rope_scaling=llama3)))[0], 131072)
        self.assertEqual(ec.context_limit(self.model(dict(text_config=dict(max_position_embeddings=262144))))[0], 262144)
        self.assertIsNone(ec.context_limit(self.model(None))[0])

    def test_resolve_max_model_len(self):
        q = self.model(dict(max_position_embeddings=40960))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ec.resolve_max_model_len(q, [120, 1500], 8192, 10240), 10240)    # the 8k runs: unchanged
            self.assertEqual(ec.resolve_max_model_len(q, [120, 300], 32768, 10240), 33068)    # raised to fit the cap
            self.assertEqual(ec.resolve_max_model_len(q, [8192], 32768, None), 40960)        # Qwen's own allocation
            with self.assertRaises(ValueError) as cm:
                ec.resolve_max_model_len(q, [8193], 32768, None)
            self.assertIn("YaRN", str(cm.exception))
            self.assertEqual(ec.resolve_max_model_len(self.model(None), [300], 32768, 10240), 33068)   # no config: vLLM checks


class TestCharToToken(unittest.TestCase):
    def test_char_tokenizer_exact(self):
        tok, text = CharTok(), "abc\n</think>\n\\boxed{5}"
        ids = [ord(c) for c in text]
        for p in range(len(text)):
            self.assertEqual(ec.char_to_token(tok, ids, p), p)
        self.assertIsNone(ec.char_to_token(tok, ids, None))
        self.assertIsNone(ec.char_to_token(tok, [], 3))

    @unittest.skipIf(U.qwen_dir() is None, "no cached Qwen tokenizer")
    def test_real_tokenizer(self):
        tok = U.load_tok(U.qwen_dir())
        text = ("<think>\nOkay, let's compute 3·4 = 12 — and check: ½ of 24 is 12.\n" * 30) + "</think>\n\nThe answer is \\boxed{12}."
        ids = tok.encode(text, add_special_tokens=False)
        full = tok.decode(ids, skip_special_tokens=True)
        self.assertEqual(full, text)
        rng = random.Random(0)
        for p in [0, 1, len(full) - 1] + rng.sample(range(len(full)), 60):
            j = ec.char_to_token(tok, ids, p)
            self.assertLessEqual(len(tok.decode(ids[:j], skip_special_tokens=True)), p)
            self.assertGreater(len(tok.decode(ids[:j + 1], skip_special_tokens=True)), p)


if __name__ == "__main__":
    unittest.main()
