"""Fake vLLM for the NatComm evaluator tests (no GPU). Put test_eval_fakes/ on PYTHONPATH.

Outputs come from the JSON file $FAKE_VLLM_SPEC: {"outputs": {request_id: [{"text": str, "finish_reason": "stop" |
"length"}, ...]}, "default": [...], "log": path}. A request with n samples takes the first n specs (cycled). As in vLLM,
token_ids = the tokenizer's encoding of the text plus the end-of-sequence id when the sample stopped; a sample longer
than max_tokens is cut there (finish_reason 'length', text = decode of the kept ids); text never contains the end token.
Every LLM() construction and every request (id, prompt token ids, sampling kwargs) is appended to spec['log'] (JSONL).
"""
import json, os

__version__ = "0.0-fake"


class SamplingParams:
    def __init__(self, **kw):
        self.kw = kw
        self.n = kw.get("n", 1)
        self.max_tokens = kw.get("max_tokens", 16)


class _Completion:
    def __init__(self, index, text, token_ids, finish_reason):
        self.index, self.text, self.token_ids, self.finish_reason = index, text, token_ids, finish_reason


class _RequestOutput:
    def __init__(self, request_id, prompt_token_ids, outputs):
        self.request_id, self.prompt_token_ids, self.outputs, self.finished = request_id, prompt_token_ids, outputs, True


def _log(spec, obj):
    if spec.get("log"):
        with open(spec["log"], "a") as f:
            f.write(json.dumps(obj) + "\n")


class _Engine:
    def __init__(self, llm):
        self.llm, self.queue = llm, []

    def add_request(self, request_id, prompt, params):
        tok = self.llm.tok
        if "prompt_token_ids" in prompt:
            pids = list(prompt["prompt_token_ids"])
        else:
            pids = tok.encode(prompt["prompt"])
        _log(self.llm.spec, dict(request=request_id, prompt_token_ids=pids, sampling=params.kw))
        self.queue.append((request_id, pids, params))

    def has_unfinished_requests(self):
        return bool(self.queue)

    def step(self):
        request_id, pids, params = self.queue.pop(0)
        spec = self.llm.spec
        outs_spec = spec.get("outputs", {}).get(request_id) or spec["default"]
        tok, eos = self.llm.tok, self.llm.tok.eos_token_id
        outs = []
        for i in range(params.n):
            o = outs_spec[i % len(outs_spec)]
            ids = tok.encode(o["text"], add_special_tokens=False)
            fr = o.get("finish_reason", "stop")
            if fr == "stop":
                ids = ids + [eos]
            if len(ids) > params.max_tokens:
                ids, fr = ids[:params.max_tokens], "length"
            text = tok.decode([x for x in ids if x != eos], skip_special_tokens=True)
            outs.append(_Completion(i, text, ids, fr))
        return [_RequestOutput(request_id, pids, outs)]


class LLM:
    def __init__(self, model, **kw):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        with open(os.environ["FAKE_VLLM_SPEC"]) as f:
            self.spec = json.load(f)
        _log(self.spec, dict(llm=dict(model=model, **kw)))
        self.llm_engine = _Engine(self)

    def get_tokenizer(self):
        return self.tok
