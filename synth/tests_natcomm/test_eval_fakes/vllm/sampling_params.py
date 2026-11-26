"""Fake vllm.sampling_params for the NatComm evaluator tests."""


class RequestOutputKind:
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2
