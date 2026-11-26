# Pre-registration: published length-reduction checkpoints under a token cap (package B)

Author: Cristiano Capone. Written 2026-11-22, before any of the models below was evaluated by us. Exploratory
extension of the Nature Communications submission "Self-training on the shortest verified solutions helps reasoning
models finish within a token cap". Before the first evaluation job, the SHA-256 of this file, of the analysis script
`synth/natcomm_ext_analysis.py`, of the generation script `synth/math_harvest.py`, of `synth/forced_answer.py` and of the
launcher `leonardo/run_natcomm_ext.sh` are written with a UTC timestamp to `logs/PREREG_EXTERNAL.sha256` on Leonardo;
the job submission times are recorded by the scheduler. No git commit is promised.

## Question
Our paper found that, under a binding token cap, the accuracy gain of a length-reducing self-training rule lies in
finishing (more samples end within the cap), not in the quality of finished answers, and that a forced answer at the cap
(the model is made to answer when the budget runs out) removes the accuracy difference. Does the same hold for published,
independently trained length-reduction methods? L1 (Aggarwal & Welleck 2025) reported the opposite at budgets of 512 and
1,024 tokens: its reinforcement-learning model far ahead of budget forcing.

## Models (public checkpoints, HuggingFace ids, main branch as downloaded on 2026-11-22; revisions recorded in the log)
| arm | model | role |
|---|---|---|
| r1base | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | starting model of the next four |
| arora_a0 | daman1209arora/alpha_0_DeepSeek-R1-Distill-Qwen-1.5B | Arora & Zanette RL without length penalty (control) |
| arora_a02 | daman1209arora/alpha_0.2_DeepSeek-R1-Distill-Qwen-1.5B | Arora & Zanette RL, length penalty alpha 0.2 |
| thinkprune | Shiyu-Lab/DeepSeek-R1-Distill-Qwen-1.5B-thinkprune-iter2k | ThinkPrune, iterative pruning to 2k |
| dler | nvidia/DLER-R1-1.5B-Research | DLER |
| dsr | agentica-org/DeepScaleR-1.5B-Preview | starting model of L1 |
| l1max4k | l3lab/L1-Qwen-1.5B-Max, prompt budget 4,096 tokens | L1-Max |
| l1max1k | l3lab/L1-Qwen-1.5B-Max, prompt budget 1,024 tokens | L1-Max (exploratory replication of L1's forcing comparison) |

## Generation (synth/math_harvest.py, eval mode)
- MATH-500 (synth/math/test500.jsonl, the paper's 500 problems and checker), 4 samples per problem, sampling seed 0.
- Temperature 0.6, top-p 0.95, top-k disabled (DeepSeek's recommended settings for R1-Distill), max_tokens 16,384.
- Weights run in bfloat16 for every arm (four checkpoints are stored in float32); vLLM, one A100 per shard.
- Chat template of each model's own tokenizer; generation prompt ends with `<think>\n` (added when the template does not
  add it); a single BOS token.
- User prompt: R1 family (r1base, arora_*, thinkprune, dler): the paper's prompt,
  `{problem}\n\nPlease reason step by step, and put your final answer within \boxed{}.`;
  DeepScaleR family (dsr, l1max*): their training prompt,
  `{problem} Let's think step by step and output the final answer within \boxed{}.`, with
  ` Think for maximum N tokens.` appended for L1 (N = 4096 or 1024).
- Saved generations; per-sample lengths, correctness and finish flags.

## Forced answers (synth/forced_answer.py, unchanged)
Budgets 1,024, 2,048, 4,096, 8,192 and 16,384 generated tokens; Qwen's thinking-budget stop text (the paper's), end
marker `</think>`, at most 1,024 answer tokens, the evaluation's sampling settings. Statuses finished / forced / continued
as in the paper.

## Scorings at each budget t (per sample)
cut-as-wrong (finished within t and correct), credited (the paper's checker on the text cut at t), forced (answer forced at
t). Tokens with forced answers counted as in the paper.

## Primary hypothesis (H1) and test
For the four method contrasts M = {arora_a02 − r1base, thinkprune − r1base, dler − r1base, l1max4k − dsr} and budgets
t ∈ {4,096, 8,192}, the accuracy gain of the method over its starting model is smaller with forced answers than with cut
samples scored as wrong. Statistic: per problem, D = mean over the 8 (contrast, budget) cells of
[forced(M) − forced(base)] − [cut-as-wrong(M) − cut-as-wrong(base)], in points; mean over problems; paired bootstrap over
the 500 problems, B = 10,000, numpy default_rng(0), 95% percentile interval, P = 2 min(share ≤ 0, share ≥ 0), two-sided.
H1 is supported if P < 0.05 and D < 0. A result with D > 0 (P < 0.05) supports the opposite (methods gain more with forcing).

## Secondary (reported for every contrast in the script, at every budget; no multiplicity correction, all labelled)
- Each contrast's gain under the three scorings, with paired-bootstrap CIs; the cut-as-wrong split into finishing and
  quality terms (dF·Qbar, dQ·Fbar, pooled shares); relative tokens with forced answers.
- arora_a02 − arora_a0 (length penalty vs the same RL without it), arora_a0 − r1base.
- Exploratory: l1max1k − dsr at 1,024 and 2,048 (L1's own regime); the paper's per-problem length check.

## Stopping rule and budget
All 8 runs evaluated and forced to completion. The launcher prints each run's overall accuracy when it finishes; no
contrast is computed before all 8 runs and their forced answers are complete. Order: r1base first as a cost calibration;
if its evaluation costs more than 30 CPUh, the author decides before anything else is submitted. Hard cap 220 CPUh for
this package (launcher tier BX). No change of models, prompts, settings, budgets or analysis after the hash is recorded;
any deviation is reported as such.
