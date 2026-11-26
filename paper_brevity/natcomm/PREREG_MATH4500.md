# Pre-registration: shortest vs random replay on 4,500 unseen MATH test problems

Written 2026-11-19, before any model has been evaluated on these problems. Author: Cristiano Capone.
This file, `synth/math_prep_conf.py` and `leonardo/run_natcomm_conf.sh` are committed to git before the first evaluation
job is submitted; the commit hash and the sha256 of the data file are recorded in the appendix below, and the
analysis will cite them.

## Why

On MATH-500 the shortest-replay student beat the random-replay student by +0.60 points (P = 0.046), and by +0.95
points over three training replicates (hierarchical bootstrap P = 0.007; t-test over replicates P = 0.053). After a Holm
correction over the eight selection contrasts of the paper, no shortest-minus-random contrast is significant (smallest
adjusted P = 0.062). This test asks the same question once, on new problems, with enough samples to answer it.

## Hypotheses

- **H1 (primary).** At an 8k-token cap, the Qwen3-4B-Thinking-2507 student fine-tuned on its shortest verified
  solutions (`math4b_S`) has higher accuracy per sample than the student fine-tuned on a random verified solution
  (`math4b_R`).
- **H2 (secondary, mechanism).** The difference in H1 lies in finishing within the cap: the finishing term of the
  Kitagawa split, ΔF·Q̄, is positive, and the quality term, ΔQ·F̄, is not positive.

No other contrast is confirmatory. Everything else computed on this set is exploratory and will be labelled so.

## Data

- **Problems.** The MATH test split (EleutherAI/hendrycks_math, all 7 subjects, all levels) minus the 500 problems of
  MATH-500 (matched on whitespace-normalised problem text), minus problems whose reference solution has no `\boxed{}`
  answer. Expected about 4,500. No model has been evaluated on them in this project. They are disjoint from the training
  pool, which comes from the MATH train split.
- **Answers.** The last `\boxed{}` of the reference solution, extracted with `math_harvest.boxed`, as for the training
  set.

## Models and generation (identical to the paper's 8k MATH-500 runs)

- Students `math4b_S/merged` and `math4b_R/merged` (training run 1; the models of the paper's Table 3, not retrained).
- vLLM, `synth/math_harvest.py` in evaluation mode, no system prompt, chat template of the model, temperature 0.6,
  max_tokens 8192, max_model_len 10240, K = 4 samples per problem, sampling seeds 0 and 1 (8 samples per problem and
  arm).
- Grading: the paper's checker (`math_harvest.py`), unchanged, including its fallback to the last `\boxed{}` anywhere
  when the reasoning is cut. A sample is *finished* if it ends within the cap (as in the paper).

## Analysis (fixed now)

- Per problem: accuracy = mean correctness over its 8 samples, per arm. A problem enters the analysis only if both arms
  have all 8 samples.
- **H1 statistic:** the mean over problems of (accuracy S − accuracy R), in percentage points.
- **H1 test:** two-sided paired bootstrap over problems, B = 10,000 resamples, seed 0; P = 2·min(share of resampled
  differences ≤ 0, share ≥ 0). H1 is supported if P < 0.05 and the difference is positive. 95% percentile interval
  reported. A paired sign-flip permutation test over problems (10,000 flips) is reported alongside as a check; the
  decision uses the bootstrap.
- **H2:** the per-problem Kitagawa split of the H1 difference (finishing term, quality term and the residual from
  cut-but-credited samples), each with a paired-bootstrap 95% interval and P, as in the paper. H2 is tested only if H1
  is supported; it is supported if the finishing term's interval excludes 0 on the positive side and the quality term's
  point estimate is ≤ 0.
- No correction is applied to H1 (single primary test). H2 is conditional on H1 (fixed-sequence).

## Power

From the MATH-500 runs, the paired standard error of S − R with 20 samples per problem is about 0.30 points on 500
problems. With 8 samples on about 4,500 problems it is expected to be about 0.16 points, giving about 95% power for a
true difference of 0.6 points and about 50% for 0.3 points (two-sided α = 0.05).

## Stopping rule and what we will not do

- Both arms are run to completion on both sampling seeds (all problems × 8 samples), whatever intermediate results show.
  Runs are ordered S s0, R s0, S s1, R s1. The S − R difference is not computed until all four runs are complete.
- Problems lost to crashes are rerun until complete; if any remain missing after the driver's retry limit, they are
  excluded from both arms and their number is reported.
- No change of cap, temperature, K, checker, problem set or model after this file is committed. If one becomes
  necessary (a bug), the original analysis is still reported, and the deviation is described.
- The result is reported whatever its direction or size, in the manuscript and the response to the editor.

## Appendix (filled in before the first evaluation job)

- Data file `synth/math/conf4500.jsonl` (built on Leonardo 2026-11-19 with `synth/math_prep_conf.py`): MATH test 5,000;
  matched to MATH-500 500; without a boxed answer 0; **4,500 problems**; sha256
  `de470032a14662dea718bc95142649e3e0733264a91104f9971a0754c674bdd2`.
- Before the first evaluation job, the sha256 of this file (as it stands after this appendix) is written to
  `logs/PREREG_MATH4500.sha256` on Leonardo, with the date; a git commit by the author follows.
