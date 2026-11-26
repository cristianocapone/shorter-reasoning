# Amendment 1 to the pre-registration of the MATH-4500 test: forced answers on the saved generations

Author: Cristiano Capone. Written 2026-11-24, before any forced answer has been produced on these generations.
Amends `PREREG_MATH4500.md` (hash `e1f37784…f15afef`), whose hypotheses H1 and H2 were tested and supported on
2026-11-20; nothing in that test changes. Before the first job of this amendment, the SHA-256 of this file, of the
analysis script `synth/natcomm_conf_forced_analysis.py`, of `synth/forced_answer.py`, of the launcher
`leonardo/run_natcomm_conf_forced.sh` and of the problem-id file are written with a UTC timestamp to
`logs/PREREG_MATH4500_AMEND1.sha256` on Leonardo; the job submission times are recorded by the scheduler.

## What is already known (stated so that the reader can judge the risk of each prediction)
- The H1 difference (S − R, the pre-registered scoring at 8k) is +1.04 points [0.75, 1.33] on 4,500 problems × 8 samples;
  the per-run overall accuracies and finishing shares are known to the author. No forced answer has been generated on
  these problems, and no contrast with forced answers has been computed.
- On MATH-500 (exploratory, 32k generations scored at budgets), forced S − R was +0.25 [−0.65, 1.15] at 4k and +0.33
  [−0.47, 1.12] at 8k.

## Question
The restart view predicts that a forced answer at the cap recovers what shortest replay gains by finishing. On MATH-500
the forced difference was not significant, but its interval (about ±0.8 points) contained the pre-registered effect.
This amendment measures the forced difference on the 4,500 pre-registered problems, where the interval is about three
times narrower.

## Data and procedure (fixed)
- Generations: the saved texts of the four pre-registered runs `nc8c_q4_{S,R}_s{0,1}` (students `math4b_S`, `math4b_R`,
  8k cap, temperature 0.6, K = 4, seeds 0 and 1), unchanged.
- Forced answers: `synth/forced_answer.py`, unchanged, at budgets 4,096 and 8,192 generated tokens, with Qwen's
  thinking-budget text and end marker, at most 1,024 answer tokens, the evaluation's sampling settings, each student
  forcing its own samples. Statuses finished / forced / continued as in the paper.
- Scorings per sample at budget t: cut-as-wrong; credited (the pre-registered checker on the text cut at t, which at
  8,192 is the pre-registered scoring); forced. Tokens with forced answers counted as in the paper.
- Per problem, the mean over the 8 samples; problems missing in any run or budget are excluded and reported.

## Hypotheses and tests (both primary; reported whatever the outcome)
- **H3.** At 8,192 tokens, the S − R difference is smaller with forced answers than with the pre-registered scoring:
  D8 = mean over problems of [(forced S − forced R) − (credited S − credited R)] < 0; paired bootstrap over problems,
  B = 10,000, numpy default_rng(0), two-sided P = 2 min(share ≤ 0, share ≥ 0); supported if D8 < 0 and P < 0.05.
- **H4 (risky).** At 8,192 tokens, forced S − R is equivalent to zero within ±0.5 points: the 90% bootstrap interval
  (two one-sided tests at 5%) lies inside (−0.5, 0.5). If it fails, the paper reports that a residual difference of that
  size is not excluded, or that shortest replay keeps a gain under forcing, whichever the interval shows.

## Secondary (labelled, no multiplicity correction)
Same quantities at 4,096; cut-as-wrong contrasts; finished shares; relative tokens with forced answers, S vs R, with a
bootstrap interval (prediction: S uses fewer tokens).

## Stopping rule and budget
All four forced runs to completion at both budgets; no contrast computed before all four are complete (the launcher
prints only row counts). Cap 150 CPUh (launcher tier CF). No change of settings or analysis after the hash is recorded;
deviations reported as such.
