# NatComm revision: Leonardo runs, launch order and budget

Written 2026-11-02 for the decision to target Nature Communications along the editor's route
(`paper_brevity/natcomm/review_20261101/editor.md`, advice step 2: cheap compute at 4B/8B). **Nothing here has been
launched.** This file covers the training side (new SFT sets, controls), the drivers and the compute budget. The
evaluation code (per-sample records, saved texts, forced answers, AIME data) was written in parallel:
`synth/math_harvest.py`, `synth/code_harvest.py`, `synth/eval_common.py`, `synth/forced_answer.py`, `synth/aime_prep.py`.

Two tiers:

* **CORE**: what the editor's decision point needs ("if shortest replay still beats random replay and keep-all under
  forced-answer evaluation ..."): Qwen3-4B on MATH-500 at a 32k cap with per-sample records and saved texts for
  S/R/L/untouched, keep-all, sham, token-matched and low-temperature arms; forced answers on all of them; AIME 2024+2025
  at 32k; the R1-Llama-8B sham.
* **EXTENDED**: R1-Llama-8B keep-all and two more training runs per arm, APPS two more training runs per arm, R1-Llama
  forced answers, the selection-strength dose-response.

Budget in one line: CORE about **1,260 CPUh (19% of the 6,700 left; range 930-2,210)**, CORE-lean about 1,000 CPUh (15%),
EXTENDED about **1,060 CPUh (16%; range 820-1,760)**. The linear pace from 2026-11-02 to 2027-06-09 is 931 CPUh per
month, so CORE takes about 1.4 months of it, CORE-lean about 1.1, and CORE + EXTENDED about 2.5 (section 6).

---

## 1. Before any GPU job

1. **Sync** to `$HOME/arcagi2` on Leonardo (a separate step diffs local and cluster copies):
   * new, this plan: `synth/natcomm_sft_sets.py`, `synth/natcomm_check.py`, `synth/natcomm_sham_check.py`,
     `leonardo/natcomm_lib.sh`, `leonardo/run_natcomm_{sets,q4,aime,l8,apps,dose}.sh`,
     `leonardo/job_natcomm_{eval,forced}.slurm`, `leonardo/natcomm_cost.py`;
   * evaluation code: `synth/{math_harvest,code_harvest,eval_common,forced_answer,aime_prep}.py`. The drivers refuse to
     start without it: `nc_preflight_eval` looks for `--save-texts` and `fins` in the solver, `nc_preflight_forced` for
     the four options of `forced_answer.py`, and `nc_eval` stops after the first job if any record lacks `lens/oks/fins`.
   * used unchanged, check them in the diff: `distill/sft_student.py`, `distill/merge_adapter.py`,
     `leonardo/job_sft.slurm`, `leonardo/job_merge.slurm`. The drivers rely on: job_sft.slurm reading TAG, MODEL, DATA,
     NPROC, MAXLEN, EPOCHS, EXTRA and writing `$WORK/ccapone0/students/$TAG`; logs at `logs/%x-%j.out`; sft_student.py
     printing `step a/b | ...`, `checkpoint saved at step N` and a line starting with `DONE`, saving `adapter/` with
     `--no-merge`, honouring `--deadline-min`, `--seed`, `--max-steps`, `--lr`; merge_adapter.py printing `MERGED`.
2. `bash leonardo/budget.sh` (pace and month spend; `saldo` lags a day).
3. Disk on `$WORK`: every merged Qwen3-4B student is about 8 GB and every R1-Llama-8B student about 16 GB. CORE adds 4
   + 1 merged models (about 50 GB); everything adds about 220 GB. Saved texts are small (about 15 MB gzip per 32k
   arm-seed).
4. Models: `$WORK/ccapone0/hf/qwen3_4b_thinking/config.json` `max_position_embeddings` must be at least prompt + 32,768
   (40,960 for Qwen3-4B, 262,144 for Qwen3-4B-Thinking-2507; `run_phys_smoke.sh` ran it with MAXLEN=65536, which only
   the latter allows, so check which checkpoint it is for the Methods). The R1-Llama tokenizer workaround
   (`tokenizer_class: PreTrainedTokenizerFast`) is checked by `run_natcomm_sets.sh` and `run_natcomm_l8.sh`.
5. AIME data: `run_natcomm_aime.sh` runs `synth/aime_prep.py` on the login node (it needs internet) if
   `synth/aime/aime24_25.jsonl` is missing.

All drivers are started from the repo root on a login node, one `nohup` per lane, and are idempotent: relaunching a lane
skips what is marked done and resumes the rest. A lane holds a lock in `logs/locks/` (login nodes are round-robin, so a
relaunch can land on another node; a stale lock of a dead process on the same node is taken over).

---

## 2. Launch order

Costs are central estimates (section 6); "dbg" = `boost_qos_dbg` (30 min, 2 running jobs per user), "normal" = 1-GPU
array tasks on the normal QOS (one shard per A100, TP=1).

| step | command (`nohup bash ... > logs/<lane>.log 2>&1 &`) | QOS | tier | review item | A100-h | CPUh |
|---|---|---|---|---|---|---|
| 1 | `leonardo/run_natcomm_sets.sh` | lrd_all_serial (free) | - | 6, 9, 14 | 0 | 0 |
| 2a | `leonardo/run_natcomm_q4.sh train` | dbg | CORE | 6, 9 | 8 | 64 |
| 2b | `leonardo/run_natcomm_q4.sh e32s0` | normal | CORE | 3, 6, 7, 8, 9 | 57 | 453 |
| 2c | `leonardo/run_natcomm_l8.sh sham` | dbg | CORE | 9 | 6 | 48 |
| 3 | calibration: read `runs_synth/agent_nc32_q4_base_s0/EVAL_DONE` (CPUh line) | - | - | - | - | - |
| 4a | `leonardo/run_natcomm_q4.sh e32s1` (`LEAN=1` for CORE-lean) | normal | CORE | as 2b | 57 | 453 |
| 4b | `leonardo/run_natcomm_aime.sh` (`AIME_ARMS="base S L"` for CORE-lean) | normal | CORE | 8 | 31 | 245 |
| 5 | analyses for the editor's decision point (evaluation-side agent) | - | - | - | - | - |
| 6a | `leonardo/run_natcomm_l8.sh ka` | dbg | EXT | 6 | 14 | 113 |
| 6b | `leonardo/run_natcomm_l8.sh rep1`, then `rep2` | dbg | EXT | 7 | 37 | 293 |
| 6c | `leonardo/run_natcomm_l8.sh texts` | dbg | EXT | 6 | 14 | 111 |
| 6d | `leonardo/run_natcomm_apps.sh rep1`, then `rep2` | dbg | EXT | 7, 10 | 60 | 477 |
| 6e | `leonardo/run_natcomm_dose.sh` | dbg | EXT | 14 | 8 | 67 |

The 2b/4a split assumes both seed lanes run the full arm list; the eval and forced costs of the 9 arms x 2 seeds are
807 + 98 CPUh. Step 6c's figure includes the forced answers on all 19 R1-Llama text sets (6a, 6b, 6c).

**Step 1 (sets, CPU only).** Builds `runs_synth/natcomm_sets/{q4,l8,c4}_<set>.jsonl` and `<prefix>_stats.json`
(section 3) on `lrd_all_serial`, with exact token counts from each model's tokenizer and `--verify` (below). Fallback if
the serial partition refuses the job: `NC_SERIAL=0` runs the same commands on the login node (each well under its CPU
limit; the sham weight check later is the heavier one). Re-print a table: `python3 synth/natcomm_sft_sets.py --show
runs_synth/natcomm_sets/q4_stats.json`.

**Step 2 (CORE wave A).** Three lanes in parallel. `q4 train` (debug QOS) trains in the order keep-all, sham (then the
CPU sham check), Rtm, Ltm, each followed by its merge. `q4 e32s0` (normal QOS) evaluates sampling seed 0 at
max_tokens 32768, K=4, per-sample records and saved texts, in the order **base, S, R, L** (existing models, the decisive
comparison) then **KA, sham, Rtm, Ltm** (each waits for the train lane's `MERGE_DONE`, up to 48 h) and **baseT04**
(untouched model at temperature 0.4), each followed by forced answers at 4k/8k/16k/32k. `l8 sham` trains the R1-Llama
zero-step adapter, runs the sham check and evaluates it at 8k with seeds 0-2.

**Step 3 (calibration).** There is no local measurement of vLLM decoding for these models (section 6), so the first 32k
evaluation is the calibration: the untouched model's seed-0 run is estimated at 45 CPUh (35-74). Its real cost is in its
`EVAL_DONE` and in `logs/natcomm_cost.tsv` (`python3 synth/natcomm_check.py cost`). If it is above about 75 CPUh, run
CORE-lean: `LEAN=1` for e32s1 (Rtm, Ltm, sham and the low-temperature run at seed 0 only) and `AIME_ARMS="base S L"`.
The same check applies to AIME: the untouched model goes first (estimated 55 CPUh, 29-140; the 18k-token mean is an
assumption).

**Step 4 (CORE wave B).** The seed-1 lane and AIME (60 problems, 8 samples each, sampling seed 0; `SEEDS="0 1"` adds 8
more per problem, OPT tier).

**Step 6 (EXTENDED).** After the decision point and at the linear pace (December-January). The R1-Llama `texts` lane
re-evaluates the run-1 arms and the untouched model at seed 0 with saved texts so that every R1-Llama arm also gets a
forced-answer evaluation at its 8k cap. Lanes with different names never share tags, so any two may run at the same
time; the debug QOS limit of two running jobs then interleaves their jobs.

Monitoring: `tail -f logs/<lane>.log`; `squeue -u $USER -o "%.12i %.32j %.8T %.10M %R"`; `python3 synth/natcomm_check.py
summary <tag>`; `python3 synth/natcomm_check.py cost` (CPUh and A100-h per tag and in total).

---

## 3. The new training sets (`synth/natcomm_sft_sets.py`)

All sets are built from the existing harvests, are written in the harvest format and train with `sft_student.py`
without `--shortest-per-task` (every trajectory in the file is one example), with the original arms' recipe: LoRA rank
32, alpha 64, learning rate 5e-5, one epoch, accum 2 on 4 GPUs (batch 8), max length 12,288, training seed 0.

| set | definition | serves |
|---|---|---|
| S, R, L | shortest / random / longest verified solution per problem: an exact re-implementation of `sft_student.py --shortest-per-task --select X --seed 0` (stable sort on `tokens`, same tie-breaking, same `random.Random(seed).sample` calls in sorted task order). `--verify` asserts that the files give `build_samples` exactly the samples of the original selection, so the stats describe what math4b_S/R/L, mathl8_S/R/L and code4b_S/R/L trained on | must_fix 9 (statistics) |
| KA (keep-all) | every distinct verified solution of every problem; duplicates = identical assistant text after whitespace normalisation (`" ".join(text.split())`) | must_fix 6 (1) |
| Rtm, Ltm (token-matched) | R and L restricted to the problems of one fixed random permutation (`random.Random(20261102)`), taken in order until their loss tokens reach S's total; the next problem is added only if that brings the total closer. The permutation is shared, so Ltm's problems are a prefix of Rtm's | must_fix 9 (token-matched control) |
| sham | the S file, trained with `--lr 0 --max-steps 1` (zero-step: PEFT initialises lora_B to 0), merged by the same `merge_adapter.py` | must_fix 9 (zero-step control) |
| Sk2 | per problem, two verified solutions drawn without replacement (`random.Random(20261102)`; one if only one exists), keep the shorter: between R (k = 1) and S (k = all); same problems as S/R/L, differs from S only on problems with at least 3 verified solutions | must_fix 14 (selection-strength dose-response) |

`--verify` also checks that every file's example count and loss tokens equal what `sft_student.build_samples` builds
from it with the same tokenizer, `SFT_END`, `SFT_NO_SYS` and max length.

**Loss normalisation (for the Methods; read from `distill/sft_student.py`).** Each example's loss is the mean
cross-entropy over its own target tokens (the replayed solution, reasoning included, plus the end-of-turn marker); the
two examples a GPU processes per step are averaged and the gradients are averaged over the four GPUs, so every example
has the same weight whatever its length. S, R and L therefore have the same number of examples and optimiser steps and
differ in loss tokens; Rtm and Ltm match S in loss tokens and have fewer examples and steps. Together they bracket a
"training dose" account in both units.

**Token-matched control, method statement.** "For random and longest replay we kept the training problems of one fixed
random permutation, in order, until the total number of loss tokens (target tokens: the replayed solution and the
end-of-turn marker) reached that of shortest replay, adding the next problem only if it brought the total closer; the
two arms then match shortest replay in loss tokens to within half of one example, with fewer examples and optimiser
steps."

**What the statistics will say.** The harvest files are only on Leonardo, so the real counts come from step 1
(`runs_synth/natcomm_sets/*_stats.json`: problems, examples, loss and context tokens, optimiser steps, target-length
mean, median, p10, p90, maximum, length tilt relative to each problem's mean verified length, problems by number of
verified and distinct solutions, problems where S and L differ). Known in advance from the harvest records
(`PAPER_BREVITY_DRAFT.md`): Qwen3-4B MATH has 1,345 problems with a verified solution and 59.5% of its 8,000 samples
correct, i.e. 4,760 verified samples, so S, R and L have 1,345 examples (169 optimiser steps at batch 8) and keep-all at
most 4,760 (at most 595 steps); APPS has 765 problems and about 2,456 verified samples (30.7% of 8,000), i.e. 765
examples (96 steps) against at most 2,456 (307 steps). There is no local record of the R1-Llama harvest counts. On a
200-problem synthetic harvest (the tests' fixture generator, real Qwen2.5 tokenizer) the builder gave, for orientation
only, Rtm and Ltm within 0.2% of S's loss tokens with 73% and 55% of S's problems, and Sk2's mean length tilt (-0.19)
between S's (-0.29) and R's (-0.02).

Table for the paper (one row per setting x arm, from the stats files): problems | examples | loss tokens | x S |
optimiser steps | target mean / median / p90 | length tilt | skipped (> max length).

**Not covered here.** The ARC training-set statistics of must_fix 5 and 9 (multi-turn trajectories, the 15,000-token
filter and the exclusion of seeded puzzles; `natcomm_sft_sets.py` counts multi-turn examples but is not wired to the ARC
harvests), KL to the base model and output entropy (must_fix 9; the saved texts allow scoring them offline), and a
re-harvest for Qwen3-4B (must_fix 7, "ideally").

---

## 4. Tags

Every new tag starts with `nc`; no existing tag does (`synth/paper_stats.py` reads `math_eval_*`, `math16_eval_*`,
`math_l8_eval_*`, `code_eval_*`, `brevcl_*`, `brevctl_*`; students are `math4b_*`, `mathl8_*`, `code4b_*`, `brev*`).
`test_train_drivers.py` checks the prefixes.

| what | tags (`runs_synth/agent_<tag>/` or `$WORK/ccapone0/students/<tag>/`) | format for a paper_stats-style loader |
|---|---|---|
| Qwen3-4B students | `nc_q4_KA`, `nc_q4_sham`, `nc_q4_Rtm`, `nc_q4_Ltm`, `nc_q4_Sk2` | |
| R1-Llama students | `nc_l8_sham`, `nc_l8_KA`, `nc_l8_{S,R,L}_rep{1,2}` | |
| APPS students | `nc_c4_{S,R,L}_rep{1,2}` | |
| MATH-500 32k | `nc32_q4_{base,S,R,L,KA,sham,Rtm,Ltm,baseT04}_s{0,1}` | `nc32_q4_{a}_s{s}` |
| AIME 32k | `nc32_aime_q4_{base,S,L,R}_s0` | `nc32_aime_q4_{a}_s{s}` |
| R1-Llama 8k | `nc8_l8_{sham,KA}_s{0,1,2}`, `nc8_l8_{S,R,L}rep{1,2}_s{0,1}`, `nc8_l8_{base,S,R,L}_s0` | `nc8_l8_{a}_s{s}`, `nc8_l8_{a}rep{r}_s{s}` |
| APPS 8k | `nc8_c4_{S,R,L}rep{1,2}_s{0,1}` | `nc8_c4_{a}rep{r}_s{s}` |
| dose-response 8k | `nc8_q4_Sk2_s{0,1,2}` | |
| SFT sets | `runs_synth/natcomm_sets/{q4,l8,c4}_<set>.jsonl`, `<prefix>_stats.json` | |

Each evaluation directory holds `shard_<i>.jsonl` (records with `lens oks fins ans_tok`), `shard_<i>.texts.jsonl.gz`,
`forced.jsonl` (+ `.meta.json`), and the markers below. Run 1 of the R1-Llama and APPS replicate analyses is the
original arm (`math_l8_eval_*`, `code_eval_*`), as `math_eval_*` is run 1 for Qwen3-4B.

**Sampling seeds.** The 32k runs use seeds 0-1, as specified. `math_harvest.py` seeds every request with `--seed`, and
the cap does not change the random stream, so a 32k sample may coincide with the 8k (or 16k) sample of the same seed
until numerical differences make them diverge. For an independent check that truncation at 8k reproduces the 8k runs,
compare with the 8k seeds 2-4; the 16k runs exist only for two seeds, so a fully independent 16k check needs other seeds
(for example 5-6) at extra cost.

---

## 5. Markers and safeguards (`leonardo/natcomm_lib.sh`)

* **TRAIN_DONE** (`$WORK/ccapone0/students/<tag>/TRAIN_DONE`) is written only when the job's log has a line starting with
  `DONE`, `sacct` reports `COMPLETED`, and the last `checkpoint saved at step N` equals the total steps of the `step a/N`
  lines. It records the job, steps, the adapter's sha256, the SFT arguments and the data file's sha256. An existing
  `adapter/` proves nothing: `sft_student.py` saves it every 100 steps, and on 2026-10-30 an squeue glitch made
  `sft_chain.sh` merge a mid-training adapter.
* **Waiting**: a job is over only when `squeue` no longer lists it **and** `sacct` reports a terminal state; if squeue is
  empty while sacct still says RUNNING (the glitch), the driver keeps waiting.
* **MERGE_DONE** only after `merge_adapter.py` printed `MERGED` for the adapter whose sha256 is in TRAIN_DONE (a changed
  adapter is refused).
* **EVAL_DONE** when every id has a record with K per-sample entries and, with saved texts, `natcomm_check.py texts`
  finds all K generations of every record with `n_tokens == lens[k]`. Resubmissions cover only unfinished shards; two
  iterations without progress, or a Traceback that is not the time-limit shutdown, stop the lane. It records the summary
  and the CPUh spent.
* **FORCED_DONE** when `forced.jsonl` holds every sample x budget row of the current configuration (`forced_answer.py`
  exits 0 when its time slice ends, so "job COMPLETED" alone is not enough); it is resubmitted, and resumes, until then.
* **No double submission**: before `sbatch`, a queued or running job with the same name (`nc_sft_<tag>`, `nc_merge_<tag>`,
  `nc_ev_<tag>`, `nc_fa_<tag>`, `nc_cpu_<name>`) is adopted instead.
* **Cost log and caps**: every finished job appends `date tier tag kind job state elapsed_s CPUh A100h` to
  `logs/natcomm_cost.tsv` (CPUh = Slurm billing x hours, as `budget.sh` computes it). A lane stops before submitting once
  its tier has spent `NC_CAP_CORE` (default 1,600 CPUh) or `NC_CAP_EXT` (default 1,400 CPUh). Two consecutive failed
  steps stop a lane (`nc_try`).
* **Time limits**: SFT and 8k evaluations run as 30-min debug jobs (the solver and the trainer stop themselves before
  the limit: `DIRECT_SLICE_MIN=24`, `--deadline-min 18`). 32k evaluations use 1-GPU array tasks of 3 h (AIME 4 h) with
  slices of 165 (225) min, about twice the expected 1.4 h (1.7 h) per shard: at 32k a sample can need more than a 24-min
  debug slice at full batch, and a problem is written only when all its samples finish, so a debug chain could redo the
  longest problems forever. Forced answers: 1 GPU, 2 h, 105-min slices. Merges: debug QOS, 30 min.

---

## 6. Cost

**Method** (`leonardo/natcomm_cost.py`; `--md` prints every item). Generated tokens per run = problems x samples x mean
capped length. Mean lengths at 8k and 16k come from `paper_stats.json` (`tokens`, `finish`). Beyond 8k the survival is
fitted per arm as S(x) = q + a exp(-lambda (x - 8192)) to the 8k and 16k finishing rates and to the 8k-16k difference of
capped means (q = share of samples that never stop): the cut samples keep going at 32k, and the model gives mean lengths
at 32k of 6,207 (S), 6,507 (R), 6,781 (L) and 6,530 (untouched) tokens per sample, with 0.4-1.8% still cut at 32k.
Decode throughput is a KV-cache-bound roofline per A100 (bf16 weights, 90% of 64 GB, KV bytes per token from the model
configs, effective bandwidth 1.0 TB/s in the central scenario, 1.3 and 0.65 TB/s in the low and high cost scenarios),
which gives about 1,440-1,510 tokens/s per A100 at the 8k regime's mean contexts and 770-830 at 32k for Qwen3-4B. Job efficiency covers
engine start, the thin batch at the end and, for debug chains, the work lost at each slice end. SFT throughput: 5.5k
tokens/s on 4 A100 for Qwen3-4B (local measurement of 6.3-6.5k for a LoRA SFT of Qwen3-4B on 4 A100, HANDOFF_HYBRID.md
2026-09-28, less 15% for the full-vocabulary loss head) and 3.0k for the 8B (scaled by parameters). The only local
vLLM measurements are for larger models (the 27B agent at about 900 tokens/s on 2 A100 and gpt-oss-120b at 1,829 on 2
A100, EXPERIMENT_SYNTHESIS.md), so **decode throughput is
the main uncertainty and step 3 calibrates it**. Other assumptions: log-normal finished lengths below 8k with CV 0.5;
AIME mean 18k tokens per sample with 25% cut at 32k (14k-24k and 15-40% in the scenarios; no AIME run of this model
exists); about 300 answer tokens per forced answer; training-set sizes in section 3; R1-Llama harvest of about 1,500
problems and 5,600 verified samples (unknown locally); APPS about 60% of samples cut at 8k. 1 A100-hour = 8 CPUh.

| tier | group | A100-h (central) | range | CPUh (central) | % of 6,700 |
|---|---|---|---|---|---|
| CORE | Qwen3-4B MATH-500 32k evals (9 arms x 2 seeds) | 100.9 | 77.3-164.1 | 807 | 12.0 |
| CORE | forced answers on those texts (4k/8k/16k/32k) | 12.3 | 10.8-16.2 | 98 | 1.5 |
| CORE | Qwen3-4B SFT: keep-all, Rtm, Ltm, sham (+ merges) | 8.0 | 7.1-10.7 | 64 | 1.0 |
| CORE | AIME 24+25 32k evals (base/S/L/R, K=8) + forced | 30.7 | 16.4-75.5 | 245 | 3.7 |
| CORE | R1-Llama-8B sham: SFT + merge + 8k evals s0-2 | 5.9 | 4.8-9.5 | 48 | 0.7 |
| EXT | R1-Llama-8B keep-all: SFT + 8k evals s0-2 | 14.2 | 11.3-21.1 | 113 | 1.7 |
| EXT | R1-Llama-8B replicates: 6 SFT + 12 evals | 36.6 | 29.0-55.4 | 293 | 4.4 |
| EXT | R1-Llama-8B run-1 re-evals with texts + forced answers on all 19 R1-Llama text sets | 13.9 | 11.6-20.4 | 111 | 1.7 |
| EXT | APPS replicates: 6 SFT + 12 evals | 59.7 | 44.5-108.5 | 477 | 7.1 |
| EXT | Sk2 dose-response: SFT + 8k evals s0-2 + forced | 8.4 | 6.4-14.5 | 67 | 1.0 |
| OPT | AIME second sampling seed | 30.7 | 16.4-75.5 | 245 | 3.7 |
| OPT | S trained to L's loss tokens ("shortest upweighted"; not scripted) | 14.6 | 11.2-22.4 | 116 | 1.7 |

| tier | A100-h low / central / high | CPUh central (range) | share of the 6,700 CPUh left | months of linear pace (931 CPUh/month) |
|---|---|---|---|---|
| CORE | 116 / 158 / 276 | 1,262 (931-2,207) | 18.8% (14-33%) | 1.36 |
| CORE-lean | 93 / 125 / 217 | 999 (741-1,735) | 14.9% (11-26%) | 1.07 |
| EXTENDED | 103 / 133 / 220 | 1,061 (822-1,760) | 15.8% (12-26%) | 1.14 |
| CORE + EXTENDED | 219 / 291 / 496 | 2,324 (1,753-3,967) | 34.7% (26-59%) | 2.50 |
| OPT | 28 / 45 / 98 | 362 (221-783) | 5.4% (3-12%) | 0.39 |

Linear allowance from 2026-11-02: 857 CPUh to 2026-11-30, 1,316 to 2026-12-15, 1,805 to 2026-12-31. So:

* CORE at the central estimate (1,262 CPUh) fits the linear allowance up to mid-December, the editor's submission
  window; at the high estimate it would take until about mid-January. CORE-lean (999) fits by early December.
* CORE + EXTENDED (2,324) would be about 520 CPUh (two to three weeks of pace) ahead of the linear allowance at the end
  of December: run EXTENDED in December-January. The largest single item is the APPS replicate evaluations (52 A100-h,
  because about 60% of APPS samples run to the 8k cap); one sampling seed per replicate would halve it.
* The caps (`NC_CAP_CORE` 1,600, `NC_CAP_EXT` 1,400 CPUh) stop the lanes about 25-30% above the central estimates; set
  them lower to enforce the pace more tightly.

---

## 7. Open decisions and risks

* **Throughput** is modelled, not measured (section 6); calibrate on `nc32_q4_base_s0` before wave B.
* **Low temperature** for the sharpening test is 0.4 (`LOWT`); the review asks only for "a lower temperature". 0.4 is a
  moderate reduction from 0.6; a stronger probe (0.2-0.3) risks more repetition loops, which would make the comparison
  with S harder to read.
* **System prompt in the Qwen3-4B training contexts.** With `SFT_NO_SYS` unset, `sft_student.py` puts the ARC system
  prompt ("You are an expert at solving ARC ...") in the context of every MATH and APPS example, while the evaluation
  prompt has none. The original Qwen3-4B arms were trained that way (run_math_brev.sh, run_paper_extra.sh do not set it),
  so the new Qwen3-4B arms keep it for comparability; the Methods should say so (check the cluster copy of
  sft_student.py in the sync diff). R1-Llama arms are trained without a system prompt (`SFT_NO_SYS=1`).
* **Sampling seeds of the 32k runs** coincide with those of the 8k and 16k runs (section 4).
* **AIME length** is assumed; the untouched model's AIME run measures it first.
* **Forced answers at the cap**: the 32k text sets get budgets 4k, 8k, 16k and 32k (32k = forced at the cap;
  `forced_answer.py` forces a sample cut exactly at the budget), the 8k text sets 4k and 8k.
* **Debug QOS**: two running jobs per user; the training lanes, 8k evaluation lanes and merges share it.
* **lrd_all_serial** has not been used by these drivers before; `NC_SERIAL=0` falls back to the login node.

---

## 8. Local tests

    env -u PYTHONPATH .venv_mps/bin/python -m unittest discover -s synth/tests_natcomm -p 'test_train_*.py' -v

51 tests, about 2 minutes on a laptop, no GPU or network:
* `test_train_sets.py`: the builder on the synthetic harvest of `test_train_fixture.py` (0-4 verified samples per problem,
  token ties, verbatim and whitespace duplicates, a problem harvested in two shards, multi-turn trajectories): S/R/L equal
  `sft_student.py`'s selection for seeds 0-2, keep-all deduplication, token matching within half an example and nesting,
  Sk2 between R and S, statistics, refusal to overwrite a set that changed, and `--verify` with a real Qwen-family
  tokenizer (cached Qwen2.5-0.5B-Instruct).
* `test_train_drivers.py`: `bash -n` on every driver and Slurm script, tag prefixes, and `natcomm_lib.sh` against a stub
  Slurm (`test_train_slurm_stub.py`): the squeue glitch, deadline checkpoints, a crash after the periodic adapter save,
  DONE with a FAILED job, adapter changed before merge, shard-selective resubmission in array and debug modes, missing
  per-sample fields, stalls, crash detection, resumed forced answers, adoption of a running job, lane locks, the budget
  guard and the q4 evaluation lane's order.
* `test_train_sham.py`: the sham arm end to end on a tiny random Qwen2 model with the real Qwen2.5 tokenizer files:
  `sft_student.py --lr 0 --max-steps 1` then `merge_adapter.py` gives bit-identical weights and identical tokenizer
  behaviour (the re-saved tokenizer files differ in format under transformers 5.17, which the check reports as
  metadata), while one step at lr 1e-2 fails the check.
