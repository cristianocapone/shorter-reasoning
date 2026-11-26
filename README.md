# Language models gain from shorter reasoning mostly by finishing in time

Code, pre-registrations, summary data and figure scripts for the manuscript

> C. Capone, *Language models gain from shorter reasoning mostly by finishing in time* (submitted, 2026).

Under a limit on generated tokens, an accuracy gain from shorter reasoning can mean that a model reasons better, or
only that more of its attempts finish before the limit. A forced answer at the limit separates the two. The paper
tests this on four published length-reduction methods and on a controlled self-training experiment (fine-tuning on
the shortest, a random or the longest of a model's own verified solutions), with three pre-registered analyses.

## Layout

| path | content |
|---|---|
| `paper_brevity/natcomm/` | manuscript (`main_v2.tex`, `main_v2.pdf`) and Supplementary Information (`si_v2.tex`, `si_v2.pdf`); pre-registrations `PREREG_MATH4500.md`, `PREREG_MATH4500_AMEND1.md`, `PREREG_EXTERNAL.md` |
| `paper_brevity/data/` | summary statistics behind every table and figure (`*.json`); SHA-256 hash logs of the pre-registrations and scheduler records (`prereg/`, `ext_runs/`, `PREREG_MATH4500_AMEND1.sha256`) |
| `paper_brevity/make_*.py`, `paper_brevity/figures/` | figure scripts and the figures they produce |
| `synth/` | harvesting, selection, fine-tuning sets, evaluation (`math_harvest.py`, `code_harvest.py`), forced answers (`forced_answer.py`) and analyses (`natcomm_*.py`, `paper_stats.py`) |
| `synth/math/` | the 4,500 MATH test problems of the pre-registered test (`synth/math_prep_conf.py`); MATH-500 and the training pool are rebuilt by `synth/math_prep.py` |
| `leonardo/` | Slurm launchers used on the CINECA Leonardo Booster cluster (`natcomm_lib.sh`, `run_natcomm_*.sh`, `job_natcomm_*.slurm`) |

## Pre-registered analyses

| analysis | pre-registration | analysis script | result |
|---|---|---|---|
| shortest vs random replay, 4,500 MATH problems | `PREREG_MATH4500.md` | `synth/natcomm_conf_analysis.py` | `paper_brevity/data/conf4500_result.json` |
| amendment: forced answers on the same generations | `PREREG_MATH4500_AMEND1.md` | `synth/natcomm_conf_forced_analysis.py` | `paper_brevity/data/conf4500_forced.json` |
| published length-reduction methods | `PREREG_EXTERNAL.md` | `synth/natcomm_ext_analysis.py` | `paper_brevity/data/natcomm_ext.json` |

The hashes of each pre-registration and of its scripts were logged with a UTC timestamp before the first job of each
analysis; the logs and the scheduler records are in `paper_brevity/data/`. The Methods of the manuscript list every
deviation.

## Reproducing

- Figures: `python paper_brevity/make_concept_fig.py`, `make_cover_nc2.py`, `make_fig_cap.py`, `make_fig_ext.py`
  (each reads `paper_brevity/data/`).
- Manuscript: in `paper_brevity/natcomm/`, compile `si_v2.tex` first, then `main_v2.tex` (pdflatex + bibtex).
- Analyses: the analysis scripts above run on the per-sample records (`runs_synth/agent_<tag>/`), which are deposited
  separately with the generations (see the Data availability statement of the manuscript).
- Generation and fine-tuning require GPUs (vLLM, PEFT); the launchers in `leonardo/` document the exact settings.

## Licence

Code: MIT (`LICENSE`). Manuscript, figures and data: CC BY 4.0 (`LICENSE-CC-BY-4.0.md`).

## Acknowledgements

CINECA award under the ISCRA initiative (ISCRA Class C project LUCID, HP10CLEJAD) for computing resources on Leonardo
Booster; APE Lab, INFN, Sezione di Roma, for access to a server with eight NVIDIA V100 GPUs.
