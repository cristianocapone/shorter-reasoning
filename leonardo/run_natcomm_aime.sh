#!/bin/bash
# NatComm revision, CORE: a harder benchmark at a realistic budget where the cap still binds (review must_fix 8).
# AIME 2024 + 2025 (60 problems, synth/aime_prep.py), Qwen3-4B untouched / S / L / R (run-1 students), max_tokens 32768,
# 8 samples per problem (MATH_K=8), sampling seed 0 (SEEDS="0 1" adds 8 more per problem), per-sample records and saved
# texts; then forced answers at 4k/8k/16k/32k (32k = forced at the cap). Normal QOS, one A100 per shard (15 problems).
# The untouched model goes first: its tokens and A100-hours calibrate the cost of the three students (leonardo/
# NATCOMM_RUNS.md assumes 18k tokens per sample; logs/natcomm_cost.tsv has the measurement).
# Tags: nc32_aime_q4_<arm>_s<seed>.
#   nohup bash leonardo/run_natcomm_aime.sh > logs/run_natcomm_aime.log 2>&1 &
cd "$(dirname "$0")/.."; source leonardo/natcomm_lib.sh
M4=$WORK/ccapone0/hf/qwen3_4b_thinking; ST=$NC_ST
NC_TIER=CORE; NC_CAP_CORE=${NC_CAP_CORE:-1600}
AD=${AIME_DATA:-synth/aime/aime24_25.jsonl}; AI=${AIME_IDS:-synth/aime/aime24_25_ids.txt}
nc_lock aime || exit 1
if [ ! -f "$AD" ] || [ ! -f "$AI" ]; then              # needs internet: login nodes have it, compute nodes do not
  nc_log "building $AD with synth/aime_prep.py"; $NC_PY synth/aime_prep.py || exit 1
fi
[ "$(grep -c . "$AI")" = 60 ] || { nc_log "$AI does not list 60 problems"; exit 1; }
nc_preflight_eval synth/math_harvest.py || exit 1
nc_preflight_forced || exit 1
E_SOLVER=synth/math_harvest.py; E_DATA=$AD; E_IDS=$AI; E_K=${AIME_K:-8}; E_TEMP=0.6
E_MAXTOK=32768; E_MAXLEN=34816; E_NSH=4; E_MODE=array; E_TIME=${E_TIME:-04:00:00}; E_SLICE=${E_SLICE:-225}; E_TEXTS=1
FB=${F_BUDGETS:-4096,8192,16384,32768}
for seed in ${SEEDS:-0}; do
  for arm in ${AIME_ARMS:-base S L R}; do                   # AIME_ARMS="base S L" = CORE-lean
    if [ "$arm" = base ]; then m=$M4; else m=$ST/math4b_$arm/merged; fi
    nc_try nc_eval "nc32_aime_q4_${arm}_s$seed" "$m" "$seed" && nc_try nc_forced "nc32_aime_q4_${arm}_s$seed" "$m" "$FB"
  done
done
nc_log "AIME done | CORE spent so far $(nc_spent CORE) CPUh (cap $NC_CAP_CORE)"
