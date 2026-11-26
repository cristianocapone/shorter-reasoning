#!/bin/bash
# NatComm revision, pre-registered confirmatory test (paper_brevity/natcomm/PREREG_MATH4500.md): shortest vs random replay,
# Qwen3-4B-Thinking-2507 students math4b_S / math4b_R (run 1, unchanged since the 8k paper runs), on the ~4,500 MATH test
# problems outside MATH-500 (synth/math/conf4500.jsonl, built by synth/math_prep_conf.py), at the paper's 8k settings:
# max_tokens 8192, max_model_len 10240, temperature 0.6, K=4 samples per problem, sampling seeds 0 and 1.
# Order S s0, R s0, S s1, R s1, so that both arms always have the same seeds done. Tags nc8c_q4_<arm>_s<seed>.
# Budget tier OPT, capped by NC_CAP_OPT (default 700 CPUh; central estimate 500).
#   nohup bash leonardo/run_natcomm_conf.sh > logs/run_natcomm_conf.log 2>&1 &
cd "$(dirname "$0")/.."; source leonardo/natcomm_lib.sh
ST=$NC_ST
NC_TIER=OPT; NC_CAP_OPT=${NC_CAP_OPT:-700}
nc_lock conf || exit 1
nc_preflight_eval synth/math_harvest.py || exit 1
nc_need synth/math/conf4500.jsonl "run synth/math_prep_conf.py on a login node first" || exit 1
nc_need paper_brevity/natcomm/PREREG_MATH4500.md "the pre-registration must exist before any evaluation" || exit 1
E_SOLVER=synth/math_harvest.py; E_DATA=synth/math/conf4500.jsonl; E_IDS=synth/math/conf4500_ids.txt; E_K=4
E_MAXTOK=8192; E_MAXLEN=10240; E_TEMP=0.6; E_NSH=8; E_MODE=array; E_TIME=${E_TIME:-03:00:00}; E_SLICE=${E_SLICE:-165}
E_TEXTS=1
for seed in 0 1; do
  for arm in S R; do
    nc_try nc_eval "nc8c_q4_${arm}_s$seed" "$ST/math4b_$arm/merged" "$seed"
  done
done
nc_log "confirmatory lane finished | OPT spent so far $(nc_spent OPT) CPUh (cap $NC_CAP_OPT)"
