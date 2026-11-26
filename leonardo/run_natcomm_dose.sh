#!/bin/bash
# NatComm revision, training-dose check (2026-11-20). On 2026-11-20 the token-matched arms reproduced the shortest-replay
# effect at 8k on the 32k texts (Rtm-R +1.60 points, Ltm-L +1.65; one sampling seed, one training run). This lane tests
# whether that is real or noise between training runs:
#   1. a second token-matched replicate: Rtm/Ltm rebuilt with a different problem permutation (--tm-seed 20261120,
#      prefix q4b), trained with the original arms' recipe -> students nc_q4b_Rtm, nc_q4b_Ltm;
#   2. evaluations at max_tokens 32768, K=4, as the e32 lanes (texts saved, no forced answers):
#        nc32_q4_Rtm_s1, nc32_q4_Ltm_s1   (the existing token-matched students, sampling seed 1)
#        nc32_q4_q4bRtm_s0, nc32_q4_q4bLtm_s0 (the new replicate, sampling seed 0)
#      tags q4bRtm/q4bLtm sit in the nc32_q4_<arm>_s<seed> scheme read by synth/natcomm_32k_analysis.py.
# Budget tier CORE, cap NC_CAP_CORE (default 1600; about 780 spent before this lane, this lane about 150).
#   nohup bash leonardo/run_natcomm_dose.sh > logs/run_natcomm_dose.log 2>&1 &
cd "$(dirname "$0")/.."; source leonardo/natcomm_lib.sh
M4=$WORK/ccapone0/hf/qwen3_4b_thinking; ST=$NC_ST; OUT=runs_synth/natcomm_sets
NC_TIER=CORE; NC_CAP_CORE=${NC_CAP_CORE:-1600}
SFT="--lora-r 32 --lora-alpha 64 --lr 5e-5 --accum 2 --no-merge"            # the original arms' recipe (run_math_brev.sh)
nc_lock dose || exit 1
nc_preflight_eval synth/math_harvest.py || exit 1
nc_need runs_synth/agent_math_harv_4b/shard_0.jsonl "Qwen3-4B MATH harvest" || exit 1

# 1. second token-matched replicate
if [ ! -f "$OUT/q4b_Rtm.jsonl" ] || [ ! -f "$OUT/q4b_Ltm.jsonl" ]; then
  NC_CPU_TAIL=14 NC_CPU_TIME=02:00:00 nc_cpu sets_q4b "$NC_PY synth/natcomm_sft_sets.py --out-dir $OUT --max-len 12288 --verify \
    --harvest 'runs_synth/agent_math_harv_4b/shard_*.jsonl' --prefix q4b --tokenizer $M4 --sets Rtm,Ltm --tm-seed 20261120" || exit 1
fi
E_SOLVER=synth/math_harvest.py; E_DATA=synth/math/test500.jsonl; E_IDS=synth/math/test500_ids.txt; E_K=4
E_MAXTOK=32768; E_MAXLEN=34816; E_NSH=4; E_MODE=array; E_TIME=${E_TIME:-03:00:00}; E_SLICE=${E_SLICE:-165}; E_TEXTS=1
E_TEMP=0.6

# 2a. existing token-matched students at sampling seed 1 (no training needed: start at once)
nc_try nc_eval nc32_q4_Rtm_s1 "$ST/nc_q4_Rtm/merged" 1
nc_try nc_eval nc32_q4_Ltm_s1 "$ST/nc_q4_Ltm/merged" 1
# 1b + 2b. the new replicate: train, merge, evaluate at sampling seed 0
nc_try nc_student nc_q4b_Rtm "$M4" "$OUT/q4b_Rtm.jsonl" "$SFT" && nc_try nc_eval nc32_q4_q4bRtm_s0 "$ST/nc_q4b_Rtm/merged" 0
nc_try nc_student nc_q4b_Ltm "$M4" "$OUT/q4b_Ltm.jsonl" "$SFT" && nc_try nc_eval nc32_q4_q4bLtm_s0 "$ST/nc_q4b_Ltm/merged" 0
nc_log "dose lane finished | CORE spent so far $(nc_spent CORE) CPUh (cap $NC_CAP_CORE)"
