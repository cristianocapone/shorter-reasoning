#!/bin/bash
# NatComm revision, EXTENDED: training replicates for APPS (review must_fix 7). Shortest, random (redrawn by the seed) and
# longest replay retrained from the same Qwen3-4B APPS harvest (code_harv_4b) with training seed 1 (lane rep1) or 2 (rep2),
# recipe of run_paper_extra.sh lane B, merged, and evaluated at the 8k cap with synth/code_harvest.py (K=4, T 0.6, sampling
# seeds 0-1, per-sample records and texts) as 30-min debug jobs, one A100 per shard. No forced answers (code: no single
# answer to force). CODE_WORKERS=8 test threads per shard (4 shards share the node's 32 cores).
# Tags: students nc_c4_<S|R|L>_rep<r>; evaluations nc8_c4_<S|R|L>rep<r>_s<seed>.
#   nohup bash leonardo/run_natcomm_apps.sh rep1 > logs/run_natcomm_apps_rep1.log 2>&1 &
cd "$(dirname "$0")/.."; source leonardo/natcomm_lib.sh
LANE=${1:?lane: rep1 | rep2}; REP=${LANE#rep}
M4=$WORK/ccapone0/hf/qwen3_4b_thinking; ST=$NC_ST
NC_TIER=EXT; NC_CAP_EXT=${NC_CAP_EXT:-1400}
SFT="--lora-r 32 --lora-alpha 64 --lr 5e-5 --accum 2 --no-merge"
export CODE_WORKERS=${CODE_WORKERS:-8}
nc_lock "apps_$LANE" || exit 1
nc_preflight_eval synth/code_harvest.py || exit 1
nc_need runs_synth/agent_code_harv_4b/shard_0.jsonl "APPS harvest" || exit 1
E_SOLVER=synth/code_harvest.py; E_DATA=synth/code/test500.jsonl; E_IDS=synth/code/test500_ids.txt; E_K=4; E_TEMP=0.6
E_MAXTOK=8192; E_MAXLEN=12288; E_NSH=4; E_MODE=dbg; E_SLICE=24; E_TEXTS=1; E_MAXIT=${E_MAXIT:-25}
for a in S:shortest R:random L:longest; do
  nc_try nc_student "nc_c4_${a%%:*}_rep$REP" "$M4" "runs_synth/agent_code_harv_4b/shard_*.jsonl" \
    "--shortest-per-task --select ${a#*:} $SFT --seed $REP"
done
for s in 0 1; do
  for a in S R L; do
    t=nc_c4_${a}_rep$REP
    [ -f "$ST/$t/MERGE_DONE" ] && nc_try nc_eval "nc8_c4_${a}rep${REP}_s$s" "$ST/$t/merged" "$s"
  done
done
nc_log "apps lane $LANE finished | EXT spent so far $(nc_spent EXT) CPUh (cap $NC_CAP_EXT)"
