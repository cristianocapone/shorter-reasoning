#!/bin/bash
# NatComm revision, CORE: Qwen3-4B on MATH-500 at a 32k cap (review must_fix 3, 6, 7, 9; the editor's decision point).
# One driver per lane; each lane is idempotent (markers, see leonardo/natcomm_lib.sh) and can be relaunched at any time.
#   train   LoRA SFT with the original arms' recipe on the sets of run_natcomm_sets.sh, then merge (debug QOS):
#             nc_q4_KA   keep-all (every distinct verified solution)
#             nc_q4_sham zero-step control: --lr 0 --max-steps 1, merged through the same pipeline, then the CPU check
#                        synth/natcomm_sham_check.py (weights bit-identical? tokenizer behaviour identical?)
#             nc_q4_Rtm, nc_q4_Ltm  token-matched: R and L on S's number of loss tokens (fewer problems)
#   e32s0   evaluations at max_tokens 32768, K=4, sampling seed 0, per-sample records + saved texts (normal QOS, one A100
#           per shard), in priority order base S R L | KA sham Rtm Ltm | base at T=$LOWT; after each, forced answers at
#           4k/8k/16k/32k on its texts. S/R/L are the run-1 students (math4b_S/R/L), untouched since the 8k paper runs.
#   e32s1   the same with sampling seed 1 (runs in parallel with e32s0); LEAN=1 stops it after KA (CORE-lean)
# Post-hoc truncation of the 32k generations gives A(t), F(t), Q(t) for every t <= 32k, including 8k and 16k (the R arm at
# 16k that the paper lacks). Tags: nc32_q4_<arm>_s<seed>; students $NC_ST/nc_q4_<arm>.
#   nohup bash leonardo/run_natcomm_q4.sh train > logs/run_natcomm_q4_train.log 2>&1 &
#   nohup bash leonardo/run_natcomm_q4.sh e32s0 > logs/run_natcomm_q4_e32s0.log 2>&1 &
#   nohup bash leonardo/run_natcomm_q4.sh e32s1 > logs/run_natcomm_q4_e32s1.log 2>&1 &
cd "$(dirname "$0")/.."; source leonardo/natcomm_lib.sh
LANE=${1:?lane: train | e32s0 | e32s1}
M4=$WORK/ccapone0/hf/qwen3_4b_thinking; ST=$NC_ST; SETS=runs_synth/natcomm_sets
NC_TIER=CORE; NC_CAP_CORE=${NC_CAP_CORE:-1600}
SFT="--lora-r 32 --lora-alpha 64 --lr 5e-5 --accum 2 --no-merge"            # the original arms' recipe (run_math_brev.sh)
SHAM="--lora-r 32 --lora-alpha 64 --lr 0 --accum 2 --max-steps 1 --no-merge"
LOWT=${LOWT:-0.4}
nc_lock "q4_$LANE" || exit 1
case $LANE in
train)
  for s in KA S Rtm Ltm; do nc_need $SETS/q4_$s.jsonl "run leonardo/run_natcomm_sets.sh first" || exit 1; done
  nc_try nc_student nc_q4_KA "$M4" $SETS/q4_KA.jsonl "$SFT"
  if nc_try nc_student nc_q4_sham "$M4" $SETS/q4_S.jsonl "$SHAM" && [ ! -f "$ST/nc_q4_sham/sham_check.json" ]; then
    NC_CPU_TAIL=12 nc_cpu sham_q4 "$NC_PY synth/natcomm_sham_check.py $M4 $ST/nc_q4_sham/merged --weights --json $ST/nc_q4_sham/sham_check.json"
  fi
  nc_try nc_student nc_q4_Rtm "$M4" $SETS/q4_Rtm.jsonl "$SFT"
  nc_try nc_student nc_q4_Ltm "$M4" $SETS/q4_Ltm.jsonl "$SFT"
  ;;
e32s0|e32s1)
  seed=${LANE#e32s}
  nc_preflight_eval synth/math_harvest.py || exit 1
  nc_preflight_forced || exit 1
  E_SOLVER=synth/math_harvest.py; E_DATA=synth/math/test500.jsonl; E_IDS=synth/math/test500_ids.txt; E_K=4
  # array mode (normal QOS) only: at 32k a sample can need more than a 24-min debug slice at full batch, and a problem is
  # written only when all its samples finish, so debug-QOS chains could redo the longest problems forever
  E_MAXTOK=32768; E_MAXLEN=34816; E_NSH=4; E_MODE=array; E_TIME=${E_TIME:-03:00:00}; E_SLICE=${E_SLICE:-165}; E_TEXTS=1
  FB=${F_BUDGETS:-4096,8192,16384,32768}
  run() {  # ARM MODEL [TEMPERATURE] [MARKER to wait for: the train lane's MERGE_DONE]
    E_TEMP=${3:-0.6}
    if [ -n "${4:-}" ]; then nc_wait_marker "$4" 48 || return 1; fi
    nc_eval "nc32_q4_$1_s$seed" "$2" "$seed" && nc_forced "nc32_q4_$1_s$seed" "$2" "$FB"
  }
  nc_try run base "$M4"
  nc_try run S "$ST/math4b_S/merged"
  nc_try run R "$ST/math4b_R/merged"
  nc_try run L "$ST/math4b_L/merged"
  nc_try run KA "$ST/nc_q4_KA/merged" 0.6 "$ST/nc_q4_KA/MERGE_DONE"
  if [ "$seed" = 1 ] && [ -n "${LEAN:-}" ]; then          # CORE-lean (NATCOMM_RUNS.md): controls at seed 0 only
    nc_log "LEAN=1: Rtm, Ltm, sham and the low-temperature run are evaluated at seed 0 only"; exit 0
  fi
  nc_try run sham "$ST/nc_q4_sham/merged" 0.6 "$ST/nc_q4_sham/MERGE_DONE"
  nc_try run Rtm "$ST/nc_q4_Rtm/merged" 0.6 "$ST/nc_q4_Rtm/MERGE_DONE"
  nc_try run Ltm "$ST/nc_q4_Ltm/merged" 0.6 "$ST/nc_q4_Ltm/MERGE_DONE"
  nc_try run "baseT${LOWT/./}" "$M4" "$LOWT"
  ;;
*) echo "unknown lane $LANE"; exit 1;;
esac
nc_log "q4 lane $LANE finished | CORE spent so far $(nc_spent CORE) CPUh (cap $NC_CAP_CORE)"
