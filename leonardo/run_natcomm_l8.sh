#!/bin/bash
# NatComm revision: DeepSeek-R1-Distill-Llama-8B on MATH-500 at the 8k cap, evaluated as in run_math_llama.sh (T 0.6, 4 samples
# per problem, cap 8192; SFT without system prompt and with the Llama end token, same LoRA recipe), now with per-sample
# records and saved texts. Evaluations run as 30-min debug jobs, one A100 per shard (4 shards). Lanes:
#   sham   CORE  zero-step adapter (--lr 0 --max-steps 1) through the exact pipeline -> CPU sham check (weights; tokenizer
#                behaviour, i.e. the PreTrainedTokenizerFast workaround of the transformers-5 space-dropping bug) -> 8k
#                evaluation, seeds 0-2, to compare with the untouched model's seeds 0-2 (math_l8_eval_base_s*)
#   ka     EXT   keep-all SFT (every distinct verified solution) -> seeds 0-2 -> forced answers at 4k and 8k (8k = the cap)
#   rep1   EXT   training replicate 2 of S, R (redrawn) and L: sft_student --shortest-per-task --seed 1 on the same harvest,
#   rep2   EXT   as run_math_reps.sh did for Qwen3-4B (seed 2 for rep2); two sampling seeds each; forced answers
#   texts  EXT   run-1 arms (mathl8_S/R/L) and the untouched model re-evaluated at seed 0 with saved texts, then forced
#                answers, so that every R1-Llama arm also has a forced-answer evaluation at the cap
# Tags: students nc_l8_sham, nc_l8_KA, nc_l8_<S|R|L>_rep<r>; evaluations nc8_l8_<arm>_s<seed>, nc8_l8_<S|R|L>rep<r>_s<seed>.
#   nohup bash leonardo/run_natcomm_l8.sh sham > logs/run_natcomm_l8_sham.log 2>&1 &
cd "$(dirname "$0")/.."; source leonardo/natcomm_lib.sh
LANE=${1:?lane: sham | ka | rep1 | rep2 | texts}
M8=$WORK/ccapone0/hf/r1_llama_8b; ST=$NC_ST; SETS=runs_synth/natcomm_sets
export SFT_NO_SYS=1 SFT_END="<｜end▁of▁sentence｜>"       # as run_math_llama.sh (distill/sft_student.py reads both)
SFT="--lora-r 32 --lora-alpha 64 --lr 5e-5 --accum 2 --no-merge"
SHAM="--lora-r 32 --lora-alpha 64 --lr 0 --accum 2 --max-steps 1 --no-merge"
NC_CAP_CORE=${NC_CAP_CORE:-1600}; NC_CAP_EXT=${NC_CAP_EXT:-1400}
grep -q PreTrainedTokenizerFast "$M8/tokenizer_config.json" \
  || { nc_log "$M8/tokenizer_config.json lacks tokenizer_class PreTrainedTokenizerFast (transformers-5 trap)"; exit 1; }
nc_lock "l8_$LANE" || exit 1
nc_preflight_eval synth/math_harvest.py || exit 1
E_SOLVER=synth/math_harvest.py; E_DATA=synth/math/test500.jsonl; E_IDS=synth/math/test500_ids.txt; E_K=4; E_TEMP=0.6
E_MAXTOK=8192; E_MAXLEN=10240; E_NSH=4; E_MODE=dbg; E_SLICE=24; E_TEXTS=1; E_MAXIT=${E_MAXIT:-20}
FB=${F_BUDGETS:-4096,8192}
ev() {  # TAG MODEL SEED [forced]
  nc_eval "$1" "$2" "$3" || return 1
  if [ -n "${4:-}" ]; then nc_forced "$1" "$2" "$FB"; fi
}
case $LANE in
sham)
  NC_TIER=CORE
  nc_need $SETS/l8_S.jsonl "run leonardo/run_natcomm_sets.sh first" || exit 1
  nc_try nc_student nc_l8_sham "$M8" $SETS/l8_S.jsonl "$SHAM" || exit 1
  if [ ! -f "$ST/nc_l8_sham/sham_check.json" ]; then
    NC_CPU_TAIL=12 nc_cpu sham_l8 "$NC_PY synth/natcomm_sham_check.py $M8 $ST/nc_l8_sham/merged --weights --json $ST/nc_l8_sham/sham_check.json"
  fi
  for s in 0 1 2; do nc_try ev "nc8_l8_sham_s$s" "$ST/nc_l8_sham/merged" "$s"; done
  ;;
ka)
  NC_TIER=EXT
  nc_need $SETS/l8_KA.jsonl "run leonardo/run_natcomm_sets.sh first" || exit 1
  nc_try nc_student nc_l8_KA "$M8" $SETS/l8_KA.jsonl "$SFT" || exit 1
  for s in 0 1 2; do nc_try ev "nc8_l8_KA_s$s" "$ST/nc_l8_KA/merged" "$s" forced; done
  ;;
rep1|rep2)
  NC_TIER=EXT; REP=${LANE#rep}
  for a in S:shortest R:random L:longest; do
    nc_try nc_student "nc_l8_${a%%:*}_rep$REP" "$M8" "runs_synth/agent_math_harv_l8/shard_*.jsonl" \
      "--shortest-per-task --select ${a#*:} $SFT --seed $REP"
  done
  for s in 0 1; do
    for a in S R L; do
      t=nc_l8_${a}_rep$REP
      [ -f "$ST/$t/MERGE_DONE" ] && nc_try ev "nc8_l8_${a}rep${REP}_s$s" "$ST/$t/merged" "$s" forced
    done
  done
  ;;
texts)
  NC_TIER=EXT
  nc_try ev nc8_l8_base_s0 "$M8" 0 forced
  for a in S R L; do nc_try ev "nc8_l8_${a}_s0" "$ST/mathl8_$a/merged" 0 forced; done
  ;;
*) echo "unknown lane $LANE"; exit 1;;
esac
nc_log "l8 lane $LANE finished | $NC_TIER spent so far $(nc_spent "$NC_TIER") CPUh"
