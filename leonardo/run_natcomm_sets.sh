#!/bin/bash
# NatComm revision, step 1 (CPU only, on the budget-free lrd_all_serial partition): build the new SFT training sets from
# the EXISTING harvests and the per-arm training-set statistics (review must_fix 6, 9, 14) with synth/natcomm_sft_sets.py:
#   q4 = Qwen3-4B MATH harvest (math_harv_4b):  S R L (= the original arms) KA (keep-all) Rtm Ltm (token-matched) Sk2
#   l8 = R1-Llama-8B MATH harvest (math_harv_l8), with SFT_NO_SYS and the Llama end token, as run_math_llama.sh trained
#   c4 = Qwen3-4B APPS harvest (code_harv_4b): S R L KA, for the training-set table only
# Exact loss/context tokens with each model's tokenizer, and --verify: the S/R/L files must give distill/sft_student.py
# exactly the samples of --shortest-per-task --select X --seed 0 (what math4b_S/R/L, mathl8_S/R/L, code4b_S/R/L trained on),
# and every file's example and loss-token counts must equal what sft_student.py builds from it. Set files are never
# overwritten with different content (the builder refuses). Output: runs_synth/natcomm_sets/<prefix>_<set>.jsonl and
# <prefix>_stats.json (examples, loss tokens, optimiser steps, target-length quantiles, length tilt, problems by number of
# verified/distinct solutions).
#   nohup bash leonardo/run_natcomm_sets.sh > logs/run_natcomm_sets.log 2>&1 &
cd "$(dirname "$0")/.."; source leonardo/natcomm_lib.sh
M4=$WORK/ccapone0/hf/qwen3_4b_thinking; M8=$WORK/ccapone0/hf/r1_llama_8b; OUT=runs_synth/natcomm_sets
NC_TIER=CORE; NC_CPU_TAIL=14; NC_CPU_TIME=02:00:00
nc_lock sets || exit 1
nc_need runs_synth/agent_math_harv_4b/shard_0.jsonl "Qwen3-4B MATH harvest" || exit 1
nc_need runs_synth/agent_math_harv_l8/shard_0.jsonl "R1-Llama-8B MATH harvest" || exit 1
nc_need runs_synth/agent_code_harv_4b/shard_0.jsonl "Qwen3-4B APPS harvest" || exit 1
grep -q PreTrainedTokenizerFast "$M8/tokenizer_config.json" \
  || { nc_log "$M8/tokenizer_config.json lacks the PreTrainedTokenizerFast workaround: token counts would be wrong"; exit 1; }
mkdir -p "$OUT"
B="$NC_PY synth/natcomm_sft_sets.py --out-dir $OUT --max-len 12288 --verify"
nc_cpu sets_q4 "$B --harvest 'runs_synth/agent_math_harv_4b/shard_*.jsonl' --prefix q4 --tokenizer $M4" || exit 1
nc_cpu sets_l8 "SFT_NO_SYS=1 SFT_END='<｜end▁of▁sentence｜>' $B --harvest 'runs_synth/agent_math_harv_l8/shard_*.jsonl' --prefix l8 --tokenizer $M8" || exit 1
nc_cpu sets_c4 "$B --harvest 'runs_synth/agent_code_harv_4b/shard_*.jsonl' --prefix c4 --tokenizer $M4 --sets S,R,L,KA" || exit 1
nc_log "sets built: $(ls "$OUT"/*.jsonl | grep -c .) files; statistics in $OUT/{q4,l8,c4}_stats.json"   # grep -c: BSD wc pads
