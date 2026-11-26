#!/bin/bash
# NatComm revision, package B: published length-reduction checkpoints on MATH-500 under a token cap
# (pre-registration paper_brevity/natcomm/PREREG_EXTERNAL.md; analysis synth/natcomm_ext_analysis.py).
#   download  login node: fetch the 7 checkpoints into $EXT (HF hub online)
#   prereg    write the SHA-256 of the pre-registration, scripts and launcher with a UTC timestamp to logs/PREREG_EXTERNAL.sha256
#   calib     r1base only (cost calibration: stop if its evaluation costs > 30 CPUh)
#   run       every arm: 16k evaluation (K=4, seed 0) then forced answers at 1k/2k/4k/8k/16k
#   nohup bash leonardo/run_natcomm_ext.sh run > logs/run_natcomm_ext.log 2>&1 &
cd "$(dirname "$0")/.."; source leonardo/natcomm_lib.sh
LANE=${1:?lane: download | prereg | calib | run}
EXT=$WORK/ccapone0/hf/ext
NC_TIER=BX; NC_CAP_BX=${NC_CAP_BX:-220}
P_DSR="{problem} Let's think step by step and output the final answer within \\boxed{{}}."
model_of() {
  case $1 in
    r1base) echo deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B;;
    arora_a0) echo daman1209arora/alpha_0_DeepSeek-R1-Distill-Qwen-1.5B;;
    arora_a02) echo daman1209arora/alpha_0.2_DeepSeek-R1-Distill-Qwen-1.5B;;
    thinkprune) echo Shiyu-Lab/DeepSeek-R1-Distill-Qwen-1.5B-thinkprune-iter2k;;
    dler) echo nvidia/DLER-R1-1.5B-Research;;
    dsr) echo agentica-org/DeepScaleR-1.5B-Preview;;
    l1max4k|l1max1k) echo l3lab/L1-Qwen-1.5B-Max;;
  esac
}
ARMS="r1base arora_a0 arora_a02 thinkprune dler dsr l1max4k l1max1k"
case $LANE in
download)
  for arm in $ARMS; do
    id=$(model_of $arm); d=$EXT/${id//\//__}
    [ -f "$d/config.json" ] && { nc_log "$id present"; continue; }
    $NC_PY -c "from huggingface_hub import snapshot_download as s; p=s('$id', local_dir='$d', allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.py']); print(p)" \
      && $NC_PY -c "from huggingface_hub import HfApi; print('$id', HfApi().model_info('$id').sha)" >> logs/ext_model_revisions.txt
  done
  cat logs/ext_model_revisions.txt;;
prereg)
  f=logs/PREREG_EXTERNAL.sha256
  [ -f "$f" ] && { nc_log "$f exists, not overwritten"; cat "$f"; exit 0; }
  { date -u +%FT%TZ
    for x in paper_brevity/natcomm/PREREG_EXTERNAL.md synth/natcomm_ext_analysis.py synth/math_harvest.py \
             synth/forced_answer.py leonardo/run_natcomm_ext.sh synth/math/test500.jsonl; do
      echo "$(nc_sha "$x")  $x"; done; } > "$f"
  cat "$f";;
calib|run)
  nc_lock "ext_$LANE" || exit 1
  [ -f logs/PREREG_EXTERNAL.sha256 ] || { nc_log "run the prereg lane first"; exit 1; }
  nc_preflight_eval synth/math_harvest.py || exit 1
  nc_preflight_forced || exit 1
  E_SOLVER=synth/math_harvest.py; E_DATA=synth/math/test500.jsonl; E_IDS=synth/math/test500_ids.txt; E_K=4
  E_MAXTOK=16384; E_MAXLEN=18432; E_NSH=2; E_MODE=array; E_TIME=${E_TIME:-02:00:00}; E_SLICE=${E_SLICE:-105}; E_TEXTS=1
  E_TEMP=0.6; E_EXTRA="--think-prefill --strip-bos --top-k -1"
  export VLLM_DTYPE=bfloat16          # four checkpoints are stored in float32: run every arm in bfloat16
  FB=1024,2048,4096,8192,16384
  one() {  # ARM
    local arm=$1 id m
    id=$(model_of $arm); m=$EXT/${id//\//__}
    case $arm in
      dsr) export MATH_PROMPT="$P_DSR";;
      l1max4k) export MATH_PROMPT="$P_DSR Think for maximum 4096 tokens.";;
      l1max1k) export MATH_PROMPT="$P_DSR Think for maximum 1024 tokens.";;
      *) unset MATH_PROMPT;;            # R1 family: math_harvest.py's default prompt (the paper's)
    esac
    nc_log "ncx_${arm}_s0: model $id, prompt ${MATH_PROMPT:-default}"
    nc_eval "ncx_${arm}_s0" "$m" 0 || return 1
    nc_log "ncx_${arm}_s0: $(tail -1 runs_synth/agent_ncx_${arm}_s0/EVAL_DONE 2>/dev/null) | eval CPUh $(nc_tag_cpuh ncx_${arm}_s0)"
    nc_forced "ncx_${arm}_s0" "$m" "$FB"
  }
  if [ "$LANE" = calib ]; then one r1base; exit $?; fi
  if [ ! -f runs_synth/agent_ncx_r1base_s0/FORCED_DONE ]; then nc_log "run the calib lane first"; exit 1; fi
  for arm in $ARMS; do nc_try one "$arm"; done
  nc_log "ext lane finished | BX spent $(nc_spent BX) CPUh (cap $NC_CAP_BX)";;
*) echo "unknown lane $LANE"; exit 1;;
esac
