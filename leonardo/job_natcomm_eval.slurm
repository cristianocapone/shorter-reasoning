#!/bin/bash
# NatComm revision (2026-11): evaluation shards of synth/math_harvest.py or synth/code_harvest.py, ONE SHARD PER GPU (TP=1:
# the 4B/8B models fit one A100, and four independent engines decode more tokens per GPU than TP=2 pairs).
#   array mode (normal QOS, the defaults below): sbatch --array=0-3 ...  -> shard = SLURM_ARRAY_TASK_ID on the task's GPU
#   multi mode (debug QOS): sbatch --qos=boost_qos_dbg --time=00:30:00 --gres=gpu:N --cpus-per-task=8N --mem=120NG with
#                           NC_SHARDS=a:b:...  -> one shard per allocated GPU
# Submitted by leonardo/natcomm_lib.sh (nc_eval), which sets every NC_* variable. Records go to
# runs_synth/agent_$NC_TAG/shard_<i>.jsonl (resumable: finished problems are skipped), saved generations (NC_TEXTS=1) to
# shard_<i>.texts.jsonl.gz next to them (appended; readers keep the last copy of a redone problem). The solver exits by
# itself after NC_SLICE minutes, before the time limit; whatever is unfinished then is redone by the next job.
# env: NC_TAG NC_MODEL NC_SOLVER NC_DATA NC_IDS NC_NSH NC_K NC_MAXTOK NC_MAXLEN NC_TEMP NC_SEED NC_SLICE NC_TEXTS NC_SHARDS NC_EXTRA
#SBATCH --job-name=nc_eval
#SBATCH --account=IscrC_LUCID
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
module purge; module load python/3.11.7
source $WORK/ccapone0/venv_vllm2/bin/activate
export LD_LIBRARY_PATH=$WORK/ccapone0/cuda_compat/13-0/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
export MATH_DATA=${NC_DATA:?} MATH_K=${NC_K:?} MATH_EVAL=1 DIRECT_SLICE_MIN=${NC_SLICE:?}
D=runs_synth/agent_${NC_TAG:?}; mkdir -p "$D"
JID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}              # the id the driver knows (array master id in array mode)
echo "$(date +%F_%T) $NC_TAG on $(hostname): model $NC_MODEL seed $NC_SEED T $NC_TEMP cap $NC_MAXTOK k $NC_K GPUs ${CUDA_VISIBLE_DEVICES:-?}"
run() {  # SHARD [GPU] -> one solver process; GPU = one entry of the job's CUDA_VISIBLE_DEVICES (multi mode)
  local dev=${2:-$CUDA_VISIBLE_DEVICES}
  CUDA_VISIBLE_DEVICES=$dev python -u ${NC_SOLVER:?} --model ${NC_MODEL:?} --tp 1 --ids ${NC_IDS:?} \
    --shard $1 --n-shards ${NC_NSH:?} --out $D/shard_$1.jsonl --max-tokens ${NC_MAXTOK:?} --max-model-len ${NC_MAXLEN:?} \
    --temperature ${NC_TEMP:?} --seed ${NC_SEED:?} --data $NC_DATA ${NC_TEXTS:+--save-texts auto} ${NC_EXTRA:-} \
    > logs/nc_ev_${NC_TAG}-${JID}_s$1.out 2>&1
}
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  run "$SLURM_ARRAY_TASK_ID"
else
  IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:?}"
  i=0
  for s in $(echo "${NC_SHARDS:?}" | tr ':' ' '); do
    run "$s" "${GPUS[$i]:?more shards than GPUs}" &
    i=$((i + 1)); sleep 20
  done
  wait
fi
echo "$(date +%F_%T) $NC_TAG: job $SLURM_JOB_ID done"
