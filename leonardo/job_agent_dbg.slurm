#!/bin/bash
# Same as job_agent.slurm but for the debug QOS (starts at once, 30 min): one node, two shards side by side
# (2 A100 each). solve_agent.py writes every task as soon as it is finished and resumes, so chained 30-min jobs
# make progress while the normal queue is congested. env: TAG, SHARD_A, SHARD_B, NSH, IDS, ARC_DIR, THREADS, ROUNDS,
#                                                          MAXTOK, MAXLEN, TPB, CHATKW, HARVEST
#SBATCH --job-name=agent_dbg
#SBATCH --account=IscrC_LUCID
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
module purge; module load python/3.11.7
source $WORK/ccapone0/venv_vllm2/bin/activate
export LD_LIBRARY_PATH=$WORK/ccapone0/cuda_compat/13-0/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
export BARC_SEEDS="$SLURM_SUBMIT_DIR/external_rules/barc_repo/seeds" BARC_TIMEOUT=${BARC_TIMEOUT:-10}
TAG=${TAG:?}; mkdir -p runs_synth/agent_$TAG
run() {  # shard, gpus, port
  CUDA_VISIBLE_DEVICES=$2 VLLM_PORT=$3 python -u synth/solve_agent.py --model ${MODEL:-$WORK/ccapone0/hf/qwen3.8_27b} --tp 2 \
    --arc-dir ${ARC_DIR:-ARC-AGI-2/data/training} --ids ${IDS:-synth/arc2_train_heldout_clean.txt} \
    --threads ${THREADS:-8} --rounds ${ROUNDS:-8} --max-tokens ${MAXTOK:-12000} --tasks-per-batch ${TPB:-6} --workers 12 \
    ${MAXLEN:+--max-model-len $MAXLEN} ${SEED:+--seed $SEED} ${TERSE:+--terse} ${TOKBUDGET:+--task-token-budget $TOKBUDGET} ${CHATKW:+--chat-kwargs "$CHATKW"} \
    ${HARVEST:+--harvest} ${KEEP:+--keep-transcripts $KEEP} --shard $1 --n-shards ${NSH:-4} --out runs_synth/agent_$TAG/shard_$1.jsonl > logs/agent_dbg-${SLURM_JOB_ID}_s$1.out 2>&1
}
run ${SHARD_A:?} 0,1 29500 &
sleep 20
run ${SHARD_B:?} 2,3 29600 &
wait
