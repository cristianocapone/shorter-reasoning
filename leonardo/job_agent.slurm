#!/bin/bash
# T1: open-weight reasoning teacher in a code loop (synth/solve_agent.py). One array task = 2 A100 (TP=2).
# env: TAG, IDS, ARC_DIR, THREADS, ROUNDS, NSH (= array size), MAXTOK, MODEL, HARVEST=1 (distillation data), KEEP,
#      MAXLEN (max model len), TPB (tasks per batch), CHATKW (JSON chat_template_kwargs, e.g. '{"reasoning_effort": "high"}' for gpt-oss)
#SBATCH --job-name=agent
#SBATCH --account=IscrC_LUCID
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=240G
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
module purge; module load python/3.11.7
source $WORK/ccapone0/venv_vllm2/bin/activate
# driver 535 + cu130 wheels: user-space forward-compat libcuda (see leonardo/README.md)
export LD_LIBRARY_PATH=$WORK/ccapone0/cuda_compat/13-0/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
export BARC_SEEDS="$SLURM_SUBMIT_DIR/external_rules/barc_repo/seeds" BARC_TIMEOUT=${BARC_TIMEOUT:-10}
TAG=${TAG:?}; mkdir -p runs_synth/agent_$TAG
python -u synth/solve_agent.py --model ${MODEL:-$WORK/ccapone0/hf/qwen3.8_27b} --tp 2 \
  --arc-dir ${ARC_DIR:-ARC-AGI-2/data/training} --ids ${IDS:-synth/arc2_train_heldout_clean.txt} \
  --threads ${THREADS:-8} --rounds ${ROUNDS:-8} --max-tokens ${MAXTOK:-12000} --tasks-per-batch ${TPB:-6} \
  ${HARVEST:+--harvest} --keep-transcripts ${KEEP:-2} ${CHATKW:+--chat-kwargs "$CHATKW"} ${MAXLEN:+--max-model-len $MAXLEN} ${TOKBUDGET:+--task-token-budget $TOKBUDGET} ${SHARE:+--share-best $SHARE} ${TERSE:+--terse} \
  --shard $SLURM_ARRAY_TASK_ID --n-shards ${NSH:-4} --out runs_synth/agent_$TAG/shard_$SLURM_ARRAY_TASK_ID.jsonl
