# Sourced by job scripts. No module purge side effects for interactive use.
module purge
# Names verified on Leonardo 2026-09-14 (base profile, spack 0.22 stack).
module load python/3.11.7
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
# $WORK was not yet writable on 2026-09-14 (project storage still provisioning): fall back to $SCRATCH.
if [ -n "$ARC_BASE" ]; then :; elif [ -w "$WORK" ]; then ARC_BASE="$WORK"; else ARC_BASE="$SCRATCH"; fi
export ARC_BASE
source "$ARC_BASE/venv_arc/bin/activate"

# Compute nodes are offline: force every library into offline mode.
export HF_HOME="$ARC_BASE/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
