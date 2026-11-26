#!/bin/bash
# NatComm existing-data analyses (synth/natcomm_analyses.py) on the records in runs_synth/. Budget-free partition.
#   sbatch leonardo/job_natcomm_analyses.slurm            (from $HOME/arcagi2)
#SBATCH --job-name=nc_analyses
#SBATCH --account=IscrC_LUCID
#SBATCH --partition=lrd_all_serial
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
export ARC_BASE="$HOME"
source "$SLURM_SUBMIT_DIR/leonardo/env.sh"
cd "$SLURM_SUBMIT_DIR"
python -u synth/natcomm_analyses.py
echo "ANALYSES DONE rc=$?"
