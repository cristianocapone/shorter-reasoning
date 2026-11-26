#!/bin/bash
# NatComm revision (2026-11): forced answers (synth/forced_answer.py) on one evaluation tag's saved generations, 1 A100.
# For every saved sample longer than a budget b the generation is cut at b tokens and the model is made to answer; the
# script writes per-sample forced-answer correctness for every budget (interface of the evaluation code, 2026-11-02).
# forced_answer.py resumes from --out and exits by itself after DIRECT_SLICE_MIN (= NC_SLICE) minutes, before the time limit;
# nc_forced resubmits until every sample x budget row is written.
# Submitted by leonardo/natcomm_lib.sh (nc_forced). env: NC_TAG NC_MODEL NC_TEXTS_GLOB NC_BUDGETS NC_OUT NC_SLICE NC_EXTRA
#SBATCH --job-name=nc_forced
#SBATCH --account=IscrC_LUCID
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
module purge; module load python/3.11.7
source $WORK/ccapone0/venv_vllm2/bin/activate
export LD_LIBRARY_PATH=$WORK/ccapone0/cuda_compat/13-0/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
export DIRECT_SLICE_MIN=${NC_SLICE:-0}
echo "$(date +%F_%T) forced answers for ${NC_TAG:?} on $(hostname): budgets ${NC_BUDGETS:?}"
python -u synth/forced_answer.py --model ${NC_MODEL:?} --texts "${NC_TEXTS_GLOB:?}" --budgets ${NC_BUDGETS} \
  --out ${NC_OUT:?} ${NC_EXTRA:-} > logs/nc_fa_${NC_TAG}-${SLURM_JOB_ID}_py.out 2>&1
rc=$?
echo "$(date +%F_%T) exit $rc"
exit $rc
