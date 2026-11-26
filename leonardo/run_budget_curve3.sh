#!/bin/bash
# Part 3: finish qB / qC (their chains burned iterations on the 10-02 disk-quota errors; outputs resume per task).
cd "$(dirname "$0")/.."
CH=$HOME/arcagi2/runs_nvarc/dev40_challenges.json; M=$WORK/ccapone0/hf/nvarc_qwen3_4b
until grep -q "part 2 complete" logs/budget_curve2.log 2>/dev/null; do sleep 300; done
while squeue -u ccapone0 -h -o %j | grep -q nvarc_ttt; do sleep 120; done
Q="NVARC_AUG_N=4 NVARC_EVAL_N=1"
env $Q NVARC_LORA_SEED=137 NVARC_TRAIN_AUG_SEED=17 NVARC_EVAL_AUG_SEED=29 NVARC_SCORE_SEED_OFFSET=7 bash leonardo/ttt_chain.sh ttt_dev40_qB $M $CH 40 10 >> logs/tttchain_ttt_dev40_qB.log 2>&1 &
sleep 60
env $Q NVARC_LORA_SEED=271 NVARC_TRAIN_AUG_SEED=31 NVARC_EVAL_AUG_SEED=37 NVARC_SCORE_SEED_OFFSET=11 bash leonardo/ttt_chain.sh ttt_dev40_qC $M $CH 40 10 >> logs/tttchain_ttt_dev40_qC.log 2>&1 &
wait
echo "$(date +%F_%T) budget curve part 3 complete"
