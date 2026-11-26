#!/bin/bash
# Budget curve, part 2: three-eighths-cost passes (AUG_N 6 = 48 TTT sequences, 8 views), 3 seeds, after part 1.
cd "$(dirname "$0")/.."
CH=$HOME/arcagi2/runs_nvarc/dev40_challenges.json; M=$WORK/ccapone0/hf/nvarc_qwen3_4b
until grep -q "budget curve complete" logs/budget_curve.log 2>/dev/null; do sleep 300; done
wait_free() { while squeue -u ccapone0 -h -o %j | grep -q nvarc_ttt; do sleep 120; done; }
E="NVARC_AUG_N=6 NVARC_EVAL_N=1"
wait_free
env $E bash leonardo/ttt_chain.sh ttt_dev40_eA $M $CH 40 10 > logs/tttchain_ttt_dev40_eA.log 2>&1 &
sleep 60
env $E NVARC_LORA_SEED=137 NVARC_TRAIN_AUG_SEED=17 NVARC_EVAL_AUG_SEED=29 NVARC_SCORE_SEED_OFFSET=7 bash leonardo/ttt_chain.sh ttt_dev40_eB $M $CH 40 10 > logs/tttchain_ttt_dev40_eB.log 2>&1 &
wait; wait_free
env $E NVARC_LORA_SEED=271 NVARC_TRAIN_AUG_SEED=31 NVARC_EVAL_AUG_SEED=37 NVARC_SCORE_SEED_OFFSET=11 bash leonardo/ttt_chain.sh ttt_dev40_eC $M $CH 40 10 > logs/tttchain_ttt_dev40_eC.log 2>&1
echo "$(date +%F_%T) budget curve part 2 complete"
