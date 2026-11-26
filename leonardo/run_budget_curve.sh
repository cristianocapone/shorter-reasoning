#!/bin/bash
# Budget curve for NVARC+ style pooling on dev-40: more independent passes vs cheaper passes.
# Runs chains two at a time (debug QOS = 2 jobs/user), after any running nvarc_ttt jobs finish.
cd "$(dirname "$0")/.."
CH=$HOME/arcagi2/runs_nvarc/dev40_challenges.json; M=$WORK/ccapone0/hf/nvarc_qwen3_4b
wait_free() { while squeue -u ccapone0 -h -o %j | grep -q nvarc_ttt; do sleep 120; done; }
run2() {  # run2 TAG1 "ENV1" TAG2 "ENV2"
  wait_free
  env $2 bash leonardo/ttt_chain.sh $1 $M $CH 40 10 > logs/tttchain_$1.log 2>&1 &
  sleep 60
  env $4 bash leonardo/ttt_chain.sh $3 $M $CH 40 10 > logs/tttchain_$3.log 2>&1 &
  wait
  echo "$(date +%F_%T) done $1 $3"
}
H="NVARC_AUG_N=8 NVARC_EVAL_N=1"; Q="NVARC_AUG_N=4 NVARC_EVAL_N=1"
run2 ttt_dev40_halfC "$H NVARC_LORA_SEED=271 NVARC_TRAIN_AUG_SEED=31 NVARC_EVAL_AUG_SEED=37 NVARC_SCORE_SEED_OFFSET=11" \
     ttt_dev40_qA    "$Q"
run2 ttt_dev40_qB    "$Q NVARC_LORA_SEED=137 NVARC_TRAIN_AUG_SEED=17 NVARC_EVAL_AUG_SEED=29 NVARC_SCORE_SEED_OFFSET=7" \
     ttt_dev40_qC    "$Q NVARC_LORA_SEED=271 NVARC_TRAIN_AUG_SEED=31 NVARC_EVAL_AUG_SEED=37 NVARC_SCORE_SEED_OFFSET=11"
run2 ttt_dev40_qD    "$Q NVARC_LORA_SEED=409 NVARC_TRAIN_AUG_SEED=43 NVARC_EVAL_AUG_SEED=47 NVARC_SCORE_SEED_OFFSET=13" \
     ttt_dev40_fullB "NVARC_LORA_SEED=137 NVARC_TRAIN_AUG_SEED=17 NVARC_EVAL_AUG_SEED=29 NVARC_SCORE_SEED_OFFSET=7"
echo "$(date +%F_%T) budget curve complete"
