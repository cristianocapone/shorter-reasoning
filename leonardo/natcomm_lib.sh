# shellcheck shell=bash
# Shared functions of the NatComm revision drivers (leonardo/run_natcomm_*.sh, 2026-11). Source from the repo root on a
# Leonardo login node. Plan, tags and costs: leonardo/NATCOMM_RUNS.md.
#
# Every step is idempotent and leaves a marker; a relaunched driver skips what is marked and resumes the rest:
#   $NC_ST/<tag>/TRAIN_DONE    only when a job's log says DONE, sacct says COMPLETED and the last checkpoint is at the
#                              final step. An existing adapter/ is NOT evidence that training finished: sft_student.py
#                              saves adapter/ every 100 steps, and on 2026-10-30 an squeue glitch made a chain merge a
#                              mid-training adapter. TRAIN_DONE records the job, the steps and the adapter's sha256.
#   $NC_ST/<tag>/MERGE_DONE    after job_merge.slurm printed MERGED for exactly the adapter whose sha256 is in TRAIN_DONE.
#   runs_synth/agent_<tag>/EVAL_DONE    every id has a record with the per-sample fields (and all texts, if saved).
#   runs_synth/agent_<tag>/FORCED_DONE  synth/forced_answer.py finished on that tag's saved texts.
# A job is over only when squeue no longer lists it AND sacct reports a terminal state. A (re)submission first adopts a
# queued or running job of the same name, so a restarted driver never starts a second copy. Every finished job's billing
# is appended to $NC_COST (date tier tag kind job state elapsed_s CPUh A100h), and NC_CAP_<TIER> (CPUh) stops a lane
# before it submits beyond its tier's cap. Bash 3.2-compatible syntax (the tests run on macOS); Leonardo runs bash 4.

NC_ST=${NC_ST:-$WORK/ccapone0/students}
NC_PY=${NC_PY:-$WORK/ccapone0/venv_vllm2/bin/python}   # venv python (transformers, torch): builder, sham check
NC_PY3=${NC_PY3:-python3}                               # any python >= 3.6: synth/natcomm_check.py
NC_POLL=${NC_POLL:-60}                                  # seconds between queue polls
NC_COST=${NC_COST:-logs/natcomm_cost.tsv}
NC_TIER=${NC_TIER:-CORE}
NC_ACCOUNT=${NC_ACCOUNT:-IscrC_LUCID}
mkdir -p logs

nc_log() { echo "$(date +%F_%T) $*"; }

nc_sha() {  # FILE -> sha256 hex
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -c1-64; else shasum -a 256 "$1" | cut -c1-64; fi
}

nc_lock() {  # LANE -> one driver per lane (login nodes are round-robin: a relaunch can land on another node)
  local l="logs/locks/$1" host pid
  mkdir -p logs/locks
  if [ -d "$l" ]; then
    read -r host pid _ 2>/dev/null < "$l/owner"
    if [ "$host" = "$(hostname)" ] && [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      nc_log "lane $1: stale lock of dead pid $pid on this node, taking it over"; rm -rf "$l"
    fi
  fi
  if mkdir "$l" 2>/dev/null; then
    echo "$(hostname) $$ $(date +%F_%T)" > "$l/owner"
    trap 'rm -rf "'"$l"'"' EXIT
    return 0
  fi
  nc_log "lane $1 is locked by: $(cat "$l/owner" 2>/dev/null). If that driver is dead: rm -r $l"
  return 1
}

nc_states() {  # JOB -> sorted unique sacct states of the job (all array elements), space-separated
  sacct -n -X -P -j "$1" -o State 2>/dev/null | awk 'NF {print $1}' | sort -u | tr '\n' ' ' | sed 's/ *$//'
}

nc_wait() {  # JOB -> echoes the final state(s) once squeue AND sacct agree that the job is over
  local j=$1 st miss=0
  while :; do
    if squeue -h -j "$j" 2>/dev/null | grep -q .; then miss=0; sleep "$NC_POLL"; continue; fi
    st=$(nc_states "$j")
    if [ -n "$st" ] && ! echo " $st " | grep -qE ' (PENDING|RUNNING|REQUEUED|REQUEUE_HOLD|REQUEUE_FED|RESIZING|SUSPENDED|CONFIGURING|COMPLETING|SIGNALING|STAGE_OUT|STOPPED) '; then
      echo "$st"; return 0
    fi
    miss=$((miss + 1))          # squeue says gone but sacct says running/unknown: an squeue glitch or sacct lag; wait
    if [ "$miss" -ge "${NC_WAIT_MAXMISS:-180}" ]; then echo "UNCONFIRMED"; return 1; fi
    sleep "$NC_POLL"
  done
}

nc_submit() {  # NAME SBATCH_ARGS... -> prints the job id; adopts a queued/running job with the same name
  local name=$1 j try
  shift
  j=$(squeue -h -u "$USER" -n "$name" -o %A 2>/dev/null | head -1)
  if [ -n "$j" ]; then echo "$j"; return 0; fi
  for try in $(seq 1 "${NC_SUBMIT_TRIES:-90}"); do
    j=$(sbatch --parsable --job-name="$name" "$@" 2>/dev/null | head -1 | cut -d';' -f1)
    if echo "$j" | grep -qE '^[0-9]+$'; then echo "$j"; return 0; fi
    sleep "$NC_POLL"
  done
  return 1
}

nc_cost() {  # TAG KIND JOB -> append the job's billing (CPUh = billing x hours; A100-h = GPUs x hours) to $NC_COST
  mkdir -p "$(dirname "$NC_COST")"
  sacct -n -X -P -j "$3" -o JobID,State,ElapsedRaw,AllocTRES 2>/dev/null | awk -F'|' -v d="$(date +%F_%T)" \
    -v ti="$NC_TIER" -v t="$1" -v k="$2" 'NF >= 4 {
      b = 0; g = 0
      if (match($4, /billing=[0-9]+/)) b = substr($4, RSTART + 8, RLENGTH - 8)
      if (match($4, /gres\/gpu=[0-9]+/)) g = substr($4, RSTART + 9, RLENGTH - 9)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%d\t%.3f\t%.3f\n", d, ti, t, k, $1, $2, $3, b * $3 / 3600, g * $3 / 3600 }' >> "$NC_COST"
}

nc_spent() {  # [TIER] -> CPUh logged so far (all tiers without an argument), integer
  [ -f "$NC_COST" ] || { echo 0; return; }
  awk -F'\t' -v t="${1:-}" '(t == "" || $2 == t) {s += $8} END {printf "%d", s + 0.5}' "$NC_COST"
}

nc_tag_cpuh() {  # TAG -> CPUh logged for that tag
  [ -f "$NC_COST" ] || { echo 0; return; }
  awk -F'\t' -v t="$1" '$3 == t {s += $8} END {printf "%.1f", s}' "$NC_COST"
}

nc_guard() {  # stop before submitting once the lane's tier has spent its cap (NC_CAP_CORE, NC_CAP_EXT, NC_CAP_OPT)
  local v="NC_CAP_$NC_TIER" cap spent
  cap=${!v:-}
  [ -z "$cap" ] && return 0
  spent=$(nc_spent "$NC_TIER")
  if [ "$spent" -ge "$cap" ]; then
    nc_log "budget guard: tier $NC_TIER has used $spent CPUh >= cap $cap ($v); not submitting"; return 1
  fi
  return 0
}

nc_train() {  # TAG BASE_MODEL DATA "SFT ARGS" -> resumable LoRA SFT as 30-min debug jobs; TRAIN_DONE on real completion
  local tag=$1 base=$2 data=$3 args=$4 d="$NC_ST/$1" it j st f fin tot sha dsha
  if [ -f "$d/TRAIN_DONE" ]; then nc_log "$tag: trained earlier ($(head -1 "$d/TRAIN_DONE"))"; return 0; fi
  [ -f "$base/config.json" ] || { nc_log "$tag: base model $base not found"; return 1; }
  for it in $(seq 1 "${NC_TRAIN_MAXIT:-30}"); do
    nc_guard || return 1
    j=$(export TAG="$tag" MODEL="$base" DATA="$data" NPROC=4 MAXLEN="${NC_SFT_MAXLEN:-12288}" EPOCHS=1 \
               EXTRA="$args --deadline-min ${NC_DL:-18}"
        nc_submit "nc_sft_$tag" --qos=boost_qos_dbg --time=00:30:00 --export=ALL leonardo/job_sft.slurm) \
      || { nc_log "$tag: sbatch failed for 90 min"; return 1; }
    nc_log "$tag: training job $j (iteration $it)"
    st=$(nc_wait "$j"); nc_cost "$tag" sft "$j"
    f="logs/nc_sft_$tag-$j.out"
    grep -hE "resumed at|checkpoint saved|step [0-9]+/[0-9]+ \||^DONE|Error|Traceback" "$f" 2>/dev/null | tail -3 | sed 's/^/    /'
    if grep -q "^DONE" "$f" 2>/dev/null; then
      fin=$(grep -oE "checkpoint saved at step [0-9]+" "$f" | tail -1 | awk '{print $NF}')
      tot=$(grep -oE "step [0-9]+/[0-9]+" "$f" | tail -1 | cut -d/ -f2)
      if [ "$st" = COMPLETED ] && [ -n "$fin" ] && [ "$fin" = "$tot" ] && [ -s "$d/adapter/adapter_model.safetensors" ]; then
        sha=$(nc_sha "$d/adapter/adapter_model.safetensors")
        dsha=$([ -f "$data" ] && nc_sha "$data" || echo "glob")
        printf 'job=%s steps=%s date=%s\nsha256=%s\nargs=%s\ndata=%s data_sha256=%s\n' "$j" "$tot" "$(date +%F_%T)" \
          "$sha" "$args" "$data" "$dsha" > "$d/TRAIN_DONE.tmp" && mv "$d/TRAIN_DONE.tmp" "$d/TRAIN_DONE"
        nc_log "$tag: training finished ($tot steps); TRAIN_DONE written"
        return 0
      fi
      nc_log "$tag: log says DONE but state=$st, last checkpoint step=$fin of $tot: no TRAIN_DONE, inspect $f"
      return 1
    fi
    if ! grep -q "checkpoint saved at step" "$f" 2>/dev/null; then
      nc_log "$tag: job $j ended $st without a checkpoint: stop (see $f)"; return 1
    fi
    sleep "${NC_PAUSE:-10}"
  done
  nc_log "$tag: training not finished after ${NC_TRAIN_MAXIT:-30} jobs"; return 1
}

nc_merge() {  # TAG BASE_MODEL -> merge the TRAIN_DONE adapter (debug QOS, 1 GPU); MERGE_DONE
  local tag=$1 base=$2 d="$NC_ST/$1" j st sha
  if [ -f "$d/MERGE_DONE" ]; then nc_log "$tag: merged earlier"; return 0; fi
  [ -f "$d/TRAIN_DONE" ] || { nc_log "$tag: no TRAIN_DONE marker: refusing to merge"; return 1; }
  sha=$(nc_sha "$d/adapter/adapter_model.safetensors")
  grep -q "^sha256=$sha" "$d/TRAIN_DONE" || { nc_log "$tag: adapter changed after TRAIN_DONE: refusing to merge"; return 1; }
  nc_guard || return 1
  j=$(export BASE="$base" ADAPTER="$d/adapter" OUT="$d/merged"
      nc_submit "nc_merge_$tag" --export=ALL leonardo/job_merge.slurm) || { nc_log "$tag: merge sbatch failed"; return 1; }
  nc_log "$tag: merge job $j"
  st=$(nc_wait "$j"); nc_cost "$tag" merge "$j"
  if [ "$st" = COMPLETED ] && grep -q "^MERGED" "logs/nc_merge_$tag-$j.out" 2>/dev/null && [ -f "$d/merged/config.json" ] \
     && ls "$d"/merged/*.safetensors >/dev/null 2>&1; then
    printf 'job=%s adapter_sha256=%s date=%s\n' "$j" "$sha" "$(date +%F_%T)" > "$d/MERGE_DONE"
    nc_log "$tag: merged"; return 0
  fi
  nc_log "$tag: merge job $j ended $st without MERGED (logs/nc_merge_$tag-$j.out)"; return 1
}

nc_student() {  # TAG BASE DATA "SFT ARGS" -> train + merge
  nc_train "$1" "$2" "$3" "$4" && nc_merge "$1" "$2"
}

nc_wait_marker() {  # FILE [MAX_HOURS] -> wait until another lane wrote FILE
  local f=$1 max=$(( ${2:-48} * 3600 )) t=0
  while [ ! -f "$f" ]; do
    if [ "$t" -ge "$max" ]; then nc_log "gave up waiting for $f"; return 1; fi
    sleep "$NC_POLL"; t=$((t + NC_POLL + 1))
  done
}

# Evaluation settings (set by the driver before nc_eval):
#   E_SOLVER synth/math_harvest.py | synth/code_harvest.py   E_DATA problems jsonl   E_IDS id list   E_K samples
#   E_MAXTOK cap   E_MAXLEN --max-model-len (floor)   E_TEMP   E_NSH shards   E_TEXTS 1 = --save-texts auto
#   E_MODE array (normal QOS, one 1-GPU array task per shard, E_TIME, E_SLICE minutes) |
#          dbg (debug QOS, one job, one GPU per unfinished shard, 30 min, E_SLICE=24)   E_MAXIT iterations
nc_eval() {  # TAG MODEL SEED
  local tag=$1 model=$2 seed=$3 dir="runs_synth/agent_$1" it j n bad prev=-1 stall=0 f s
  local total n_done complete incomplete n_records n_bad_fields
  if [ -f "$dir/EVAL_DONE" ]; then nc_log "$tag: evaluated earlier"; return 0; fi
  [ -f "$model/config.json" ] || { nc_log "$tag: model $model not found"; return 1; }
  mkdir -p "$dir"
  for it in $(seq 1 "${E_MAXIT:-10}"); do
    s=$($NC_PY3 synth/natcomm_check.py status "$tag" --ids "$E_IDS" --nsh "$E_NSH" --k "$E_K" --fields) || return 1
    eval "$s"
    nc_log "$tag: $n_done/$total problems (iteration $it)"
    if [ "$n_bad_fields" -gt 0 ]; then
      nc_log "$tag: $n_bad_fields records lack lens/oks/fins: the cluster's $E_SOLVER predates the NatComm interface; stop"
      return 1
    fi
    [ "$complete" = 1 ] && break
    if [ "$n_done" -le "$prev" ]; then
      stall=$((stall + 1))
      if [ "$stall" -ge 2 ]; then nc_log "$tag: no progress in two iterations: stop"; return 1; fi
    else
      stall=0
    fi
    prev=$n_done
    nc_guard || return 1
    j=$(export NC_TAG="$tag" NC_MODEL="$model" NC_SEED="$seed" NC_SOLVER="$E_SOLVER" NC_DATA="$E_DATA" NC_IDS="$E_IDS" \
               NC_NSH="$E_NSH" NC_K="$E_K" NC_MAXTOK="$E_MAXTOK" NC_MAXLEN="$E_MAXLEN" NC_TEMP="$E_TEMP" \
               NC_SLICE="$E_SLICE" NC_TEXTS="${E_TEXTS:-}" NC_EXTRA="${E_EXTRA:-}" NC_SHARDS="$incomplete"
        if [ "$E_MODE" = array ]; then
          nc_submit "nc_ev_$tag" --qos=normal --time="$E_TIME" --array="$(echo "$incomplete" | tr ':' ',')" \
            --gres=gpu:1 --cpus-per-task=8 --mem=120G --export=ALL leonardo/job_natcomm_eval.slurm
        else
          n=$(echo "$incomplete" | tr ':' '\n' | grep -c .)
          nc_submit "nc_ev_$tag" --qos=boost_qos_dbg --time=00:30:00 --gres=gpu:"$n" --cpus-per-task=$((8 * n)) \
            --mem=$((120 * n))G --export=ALL leonardo/job_natcomm_eval.slurm
        fi) || { nc_log "$tag: sbatch failed for 90 min"; return 1; }
    nc_log "$tag: eval job $j ($E_MODE, shards $incomplete, seed $seed, T $E_TEMP, cap $E_MAXTOK)"
    nc_wait "$j" > /dev/null; nc_cost "$tag" eval "$j"
    bad=0
    for f in logs/nc_ev_"$tag"-"$j"_s*.out; do         # real crashes only, not the shutdown at the time limit
      [ -f "$f" ] || continue
      if grep -q Traceback "$f" && ! grep -q "RuntimeError: cancelled" "$f"; then
        nc_log "$tag: crash in $f"; grep -E "Error|error" "$f" | tail -2 | sed 's/^/    /'; bad=1
      fi
    done
    [ "$bad" = 1 ] && return 1
  done
  s=$($NC_PY3 synth/natcomm_check.py status "$tag" --ids "$E_IDS" --nsh "$E_NSH" --k "$E_K" --fields) || return 1
  eval "$s"
  if [ "$complete" != 1 ] || [ "$n_bad_fields" -gt 0 ]; then nc_log "$tag: incomplete after ${E_MAXIT:-10} iterations"; return 1; fi
  if [ -n "${E_TEXTS:-}" ]; then
    $NC_PY3 synth/natcomm_check.py texts "$tag" --k "$E_K" || { nc_log "$tag: saved texts incomplete"; return 1; }
  fi
  s=$($NC_PY3 synth/natcomm_check.py summary "$tag" --k "$E_K")
  printf 'date=%s model=%s seed=%s temp=%s cap=%s k=%s solver=%s data=%s\n%s\nCPUh=%s\n' "$(date +%F_%T)" "$model" \
    "$seed" "$E_TEMP" "$E_MAXTOK" "$E_K" "$E_SOLVER" "$E_DATA" "$s" "$(nc_tag_cpuh "$tag")" > "$dir/EVAL_DONE"
  nc_log "$tag: $s | $(nc_tag_cpuh "$tag") CPUh"
  return 0
}

nc_forced() {  # TAG MODEL BUDGETS -> synth/forced_answer.py on the tag's saved texts (normal QOS, 1 GPU); FORCED_DONE
  local tag=$1 model=$2 budgets=$3 dir="runs_synth/agent_$1" out j st it s f_rows f_expected f_complete prev=0
  if [ -f "$dir/FORCED_DONE" ]; then nc_log "$tag: forced answers done earlier"; return 0; fi
  [ -f "$dir/EVAL_DONE" ] || { nc_log "$tag: no EVAL_DONE, forced answers need the finished evaluation"; return 1; }
  nc_preflight_forced || return 1
  out="$dir/forced.jsonl"
  for it in $(seq 1 "${F_MAXIT:-4}"); do
    nc_guard || return 1
    j=$(export NC_TAG="$tag" NC_MODEL="$model" NC_TEXTS_GLOB="$dir/shard_*.texts.jsonl.gz" NC_BUDGETS="$budgets" \
               NC_OUT="$out" NC_SLICE="${F_SLICE:-105}" NC_EXTRA="${F_EXTRA:-}"
        nc_submit "nc_fa_$tag" --qos=normal --time="${F_TIME:-02:00:00}" --export=ALL leonardo/job_natcomm_forced.slurm) \
      || { nc_log "$tag: forced sbatch failed"; return 1; }
    nc_log "$tag: forced-answer job $j (budgets $budgets, iteration $it)"
    st=$(nc_wait "$j"); nc_cost "$tag" forced "$j"
    s=$($NC_PY3 synth/natcomm_check.py forced "$tag" --out "$out") || return 1
    eval "$s"
    nc_log "$tag: forced rows $f_rows/$f_expected ($st)"
    if [ "$f_complete" = 1 ]; then
      printf 'job=%s budgets=%s rows=%s date=%s\n' "$j" "$budgets" "$f_rows" "$(date +%F_%T)" > "$dir/FORCED_DONE"
      return 0
    fi
    if [ "$st" != COMPLETED ] || [ "$f_rows" -le "$prev" ]; then
      nc_log "$tag: forced-answer job $j ended $st with $f_rows/$f_expected rows (logs/nc_fa_$tag-${j}_py.out)"; return 1
    fi
    prev=$f_rows
  done
  nc_log "$tag: forced answers incomplete after ${F_MAXIT:-4} jobs"; return 1
}

nc_cpu() {  # NAME "COMMAND" -> run on lrd_all_serial (budget-free CPU partition) and wait; NC_SERIAL=0 runs it here
  local name=$1 cmd=$2 j st
  if [ "${NC_SERIAL:-1}" = 0 ]; then nc_log "cpu $name on $(hostname): $cmd"; bash -c "$cmd"; return $?; fi
  j=$(nc_submit "nc_cpu_$name" --account="$NC_ACCOUNT" --partition=lrd_all_serial --time="${NC_CPU_TIME:-01:00:00}" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=30G --output="logs/nc_cpu_$name-%j.out" --wrap="cd $PWD && $cmd") \
    || { nc_log "cpu $name: sbatch failed"; return 1; }
  st=$(nc_wait "$j"); nc_cost "$name" cpu "$j"
  if [ "$st" != COMPLETED ]; then nc_log "cpu $name: job $j $st"; tail -5 "logs/nc_cpu_$name-$j.out" 2>/dev/null; return 1; fi
  nc_log "cpu $name done (logs/nc_cpu_$name-$j.out)"
  [ "${NC_CPU_TAIL:-0}" -gt 0 ] && tail -n "$NC_CPU_TAIL" "logs/nc_cpu_$name-$j.out" | sed 's/^/    /'
  return 0
}

NC_FAILS=0
nc_try() {  # CMD ARGS... -> run it; NC_MAX_FAILS (2) consecutive failures stop the driver (a systematic problem would
            # otherwise cost one job per arm). Call it directly, never inside $( ): it may exit.
  if "$@"; then NC_FAILS=0; return 0; fi
  NC_FAILS=$((NC_FAILS + 1))
  if [ "$NC_FAILS" -ge "${NC_MAX_FAILS:-2}" ]; then nc_log "$NC_FAILS consecutive failures (last: $1 $2): stopping this lane"; exit 1; fi
  return 1
}

nc_need() {  # FILE MESSAGE -> fail fast when a prerequisite is missing
  [ -e "$1" ] || { nc_log "missing $1: $2"; return 1; }
}

nc_preflight_eval() {  # SOLVER -> the cluster copy of the solver must have the NatComm interface (per-sample fields, texts)
  local s=$1
  nc_need "$s" "solver" || return 1
  nc_need synth/eval_common.py "sync the NatComm evaluation code" || return 1
  if ! grep -q -- '"--save-texts"' "$s" || ! grep -q 'fins' "$s"; then
    nc_log "$s lacks --save-texts / per-sample fields: sync the NatComm evaluation code to the cluster first"; return 1
  fi
}

nc_preflight_forced() {
  nc_need synth/forced_answer.py "sync the NatComm evaluation code" || return 1
  local x
  for x in --model --texts --budgets --out; do
    grep -q -- "\"$x\"" synth/forced_answer.py || { nc_log "synth/forced_answer.py has no $x option: interface changed?"; return 1; }
  done
}
