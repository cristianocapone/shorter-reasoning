#!/bin/bash
# NatComm revision, amendment 1 to the pre-registered MATH-4500 test (paper_brevity/natcomm/PREREG_MATH4500_AMEND1.md):
# forced answers at 4k and 8k on the saved generations of nc8c_q4_{S,R}_s{0,1}; analysis synth/natcomm_conf_forced_analysis.py.
#   prereg  write the SHA-256 of the amendment, scripts and launcher with a UTC timestamp to logs/PREREG_MATH4500_AMEND1.sha256
#   run     forced answers for the four runs (tier CF, cap 150 CPUh); prints row counts only
#   nohup bash leonardo/run_natcomm_conf_forced.sh run > logs/run_natcomm_conf_forced.log 2>&1 &
cd "$(dirname "$0")/.."; source leonardo/natcomm_lib.sh
LANE=${1:?lane: prereg | run}
ST=$NC_ST
NC_TIER=CF; NC_CAP_CF=${NC_CAP_CF:-150}
case $LANE in
prereg)
  f=logs/PREREG_MATH4500_AMEND1.sha256
  [ -f "$f" ] && { nc_log "$f exists, not overwritten"; cat "$f"; exit 0; }
  { date -u +%FT%TZ
    for x in paper_brevity/natcomm/PREREG_MATH4500_AMEND1.md synth/natcomm_conf_forced_analysis.py synth/forced_answer.py \
             leonardo/run_natcomm_conf_forced.sh synth/math/conf4500_ids.txt; do
      echo "$(nc_sha "$x")  $x"; done; } > "$f"
  cat "$f";;
run)
  nc_lock conf_forced || exit 1
  [ -f logs/PREREG_MATH4500_AMEND1.sha256 ] || { nc_log "run the prereg lane first"; exit 1; }
  nc_preflight_forced || exit 1
  for seed in 0 1; do
    for arm in S R; do
      nc_try nc_forced "nc8c_q4_${arm}_s$seed" "$ST/math4b_$arm/merged" 4096,8192
    done
  done
  nc_log "conf forced lane finished | CF spent $(nc_spent CF) CPUh (cap $NC_CAP_CF)";;
*) echo "unknown lane $LANE"; exit 1;;
esac
