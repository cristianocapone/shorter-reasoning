#!/bin/bash
# Run on a Leonardo login node. Budget + linear-pace check for IscrC_LUCID.
# Usage: bash leonardo/budget.sh
PROJECT=IscrC_LUCID

echo "=== saldo -b (Booster) ==="
saldo -b
echo

row=$(saldo -b 2>/dev/null | awk -v p="$PROJECT" '$1==p')
if [ -z "$row" ]; then echo "riga $PROJECT non trovata in saldo"; exit 1; fi

read -r _ start end total localc totc totpct mtot mcons <<<"$row"

day=$(date +%-d)
days_in_month=$(cal $(date +%m) $(date +%Y) | awk 'NF{last=$NF} END{print last}')
month_elapsed=$(awk -v d="$day" -v n="$days_in_month" 'BEGIN{printf "%.1f", 100*d/n}')
month_used=$(awk -v c="$mcons" -v t="$mtot" 'BEGIN{ if(t>0) printf "%.1f", 100*c/t; else print "0.0"}')
# User rule 2026-09-25: up to 3x the linear pace (was 2x on 09-23) until the ARC Prize deadline (2026-11-02), then linear again.
PACE_MULT=$([ "$(date +%Y%m%d)" -le 20261102 ] && echo 4.5 || echo 1)   # 3x linear + 50 % tolerance (user, 2026-09-28)
gap=$(awk -v u="$month_used" -v e="$month_elapsed" -v m="$PACE_MULT" 'BEGIN{printf "%.1f", u-m*e}')

echo "=== ritmo consentito: ${PACE_MULT}x lineare (mese corrente) ==="
printf "%-28s %s\n" "mese trascorso"          "${month_elapsed}%"
printf "%-28s %s\n" "quota mensile consumata" "${month_used}%  (${mcons} / ${mtot} CPUh)"
printf "%-28s %s\n" "scarto vs ${PACE_MULT}x lineare"  "${gap} punti"
printf "%-28s %s\n" "totale progetto"         "${totc} / ${total} CPUh (${totpct}%)"
echo

awk -v g="$gap" 'BEGIN{
  if (g > 20)       print "OLTRE IL RITMO CONSENTITO: sospendere i job grandi.";
  else if (g < -20) print "INDIETRO COL RITMO: non accumulare, lanciare subito il lavoro pronto.";
  else              print "IN LINEA.";
}'
echo
echo "Promemoria: nodo pieno = 32 CPUh/h; 1 GPU + 8 core + 120G = 8 CPUh/h."

# saldo lags about one day. Live estimate from Slurm accounting since the 9th of this month:
echo
echo "=== stima live da sacct (billing x ore), mese corrente ==="
sacct -S "$(date +%Y-%m)-09" -X -n -o AllocTRES%60,ElapsedRaw 2>/dev/null | awk '
  { b=0; if (match($1, /billing=[0-9]+/)) b=substr($1, RSTART+8, RLENGTH-8); s += b * $2 / 3600 }
  END { printf "  %.0f CPUh = %.0f ore A100 equivalenti\n", s, s/8 }'
