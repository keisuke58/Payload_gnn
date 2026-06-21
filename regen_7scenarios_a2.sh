#!/bin/bash
# a2(healthy 2.0mm)収束確認後に実行。7欠陥シナリオを adhesive 2.0mm で再生成(QAEトークン・solveはしない)
set -e
cd /home/nishioka/Payload2026
export PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:$PATH
export TMPDIR=/home/nishioka/render_scratch
declare -A MAP=( [debond]=debonding [fod]=fod [impact]=impact [delam]=delam [acoustic]=acoustic [inner]=inner_debond [thermal]=thermal_progression )
for tag in debond fod impact delam acoustic inner thermal; do
  pf="params_${tag}.json"; [ "$tag" = "inner" ] && pf="params_inner.json"; [ "$tag" = "thermal" ] && pf="params_thermal.json"
  [ "$tag" = "debond" ] && pf="params_debonding.json"
  job="H3_${tag}_a2"
  echo "[$(date '+%F %T')] generating $job (adhesive 2.0mm, $pf)"
  abq2024 cae noGUI=src/generate_cohesive_fairing.py -- \
    --job_name "$job" --param_file "$pf" --adhesive_thickness 2.0 --no_run \
    > /home/nishioka/render_scratch/regen_${job}.log 2>&1
  echo "  rc=$? -> $(ls -la $job.inp 2>/dev/null | awk '{print $5}') bytes"
done
echo "ALL_GEN_DONE $(date '+%F %T')"
