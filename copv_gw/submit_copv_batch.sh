#!/bin/bash
# Submit the 5-freq COPV batch as a strictly-serial chain (one QEX token at a time).
# Usage: bash submit_copv_batch.sh [DEPJOB]
cd /home/nishioka/Payload2026/copv_gw
DEP="${1:-}"
FREQS="60 120 180 260 300"
prev="$DEP"
for f in $FREQS; do
  if [ -n "$prev" ]; then
    id=$(qsub -W depend=afterany:${prev} solve_${f}k.pbs)
  else
    id=$(qsub solve_${f}k.pbs)
  fi
  echo "submitted ${f}kHz -> ${id}  (after ${prev:-none})"
  prev=$(echo "$id" | cut -d. -f1)
done
echo "chain submitted; qstat to watch."
