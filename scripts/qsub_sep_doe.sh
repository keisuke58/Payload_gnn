#!/bin/bash
# qsub_sep_doe.sh — 分離 DOE バッチ投入
# Usage: bash scripts/qsub_sep_doe.sh

SCRIPT=~/Payload2026/scripts/qsub_fairing_sep_run.sh
EXTRACT=~/Payload2026/scripts/qsub_extract_sep_graph.sh

echo "=== Fairing Separation DOE Batch ==="

# Stuck bolt sweep (1,2,4,5,8,9,12)
for n in 1 2 4 5 8 9 12; do
    name="Sep-DOE-S$(printf '%02d' $n)"
    echo "  $name: n_stuck_bolts=$n"
    qsub -v JOB_NAME=$name,EXTRA_ARGS="--n_stuck_bolts $n" $SCRIPT
done

# Spring stiffness variation with 3 stuck bolts
for k in 300 800 2000 8000; do
    name="Sep-DOE-K${k}"
    echo "  $name: n_stuck=3, spring=$k"
    qsub -v JOB_NAME=$name,EXTRA_ARGS="--n_stuck_bolts 3 --spring_stiffness $k" $SCRIPT
done

# Additional normal case (for augmentation)
echo "  Sep-DOE-N00: normal (0 stuck)"
qsub -v JOB_NAME=Sep-DOE-N00 $SCRIPT

echo ""
echo "Total: 12 jobs submitted"
echo "After completion, run graph extraction:"
echo '  for job in Sep-DOE-S01 Sep-DOE-S02 Sep-DOE-S04 Sep-DOE-S05 Sep-DOE-S08 Sep-DOE-S09 Sep-DOE-S12 Sep-DOE-K300 Sep-DOE-K800 Sep-DOE-K2000 Sep-DOE-K8000 Sep-DOE-N00; do'
echo '    qsub -v JOB_NAME=$job scripts/qsub_extract_sep_graph.sh'
echo '  done'
