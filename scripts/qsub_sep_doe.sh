#!/bin/bash
# qsub_sep_doe.sh — 分離 DOE 12ケース一括投入
#
# Usage:
#   bash scripts/qsub_sep_doe.sh         # 投入
#   bash scripts/qsub_sep_doe.sh status  # 進捗確認
#   bash scripts/qsub_sep_doe.sh dry     # 確認のみ

MODE="${1:-submit}"
SCRIPT=~/Payload2026/scripts/qsub_fairing_sep_run.sh
WORK_DIR=~/Payload2026/abaqus_work

declare -a CASES=(
    "Sep-DOE-S01 --n_stuck_bolts 1"
    "Sep-DOE-S02 --n_stuck_bolts 2"
    "Sep-DOE-S04 --n_stuck_bolts 4"
    "Sep-DOE-S05 --n_stuck_bolts 5"
    "Sep-DOE-S08 --n_stuck_bolts 8"
    "Sep-DOE-S09 --n_stuck_bolts 9"
    "Sep-DOE-S12 --n_stuck_bolts 12"
    "Sep-DOE-K300  --n_stuck_bolts 3 --spring_stiffness 300"
    "Sep-DOE-K2000 --n_stuck_bolts 3 --spring_stiffness 2000"
    "Sep-DOE-K8000 --n_stuck_bolts 3 --spring_stiffness 8000"
    "Sep-DOE-S06K300  --n_stuck_bolts 6 --spring_stiffness 300"
    "Sep-DOE-S06K8000 --n_stuck_bolts 6 --spring_stiffness 8000"
)

if [ "$MODE" = "status" ]; then
    echo "=== Sep DOE Status ==="
    for entry in "${CASES[@]}"; do
        job=$(echo "$entry" | awk '{print $1}')
        if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
            echo "  DONE $job"
        elif [ -f "$WORK_DIR/$job.odb" ]; then
            echo "  RUN  $job"
        else
            echo "  WAIT $job"
        fi
    done
    exit 0
fi

echo "=== Sep DOE: ${#CASES[@]} cases ==="
for entry in "${CASES[@]}"; do
    job=$(echo "$entry" | awk '{print $1}')
    args=$(echo "$entry" | cut -d' ' -f2-)
    if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
        echo "  SKIP $job (already done)"
        continue
    fi
    if [ "$MODE" = "dry" ]; then
        echo "  [DRY] qsub -v JOB_NAME=$job,EXTRA_ARGS=\"$args\" $SCRIPT"
    else
        qsub -v JOB_NAME=$job,EXTRA_ARGS="$args" $SCRIPT
        echo "  Submitted: $job ($args)"
    fi
done
echo "Done."
