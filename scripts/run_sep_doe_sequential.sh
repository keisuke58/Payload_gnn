#!/bin/bash
# run_sep_doe_sequential.sh — 分離 DOE を 1 個ずつ順次実行
# Usage: nohup bash scripts/run_sep_doe_sequential.sh > results/sep_doe.log 2>&1 &

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

echo "[$(date)] Starting Sep DOE: ${#CASES[@]} cases"

for entry in "${CASES[@]}"; do
    job=$(echo "$entry" | awk '{print $1}')
    args=$(echo "$entry" | cut -d' ' -f2-)

    if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
        echo "[$(date)] SKIP $job (done)"
        continue
    fi

    echo "[$(date)] Submitting: $job"
    JOB_ID=$(qsub -v JOB_NAME=$job,EXTRA_ARGS="$args" $SCRIPT 2>&1)
    echo "  Job ID: $JOB_ID"

    # Wait for completion
    while true; do
        sleep 30
        if ! qstat $JOB_ID > /dev/null 2>&1; then
            break
        fi
    done

    if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
        echo "[$(date)] DONE $job"
    else
        echo "[$(date)] FAILED $job"
    fi
done

echo "[$(date)] All done."
