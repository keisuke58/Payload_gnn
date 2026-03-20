#!/bin/bash
# run_gw_auto.sh — GW ジョブを 4 個ずつ自動投入
# Usage: nohup bash scripts/run_gw_auto.sh > results/gw_auto.log 2>&1 &

WORK_DIR=~/Payload2026/abaqus_work
SCRIPT=~/Payload2026/scripts/qsub_gw_single.sh
BATCH_SIZE=4

# Get list of remaining jobs (excluding already completed)
get_remaining() {
    comm -23 \
        <(find $WORK_DIR -maxdepth 1 -name "Job-GW-Fair-????.inp" | sed 's/.*Fair-\([0-9]*\).*/\1/' | sort -n) \
        <(find $WORK_DIR -maxdepth 1 -name "Job-GW-Fair-????.odb" | sed 's/.*Fair-\([0-9]*\).*/\1/' | sort -n)
}

echo "[$(date)] Starting GW auto-runner (batch_size=$BATCH_SIZE)"

while true; do
    remaining=($(get_remaining))
    n_remain=${#remaining[@]}
    
    if [ $n_remain -eq 0 ]; then
        echo "[$(date)] All GW jobs completed!"
        break
    fi
    
    echo "[$(date)] Remaining: $n_remain jobs"
    
    # Submit batch
    submitted=0
    for id in "${remaining[@]}"; do
        if [ $submitted -ge $BATCH_SIZE ]; then
            break
        fi
        job="Job-GW-Fair-${id}"
        # Skip if already running
        if qstat 2>/dev/null | grep -q "$job"; then
            continue
        fi
        # Clean up old lock files
        rm -f "$WORK_DIR/${job}.lck" 2>/dev/null
        qsub -v JOB_NAME=$job -N $job $SCRIPT
        echo "  Submitted: $job"
        ((submitted++))
    done
    
    if [ $submitted -eq 0 ]; then
        echo "  Waiting for running jobs..."
    fi
    
    # Wait for current batch to finish
    while qstat 2>/dev/null | grep -q "Job-GW-Fair"; do
        sleep 120
    done
    echo "[$(date)] Batch complete"
done

echo "[$(date)] GW auto-runner finished"
