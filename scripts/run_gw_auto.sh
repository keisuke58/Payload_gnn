#!/bin/bash
# run_gw_auto.sh — GW ジョブを自動投入 (新設定 field output 100μs)
# Usage: nohup bash scripts/run_gw_auto.sh > results/gw_auto.log 2>&1 &
#
# - 常に BATCH_SIZE 個が走るよう維持
# - 完了したら次を自動補充
# - 新設定 (field output 1e-04) が適用済みの INP のみ対象
# - 旧設定 ODB (>50GB) はスキップしない（INP は修正済みなので再実行で上書き）

WORK_DIR=~/Payload2026/abaqus_work
SCRIPT=~/Payload2026/scripts/qsub_gw_single.sh
BATCH_SIZE=${1:-4}  # 引数で変更可能 (default: 4)
POLL_INTERVAL=120   # 秒

# 完了判定: .sta に COMPLETED があり、ODB が 10GB 未満 (新設定) か存在
is_completed() {
    local job=$1
    local sta="$WORK_DIR/${job}.sta"
    local odb="$WORK_DIR/${job}.odb"
    if grep -q "COMPLETED SUCCESSFULLY" "$sta" 2>/dev/null; then
        # 新設定の ODB は ~2GB。旧設定は 190GB だが INP 修正済みなので
        # 再実行不要（CSV 抽出済み）
        return 0
    fi
    return 1
}

# 未完了ジョブ一覧
get_remaining() {
    for inp in $WORK_DIR/Job-GW-Fair-????.inp; do
        local job=$(basename "$inp" .inp)
        if ! is_completed "$job"; then
            echo "${job#Job-GW-Fair-}"
        fi
    done | sort -n
}

# 現在の qsub で走っている GW ジョブ数
count_running() {
    qstat 2>/dev/null | grep -c "Job-GW-Fair" || echo 0
}

echo "[$(date)] Starting GW auto-runner (batch_size=$BATCH_SIZE, poll=${POLL_INTERVAL}s)"
echo "  INP field output: $(grep -l 'time interval=1e-04' $WORK_DIR/Job-GW-Fair-0000.inp 2>/dev/null && echo '1e-04 (new)' || echo '1e-06 (OLD!)')"

while true; do
    remaining=($(get_remaining))
    n_remain=${#remaining[@]}

    if [ $n_remain -eq 0 ]; then
        # Wait for running jobs to finish
        n_run=$(count_running)
        if [ "$n_run" -gt 0 ]; then
            echo "[$(date)] All submitted. Waiting for $n_run running jobs..."
            sleep $POLL_INTERVAL
            continue
        fi
        echo "[$(date)] All GW jobs completed!"
        break
    fi

    n_run=$(count_running)
    n_to_submit=$((BATCH_SIZE - n_run))

    if [ $n_to_submit -le 0 ]; then
        sleep $POLL_INTERVAL
        continue
    fi

    echo "[$(date)] Remaining: $n_remain | Running: $n_run | Submitting: $n_to_submit"

    submitted=0
    for id in "${remaining[@]}"; do
        if [ $submitted -ge $n_to_submit ]; then
            break
        fi
        job="Job-GW-Fair-${id}"
        # Skip if already in queue
        if qstat 2>/dev/null | grep -q "$job"; then
            continue
        fi
        # Verify INP has new setting
        if grep -q "field, time interval=1e-06" "$WORK_DIR/${job}.inp" 2>/dev/null; then
            echo "  WARNING: $job still has old field output setting, fixing..."
            sed -i '0,/\*Output, field, time interval=1e-06/{s/\*Output, field, time interval=1e-06/*Output, field, time interval=1e-04/}' "$WORK_DIR/${job}.inp"
        fi
        # Clean up stale files
        rm -f "$WORK_DIR/${job}.lck" 2>/dev/null
        qsub -v JOB_NAME=$job -N $job $SCRIPT
        echo "  Submitted: $job"
        ((submitted++))
    done

    sleep $POLL_INTERVAL
done

echo "[$(date)] GW auto-runner finished"
