#!/bin/bash
# qsub_gw_autosubmit.sh — 5個ずつ自動投入し続ける
#
# Usage:
#   nohup bash scripts/qsub_gw_autosubmit.sh > abaqus_work/logs/autosubmit.log 2>&1 &
#
# 動作:
#   1. nishioka の PBS ジョブ数を確認
#   2. 5個未満なら、未完了ジョブを追加投入（合計5個まで）
#   3. 10分待って繰り返し
#   4. 全100サンプル完了で終了

BATCH_SIZE=5
CHECK_INTERVAL=600  # 10分
N_SAMPLES=100
WORK_DIR=~/Payload2026/abaqus_work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QSUB_SCRIPT="$SCRIPT_DIR/qsub_gw_single.sh"
CSV_DIR="$WORK_DIR/gw_fairing_dataset"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# 完了済みサンプル数を数える
count_done() {
    local n=0
    for ((i=0; i<N_SAMPLES; i++)); do
        job=$(printf "Job-GW-Fair-%04d" $i)
        grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null && ((n++))
    done
    echo $n
}

# nishioka の実行中 PBS ジョブ数
count_running() {
    qstat -u nishioka 2>/dev/null | grep " R " | wc -l
}

# キュー待ち含む nishioka の全 PBS ジョブ数
count_queued() {
    qstat -u nishioka 2>/dev/null | grep -E " [RQ] " | wc -l
}

# 未完了ジョブの ID リスト（投入順）
get_pending() {
    for ((i=0; i<N_SAMPLES; i++)); do
        job=$(printf "Job-GW-Fair-%04d" $i)
        if [ -f "$WORK_DIR/$job.inp" ] && \
           ! grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
            # PBS に既に投入済みか確認
            if ! qstat -u nishioka 2>/dev/null | grep -q "$job"; then
                echo $i
            fi
        fi
    done
}

log "=== GW FEM Auto-Submit Started ==="
log "Batch size: $BATCH_SIZE, Check interval: ${CHECK_INTERVAL}s"

while true; do
    n_done=$(count_done)
    n_queued=$(count_queued)

    log "Status: $n_done/$N_SAMPLES done, $n_queued in PBS queue"

    # 全完了チェック
    if [ "$n_done" -ge "$N_SAMPLES" ]; then
        log "=== ALL $N_SAMPLES SAMPLES COMPLETED ==="
        # CSV 数も確認
        n_csv=$(ls "$CSV_DIR"/Job-GW-Fair-[0-9]*_sensors.csv 2>/dev/null | wc -l)
        log "CSV files: $n_csv/$N_SAMPLES"
        break
    fi

    # 投入数を計算（BATCH_SIZE - 現在のキュー数）
    to_submit=$((BATCH_SIZE - n_queued))

    if [ "$to_submit" -le 0 ]; then
        log "Queue full ($n_queued/$BATCH_SIZE). Waiting..."
    else
        # 未投入の pending ジョブを取得
        pending=($(get_pending))
        n_pending=${#pending[@]}

        if [ "$n_pending" -eq 0 ]; then
            log "No pending jobs to submit. All in queue or done. Waiting..."
        else
            # to_submit 個だけ投入
            if [ "$to_submit" -gt "$n_pending" ]; then
                to_submit=$n_pending
            fi

            log "Submitting $to_submit jobs..."
            for ((j=0; j<to_submit; j++)); do
                idx=${pending[$j]}
                job=$(printf "Job-GW-Fair-%04d" $idx)
                result=$(qsub -v "JOB_NAME=$job" -N "$job" "$QSUB_SCRIPT" 2>&1)
                log "  Submitted: $job → $result"
            done
        fi
    fi

    sleep $CHECK_INTERVAL
done

log "=== Auto-Submit Finished ==="
