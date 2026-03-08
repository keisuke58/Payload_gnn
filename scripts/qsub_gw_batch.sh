#!/bin/bash
# qsub_gw_batch.sh — 未完了の GW FEM ジョブを qsub で一括投入
#
# Usage:
#   bash scripts/qsub_gw_batch.sh          # 未完了ジョブを投入
#   bash scripts/qsub_gw_batch.sh status   # 進捗確認
#   bash scripts/qsub_gw_batch.sh dry      # 何が投入されるか確認（投入しない）

set -e

WORK_DIR=~/Payload2026/abaqus_work
CSV_DIR=~/Payload2026/abaqus_work/gw_fairing_dataset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QSUB_SCRIPT="$SCRIPT_DIR/qsub_gw_single.sh"
N_SAMPLES=100

MODE="${1:-submit}"

# ==============================================================================
# status: 進捗確認
# ==============================================================================
if [ "$MODE" = "status" ]; then
    echo "=== GW Defect Job Status ==="
    n_inp=0; n_done=0; n_csv=0; n_running=0
    running_jobs=""

    for ((i=0; i<N_SAMPLES; i++)); do
        job=$(printf "Job-GW-Fair-%04d" $i)
        [ -f "$WORK_DIR/$job.inp" ] && ((n_inp++))
        if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
            ((n_done++))
        elif [ -f "$WORK_DIR/$job.odb" ]; then
            pct=$(tail -1 "$WORK_DIR/$job.sta" 2>/dev/null | awk '{print $2}')
            running_jobs="$running_jobs  $job: step_time=$pct/3.920E-03\n"
            ((n_running++))
        fi
        [ -f "$CSV_DIR/${job}_sensors.csv" ] && ((n_csv++))
    done

    echo "INP:       $n_inp / $N_SAMPLES"
    echo "FEM完了:   $n_done / $N_SAMPLES"
    echo "実行中:    $n_running"
    echo "CSV抽出:   $n_csv / $N_SAMPLES"

    if [ -n "$running_jobs" ]; then
        echo ""
        echo "実行中ジョブ:"
        echo -e "$running_jobs"
    fi

    echo ""
    echo "PBS キュー (nishioka のジョブ):"
    qstat -u nishioka 2>/dev/null || echo "  (なし)"
    exit 0
fi

# ==============================================================================
# submit / dry: 未完了ジョブを投入
# ==============================================================================
echo "=== GW Defect qsub Batch Submission ==="

# 投入対象を収集
pending=()
for ((i=0; i<N_SAMPLES; i++)); do
    job=$(printf "Job-GW-Fair-%04d" $i)
    # INP が存在し、まだ完了していないもの
    if [ -f "$WORK_DIR/$job.inp" ] && \
       ! grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
        pending+=($i)
    fi
done

echo "未完了: ${#pending[@]} ジョブ"

if [ ${#pending[@]} -eq 0 ]; then
    echo "全ジョブ完了済みです。"
    exit 0
fi

if [ "$MODE" = "dry" ]; then
    echo ""
    echo "投入予定 (dry run):"
    for idx in "${pending[@]}"; do
        job=$(printf "Job-GW-Fair-%04d" $idx)
        echo "  qsub -v JOB_NAME=$job -N $job $QSUB_SCRIPT"
    done
    echo ""
    echo "実行するには: bash scripts/qsub_gw_batch.sh"
    exit 0
fi

# 実際に投入
echo ""
submitted=0
for idx in "${pending[@]}"; do
    job=$(printf "Job-GW-Fair-%04d" $idx)
    qsub -v "JOB_NAME=$job" -N "$job" "$QSUB_SCRIPT" 2>&1
    ((submitted++))
done

echo ""
echo "$submitted ジョブを投入しました。"
echo ""
echo "確認: qstat -u nishioka"
echo "進捗: bash scripts/qsub_gw_batch.sh status"
