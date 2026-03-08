#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -V

# qsub_gw_single.sh — 1つの GW FEM ジョブを実行
# Usage: qsub -v JOB_NAME=Job-GW-Fair-0001 scripts/qsub_gw_single.sh

WORK_DIR=~/Payload2026/abaqus_work
CPUS=4
ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus
EXTRACT_SCRIPT=~/Payload2026/scripts/extract_gw_history.py
CSV_DIR=~/Payload2026/abaqus_work/gw_fairing_dataset

export LD_PRELOAD=/home/nishioka/libfake_x11.so

if [ -z "$JOB_NAME" ]; then
    echo "ERROR: JOB_NAME not set. Use: qsub -v JOB_NAME=Job-GW-Fair-XXXX"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $JOB_NAME on $(hostname) (PID $$)"
echo "  CPUs: $CPUS, PBS_JOBID: $PBS_JOBID"

cd "$WORK_DIR"

# すでに完了していたらスキップ
if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$JOB_NAME.sta" 2>/dev/null; then
    echo "[$(date)] SKIP $JOB_NAME (already completed)"
    exit 0
fi

# PBS がコア割り当てを管理するため、他ユーザーチェックは不要
# (PBS 外で直接実行する場合のみ注意)

# ロックファイル削除（前回クラッシュ時の残骸）
rm -f "$WORK_DIR/$JOB_NAME.lck"

# FEM 実行
$ABAQUS job=$JOB_NAME input=$JOB_NAME.inp cpus=$CPUS interactive 2>&1

# 完了チェック
if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$JOB_NAME.sta" 2>/dev/null; then
    echo "[$(date)] FEM DONE $JOB_NAME"

    # ODB → CSV 抽出
    echo "[$(date)] Extracting sensors..."
    $ABAQUS python "$EXTRACT_SCRIPT" "$JOB_NAME.odb" 2>&1 | tail -5
    if [ -f "$WORK_DIR/${JOB_NAME}_sensors.csv" ]; then
        mkdir -p "$CSV_DIR"
        cp "$WORK_DIR/${JOB_NAME}_sensors.csv" "$CSV_DIR/"
        echo "[$(date)] CSV extracted: ${JOB_NAME}_sensors.csv"
    else
        echo "[$(date)] WARNING: CSV extraction failed"
    fi
else
    echo "[$(date)] FAIL $JOB_NAME"
    tail -20 "$WORK_DIR/$JOB_NAME.sta" 2>/dev/null
    exit 1
fi

echo "[$(date)] ALL DONE $JOB_NAME"
