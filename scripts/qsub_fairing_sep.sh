#!/bin/bash
#PBS -l nodes=1:ppn=8
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -V

# qsub_fairing_sep.sh — フェアリング分離ダイナミクス INP 生成 + 解析
# Usage: qsub -v JOB_NAME=Sep-Test scripts/qsub_fairing_sep.sh
#        qsub -v JOB_NAME=Sep-Abnormal,EXTRA_ARGS="--n_stuck_bolts 3" scripts/qsub_fairing_sep.sh

WORK_DIR=~/Payload2026/abaqus_work
SRC_DIR=~/Payload2026/src
CPUS=8
ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus

export LD_PRELOAD=/home/nishioka/libfake_x11.so

if [ -z "$JOB_NAME" ]; then
    echo "ERROR: JOB_NAME not set. Use: qsub -v JOB_NAME=Sep-Test"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $JOB_NAME on $(hostname) (PID $$)"
echo "  CPUs: $CPUS, PBS_JOBID: $PBS_JOBID"

cd "$WORK_DIR"

# ロックファイル削除
rm -f "$WORK_DIR/$JOB_NAME.lck"

# Phase 1: INP 生成
echo ""
echo "=== Phase 1: Generating INP ==="
$ABAQUS cae noGUI=$SRC_DIR/generate_fairing_separation.py -- \
    --job "$JOB_NAME" --no_run $EXTRA_ARGS 2>&1

if [ ! -f "$WORK_DIR/$JOB_NAME.inp" ]; then
    echo "ERROR: INP generation failed"
    exit 1
fi
echo "INP generated: $JOB_NAME.inp ($(wc -l < $JOB_NAME.inp) lines)"

# Phase 2: Abaqus/Explicit 解析
echo ""
echo "=== Phase 2: Running Abaqus/Explicit ==="
$ABAQUS job=$JOB_NAME cpus=$CPUS interactive 2>&1

# 結果確認
if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$JOB_NAME.sta" 2>/dev/null; then
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS $JOB_NAME"
    echo "  ODB: $WORK_DIR/$JOB_NAME.odb"
    # ODB サイズ
    ls -lh "$WORK_DIR/$JOB_NAME.odb" 2>/dev/null
else
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED or INCOMPLETE $JOB_NAME"
    echo "  Check: $WORK_DIR/$JOB_NAME.msg"
    echo "  Last 20 lines of .dat:"
    tail -20 "$WORK_DIR/$JOB_NAME.dat" 2>/dev/null
    exit 1
fi
