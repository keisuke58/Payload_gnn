#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -V

# qsub_gw_valid_v3.sh — GW Validation モデル再生成 + 解析
# mesh_seed=5.0 で再生成し、そのまま solver 実行 → CSV 抽出
#
# Usage:
#   qsub -N GW-Valid-H-v3  -v MODEL_TYPE=healthy   scripts/qsub_gw_valid_v3.sh
#   qsub -N GW-Valid-D30-v3 -v MODEL_TYPE=debond30  scripts/qsub_gw_valid_v3.sh

BASE_DIR=~/Payload2026/abaqus_work
CPUS=4
ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus
GEN_SCRIPT=~/Payload2026/src/generate_gw_fairing.py
EXTRACT_SCRIPT=~/Payload2026/scripts/extract_gw_history.py
CSV_DIR=~/Payload2026/abaqus_work/gw_fairing_dataset

MESH_SEED=5.0

export LD_PRELOAD=/home/nishioka/libfake_x11.so

if [ -z "$MODEL_TYPE" ]; then
    echo "ERROR: MODEL_TYPE not set. Use: -v MODEL_TYPE=healthy or debond30"
    exit 1
fi

# --- Job name & defect parameters ---
case "$MODEL_TYPE" in
    healthy)
        JOB_NAME=Job-GW-Valid-Healthy-v3
        DEFECT_ARG=""
        ;;
    debond30)
        JOB_NAME=Job-GW-Valid-Debond30-v3
        # Defect at center of sector, radius=30mm
        DEFECT_ARG="--defect {\"z_center\":1500,\"theta_deg\":15,\"radius\":30}"
        ;;
    *)
        echo "ERROR: Unknown MODEL_TYPE=$MODEL_TYPE (use: healthy, debond30)"
        exit 1
        ;;
esac

# --- Use job-specific subdirectory to avoid CAE temp file collision ---
WORK_DIR="$BASE_DIR/$JOB_NAME"
mkdir -p "$WORK_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $JOB_NAME on $(hostname) (PID $$)"
echo "  MODEL_TYPE=$MODEL_TYPE, CPUs=$CPUS, MESH_SEED=$MESH_SEED"
echo "  WORK_DIR=$WORK_DIR"
echo "  PBS_JOBID: $PBS_JOBID"

cd "$WORK_DIR"

# すでに完了していたらスキップ
if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$JOB_NAME.sta" 2>/dev/null; then
    echo "[$(date)] SKIP $JOB_NAME (already completed)"
    exit 0
fi

# 前回の残骸を削除
rm -f "$WORK_DIR/$JOB_NAME.lck"

# --- Step 1: CAE でモデル再生成 (INP 出力) ---
echo "[$(date)] Generating model with mesh_seed=$MESH_SEED ..."
$ABAQUS cae noGUI=$GEN_SCRIPT -- \
    --job $JOB_NAME --mesh_seed $MESH_SEED --no_run $DEFECT_ARG 2>&1

if [ ! -f "$WORK_DIR/$JOB_NAME.inp" ]; then
    echo "[$(date)] FAIL: INP generation failed"
    exit 1
fi
echo "[$(date)] INP generated: $JOB_NAME.inp ($(du -h $JOB_NAME.inp | cut -f1))"

# --- Verify INP header matches job name ---
if ! head -2 "$JOB_NAME.inp" | grep -q "$JOB_NAME"; then
    echo "[$(date)] ERROR: INP header mismatch (temp file collision?)"
    head -2 "$JOB_NAME.inp"
    exit 1
fi

# --- Step 2: Solver 実行 ---
echo "[$(date)] Running solver ..."
$ABAQUS job=$JOB_NAME cpus=$CPUS interactive 2>&1

# --- Step 3: 完了チェック + CSV 抽出 ---
if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$JOB_NAME.sta" 2>/dev/null; then
    echo "[$(date)] FEM DONE $JOB_NAME"

    # Stable time increment 情報をログに出力
    grep -A 8 "STABLE TIME INCREMENT" "$JOB_NAME.sta" 2>/dev/null | head -12
    grep "overclosure" "$JOB_NAME.sta" 2>/dev/null | head -3

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
    # Diagnostic output
    grep -A 8 "STABLE TIME INCREMENT" "$JOB_NAME.sta" 2>/dev/null | head -12
    grep "overclosure" "$JOB_NAME.sta" 2>/dev/null | head -3
    tail -20 "$JOB_NAME.sta" 2>/dev/null
    exit 1
fi

echo "[$(date)] ALL DONE $JOB_NAME"
