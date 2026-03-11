#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -V

# qsub_gw_v3_single.sh — 1つの GW v3 FEM ジョブ (CAE生成 + solver + CSV抽出)
#
# v3 format: 104 sensors (8x13 grid), mesh_seed=5.0
#
# Usage:
#   # Defective:
#   qsub -N Job-GW-v3-0000 \
#        -v JOB_NAME=Job-GW-v3-0000,DEFECT='{"z_center":1298.8,"theta_deg":6.21,"radius":49.4,"defect_type":"debonding"}' \
#        scripts/qsub_gw_v3_single.sh
#
#   # Healthy:
#   qsub -N Job-GW-v3-H0 -v JOB_NAME=Job-GW-v3-H0 scripts/qsub_gw_v3_single.sh

BASE_DIR=~/Payload2026/abaqus_work
CPUS=4
ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus
GEN_SCRIPT=~/Payload2026/src/generate_gw_fairing.py
EXTRACT_SCRIPT=~/Payload2026/scripts/extract_gw_history.py
CSV_DIR=~/Payload2026/abaqus_work/gw_v3_dataset
MESH_SEED=5.0

export LD_PRELOAD=/home/nishioka/libfake_x11.so

if [ -z "$JOB_NAME" ]; then
    echo "ERROR: JOB_NAME not set."
    exit 1
fi

# Job-specific subdirectory to avoid CAE temp file collision
WORK_DIR="$BASE_DIR/$JOB_NAME"
mkdir -p "$WORK_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $JOB_NAME on $(hostname) (PID $$)"
echo "  CPUs=$CPUS, MESH_SEED=$MESH_SEED"
echo "  DEFECT=$DEFECT"
echo "  WORK_DIR=$WORK_DIR"
echo "  PBS_JOBID: $PBS_JOBID"

cd "$WORK_DIR"

# Skip if already completed
if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$JOB_NAME.sta" 2>/dev/null; then
    echo "[$(date)] SKIP $JOB_NAME (already completed)"
    # Still extract CSV if missing
    if [ ! -f "$CSV_DIR/${JOB_NAME}_sensors.csv" ]; then
        echo "[$(date)] Re-extracting CSV..."
        $ABAQUS python "$EXTRACT_SCRIPT" "$JOB_NAME.odb" 2>&1 | tail -5
        if [ -f "$WORK_DIR/${JOB_NAME}_sensors.csv" ]; then
            mkdir -p "$CSV_DIR"
            cp "$WORK_DIR/${JOB_NAME}_sensors.csv" "$CSV_DIR/"
        fi
    fi
    exit 0
fi

# Clean up stale lock files
rm -f "$WORK_DIR/$JOB_NAME.lck"

# --- Step 1: CAE model generation (INP output) ---
echo "[$(date)] Generating v3 model (grid 104 sensors, mesh_seed=$MESH_SEED)..."

if [ -n "$DEFECT" ]; then
    # Defective model
    $ABAQUS cae noGUI=$GEN_SCRIPT -- \
        --job $JOB_NAME --mesh_seed $MESH_SEED --sensor_layout grid \
        --no_run --defect "$DEFECT" 2>&1
else
    # Healthy model
    $ABAQUS cae noGUI=$GEN_SCRIPT -- \
        --job $JOB_NAME --mesh_seed $MESH_SEED --sensor_layout grid \
        --no_run 2>&1
fi

if [ ! -f "$WORK_DIR/$JOB_NAME.inp" ]; then
    echo "[$(date)] FAIL: INP generation failed"
    exit 1
fi
echo "[$(date)] INP generated: $JOB_NAME.inp ($(du -h $JOB_NAME.inp | cut -f1))"

# Verify INP header
if ! head -2 "$JOB_NAME.inp" | grep -q "$JOB_NAME"; then
    echo "[$(date)] ERROR: INP header mismatch (temp file collision?)"
    head -2 "$JOB_NAME.inp"
    exit 1
fi

# --- Step 2: Solver ---
echo "[$(date)] Running solver..."
$ABAQUS job=$JOB_NAME cpus=$CPUS interactive 2>&1

# --- Step 3: Check completion + CSV extraction ---
if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$JOB_NAME.sta" 2>/dev/null; then
    echo "[$(date)] FEM DONE $JOB_NAME"

    # Diagnostic info
    grep -A 8 "STABLE TIME INCREMENT" "$JOB_NAME.sta" 2>/dev/null | head -12
    grep "overclosure" "$JOB_NAME.sta" 2>/dev/null | head -3

    # ODB -> CSV extraction
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
    grep -A 8 "STABLE TIME INCREMENT" "$JOB_NAME.sta" 2>/dev/null | head -12
    grep "overclosure" "$JOB_NAME.sta" 2>/dev/null | head -3
    tail -20 "$JOB_NAME.sta" 2>/dev/null
    exit 1
fi

echo "[$(date)] ALL DONE $JOB_NAME"
