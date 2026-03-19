#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -V

# qsub_extract_sep.sh — フェアリング分離 ODB から結果抽出
# Usage: qsub -v JOB_NAME=Sep-Test scripts/qsub_extract_sep.sh

WORK_DIR=~/Payload2026/abaqus_work
SRC_DIR=~/Payload2026/src
OUTPUT_DIR=~/Payload2026/results/separation
ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus

if [ -z "$JOB_NAME" ]; then
    echo "ERROR: JOB_NAME not set"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Extracting $JOB_NAME on $(hostname)"

mkdir -p "$OUTPUT_DIR"

ODB_PATH="$WORK_DIR/$JOB_NAME.odb"
if [ ! -f "$ODB_PATH" ]; then
    echo "ERROR: ODB not found: $ODB_PATH"
    exit 1
fi

cd "$WORK_DIR"
$ABAQUS python "$SRC_DIR/extract_separation_results.py" \
    --odb "$ODB_PATH" --output "$OUTPUT_DIR" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done"
