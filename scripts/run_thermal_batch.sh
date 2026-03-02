#!/bin/bash
# run_thermal_batch.sh — 熱有 S12 CZM バッチ実行 (frontale用)
#
# 熱テンプレート (Job-CZM-S12-Thermal.inp) を使って
# 100サンプルのソルバー実行 + ODB抽出 を行う。
#
# Usage (frontale):
#   nohup bash scripts/run_thermal_batch.sh > thermal_batch.log 2>&1 &
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus
DOE="doe_sector12_100.json"
TEMPLATE="abaqus_work/Job-CZM-S12-Thermal.inp"
WORKDIR="abaqus_work/batch_s12_100_thermal"

echo "========================================"
echo " Thermal Batch: S12 CZM 100 samples"
echo " Template: $TEMPLATE"
echo " Output:   $WORKDIR"
echo " Started:  $(date)"
echo "========================================"

python3 src/run_sector12_batch.py \
    --doe "$DOE" \
    --template "$TEMPLATE" \
    --workdir "$WORKDIR" \
    --cpus 4 --memory "16 gb" \
    --parallel 4

echo ""
echo "=== Batch Complete: $(date) ==="

# Post-process: PyG変換 + binary変換
echo ""
echo "=== PyG Conversion (8-class) ==="
python3 src/prepare_ml_data.py \
    --input "$WORKDIR" \
    --output data/processed_s12_czm_thermal \
    --val_ratio 0.2

echo ""
echo "=== Binary Conversion ==="
python3 scripts/convert_to_binary.py \
    --input data/processed_s12_czm_thermal \
    --output data/processed_s12_czm_thermal_binary

echo ""
echo "Done: $(date)"
echo "Transfer to vancouver02:"
echo "  scp -r data/processed_s12_czm_thermal_binary vancouver02:~/Payload2026/data/"
