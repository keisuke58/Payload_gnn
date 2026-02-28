#!/bin/bash
# =============================================================================
# Quick Verification Pipeline: 20 samples with larger defects
#
# Validates the full pipeline with corrected DOE radii (r>=50mm) and
# updated feature extraction (displacement + stress + temperature).
#
# Usage:
#   bash scripts/run_verify_batch.sh              # Full pipeline
#   bash scripts/run_verify_batch.sh --skip-abaqus # Skip FEM, use existing data
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DOE_FILE="doe_verify.json"
DATASET_DIR="dataset_verify"
ML_DATA_DIR="data/verify"

SKIP_ABAQUS=false
for arg in "$@"; do
    case $arg in
        --skip-abaqus) SKIP_ABAQUS=true ;;
    esac
done

echo "=========================================="
echo " Verification Pipeline (20 samples)"
echo "=========================================="
echo " DOE:      $DOE_FILE"
echo " Dataset:  $DATASET_DIR"
echo " ML Data:  $ML_DATA_DIR"
echo ""

# ----- Step 1: Generate DOE (if not exists) -----
if [ ! -f "$DOE_FILE" ]; then
    echo "[Step 1] Generating verification DOE..."
    python src/generate_doe.py --n_samples 20 --n_healthy 1 --output "$DOE_FILE"
else
    echo "[Step 1] DOE already exists: $DOE_FILE"
fi

# ----- Step 2: Abaqus batch generation -----
if [ "$SKIP_ABAQUS" = false ]; then
    echo ""
    echo "[Step 2] Running Abaqus batch (20 samples + 1 healthy)..."
    if command -v abaqus &> /dev/null; then
        python src/run_batch.py \
            --doe "$DOE_FILE" \
            --output_dir "$DATASET_DIR" \
            --n_cpus 4 \
            --resume
    else
        echo "ERROR: Abaqus not found. Use --skip-abaqus to skip."
        exit 1
    fi
else
    echo ""
    echo "[Step 2] Skipping Abaqus (--skip-abaqus)"
fi

# ----- Step 3: Validate dataset -----
echo ""
echo "[Step 3] Validating dataset..."
python3 -c "
import os, csv
dataset_dir = '$DATASET_DIR'
n_ok = 0
n_defect_total = 0
for name in sorted(os.listdir(dataset_dir)):
    meta_path = os.path.join(dataset_dir, name, 'metadata.csv')
    nodes_path = os.path.join(dataset_dir, name, 'nodes.csv')
    if not os.path.exists(nodes_path):
        continue
    with open(nodes_path) as f:
        n_nodes = sum(1 for _ in f) - 1
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for row in csv.DictReader(f):
                meta[row['key']] = row['value']
    n_def = int(meta.get('n_defect_nodes', 0))
    n_defect_total += n_def
    print(f'  {name}: {n_nodes} nodes, {n_def} defect nodes')
    n_ok += 1
print(f'Total: {n_ok} samples, {n_defect_total} defect nodes')
if n_defect_total < 100:
    print('WARNING: Very few defect nodes. Check defect radii and mesh density.')
"

# ----- Step 4: Prepare ML data -----
echo ""
echo "[Step 4] Building graphs (prepare_ml_data)..."
python src/prepare_ml_data.py \
    --input "$DATASET_DIR" \
    --output "$ML_DATA_DIR" \
    --val_ratio 0.2

# ----- Step 5: Train GAT -----
echo ""
echo "[Step 5] Training GAT model..."
python src/train.py \
    --arch gat \
    --data_dir "$ML_DATA_DIR" \
    --output_dir runs/verify \
    --epochs 100 \
    --batch_size 4 \
    --hidden 128 \
    --layers 4 \
    --lr 1e-3 \
    --patience 30 \
    --log_every 5

echo ""
echo "=========================================="
echo " Verification complete!"
echo " Check: runs/verify/ for training logs"
echo "=========================================="
