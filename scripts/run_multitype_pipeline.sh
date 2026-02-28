#!/bin/bash
# =============================================================================
# Multi-Type Defect Pipeline: 前処理 → 4-class GNN ベンチマーク
#
# Usage:
#   bash scripts/run_multitype_pipeline.sh
#   bash scripts/run_multitype_pipeline.sh --input dataset_multitype_100
#   bash scripts/run_multitype_pipeline.sh --arch gat   # single arch
#   bash scripts/run_multitype_pipeline.sh --all        # all 4 archs (default)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT_DIR="dataset_multitype_100"
OUTPUT_DATA="data/processed_multitype_100"
OUTPUT_RUNS="runs/multitype"
ARCH=""
ALL_ARCHS=true
EPOCHS=200
LOSS="weighted_ce"

for arg in "$@"; do
    case $arg in
        --input=*) INPUT_DIR="${arg#*=}" ;;
        --output=*) OUTPUT_DATA="${arg#*=}" ;;
        --arch=*) ARCH="${arg#*=}"; ALL_ARCHS=false ;;
        --epochs=*) EPOCHS="${arg#*=}" ;;
        --loss=*) LOSS="${arg#*=}" ;;
        --all) ALL_ARCHS=true ;;
    esac
done

echo "=========================================="
echo " Multi-Type Defect Pipeline"
echo "=========================================="
echo " Input:  $INPUT_DIR"
echo " Output: $OUTPUT_DATA"
echo " Runs:   $OUTPUT_RUNS"
echo " Epochs: $EPOCHS | Loss: $LOSS"
echo ""

# ----- Phase 1: Preprocess (CSV → PyG graphs with 4-class labels) -----
if [ -f "$OUTPUT_DATA/train.pt" ]; then
    echo "[Phase 1] train.pt already exists, skipping preprocessing."
    echo "  (delete $OUTPUT_DATA to re-preprocess)"
else
    echo "[Phase 1] Preprocessing: build curvature-aware graphs..."
    python src/prepare_ml_data.py \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DATA" \
        --val_ratio 0.2
fi

echo ""

# ----- Phase 2: Train GNN(s) -----
if [ "$ALL_ARCHS" = true ]; then
    ARCHS=("gat" "gcn" "gin" "sage")
    echo "[Phase 2] Benchmark: training all 4 architectures"
else
    ARCHS=("$ARCH")
    echo "[Phase 2] Training: $ARCH"
fi

for arch in "${ARCHS[@]}"; do
    echo ""
    echo "--- Training $arch ---"
    python src/train.py \
        --arch "$arch" \
        --data_dir "$OUTPUT_DATA" \
        --output_dir "$OUTPUT_RUNS" \
        --epochs "$EPOCHS" \
        --batch_size 4 \
        --hidden 128 \
        --layers 4 \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --patience 30 \
        --loss "$LOSS" \
        --log_every 10
done

echo ""
echo "=========================================="
echo " Pipeline complete!"
echo " Results: $OUTPUT_RUNS/"
echo "=========================================="

# ----- Summary -----
echo ""
echo "=== Benchmark Summary ==="
for dir in "$OUTPUT_RUNS"/*/; do
    if [ -f "$dir/training_log.csv" ]; then
        name=$(basename "$dir")
        best=$(tail -1 "$dir/training_log.csv" | cut -d',' -f6)
        echo "  $name: val_f1=$best"
    fi
done
