#!/bin/bash
# =============================================================================
# ML Pipeline: 前処理 → 学習 (データ生成完了後)
#
# Usage:
#   bash scripts/run_ml_pipeline.sh
#   bash scripts/run_ml_pipeline.sh --input dataset_output_50mm_100
#   bash scripts/run_ml_pipeline.sh --arch gat --epochs 100
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT_DIR="dataset_output_100"
OUTPUT_DATA="data/processed_50mm_100"
OUTPUT_RUNS="runs"
ARCH="gat"
EPOCHS=200

for arg in "$@"; do
    case $arg in
        --input=*) INPUT_DIR="${arg#*=}" ;;
        --arch=*) ARCH="${arg#*=}" ;;
        --epochs=*) EPOCHS="${arg#*=}" ;;
    esac
done

echo "=========================================="
echo " ML Pipeline: Preprocess → Train"
echo "=========================================="
echo " Input:  $INPUT_DIR"
echo " Output: $OUTPUT_DATA, $OUTPUT_RUNS"
echo " Arch:   $ARCH | Epochs: $EPOCHS"
echo ""

# ----- Phase 1: Preprocess -----
echo "[Phase 1] Preparing ML data (build_graph → train/val split)..."
python src/prepare_ml_data.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DATA" \
    --val_ratio 0.2

# ----- Phase 2: Train -----
echo ""
echo "[Phase 2] Training GNN..."
python src/train.py \
    --arch "$ARCH" \
    --data_dir "$OUTPUT_DATA" \
    --output_dir "$OUTPUT_RUNS" \
    --epochs "$EPOCHS" \
    --batch_size 4 \
    --hidden 128 \
    --layers 4 \
    --lr 1e-3 \
    --patience 30 \
    --log_every 10

echo ""
echo "=========================================="
echo " Pipeline complete!"
echo " Model: $OUTPUT_RUNS/"
echo "=========================================="
