#!/bin/bash
# =============================================================================
# Run training for all GNN architectures (GCN, GAT, GIN, SAGE) and compare
#
# Usage:
#   bash scripts/run_all_models.sh
#   bash scripts/run_all_models.sh --epochs 100
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data/processed_50mm_100"
OUTPUT_BASE="runs"
EPOCHS=200
BATCH_SIZE=4
HIDDEN=128
LAYERS=4
PATIENCE=30

for arg in "$@"; do
    case $arg in
        --epochs=*) EPOCHS="${arg#*=}" ;;
        --epochs) : ;;  # next arg would be value, skip
        --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
        --hidden=*) HIDDEN="${arg#*=}" ;;
        --layers=*) LAYERS="${arg#*=}" ;;
    esac
done
# Parse --epochs N (space-separated)
args=("$@")
for i in "${!args[@]}"; do
    if [[ "${args[i]}" == "--epochs" && $((i+1)) -lt ${#args[@]} ]]; then
        EPOCHS="${args[i+1]}"
        break
    fi
done

echo "=========================================="
echo " Multi-Model Training"
echo "=========================================="
echo " Data:    $DATA_DIR"
echo " Output:  $OUTPUT_BASE/"
echo " Epochs:  $EPOCHS | Batch: $BATCH_SIZE"
echo " Hidden:  $HIDDEN | Layers: $LAYERS"
echo ""

for ARCH in gcn gat gin sage; do
    RUN_DIR="${OUTPUT_BASE}/${ARCH}_$(date +%Y%m%d_%H%M%S)"
    echo "----------------------------------------"
    echo " [${ARCH^^}] Training..."
    echo "----------------------------------------"
    python src/train.py \
        --arch "$ARCH" \
        --data_dir "$DATA_DIR" \
        --output_dir "$RUN_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --hidden "$HIDDEN" \
        --layers "$LAYERS" \
        --patience "$PATIENCE" \
        --lr 1e-3 \
        --log_every 10
    echo ""
done

echo "=========================================="
echo " All models trained. Results in $OUTPUT_BASE/"
echo "=========================================="
