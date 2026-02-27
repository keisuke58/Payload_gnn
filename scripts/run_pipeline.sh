#!/bin/bash
# =============================================================================
# Full Pipeline: Dataset Generation -> Preprocessing -> Training -> Evaluation
#
# Usage:
#   bash scripts/run_pipeline.sh              # Full pipeline
#   bash scripts/run_pipeline.sh --skip-abaqus  # Skip FEM (use existing CSV data)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

SKIP_ABAQUS=false
for arg in "$@"; do
    case $arg in
        --skip-abaqus) SKIP_ABAQUS=true ;;
    esac
done

echo "=========================================="
echo " Payload Fairing GNN Pipeline"
echo "=========================================="
echo " Project root: $PROJECT_ROOT"
echo " Skip Abaqus:  $SKIP_ABAQUS"
echo ""

# ----- Phase 1: FEM Dataset Generation -----
if [ "$SKIP_ABAQUS" = false ]; then
    echo "[Phase 1] Running Abaqus dataset generation..."
    if command -v abaqus &> /dev/null; then
        abaqus cae noGUI=src/generate_fairing_dataset.py
    else
        echo "WARNING: Abaqus not found. Skipping FEM generation."
        echo "         Use --skip-abaqus to skip this step explicitly."
    fi
fi

# ----- Phase 2: Preprocessing -----
echo ""
echo "[Phase 2] Preprocessing FEM data -> PyG graphs..."
python src/preprocess_fairing_data.py \
    --raw_dir dataset_output \
    --output_dir dataset/processed \
    --mesh_size 50.0 \
    --height 5000.0

# ----- Phase 3: Training (all architectures) -----
echo ""
echo "[Phase 3] Training GNN models..."

ARCHS=("gat" "gcn" "gin" "sage")
for arch in "${ARCHS[@]}"; do
    echo ""
    echo "--- Training: $arch ---"
    python src/train.py \
        --arch "$arch" \
        --data_dir dataset/processed \
        --output_dir runs \
        --epochs 200 \
        --batch_size 4 \
        --hidden 128 \
        --layers 4 \
        --lr 1e-3 \
        --patience 30 \
        --log_every 10
done

# ----- Phase 4: Evaluation & Comparison -----
echo ""
echo "[Phase 4] Evaluating and comparing architectures..."
python src/evaluate.py \
    --compare_dir runs \
    --data_dir dataset/processed \
    --output_dir results

# Evaluate best model on OOD set
BEST_RUN=$(python -c "
import json, os
best_f1, best_dir = 0, ''
for d in os.listdir('runs'):
    p = os.path.join('runs', d, 'best_model.pt')
    if os.path.exists(p):
        import torch
        c = torch.load(p, map_location='cpu', weights_only=False)
        if c.get('val_f1', 0) > best_f1:
            best_f1 = c['val_f1']
            best_dir = os.path.join('runs', d)
print(best_dir)
")

if [ -n "$BEST_RUN" ]; then
    echo ""
    echo "--- Best model: $BEST_RUN ---"
    python src/evaluate.py \
        --checkpoint "$BEST_RUN/best_model.pt" \
        --data_dir dataset/processed \
        --output_dir results \
        --eval_ood \
        --num_vis 10
fi

echo ""
echo "=========================================="
echo " Pipeline complete!"
echo " Results: results/"
echo " Models:  runs/"
echo "=========================================="
