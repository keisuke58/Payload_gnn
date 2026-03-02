#!/bin/bash
# sweep_binary.sh — S12 CZM binary分類 学習 + 閾値最適化
#
# Usage (vancouver02):
#   nohup bash scripts/sweep_binary.sh > sweep_binary.log 2>&1 &
set -euo pipefail

# Conda activation (vancouver02: miniforge3)
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate base
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data/processed_s12_czm_96_binary"
OUT_DIR="runs/s12_binary"
mkdir -p "$OUT_DIR"

echo "========================================"
echo " Binary Training: S12 CZM 96 (34-dim)"
echo " Data: $DATA_DIR"
echo " Started: $(date)"
echo "========================================"

python3 src/train.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR" \
    --arch sage \
    --hidden 128 --layers 4 --dropout 0.1 \
    --lr 1e-3 --epochs 200 --patience 30 \
    --batch_size 4 \
    --loss focal --focal_gamma 2.0 \
    --log_every 10

echo ""
echo "=== Threshold Optimization ==="
BEST_PT=$(find "$OUT_DIR" -name "best_model.pt" -newer "$OUT_DIR" 2>/dev/null | head -1)
if [ -n "$BEST_PT" ]; then
    python3 scripts/optimize_threshold.py \
        --checkpoint "$BEST_PT" \
        --data_dir "$DATA_DIR"
fi

echo ""
echo "Done: $(date)"
