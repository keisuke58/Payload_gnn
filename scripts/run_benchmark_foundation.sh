#!/bin/bash
# Foundation Model Benchmark: FNO vs CNN vs DPOT vs Poseidon + Dual-Stage FNO
# GPU recommended (Poseidon 128x128 SwinV2 is very slow on CPU)
#
# Usage:
#   ./scripts/gpu_submit.sh auto "bash scripts/run_benchmark_foundation.sh"

set -e
cd "$(dirname "$0")/.."

# Use miniconda python (vancouver02: /home/nishioka/miniconda3/bin/python)
export PATH="/home/nishioka/miniconda3/bin:$PATH"

DATA_DIR="abaqus_work/gw_fairing_dataset"
DOE="doe_gw_fairing.json"
EPOCHS=300
AUGMENT=10
COMMON="--data_dir $DATA_DIR --doe $DOE --epochs $EPOCHS --augment $AUGMENT --batch_size 4 --downsample 4 --log_interval 50"

echo "=== Foundation Model Benchmark ==="
echo "Epochs: $EPOCHS, Augment: ${AUGMENT}x"
echo "Device: $(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo ""

# 1. Foundation model benchmark: FNO, CNN, DPOT, Poseidon
for MODEL in fno cnn dpot poseidon; do
  echo ">>> Training: $MODEL"
  python src/benchmark_foundation.py --model $MODEL $COMMON \
    --output runs/benchmark_foundation \
    --lr 1e-3
  echo ""
done

# 2. Dual-Stage FNO (from comparison experiment, re-run on GPU)
echo ">>> Training: dual_fno (spectral+aug10)"
python src/train_fno_gw.py --model dual_fno $COMMON \
  --spectral_weight 0.1 \
  --output runs/compare_full_dual_fno
echo ""

echo "=== All Benchmark Training Complete ==="
echo ""

# Summary
echo "--- Foundation Model Results ---"
cat runs/benchmark_foundation/benchmark_results.json 2>/dev/null || echo "(no results)"
echo ""
echo "--- Dual-Stage FNO ---"
tail -1 runs/compare_full_dual_fno/train_log.jsonl 2>/dev/null || echo "(no results)"
