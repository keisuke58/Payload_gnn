#!/bin/bash
# FNO GW Surrogate: 4条件比較実験
# Baseline vs Improved (spectral loss + warmup + augmentation)
#
# N=6 samples, CPU で ~5-10 min per run

set -e
cd "$(dirname "$0")/.."

DATA_DIR="abaqus_work/gw_fairing_dataset"
DOE="doe_gw_fairing.json"
EPOCHS=300
COMMON="--data_dir $DATA_DIR --doe $DOE --epochs $EPOCHS --batch_size 4 --downsample 4 --log_interval 50"

echo "=== FNO GW Comparison (N=6, 4 conditions) ==="
echo ""

# 1. Baseline FNO (no spectral loss, no augmentation)
echo ">>> [1/4] Baseline FNO"
python src/train_fno_gw.py --model fno $COMMON \
  --spectral_weight 0.0 --augment 1 \
  --output runs/compare_baseline_fno

# 2. FNO + Spectral Loss + Warmup
echo ""
echo ">>> [2/4] FNO + Spectral Loss (0.1)"
python src/train_fno_gw.py --model fno $COMMON \
  --spectral_weight 0.1 --augment 1 \
  --output runs/compare_spectral_fno

# 3. FNO + Augmentation (10x)
echo ""
echo ">>> [3/4] FNO + Augmentation (10x)"
python src/train_fno_gw.py --model fno $COMMON \
  --spectral_weight 0.0 --augment 10 \
  --output runs/compare_aug10_fno

# 4. FNO + Spectral + Augmentation (full improved)
echo ""
echo ">>> [4/4] FNO + Spectral + Aug10 (full improved)"
python src/train_fno_gw.py --model fno $COMMON \
  --spectral_weight 0.1 --augment 10 \
  --output runs/compare_full_fno

# 5. Dual-Stage FNO + Spectral + Aug (bonus)
echo ""
echo ">>> [5/5] Dual-Stage FNO + Spectral + Aug10"
python src/train_fno_gw.py --model dual_fno $COMMON \
  --spectral_weight 0.1 --augment 10 \
  --output runs/compare_full_dual_fno

echo ""
echo "=== All runs complete ==="
echo ""

# Evaluate all
echo "=== Evaluation ==="
for d in runs/compare_*; do
  if [ -f "$d/best_model.pt" ]; then
    echo ""
    echo ">>> Evaluating: $d"
    python src/evaluate_fno_gw.py --checkpoint "$d/best_model.pt" \
      --data_dir $DATA_DIR --doe $DOE --downsample 4
  fi
done
