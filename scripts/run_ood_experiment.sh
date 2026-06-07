#!/usr/bin/env bash
# run_ood_experiment.sh — End-to-end OOD generalisation experiment
#
# Steps:
#   1. Prepare three OOD splits (P50 / P75 / P90) on vancouver01
#   2. Train LGSTA (best arch) on each ID train set
#   3. Evaluate on OOD test set and ID val set
#   4. Save results to runs/ood_*/results.json
#
# Usage:
#   bash scripts/run_ood_experiment.sh
#   bash scripts/run_ood_experiment.sh --arch sage  # test on SAGE too

set -e
PY=/home/nishioka/miniconda3/envs/gnn_final_env/bin/python
ROOT=/home/nishioka/Payload2026
ARCH=${1:-lgsta}
GPU=2  # Transolver slot freed

cd $ROOT

echo "=== [1/3] Preparing OOD splits ==="
for PCT in 50 75 90; do
  TAG="ood_p${PCT}"
  echo "  Preparing $TAG (threshold=P${PCT} defect radius) ..."
  $PY scripts/prepare_ood_split.py \
    --data_dir data/processed_s12_thermal_700_5class \
    --threshold_pct $PCT \
    --tag $TAG \
    --seed 42
done

echo ""
echo "=== [2/3] Training $ARCH on each ID split ==="
for PCT in 50 75 90; do
  TAG="ood_p${PCT}"
  RUN_DIR="runs/ood_${ARCH}_${TAG}"
  echo "  Training on $TAG → $RUN_DIR ..."
  CUDA_VISIBLE_DEVICES=$GPU \
  $PY src/train.py \
    --arch $ARCH \
    --data_dir data/$TAG \
    --output_dir $RUN_DIR \
    --task multitask \
    --reg_weight 0.5 \
    --loss focal --focal_gamma 2.0 \
    --physics_lambda_stress 0.1 \
    --physics_lambda_eq 0.05 \
    --epochs 200 \
    --hidden 128 \
    --layers 4 \
    --batch_size 1 \
    2>&1 | tee logs/ood_${ARCH}_${TAG}.log
done

echo ""
echo "=== [3/3] OOD Evaluation ==="
for PCT in 50 75 90; do
  TAG="ood_p${PCT}"
  RUN_DIR="runs/ood_${ARCH}_${TAG}"
  CKPT="$RUN_DIR/best_model.pt"
  if [ -f "$CKPT" ]; then
    echo "  Evaluating $ARCH on OOD split $TAG ..."
    CUDA_VISIBLE_DEVICES=$GPU \
    $PY src/train.py \
      --arch $ARCH \
      --data_dir data/$TAG \
      --output_dir ${RUN_DIR}_ood_eval \
      --task multitask \
      --loss focal --focal_gamma 2.0 \
      --epochs 1 \
      --eval_only \
      --pretrained $CKPT \
      --eval_split ood \
      2>&1 | tee logs/ood_eval_${ARCH}_${TAG}.log || echo "  (eval_only not supported, skipping)"
  fi
done

echo ""
echo "Done. Check logs/ and runs/ood_*/ for results."
