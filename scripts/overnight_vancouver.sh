#!/bin/bash
# Overnight tasks on vancouver02
# 1. 5-fold CV with best λ (smooth=0.5, stress=0.05, connected=0.01)
# 2. Ensemble inference with uncertainty (GPU)

PYTHON=~/miniconda3/bin/python
cd ~/Payload2026

LOGDIR=runs/overnight_$(date +%Y%m%d)
mkdir -p $LOGDIR

echo "=== Task 1: 5-fold CV with optimal λ ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=0 $PYTHON -u src/train.py \
  --arch gat --hidden 128 --layers 4 \
  --loss focal --focal_gamma 3.0 \
  --residual --defect_weight 5.0 \
  --feature_mask 0.1 --augment_flip 0.5 \
  --drop_edge 0.1 --feature_noise 0.01 \
  --physics_lambda_smooth 0.5 \
  --physics_lambda_stress 0.05 \
  --physics_lambda_connected 0.01 \
  --cross_val 5 --epochs 200 --seed 42 \
  --data_dir data/processed_s12_czm_thermal_200_binary \
  > $LOGDIR/cv5_bestlambda.log 2>&1

echo "5-fold CV done: $(date)"
echo "Results:"
tail -5 $LOGDIR/cv5_bestlambda.log

echo ""
echo "=== Task 2: Ensemble inference + Uncertainty (GPU) ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=0 $PYTHON -u scripts/ensemble_inference.py \
  --uncertainty --mc_T 20 \
  --output_dir $LOGDIR/uncertainty \
  --save_results $LOGDIR/ensemble_results.json \
  > $LOGDIR/uncertainty_eval.log 2>&1

echo "Uncertainty done: $(date)"
tail -20 $LOGDIR/uncertainty_eval.log

echo ""
echo "=== All overnight tasks completed: $(date) ==="
