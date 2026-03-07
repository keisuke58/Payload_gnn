#!/bin/bash
# 4-GPU Parallel Training Launch Script
# GPU 0: GraphMAE self-supervised pre-training (600 samples)
# GPU 1: 5-class GNN training (SAGE, 600 samples)
# GPU 2: PI-GNN (SAGE + physics losses, 5-class)
# GPU 3: FNO→GNN knowledge distillation
#
# Usage: bash scripts/launch_parallel_training.sh
# Logs: runs/parallel_*/

set -u
cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "============================================="
echo "  Parallel Training Launch: $TIMESTAMP"
echo "============================================="

# ── GPU 0: GraphMAE Pre-training ──
echo "[GPU 0] GraphMAE self-supervised pre-training..."
CUDA_VISIBLE_DEVICES=0 python src/prad/train_mae.py \
    --data_dir data/processed_s12_thermal_600 \
    --encoder_arch sage \
    --hidden 128 \
    --num_layers 4 \
    --mask_ratio 0.5 \
    --lambda_physics 0.1 \
    --epochs 300 \
    --lr 1e-3 \
    --patience 40 \
    --device cuda \
    --log_dir "runs/parallel_mae_${TIMESTAMP}" \
    --checkpoint_dir checkpoints \
    > "runs/parallel_mae_${TIMESTAMP}.log" 2>&1 &
PID_MAE=$!
echo "  PID: $PID_MAE"

# ── GPU 1: 5-class GNN (SAGE baseline) ──
echo "[GPU 1] 5-class SAGE training..."
CUDA_VISIBLE_DEVICES=1 python src/train.py \
    --arch sage \
    --data_dir data/processed_s12_thermal_600_5class \
    --hidden 128 \
    --layers 4 \
    --dropout 0.1 \
    --lr 1e-3 \
    --epochs 300 \
    --batch_size 4 \
    --loss focal \
    --focal_gamma 2.0 \
    --boundary_weight 3.0 \
    --defect_weight 5.0 \
    --feature_noise 0.01 \
    --drop_edge 0.05 \
    --augment_flip 0.5 \
    --patience 40 \
    --output_dir "runs/parallel_5class_${TIMESTAMP}" \
    > "runs/parallel_5class_${TIMESTAMP}.log" 2>&1 &
PID_5CLASS=$!
echo "  PID: $PID_5CLASS"

# ── GPU 2: PI-GNN (SAGE + physics-informed losses) ──
echo "[GPU 2] PI-GNN (physics-informed SAGE)..."
CUDA_VISIBLE_DEVICES=2 python src/train.py \
    --arch sage \
    --data_dir data/processed_s12_thermal_600_5class \
    --hidden 128 \
    --layers 4 \
    --dropout 0.1 \
    --lr 1e-3 \
    --epochs 300 \
    --batch_size 4 \
    --loss focal \
    --focal_gamma 2.0 \
    --boundary_weight 3.0 \
    --defect_weight 5.0 \
    --feature_noise 0.01 \
    --drop_edge 0.05 \
    --augment_flip 0.5 \
    --physics_lambda_smooth 0.1 \
    --physics_lambda_stress 0.1 \
    --physics_lambda_connected 0.05 \
    --patience 40 \
    --output_dir "runs/parallel_pignn_${TIMESTAMP}" \
    > "runs/parallel_pignn_${TIMESTAMP}.log" 2>&1 &
PID_PIGNN=$!
echo "  PID: $PID_PIGNN"

# ── GPU 3: FNO→GNN Distillation ──
echo "[GPU 3] FNO→GNN knowledge distillation..."
CUDA_VISIBLE_DEVICES=3 python src/prad/distill_fno2gnn.py \
    --mae_checkpoint checkpoints/prad_mae_sage.pt \
    --fno_checkpoint runs/fno_production/best_model.pt \
    --data_dir data/processed_s12_czm_thermal_200_binary \
    --fno_grid_dir data/fno_grids_200 \
    --epochs 150 \
    --lr 5e-4 \
    --device cuda \
    --log_dir "runs/parallel_distill_${TIMESTAMP}" \
    --output "checkpoints/prad_distilled_${TIMESTAMP}.pt" \
    > "runs/parallel_distill_${TIMESTAMP}.log" 2>&1 &
PID_DISTILL=$!
echo "  PID: $PID_DISTILL"

echo ""
echo "============================================="
echo "  All 4 tasks launched!"
echo "  PIDs: MAE=$PID_MAE  5CLASS=$PID_5CLASS  PIGNN=$PID_PIGNN  DISTILL=$PID_DISTILL"
echo "============================================="
echo ""
echo "Monitor:"
echo "  tail -f runs/parallel_mae_${TIMESTAMP}.log"
echo "  tail -f runs/parallel_5class_${TIMESTAMP}.log"
echo "  tail -f runs/parallel_pignn_${TIMESTAMP}.log"
echo "  tail -f runs/parallel_distill_${TIMESTAMP}.log"
echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir runs/ --port 6006"
echo ""
echo "Wait for all:"
echo "  wait $PID_MAE $PID_5CLASS $PID_PIGNN $PID_DISTILL"

# Wait for all background processes
wait $PID_MAE $PID_5CLASS $PID_PIGNN $PID_DISTILL
echo ""
echo "All tasks completed!"
