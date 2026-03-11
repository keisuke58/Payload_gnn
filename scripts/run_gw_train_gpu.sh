#!/bin/bash
# run_gw_train_gpu.sh — 4 GPU 並行学習 (vancouver02 用)
# Usage: bash scripts/run_gw_train_gpu.sh

source /home/nishioka/miniforge3/etc/profile.d/conda.sh
conda activate base

cd ~/Payload2026

# Kill previous runs if any
pkill -f "train_gw.py" 2>/dev/null
sleep 1

DATA=data/processed_gw_comprehensive
EPOCHS=300
OVERSAMPLE=10

mkdir -p runs/gw_gat_comp runs/gw_gin_comp runs/gw_sage_comp runs/gw_gcn_comp

echo "[$(date)] Launching 4 GPU training jobs..."

# GAT (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup python src/train_gw.py \
  --data_dir $DATA --arch gat --epochs $EPOCHS --batch_size 32 --lr 5e-4 \
  --hidden 128 --num_layers 3 --loss focal --focal_gamma 2.0 --class_weight \
  --oversample $OVERSAMPLE --aug_noise 0.03 --weight_decay 1e-4 --seed 42 \
  --run_dir runs/gw_gat_comp > runs/gw_gat_comp.log 2>&1 &
echo "  GAT on GPU 0 (PID $!)"

# GIN (GPU 1)
CUDA_VISIBLE_DEVICES=1 nohup python src/train_gw.py \
  --data_dir $DATA --arch gin --epochs $EPOCHS --batch_size 32 --lr 5e-4 \
  --hidden 128 --num_layers 4 --loss focal --focal_gamma 2.0 --class_weight \
  --oversample $OVERSAMPLE --aug_noise 0.03 --weight_decay 1e-4 --seed 42 \
  --run_dir runs/gw_gin_comp > runs/gw_gin_comp.log 2>&1 &
echo "  GIN on GPU 1 (PID $!)"

# GraphSAGE (GPU 2)
CUDA_VISIBLE_DEVICES=2 nohup python src/train_gw.py \
  --data_dir $DATA --arch sage --epochs $EPOCHS --batch_size 32 --lr 5e-4 \
  --hidden 128 --num_layers 3 --loss focal --focal_gamma 2.0 --class_weight \
  --oversample $OVERSAMPLE --aug_noise 0.03 --weight_decay 1e-4 --seed 42 \
  --run_dir runs/gw_sage_comp > runs/gw_sage_comp.log 2>&1 &
echo "  SAGE on GPU 2 (PID $!)"

# GCN (GPU 3)
CUDA_VISIBLE_DEVICES=3 nohup python src/train_gw.py \
  --data_dir $DATA --arch gcn --epochs $EPOCHS --batch_size 32 --lr 1e-3 \
  --hidden 64 --num_layers 3 --loss focal --focal_gamma 2.0 --class_weight \
  --oversample $OVERSAMPLE --aug_noise 0.03 --weight_decay 1e-4 --seed 42 \
  --run_dir runs/gw_gcn_comp > runs/gw_gcn_comp.log 2>&1 &
echo "  GCN on GPU 3 (PID $!)"

echo "[$(date)] All jobs launched. Check: tail -f runs/gw_*_comp.log"
