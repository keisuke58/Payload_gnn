#!/bin/bash
# DANN + MAE pretrained encoder 組み合わせ実験
# MAE pretrain 完了後に実行
PYBIN=/home/nishioka/miniconda3/bin/python
SRC=data/processed_s12_czm_thermal_200_binary
TGT=data/processed_s12_100_binary
MAE_CKPT=checkpoints/gtmae_gps.pt

cd ~/Payload2026

echo "=== Waiting for MAE pretrain to finish ==="
while pgrep -f 'train_gtmae.py pretrain' > /dev/null; do
    sleep 30
done
echo "=== MAE pretrain done ==="

echo ""
echo "=========================================="
echo "Exp 1: DANN (no pretrain) — already done"
echo "=========================================="
echo "Result: see logs/dann_sage.log"

echo ""
echo "=========================================="
echo "Exp 2: DANN + MAE pretrained encoder"
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 $PYBIN src/prad/domain_adapt.py \
    --source_dir $SRC --target_dir $TGT \
    --arch sage --hidden 128 --num_layers 4 \
    --epochs 100 --lr 5e-4 --lambda_domain 0.1 \
    --pretrained $MAE_CKPT \
    --device cuda

echo ""
echo "=========================================="
echo "Exp 3: GPS + DANN (no pretrain)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 $PYBIN src/prad/domain_adapt.py \
    --source_dir $SRC --target_dir $TGT \
    --arch gps --hidden 128 --num_layers 4 \
    --epochs 100 --lr 5e-4 --lambda_domain 0.1 \
    --device cuda

echo ""
echo "=========================================="
echo "Exp 4: GPS + DANN + MAE pretrained"
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 $PYBIN src/prad/domain_adapt.py \
    --source_dir $SRC --target_dir $TGT \
    --arch gps --hidden 128 --num_layers 4 \
    --epochs 100 --lr 5e-4 --lambda_domain 0.1 \
    --pretrained $MAE_CKPT \
    --device cuda

echo ""
echo "=========================================="
echo "Exp 5: DANN — Thermal700 → CZM+Thermal200 (harder cross-domain)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 $PYBIN src/prad/domain_adapt.py \
    --source_dir data/processed_s12_thermal_700v2_5class \
    --target_dir data/processed_s12_czm_thermal_200_binary \
    --arch sage --hidden 128 --num_layers 4 \
    --epochs 100 --lr 5e-4 --lambda_domain 0.1 \
    --run_baseline \
    --device cuda

echo ""
echo "=========================================="
echo "Exp 6: DANN + MAE — Thermal700 → CZM+Thermal200"
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 $PYBIN src/prad/domain_adapt.py \
    --source_dir data/processed_s12_thermal_700v2_5class \
    --target_dir data/processed_s12_czm_thermal_200_binary \
    --arch sage --hidden 128 --num_layers 4 \
    --epochs 100 --lr 5e-4 --lambda_domain 0.1 \
    --pretrained $MAE_CKPT \
    --device cuda

echo ""
echo "=== All experiments complete ==="
