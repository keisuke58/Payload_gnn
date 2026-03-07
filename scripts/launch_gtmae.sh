#!/bin/bash
# GPS Graph Transformer MAE: Pre-training + Fine-tuning
#
# Run on vancouver02 (GPU: RTX 4090 x4)
#
# Phase 1: Pre-train GPS+MAE on fairing + CompDam data (self-supervised)
# Phase 2: Fine-tune for defect classification (supervised)
#
# Usage:
#   bash scripts/launch_gtmae.sh
#
# Prerequisites:
#   - data/processed_s12_mixed_400/ exists (fairing training data)
#   - data/compdam_graphs.pt exists (optional, CompDam data)

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "============================================================"
echo "GPS Graph Transformer MAE Pipeline"
echo "Started: $(date)"
echo "============================================================"

# ── Phase 1: Pre-training ──
echo ""
echo "Phase 1: GPS+MAE Self-supervised Pre-training"
echo "============================================================"

COMPDAM_ARG=""
if [ -f data/compdam_graphs.pt ]; then
    COMPDAM_ARG="--compdam_graphs data/compdam_graphs.pt"
    echo "  CompDam data found: data/compdam_graphs.pt"
fi

CUDA_VISIBLE_DEVICES=0 python src/prad/train_gtmae.py pretrain \
    --data_dir data/processed_s12_mixed_400 \
    $COMPDAM_ARG \
    --hidden 128 \
    --num_layers 4 \
    --heads 4 \
    --dropout 0.1 \
    --mask_ratio 0.5 \
    --lambda_physics 0.1 \
    --epochs 300 \
    --lr 1e-3 \
    --patience 40 \
    --log_dir "runs/gtmae_pretrain_${TIMESTAMP}" \
    --checkpoint_dir checkpoints

echo ""
echo "Phase 1 complete."

# ── Phase 2: Fine-tuning ──
echo ""
echo "Phase 2: GPS Supervised Fine-tuning"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python src/prad/train_gtmae.py finetune \
    --data_dir data/processed_s12_mixed_400 \
    --pretrained checkpoints/gtmae_gps.pt \
    --hidden 128 \
    --num_layers 4 \
    --num_classes 2 \
    --epochs 150 \
    --lr 5e-4 \
    --patience 30 \
    --log_dir "runs/gtmae_finetune_${TIMESTAMP}" \
    --checkpoint_dir checkpoints

echo ""
echo "============================================================"
echo "Pipeline complete: $(date)"
echo "============================================================"
echo "  Pre-trained:  checkpoints/gtmae_gps.pt"
echo "  Fine-tuned:   checkpoints/gtmae_gps_finetuned.pt"
echo "  Logs:         runs/gtmae_*_${TIMESTAMP}/"
