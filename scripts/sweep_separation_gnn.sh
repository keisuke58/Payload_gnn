#!/bin/bash
# sweep_separation_gnn.sh — 分離データで GNN アーキテクチャ比較
# Usage: bash scripts/sweep_separation_gnn.sh [GPU_ID]
#
# Prerequisites:
#   - data/processed_separation/{train.pt, val.pt, norm_stats.pt}
#   - runs/pretrained_gat.pt (optional, for fine-tune)

GPU=${1:-0}
DATA_DIR=data/processed_separation
BASE_OUT=results/separation_sweep
PRETRAINED=runs/pretrained_gat.pt
EPOCHS=200
PATIENCE=40
SEED=42

export CUDA_VISIBLE_DEVICES=$GPU
export LD_LIBRARY_PATH=~/miniconda3/lib

echo "=== Separation GNN Architecture Sweep ==="
echo "GPU: $GPU, Data: $DATA_DIR"
echo ""

# Common args
COMMON="--data_dir $DATA_DIR --epochs $EPOCHS --patience $PATIENCE --seed $SEED --loss focal --focal_gamma 3.0 --defect_weight 10.0 --feature_noise 0.01"

# 1. GAT (baseline - from scratch)
echo "[1/8] GAT from scratch..."
python3 -u src/train.py $COMMON \
    --arch gat --hidden 64 --layers 3 --dropout 0.1 --lr 0.001 --batch_size 1 \
    --output_dir $BASE_OUT/gat_scratch \
    2>&1 | tail -5
echo ""

# 2. GAT + pretrained (fine-tune)
echo "[2/8] GAT fine-tuned..."
python3 -u src/train.py $COMMON \
    --arch gat --hidden 64 --layers 3 --dropout 0.1 --lr 0.001 --batch_size 1 \
    --pretrained $PRETRAINED --freeze_layers 1 \
    --output_dir $BASE_OUT/gat_finetune \
    2>&1 | tail -5
echo ""

# 3. GCN
echo "[3/8] GCN..."
python3 -u src/train.py $COMMON \
    --arch gcn --hidden 64 --layers 4 --dropout 0.1 --lr 0.001 --batch_size 1 \
    --output_dir $BASE_OUT/gcn \
    2>&1 | tail -5
echo ""

# 4. GraphSAGE
echo "[4/8] GraphSAGE..."
python3 -u src/train.py $COMMON \
    --arch sage --hidden 64 --layers 4 --dropout 0.1 --lr 0.001 --batch_size 1 \
    --output_dir $BASE_OUT/sage \
    2>&1 | tail -5
echo ""

# 5. GIN
echo "[5/8] GIN..."
python3 -u src/train.py $COMMON \
    --arch gin --hidden 64 --layers 4 --dropout 0.1 --lr 0.001 --batch_size 1 \
    --output_dir $BASE_OUT/gin \
    2>&1 | tail -5
echo ""

# 6. GAT + augmentation (DropEdge + NodeDrop + Flip)
echo "[6/8] GAT + full augmentation..."
python3 -u src/train.py $COMMON \
    --arch gat --hidden 64 --layers 3 --dropout 0.1 --lr 0.001 --batch_size 1 \
    --pretrained $PRETRAINED --freeze_layers 1 \
    --drop_edge 0.1 --node_drop 0.1 --augment_flip 0.3 --feature_mask 0.05 \
    --output_dir $BASE_OUT/gat_ft_aug \
    2>&1 | tail -5
echo ""

# 7. GAT + physics-informed loss
echo "[7/8] GAT + physics loss..."
python3 -u src/train.py $COMMON \
    --arch gat --hidden 64 --layers 3 --dropout 0.1 --lr 0.001 --batch_size 1 \
    --pretrained $PRETRAINED --freeze_layers 1 \
    --physics_lambda_smooth 0.1 --physics_lambda_stress 0.05 \
    --output_dir $BASE_OUT/gat_ft_physics \
    2>&1 | tail -5
echo ""

# 8. GAT large (hidden=128, layers=4)
echo "[8/8] GAT large (h=128, L=4)..."
python3 -u src/train.py $COMMON \
    --arch gat --hidden 128 --layers 4 --dropout 0.15 --lr 0.0005 --batch_size 1 \
    --output_dir $BASE_OUT/gat_large \
    2>&1 | tail -5
echo ""

# Summary
echo "=== Results Summary ==="
for d in $BASE_OUT/*/; do
    name=$(basename $d)
    if [ -f "$d/best_model.pt" ]; then
        f1=$(python3 -c "import torch; c=torch.load('$d/best_model.pt', weights_only=False, map_location='cpu'); print('%.4f' % c.get('val_f1', 0))" 2>/dev/null)
        echo "  $name: Val OptF1 = $f1"
    fi
done
echo ""
echo "Done."
