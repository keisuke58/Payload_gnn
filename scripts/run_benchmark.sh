
#!/bin/bash

# Dataset path
DATA_DIR="data/processed_25mm_100"

# Output dir
OUT_DIR="runs/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUT_DIR

echo "========================================================"
echo "Starting Benchmark: GNN vs PointNet"
echo "Dataset: $DATA_DIR"
echo "Output: $OUT_DIR"
echo "========================================================"

# 1. Run GNN (GraphSAGE)
echo ""
echo "[1/2] Training GNN (GraphSAGE)..."
python src/train.py \
    --arch sage \
    --data_dir $DATA_DIR \
    --epochs 5 \
    --batch_size 16 \
    --hidden 64 \
    --output_dir $OUT_DIR/gnn \
    2>&1 | tee $OUT_DIR/gnn_log.txt

# 2. Run PointNet
echo ""
echo "[2/2] Training PointNet (Transformer)..."
# Note: PointNet consumes more memory due to dense batch, so reduce batch size if needed
# But nodes=40k is huge for dense operations. We might hit OOM.
# If OOM, we need to sample points or use smaller batch.
python src/train_pointnet.py \
    --data_dir $DATA_DIR \
    --epochs 5 \
    --batch_size 2 \
    --output_dir $OUT_DIR/pointnet \
    2>&1 | tee $OUT_DIR/pointnet_log.txt

echo ""
echo "========================================================"
echo "Benchmark Completed."
echo "Check results in $OUT_DIR"
