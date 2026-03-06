#!/bin/bash
# pipeline_fno_gnn.sh — FNO サロゲート → GNN 学習の統合パイプライン
#
# 1. FNO を Abaqus 実データで学習 (defect_params → sensor waveform)
# 2. FNO で大量の合成波形を生成 (1000-10000 samples)
# 3. 実データ + 合成データを統合して GNN 分類器を学習
#
# Usage:
#   bash scripts/pipeline_fno_gnn.sh train_fno     # Step 1: FNO 学習
#   bash scripts/pipeline_fno_gnn.sh generate 5000 # Step 2: 合成データ 5000件
#   bash scripts/pipeline_fno_gnn.sh prepare        # Step 3: GNN 用データ準備
#   bash scripts/pipeline_fno_gnn.sh train_gnn      # Step 4: GNN 学習
#   bash scripts/pipeline_fno_gnn.sh all             # 全ステップ実行

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
DATA_DIR="abaqus_work/gw_fairing_dataset"
DOE="doe_gw_fairing.json"
FNO_RUN_DIR="runs/fno_gw_surrogate"
FNO_GEN_DIR="$FNO_RUN_DIR/generated"
COMBINED_DIR="abaqus_work/gw_combined_dataset"
GNN_OUTPUT="data/processed_gw_fno_augmented"

N_GENERATE=${2:-5000}      # default 5000 synthetic samples
SENSOR_K="10"              # sensor subset for GNN (10 sensors)
FEATURE_SET="extended"

# Step 1: Train FNO surrogate
do_train_fno() {
    echo "=== Step 1: Train FNO Surrogate ==="

    # Check available defect data
    n_defect=$(ls "$DATA_DIR"/Job-GW-Fair-0*_sensors.csv 2>/dev/null | wc -l)
    echo "Available defect samples: $n_defect"

    if [ "$n_defect" -lt 5 ]; then
        echo "ERROR: Need at least 5 defect samples. Currently have $n_defect."
        echo "Run: bash scripts/run_gw_defect_parallel.sh all"
        exit 1
    fi

    python src/train_fno_gw.py \
        --model fno \
        --data_dir "$DATA_DIR" \
        --doe "$DOE" \
        --output "$FNO_RUN_DIR" \
        --epochs 500 \
        --batch_size 8 \
        --lr 1e-3 \
        --modes 64 \
        --width 64 \
        --n_layers 4 \
        --downsample 4 \
        --residual

    echo ""
    echo "FNO model saved: $FNO_RUN_DIR/best_model.pt"
}

# Step 2: Generate synthetic waveforms
do_generate() {
    echo "=== Step 2: Generate $N_GENERATE Synthetic Waveforms ==="

    if [ ! -f "$FNO_RUN_DIR/best_model.pt" ]; then
        echo "ERROR: FNO model not found. Run train_fno first."
        exit 1
    fi

    python src/train_fno_gw.py \
        --model fno \
        --data_dir "$DATA_DIR" \
        --doe "$DOE" \
        --output "$FNO_RUN_DIR" \
        --downsample 4 \
        --residual \
        --generate "$N_GENERATE"

    echo ""
    echo "Generated $N_GENERATE waveforms in $FNO_GEN_DIR/"
}

# Step 3: Prepare combined dataset for GNN
do_prepare() {
    echo "=== Step 3: Prepare Combined Dataset for GNN ==="
    mkdir -p "$COMBINED_DIR"

    # Copy real healthy data
    echo "  Copying real healthy data..."
    for f in "$DATA_DIR"/Job-GW-Fair-Healthy_sensors.csv \
             "$DATA_DIR"/Job-GW-Fair-Healthy-A*_sensors.csv; do
        [ -f "$f" ] && cp "$f" "$COMBINED_DIR/"
    done
    n_healthy=$(ls "$COMBINED_DIR"/Job-GW-Fair-Healthy*_sensors.csv 2>/dev/null | wc -l)
    echo "  Healthy samples: $n_healthy"

    # Copy real defect data
    echo "  Copying real defect data..."
    for f in "$DATA_DIR"/Job-GW-Fair-0*_sensors.csv; do
        [ -f "$f" ] && cp "$f" "$COMBINED_DIR/"
    done
    n_real_defect=$(ls "$COMBINED_DIR"/Job-GW-Fair-0*_sensors.csv 2>/dev/null | wc -l)
    echo "  Real defect samples: $n_real_defect"

    # Copy FNO-generated defect data
    echo "  Copying FNO-generated defect data..."
    for f in "$FNO_GEN_DIR"/FNO-GW-*_sensors.csv; do
        [ -f "$f" ] && cp "$f" "$COMBINED_DIR/"
    done
    n_synth=$(ls "$COMBINED_DIR"/FNO-GW-*_sensors.csv 2>/dev/null | wc -l)
    echo "  Synthetic defect samples: $n_synth"

    total=$((n_healthy + n_real_defect + n_synth))
    echo ""
    echo "  Total combined: $total samples"
    echo "    Healthy: $n_healthy"
    echo "    Defect (real): $n_real_defect"
    echo "    Defect (FNO): $n_synth"

    # Build PyG dataset
    echo ""
    echo "  Building PyG graph dataset..."
    python src/prepare_gw_ml_data.py \
        --input "$COMBINED_DIR" \
        --no_doe \
        --output "$GNN_OUTPUT" \
        --feature_set "$FEATURE_SET" \
        --val_ratio 0.2

    echo ""
    echo "  Dataset: $GNN_OUTPUT/"
    if [ -f "$GNN_OUTPUT/train.pt" ]; then
        python -c "
import torch
train = torch.load('$GNN_OUTPUT/train.pt', weights_only=False)
val = torch.load('$GNN_OUTPUT/val.pt', weights_only=False)
print(f'  train: {len(train)} graphs')
print(f'  val:   {len(val)} graphs')
print(f'  features: {train[0].x.shape}')
"
    fi
}

# Step 4: Train GNN classifier
do_train_gnn() {
    echo "=== Step 4: Train GNN Classifier (FNO-augmented) ==="

    if [ ! -f "$GNN_OUTPUT/train.pt" ]; then
        echo "ERROR: Dataset not found. Run prepare first."
        exit 1
    fi

    python src/train_gw.py \
        --data_dir "$GNN_OUTPUT" \
        --arch GAT \
        --epochs 200 \
        --batch_size 32 \
        --lr 1e-3 \
        --hidden 64 \
        --run_name "gw_fno_augmented"

    echo ""
    echo "GNN training complete."
}

# Status
do_status() {
    echo "=== FNO→GNN Pipeline Status ==="
    echo ""
    echo "Real data:"
    echo "  Healthy CSVs: $(ls "$DATA_DIR"/Job-GW-Fair-Healthy*_sensors.csv 2>/dev/null | wc -l)"
    echo "  Defect CSVs:  $(ls "$DATA_DIR"/Job-GW-Fair-0*_sensors.csv 2>/dev/null | wc -l)"
    echo ""

    if [ -f "$FNO_RUN_DIR/best_model.pt" ]; then
        echo "FNO model: $FNO_RUN_DIR/best_model.pt"
        python -c "
import torch
ckpt = torch.load('$FNO_RUN_DIR/best_model.pt', weights_only=False)
print(f\"  Epoch: {ckpt['epoch']}\")
print(f\"  Val loss: {ckpt['val_loss']:.6f}\")
" 2>/dev/null || true
    else
        echo "FNO model: not trained yet"
    fi
    echo ""

    echo "Generated waveforms: $(ls "$FNO_GEN_DIR"/FNO-GW-*_sensors.csv 2>/dev/null | wc -l)"
    echo ""

    if [ -f "$GNN_OUTPUT/train.pt" ]; then
        echo "GNN dataset: $GNN_OUTPUT/"
        python -c "
import torch
train = torch.load('$GNN_OUTPUT/train.pt', weights_only=False)
val = torch.load('$GNN_OUTPUT/val.pt', weights_only=False)
print(f'  train: {len(train)} | val: {len(val)}')
" 2>/dev/null || true
    else
        echo "GNN dataset: not prepared yet"
    fi
}

case "${1:-status}" in
    train_fno)  do_train_fno ;;
    generate)   do_generate ;;
    prepare)    do_prepare ;;
    train_gnn)  do_train_gnn ;;
    status)     do_status ;;
    all)
        do_train_fno
        do_generate
        do_prepare
        do_train_gnn
        ;;
    *)  echo "Usage: $0 [train_fno|generate|prepare|train_gnn|status|all] [n_generate]"
        echo ""
        echo "  train_fno:  FNO サロゲート学習 (Abaqus 実データ)"
        echo "  generate N: FNO で N 件合成波形生成 (default: 5000)"
        echo "  prepare:    実データ + 合成データを統合 → PyG データセット"
        echo "  train_gnn:  GNN 分類器学習 (FNO 増強データ)"
        echo "  status:     パイプライン状態確認"
        echo "  all:        全ステップ実行"
        ;;
esac
