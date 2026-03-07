#!/bin/bash
# 3 Foundation Models 統合実行スクリプト (vancouver02 GPU用)
#
# 1. Chronos-Bolt: 時系列異常検知 (ガイド波)
# 2. AnomalyGFM:   グラフ異常検知 (ノードレベル zero-shot)
# 3. MeshGraphNet:  FEM サロゲートモデル学習
#
# Usage:
#   bash scripts/run_all_fm.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=~/miniconda3/bin/python
DEVICE=cuda

echo "============================================================"
echo " Foundation Model 統合実験"
echo " $(date)"
echo "============================================================"

# ── 1. Chronos-Bolt: GW 時系列異常検知 ──
echo ""
echo ">>> [1/3] Chronos-Bolt: Guided Wave Anomaly Detection"
echo "------------------------------------------------------------"
$PYTHON src/chronos_shm.py \
    --mode embedding \
    --device $DEVICE \
    --downsample 1 \
    --out-dir results/chronos_shm

# ── 2. AnomalyGFM: グラフノード異常検知 ──
echo ""
echo ">>> [2/3] AnomalyGFM: Graph Node Anomaly Detection (zero-shot)"
echo "------------------------------------------------------------"
$PYTHON src/anomalygfm_shm.py \
    --data_dir data/processed_s12_thermal_500 \
    --device $DEVICE \
    --max_samples 20 \
    --n_runs 10 \
    --out_dir results/anomalygfm_shm

# ── 3. MeshGraphNet: FEM サロゲートモデル ──
echo ""
echo ">>> [3/3] MeshGraphNet: FEM Surrogate Training"
echo "------------------------------------------------------------"
$PYTHON src/physicsnemo_surrogate.py \
    --mode train \
    --data_dir data/processed_s12_thermal_500 \
    --output_dir runs/surrogate \
    --device $DEVICE \
    --hidden_dim 128 \
    --n_blocks 6 \
    --epochs 200 \
    --batch_size 4

echo ""
echo "============================================================"
echo " All 3 Foundation Models completed!"
echo " $(date)"
echo "============================================================"
echo ""
echo "Results:"
echo "  Chronos:     results/chronos_shm/"
echo "  AnomalyGFM:  results/anomalygfm_shm/"
echo "  Surrogate:   runs/surrogate/"
