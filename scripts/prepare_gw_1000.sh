#!/bin/bash
# prepare_gw_1000.sh — 約700サンプルの GW データセットを準備し、センサ数×特徴量パターンを作成
#
# 前提: abaqus_work/gw_fairing_dataset に 100センサ CSV があること
#   batch_generate_gw_dataset.sh all で subset 抽出済み → s10/s20/s30/s50
#
# 出力:
#   data/processed_gw_1000_s10_baseline/, s10_extended/, s10_full/
#   data/processed_gw_1000_s20_baseline/, ...
#   data/processed_gw_1000_s30_*, data/processed_gw_1000_s50_*
#
# Augment: Healthy 1→100, Defect 100→600 (100センサで augment → subset 抽出)
#
# Usage:
#   bash scripts/prepare_gw_1000.sh                    # フル実行
#   bash scripts/prepare_gw_1000.sh --prepare-only     # augment スキップ
#   bash scripts/prepare_gw_1000.sh --sensors 10 30    # s10, s30 のみ prepare
#   bash scripts/prepare_gw_1000.sh --sensors 10 30 --prepare-only

set -e
PREPARE_ONLY=false
SENSORS="10 20 30 50"
i=1
while [ $i -le $# ]; do
  arg="${!i}"
  case "$arg" in
    --prepare-only) PREPARE_ONLY=true ;;
    --sensors)
      SENSORS=""
      ((i++))
      while [ $i -le $# ] && [[ "${!i}" != --* ]]; do
        SENSORS="${SENSORS:+$SENSORS }${!i}"
        ((i++))
      done
      ((i--))
      ;;
  esac
  ((i++))
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT_BASE="abaqus_work/gw_fairing_dataset"
DOE="doe_gw_fairing.json"

# 1. Augment: 100センサ CSV で Healthy 1→100, Defect 100→600
if [ "$PREPARE_ONLY" = false ]; then
  echo "=== Step 1: Augmentation (100 sensors) ==="
  if [ ! -f "$INPUT_BASE/Job-GW-Fair-Healthy_sensors.csv" ]; then
      echo "ERROR: Healthy CSV not found. Run batch_generate_gw_dataset.sh all first."
      exit 1
  fi
  if [ ! -f "$INPUT_BASE/Job-GW-Fair-0000_sensors.csv" ]; then
      echo "ERROR: Defect CSV (0000) not found. Run batch_generate_gw_dataset.sh all first."
      exit 1
  fi

  python src/augment_gw_healthy.py \
      --input "$INPUT_BASE/Job-GW-Fair-Healthy_sensors.csv" \
      --output "$INPUT_BASE" \
      --n_healthy 99 \
      --prefix "Job-GW-Fair-Healthy-A" \
      --seed 42

  python src/augment_gw_defect.py \
      --input "$INPUT_BASE" \
      --doe "$DOE" \
      --n_per_defect 5 \
      --output "$INPUT_BASE" \
      --seed 43

  echo "=== Step 2: Extract subset (10, 20, 30, 50) ==="
  python scripts/extract_gw_sensor_subset.py --input "$INPUT_BASE" --output "$INPUT_BASE" --k 10 20 30 50
  echo ""
else
  echo "=== Step 1-2: Skipped (--prepare-only) ==="
fi

echo "=== Step 3: Prepare (sensor × feature) ==="
for S in $SENSORS; do
  IN="$INPUT_BASE/s$S"
  if [ ! -d "$IN" ]; then
    echo "  SKIP s$S: $IN not found (run without --prepare-only first)"
    continue
  fi
  for FEAT in baseline extended full; do
    OUT="data/processed_gw_1000_s${S}_${FEAT}"
    echo "  s${S} $FEAT -> $OUT"
    python src/prepare_gw_ml_data.py \
        --input "$IN" \
        --no_doe \
        --output "$OUT" \
        --feature_set "$FEAT" \
        --val_ratio 0.2
  done
done

echo ""
echo "=== Done ==="
echo "Datasets:"
for S in $SENSORS; do
  for FEAT in baseline extended full; do
    P="data/processed_gw_1000_s${S}_${FEAT}"
    if [ -f "$P/train.pt" ]; then
      N=$(python -c "import torch; d=torch.load('$P/train.pt', weights_only=False); print(len(d))")
      M=$(python -c "import torch; d=torch.load('$P/val.pt', weights_only=False); print(len(d))")
      echo "  s${S} $FEAT: train=$N val=$M"
    fi
  done
done
