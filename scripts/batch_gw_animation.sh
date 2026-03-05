#!/bin/bash
# batch_gw_animation.sh
# 平板ガイド波: 全欠陥ケースの波面比較 GIF を一括生成
#
# Prerequisite:
#   1. extract_gw_field.py で _coords.csv, _frames.csv を抽出済み
#      abaqus python scripts/extract_gw_field.py Job-GW-Healthy-v2.odb \
#        Job-GW-Debond-v2.odb Job-GW-D2-r15.odb Job-GW-D3-r40.odb \
#        Job-GW-D4-near.odb Job-GW-D5-edge.odb
#   2. abaqus_work/ に上記 CSV が存在
#
# Usage:
#   bash scripts/batch_gw_animation.sh
#   bash scripts/batch_gw_animation.sh --fps 25 --dpi 150

set -e
cd "$(dirname "$0")/.."
WORK_DIR="${WORK_DIR:-abaqus_work}"
OUT_DIR="${OUT_DIR:-wiki_repo/images/guided_wave}"
PREFIX_HEALTHY="$WORK_DIR/Job-GW-Healthy-v2"

if [[ ! -f "${PREFIX_HEALTHY}_coords.csv" ]]; then
    echo "ERROR: ${PREFIX_HEALTHY}_coords.csv not found."
    echo "Run first: abaqus python scripts/extract_gw_field.py Job-GW-Healthy-v2.odb Job-GW-Debond-v2.odb ..."
    exit 1
fi

echo "=== GW Wave Comparison Animation Batch ==="
echo "  Healthy: $PREFIX_HEALTHY"
echo "  Output:  $OUT_DIR"
echo ""

python scripts/plot_gw_animation.py "$PREFIX_HEALTHY" --batch --out_dir "$OUT_DIR" "$@"

echo ""
echo "Done. Check $OUT_DIR/gw_wave_comparison_*.gif"
