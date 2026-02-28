#!/bin/bash
# =============================================================================
# 25mm データセットを完璧に再生成
#
# 1. 既存 INP に thermal パッチを適用
# 2. バッチ再実行（--force, --strict で物理量ゼロを検出）
# 3. 必要に応じて ODB から再抽出
#
# Usage:
#   bash scripts/regenerate_25mm.sh                    # 全400サンプル再生成
#   bash scripts/regenerate_25mm.sh --start 0 --end 5 # 最初の5サンプルのみ
#   bash scripts/regenerate_25mm.sh --extract-only    # ODB が既にある場合の再抽出のみ
# =============================================================================

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DOE="doe_25mm_400.json"
OUTPUT="dataset_output_25mm_400"
START=0
END=400
EXTRACT_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --start=*) START="${arg#*=}" ;;
        --end=*)   END="${arg#*=}" ;;
        --extract-only) EXTRACT_ONLY=true ;;
    esac
done

WORK_DIR="$PROJECT_ROOT/abaqus_work"
PATCH_SCRIPT="$PROJECT_ROOT/scripts/patch_inp_thermal.py"

echo "=========================================="
echo " 25mm データセット再生成"
echo "=========================================="
echo " DOE:    $DOE"
echo " Output: $OUTPUT"
echo " Range:  sample $START .. $END"
echo " Mode:   $([ "$EXTRACT_ONLY" = true ] && echo 'ODB再抽出のみ' || echo 'フル再生成')"
echo ""

if [ "$EXTRACT_ONLY" = true ]; then
    echo "[Phase 1] 既存 ODB から再抽出 (--strict で物理量検証)..."
    python scripts/extract_existing_odbs.py \
        --doe "$DOE" \
        --output_dir "$OUTPUT" \
        --start "$START" \
        --end "$END" \
        --strict
    echo ""
    echo "抽出完了。検証: head -3 $OUTPUT/sample_$(printf '%04d' $START)/nodes.csv"
    exit 0
fi

# 既存 INP にパッチ適用（abaqus_work 内の 25mm INP、再実行時の保険）
echo "[Phase 0] 既存 INP に thermal パッチ適用..."
for inp in "$WORK_DIR"/H3_Debond_01*.inp; do
    [ -f "$inp" ] || continue
    python "$PATCH_SCRIPT" "$inp" || true
done
echo ""

echo "[Phase 1] バッチ再生成 (--force --strict)..."
python src/run_batch.py \
    --doe "$DOE" \
    --output_dir "$OUTPUT" \
    --global_seed 25 \
    --defect_seed 8 \
    --force \
    --strict \
    --start "$START" \
    --end "$END"

echo ""
echo "=========================================="
echo " 25mm 再生成完了"
echo "=========================================="
echo " 検証: python scripts/verify_dataset_quality.py --data_dir $OUTPUT"
echo " ML:   python src/prepare_ml_data.py --input $OUTPUT --output data/processed_25mm_400"
echo ""
