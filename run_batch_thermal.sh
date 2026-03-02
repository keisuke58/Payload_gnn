#!/bin/bash
# 熱ステップ付き S12 バッチ実行
# 既存 batch_s12_100 (機械のみ) と並行して、OuterSkin z依存温度あり版を生成
#
# Usage:
#   bash run_batch_thermal.sh            # フル100サンプル (抽出込み)
#   bash run_batch_thermal.sh --test     # テスト: 1サンプルのみ
#   bash run_batch_thermal.sh --range 5-10  # 範囲指定

export PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:$PATH
cd /home/nishioka/Payload2026

TEMPLATE=abaqus_work/Job-CZM-S12-Thermal.inp
DOE=doe_sector12_100.json
WORKDIR=abaqus_work/batch_s12_100_thermal

if [ ! -f "$TEMPLATE" ]; then
  echo "ERROR: Thermal template not found: $TEMPLATE"
  echo "Run: cp abaqus_work/Job-CZM-S12-Test.inp $TEMPLATE && python3 scripts/patch_inp_thermal.py $TEMPLATE"
  exit 1
fi

EXTRA_ARGS=""
if [ "$1" = "--test" ]; then
  echo "=== TEST MODE: 1 sample only ==="
  EXTRA_ARGS="--range 1-1"
  shift
elif [ "$1" = "--range" ]; then
  EXTRA_ARGS="--range $2"
  shift 2
fi

echo "Starting thermal batch at $(date)"
echo "Template: $TEMPLATE"
echo "DOE: $DOE"
echo "Workdir: $WORKDIR"

python3 src/run_sector12_batch.py \
  --doe "$DOE" \
  --template "$TEMPLATE" \
  --workdir "$WORKDIR" \
  --cpus 4 --memory "16 gb" \
  --parallel 4 \
  $EXTRA_ARGS

echo "Thermal batch finished at $(date)"
