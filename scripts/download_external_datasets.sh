#!/bin/bash
# 必須外部データセットのダウンロード
# 詳細: docs/DATASET_SURVEY.md, wiki_repo/Dataset-Survey.md

set -e
mkdir -p data/external
cd data/external

echo "=== NASA CFRP Composites ==="
if [ ! -f "NASA_Composites.zip" ]; then
  wget "https://phm-datasets.s3.amazonaws.com/NASA/2.+Composites.zip" -O NASA_Composites.zip
fi
if [ ! -d "NASA_Composites" ]; then
  unzip -o NASA_Composites.zip -d NASA_Composites
fi
echo "NASA CFRP: OK"

echo "=== OGW #4 (Zenodo 5105861) ==="
if command -v zenodo_get &>/dev/null; then
  zenodo_get 5105861
  echo "OGW #4: downloaded via zenodo_get"
else
  echo "OGW #4: pip install zenodo-get && zenodo_get 5105861"
  echo "  または https://zenodo.org/records/5105861 から手動取得"
fi
echo "ファイル: OGW_CFRP_Stringer_Wavefield_Intact.zip, *_FirstImpact.zip, *_SecondImpact.zip"
echo ""
echo "詳細: docs/PUBLIC_DATASETS_AND_INTEGRATION.md, wiki_repo/Dataset-Survey.md"
