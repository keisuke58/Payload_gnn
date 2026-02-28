# CLAUDE.md — Project Instructions for Claude Code

## Project
JAXA H3 ロケット CFRP/Al-Honeycomb フェアリングの GNN-SHM (構造ヘルスモニタリング)。
デボンディング欠陥検出を Graph Neural Network で実現する研究プロジェクト。

## Pipeline
```
generate_doe.py → generate_fairing_dataset.py (Abaqus CAE)
  → extract_odb_results.py (Abaqus Python) → build_graph.py → prepare_ml_data.py → train.py
```

## Environment
- Abaqus 2024 (Python 3.10.5 embedded)
- PyTorch 2.10.0, PyG 2.7.0 (pip, pyenv miniconda3)
- GPU: 24GB x 4
- Abaqus 実行は `abaqus_work/` から: `abaqus cae noGUI=...`
- ODB 抽出は: `abaqus python src/extract_odb_results.py`

## Key Directories
- `src/` — メインコード (generate, extract, build_graph, train)
- `scripts/` — 検証・可視化スクリプト
- `wiki_repo/` — GitHub Wiki ソース (images/ 含む)
- `dataset_realistic_25mm_100/` — 現行データセット (N=100)
- `data/processed_*/` — PyG 前処理済みデータ
- `runs/` — 学習ログ (TensorBoard)

## Communication
- 日本語で会話する
- まず1つ生成して確認してから丁寧に進める

## Git / PR Rules
- PR の body に "Generated with Claude Code" を書かない
- コミットメッセージは日本語OK、Co-Authored-By は付ける

## Data (34-dim Node Features)
位置・幾何(10) + 変位(4) + 温度(1) + 応力(5) + 熱応力(1) + ひずみ(3) + 繊維配向(3) + 積層構成(5) + 境界フラグ(2) = 34次元
