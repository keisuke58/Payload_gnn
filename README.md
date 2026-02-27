# Payload Fairing Defect Localization using GNN

[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()

GNNとFEMを統合し、ペイロードフェアリング（CFRPハニカムサンドイッチ構造）の欠陥位置を特定する手法の開発プロジェクトです。
本リポジトリは [keisuke58/Payload_gnn](https://github.com/keisuke58/Payload_gnn) で管理されます。

## プロジェクト概要
再使用型ロケットの運用において重要となる、迅速かつ自動化された構造ヘルスモニタリング (SHM) の実現を目指します。
Frontiers in Materials に掲載された "Development of Defect Localization Method for Perforated Carbon-fiber-reinforced Plastic Specimens Using Finite Element Method and Graph Neural Network" の手法を、より複雑なフェアリング構造へ拡張します。

## 利用可能なデータセット & リソース
本プロジェクトでは、独自のデータセット生成に加え、以下の公開データセットもベンチマークや検証に利用します。

### 1. 公開データセット (Open Source Datasets)
*   **[Open Guided Waves (Zenodo)](https://zenodo.org/records/5105861)**: 
    *   CFRPプレートの超音波ガイド波計測データセット。
    *   ストリンガーの剥離（Debonding）を含むため、フェアリングの補強材剥離の模擬に最適。
*   **[Compression After Impact (CAI) Dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC11999467/)**:
    *   CFRP積層板への衝撃試験データ。衝撃損傷（BVID）のモデル化の参考に利用。

### 2. 独自データセット生成 (Synthetic Dataset)
Abaqus/Standard を用いて、実機に近い積層構成を持つフェアリングモデルを自動生成します。

*   **スクリプト**: `src/generate_fairing_dataset.py`
*   **特徴**:
    *   **形状**: 1/6 円筒バレルセクション（半径 2m, 高さ 5m）
    *   **材料**: CFRP (T300/914相当) + アルミハニカムコア
    *   **積層構成**: 準等方性積層 `[45/0/-45/90]s` を定義
    *   **要素**: Shell Elements (S4R) + Composite Shell Section

## 主な機能
1.  **Abaqusによる自動データセット生成**: 
    - 1/6円筒バレルセクションのパラメトリックモデリング
    - 剥離 (Disbond) および衝撃損傷 (Impact Damage) のシミュレーション
    - 座屈および音響荷重下の応答解析
2.  **GNNによる欠陥位置特定**:
    - FEMメッシュ情報をグラフ構造へ変換
    - 表面主応力和 (DSPSS) 分布からの欠陥推定

## ロードマップ
詳細な計画は [ROADMAP.md](ROADMAP.md) を参照してください。

## ディレクトリ構成
```
Payload2026/
├── src/
│   ├── generate_fairing_dataset.py  # Abaqus: FEM model generation + defect injection
│   ├── preprocess_fairing_data.py   # FEM CSV → PyG graph conversion pipeline
│   ├── models.py                    # GNN architectures (GCN/GAT/GIN/SAGE)
│   ├── train.py                     # Training with Focal Loss, CV, early stopping
│   ├── evaluate.py                  # Evaluation metrics, heatmaps, arch comparison
│   └── predict_api.py               # Inference API (Python + FastAPI REST)
├── scripts/
│   └── run_pipeline.sh              # End-to-end pipeline execution
├── references/                      # External repos (MeshPhysicsGNN-NeuralOps, etc.)
├── requirements.txt                 # Python dependencies
├── ROADMAP.md                       # Project Roadmap
└── README.md                        # This file
```

## クイックスタート
```bash
# 依存関係のインストール
pip install -r requirements.txt

# フルパイプライン実行 (Abaqus → 前処理 → 学習 → 評価)
bash scripts/run_pipeline.sh

# Abaqusなしで実行 (既存CSVデータ使用)
bash scripts/run_pipeline.sh --skip-abaqus

# 個別実行
python src/preprocess_fairing_data.py --raw_dir dataset_output --output_dir dataset/processed
python src/train.py --arch gat --data_dir dataset/processed --epochs 200
python src/evaluate.py --checkpoint runs/<run_name>/best_model.pt --eval_ood

# 推論API起動
MODEL_CHECKPOINT=runs/<run_name>/best_model.pt uvicorn src.predict_api:app --port 8000
```

## 開発環境
- **FEM**: Abaqus/Standard
- **GNN Framework**: PyTorch Geometric
- **Language**: Python 3.x

## 参考文献
- [Buckling Design and Analysis of a Payload Fairing 1/6th Cylindrical Barrel Section (NASA)](https://ntrs.nasa.gov/)
- [Machine Learning for Structural Health Monitoring of Aerospace Composites](https://doi.org/)
