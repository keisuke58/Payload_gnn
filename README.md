# Payload Fairing Defect Localization using GNN

[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()

**JAXA H3ロケットのCFRPハニカムサンドイッチフェアリング**を対象に、GNN（Graph Neural Network）とFEM（Abaqus）を統合した欠陥位置特定システムを開発するプロジェクトです。

## なぜH3フェアリングか

H3はJAXAの次世代基幹ロケットであり、フェアリングに**CFRP AFP（自動繊維配置）スキン + Al-Honeycomb コア**のサンドイッチ構造を採用した**JAXA初の機体**です。

| 特性 | H3 (本プロジェクト対象) | Epsilon (環境データ参考) |
|------|----------------------|----------------------|
| **スキン材** | **CFRP** (T1000クラス, AFP) | Al 7075 |
| **コア材** | Al Honeycomb | Al Honeycomb |
| **直径** | 5.2 m | 2.6 m |
| **SHMニーズ** | 高 (新材料系, CTE不整合) | 低 (Al/Al, 成熟40年) |

CFRPスキンの異方性ガイド波伝播とCTE不整合による熱応力が、デボンディングの主要駆動力です。

## パイプライン

```
Abaqus FEM (H3フェアリングモデル)
    ↓  CSV (節点応力 + 温度 + 熱応力)
PyTorch Geometric Graph 変換
    ↓  曲率・法線・熱特徴量
GNN 学習 (GCN / GAT / GIN / SAGE)
    ↓  ノードレベル欠陥確率
推論 API (FastAPI)
```

## 主な機能

1. **Shell-Solid-Shell サンドイッチ FEM**
   - H3仕様: φ5.2m, CFRP [45/0/-45/90]s + Al-Honeycomb
   - 熱解析統合: JAXA文献値ベースの温度場 → CTE不整合熱応力
   - 荷重: 軸圧縮 + 外圧 30kPa (Max Q) + 音響 147dB

2. **曲率対応グラフ構築**
   - 表面法線・主曲率・測地線距離
   - ノード特徴量 18次元 (座標 + 法線 + 曲率 + 応力 + 温度 + ノードタイプ)

3. **4種GNNアーキテクチャ比較**
   - Focal Loss (クラス不均衡対策), CosineAnnealing, Early Stopping, 5-fold CV

4. **推論API** — FastAPI REST エンドポイント

## ディレクトリ構成

```
Payload2026/
├── src/
│   ├── generate_fairing_dataset.py  # Abaqus: H3 FEM + 熱解析 + CSV出力
│   ├── build_graph.py               # 曲率対応グラフ構築 (18次元特徴量)
│   ├── preprocess_fairing_data.py   # FEM CSV → PyG Data (12次元特徴量)
│   ├── models.py                    # GNN (GCN/GAT/GIN/SAGE)
│   ├── train.py                     # 学習 (Focal Loss, CV, Early Stopping)
│   ├── evaluate.py                  # 評価・ヒートマップ・比較
│   └── predict_api.py               # 推論 API (FastAPI)
├── JAXA_LIBRARY/                    # JAXA技術文書 (Epsilon Manual, JMR-002等)
├── scripts/
│   └── run_pipeline.sh              # パイプライン一括実行
├── WIKI.md                          # H3ターゲット選定根拠
├── ROADMAP.md                       # 開発ロードマップ
└── requirements.txt
```

## クイックスタート

```bash
pip install -r requirements.txt

# フルパイプライン
bash scripts/run_pipeline.sh

# Abaqusなしで実行
bash scripts/run_pipeline.sh --skip-abaqus

# 個別実行
python src/train.py --arch gat --data_dir dataset/processed --epochs 200
python src/evaluate.py --checkpoint runs/<run>/best_model.pt --eval_ood

# 推論API
MODEL_CHECKPOINT=runs/<run>/best_model.pt uvicorn src.predict_api:app --port 8000
```

## 開発環境

- **FEM**: Abaqus/Standard 2024
- **GNN**: PyTorch Geometric 2.4+
- **Language**: Python 3.x

## 参考文献

- Epsilon Users Manual (JAXA) — 環境荷重データ
- JAXA-JMR-002E — ペイロード安全基準
- KHI ANSWERS — H3フェアリング製造技術
- [Open Guided Waves (Zenodo)](https://zenodo.org/records/5105861) — 実験データベンチマーク

## Wiki

詳細な技術ドキュメントは [Wiki](https://github.com/keisuke58/Payload_gnn/wiki) を参照してください。
