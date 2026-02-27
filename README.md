# Payload Fairing Defect Localization using GNN

[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()
[![Python](https://img.shields.io/badge/Python-3.x-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch_Geometric-2.4+-orange)]()
[![FEM](https://img.shields.io/badge/FEM-Abaqus%202024-green)]()

**JAXA H3ロケットのCFRPハニカムサンドイッチフェアリング**を対象に、GNN（Graph Neural Network）とFEM（Abaqus）を統合した**スキン-コア界面デボンディング位置特定システム**を開発するプロジェクトです。

## 背景: なぜこの研究が必要か

H3 は JAXA 初の **CFRP スキン / Al-Honeycomb コア** フェアリングを採用した基幹ロケットです。従来の全アルミ構造 (H-IIA/B, Epsilon) と異なり、CFRP-Al 間の **CTE 不整合** (CFRP: −0.3×10⁻⁶ vs Al: 23×10⁻⁶ /°C) がデボンディングの主要駆動力となります。

**2025年12月の H3 8号機事故**では、CFRP/Al-HC サンドイッチ構造の衛星搭載構造 (PSS) で製造時の接着不良 (デボンディング) が飛行中に進展し破壊に至ったことが有力要因とされており、本研究テーマの喫緊性が現実の事故によって裏付けられました。

| 特性 | H-IIA/B, Epsilon (従来) | H3 (本プロジェクト対象) |
|------|----------------------|----------------------|
| **スキン材** | Al 7075 | **CFRP** (T1000クラス, AFP) |
| **コア材** | Al Honeycomb | Al Honeycomb |
| **直径** | 4.0 m / 2.6 m | **5.2 m** |
| **CTE不整合** | ≈0 | **巨大** → 熱応力デボンディング |
| **SHMニーズ** | 低 (成熟40年) | **高** (新材料系, F8事故で顕在化) |

## パイプライン

```
Abaqus FEM (H3フェアリング: φ5.2m, CFRP [45/0/-45/90]s / Al-HC)
    ↓  CSV (節点座標 + 応力テンソル + 温度 + CTE熱応力)
PyTorch Geometric Graph 変換
    ↓  18次元ノード特徴量 (座標 + 法線 + 曲率 + 応力 + 温度 + タイプ)
    ↓  エッジ: k-NN / Delaunay (測地線距離ベース)
GNN 学習 (GCN / GAT / GIN / SAGE)
    ↓  Focal Loss + CosineAnnealing + EarlyStopping + 5-fold CV
    ↓  ノードレベル欠陥確率マップ
推論 API (FastAPI)
    ↓  REST endpoint → 3Dヒートマップ可視化
```

## 主な機能

### 1. Shell-Solid-Shell サンドイッチ FEM

- H3仕様: φ5.2m, CFRP [45/0/-45/90]s + Al-Honeycomb (~40mm パネル)
- 熱解析統合: JAXA文献値ベースの温度場 → CTE不整合熱応力
- 荷重: 軸圧縮 + 外圧 30kPa (Max Q) + 音響 147–148 dB
- 欠陥モデリング: スキン-コア界面のデボンディング領域

### 2. 曲率対応グラフ構築

- 非ユークリッド多様体上で直接動作 (2D投影による歪みなし)
- 表面法線・主曲率・測地線距離をエッジ/ノード特徴量に反映
- ノード特徴量 18次元 (座標 + 法線 + 曲率 + 応力 + 温度 + ノードタイプ)

### 3. 4種GNNアーキテクチャ比較

- **GCN**: スペクトルフィルタ — 大域パターン
- **GAT**: アテンション — 欠陥近傍への適応的重み付け
- **GIN**: 最大表現力 — 微細構造差の弁別
- **GraphSAGE**: サンプリング+集約 — スケーラビリティ
- Focal Loss (クラス不均衡対策), CosineAnnealing, Early Stopping, 5-fold CV

### 4. 推論API

- FastAPI REST エンドポイント
- チェックポイントからのモデル復元
- JSON 入力 → 欠陥確率マップ出力

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
├── abaqus_work/
│   └── extract_odb_mesh.py          # Abaqus ODB → メッシュ抽出
├── JAXA_LIBRARY/                    # JAXA技術文書 (Epsilon Manual, JMR-002等)
│   ├── H3_ROCKET_SPECS.md           # H3仕様・打ち上げ履歴
│   └── SHM_RESEARCH_SUMMARY.md      # SHM技術サーベイ
├── scripts/
│   └── run_pipeline.sh              # パイプライン一括実行
├── WIKI.md                          # 技術Wiki (H3詳細・SHM・GNN根拠)
├── LITERATURE_REVIEW.md             # 文献レビュー (OGW, GNN-SHM)
├── RESEARCH_REPORT.md               # リサーチレポート
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

## H3 ロケット打ち上げ履歴

| # | 日付 | 構成 | ペイロード | 結果 |
|:---:|:---|:---|:---|:---:|
| TF1 | 2023/03 | H3-22S | ALOS-3 | **失敗** |
| TF2 | 2024/02 | H3-22S | VEP-4 | 成功 |
| F3 | 2024/07 | H3-22S | ALOS-4 | 成功 |
| F4 | 2024/11 | H3-22S | DSN-3 | 成功 |
| F5 | 2025/02 | H3-22S | QZS-6 | 成功 |
| F6 | 2025 | **H3-30S** | 技術実証 | 成功 |
| F7 | 2025/10 | **H3-24W** | HTV-X1 | 成功 |
| F8 | 2025/12 | H3-22S | QZS-5 | **失敗** |

8号機の PSS (CFRP/Al-HC サンドイッチ) 破壊が本プロジェクトの研究対象と直接関連 → [詳細はWiki参照](WIKI.md#4-f8-事故と本研究の関連性)

## 開発環境

- **FEM**: Abaqus/Standard 2024
- **GNN**: PyTorch Geometric 2.4+
- **Language**: Python 3.x
- **API**: FastAPI + Uvicorn

## 参考文献

- Epsilon Users Manual (JAXA) — 環境荷重データ ([JAXA_LIBRARY/](JAXA_LIBRARY/))
- JAXA-JMR-002E — ペイロード安全基準, No-Growth 要件
- KHI ANSWERS — [H3フェアリング製造技術 (AFP/OoA)](https://answers.khi.co.jp/ja/mobility/20210806j-01/)
- Beyond Gravity — [Type-W フェアリング](https://www.beyondgravity.com/en/news/beyond-gravity-rocket-nose-cone-celebrates-premiere-japans-h3-rocket)
- [Open Guided Waves (Zenodo)](https://zenodo.org/records/5105861) — 実験データベンチマーク
- JAXA F8 調査 — [公式対応状況](https://www.jaxa.jp/hq-disclosure/h3f8/index_j.html)

## Wiki

詳細な技術ドキュメント (H3仕様・打ち上げ全履歴・SHMコンテキスト・GNN設計根拠・F8事故分析) は **[WIKI.md](WIKI.md)** を参照してください。
