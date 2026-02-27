# H3 Fairing FEM Dataset

Abaqus FEM から抽出した H3 フェアリング（CFRP/Al-Honeycomb サンドイッチ）の節点・要素データです。GNN 学習用の前処理パイプライン（`preprocess_fairing_data.py`）の入力形式です。

## ディレクトリ構成

```
dataset_output/
├── healthy_baseline/    # 健全ベースライン（デボンディングなし）
├── sample_0001/         # デボンディングサンプル例
└── README.md
```

## サンプル形式

各サンプルディレクトリには以下が含まれます：

| ファイル | 説明 |
|----------|------|
| `nodes.csv` | 節点データ（座標、応力、変位、欠陥ラベル） |
| `elements.csv` | 要素接続情報 |
| `metadata.csv` | サンプルメタデータ（欠陥パラメータ等） |

## nodes.csv カラム

| カラム | 説明 |
|--------|------|
| node_id | 節点ID |
| x, y, z | 座標 (mm) |
| s11, s22, s12 | 応力テンソル成分 (MPa) |
| dspss | 等価変位 |
| defect_label | 欠陥ラベル (0: 健全, 1: デボンディング) |

## サンプル一覧

| サンプル | defect_type | 備考 |
|----------|-------------|------|
| healthy_baseline | healthy | 22,220 ノード, 16,200 要素 |
| sample_0001 | debonding | θ=30°, z=2500mm, r=150mm, 77 欠陥ノード |

## 健全ベースライン検証 (欠陥挿入前必須)

```bash
python scripts/validate_healthy_baseline.py
# → 20/20 passed を確認してからデボンディングサンプル生成
```

## データ生成

```bash
# DOE パラメータ生成
python src/generate_doe.py --n_samples 500 --output doe_params.json

# バッチ実行（Abaqus 必要）
python src/run_batch.py --doe doe_params.json --output_dir dataset_output
```

個別サンプルの抽出は `extract_odb_results.py` を参照してください。
