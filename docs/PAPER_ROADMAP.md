# 論文化ロードマップ — Paper Publication Checklist

> 最終更新: 2026-02-28

## 現状サマリ

| 項目 | 状態 | 備考 |
|------|------|------|
| **FEM モデル** | ✅ 完了 | H3 Type-S 整合 (Barrel + Ogive, 1/6 セクション) |
| **データセット** | ✅ 100 サンプル生成済 | train 81 + val 20, 10,897 nodes/graph |
| **GNN 4種** | ✅ 実装・初回学習済 | GCN / GAT / GIN / SAGE |
| **比較実験** | 🔲 未実施 | UV-Net, Point Transformer との比較 |
| **アブレーション** | 🔲 未実施 | 幾何特徴量の寄与評価 |
| **ロバスト性評価** | 🔲 未実施 | ノイズ耐性、Sim-to-Real Gap |
| **ドキュメント** | ✅ 充実 | Wiki, JAXA仕様, FEM可視化 |

## 計算環境

| 環境 | スペック | 用途 |
|------|---------|------|
| **CPU** | pyenv miniconda3 / PyTorch 2.10 | 前処理、プロトタイプ、100サンプル訓練 |
| **GPU** | **24 GB VRAM × 4枚** | 大規模訓練、HP探索、メッシュ細分化実験 |

---

## Phase 1: データ生成 ✅ 完了

| タスク | 状態 |
|--------|------|
| DOE 100 サンプル設計 | ✅ |
| バッチ FEM 生成 (Abaqus) | ✅ 100/100 |
| ODB → CSV 抽出 | ✅ 変位・温度・応力 |
| PyG グラフ変換 | ✅ train.pt / val.pt |

**データ仕様**:
- グラフ: 10,897 nodes × 16 features, 85,514 edges × 5 features
- クラス比: defect 0.06% / healthy 99.94%
- 欠陥: Small 30%, Medium 40%, Large 25%, Critical 5%
- 位置: θ 5–55°, z 800–4200 mm

---

## Phase 2: ベンチマーク学習 ← **現在地**

### 2a. GNN 4 種比較 (CPU で実施可能)

**環境**: CPU — 100サンプルなら ~1時間/モデル

| モデル | パラメータ数 | コマンド |
|--------|-------------|---------|
| GCN | 61K | `python src/train.py --arch gcn --cross_val 5` |
| **GAT** | **625K** | `python src/train.py --arch gat --cross_val 5` |
| GIN | 127K | `python src/train.py --arch gin --cross_val 5` |
| SAGE | 112K | `python src/train.py --arch sage --cross_val 5` |

```bash
# 一括実行
bash scripts/run_all_models.sh
```

### 2b. 追加モデル比較 (GPU 推奨)

| モデル | タイプ | スクリプト | 備考 |
|--------|-------|------------|------|
| UV-Net (U-Net) | 2D CNN | `src/run_uv_2d.py` | 円筒展開 → 2D |
| Point Transformer | 3D 点群 | 要実装 | 大域的 Attention |

### 2c. ハイパーパラメータ探索 (GPU 4枚並列)

```bash
# Optuna で 4 GPU に 1 trial ずつ割り当て
python src/hparam_search.py --n_trials 100 --n_gpus 4
```

| パラメータ | 探索範囲 |
|-----------|---------|
| hidden | 64, 128, 256 |
| layers | 3, 4, 6 |
| lr | 1e-4 – 1e-2 |
| dropout | 0.0 – 0.3 |
| focal_alpha | auto, 0.5, 0.75, 0.9 |
| batch_size | 4, 8, 16, 32 |

**指標**: 5-Fold CV Mean F1, AUC

---

## Phase 3: データ拡張 → 大規模学習 (GPU)

### 3a. データスケールアップ

| 規模 | サンプル数 | 訓練時間 (GPU 1枚) | 目的 |
|------|-----------|-------------------|------|
| **現状** | 100 | ~5 分 | パイプライン検証 |
| **中規模** | 500–1,000 | ~30 分 | 論文用ベンチマーク |
| **大規模** | 5,000+ | ~4 時間 | 汎化性能の限界評価 |

```bash
# DOE 拡張
python src/generate_doe.py --n_samples 1000 --output doe_1000.json
# Abaqus バッチ (ボトルネック: ~5分/サンプル → 1000サンプルで ~3日)
python src/run_batch.py --doe doe_1000.json --output_dir dataset_output_1000
```

### 3b. メッシュ細分化実験

| メッシュ | ノード数/グラフ | VRAM (batch=4) | 目的 |
|---------|---------------|---------------|------|
| 50 mm (現状) | ~11K | ~200 MB | ベースライン |
| 25 mm | ~40K | ~800 MB | 精度向上 |
| 10 mm | ~250K | ~5 GB | 微細欠陥検出 |

24GB VRAM なら 10mm メッシュでも batch_size=4 で余裕。

### 3c. マルチ GPU 訓練

```python
# PyG DataParallel
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader
```

4 GPU で linear speedup → 大規模データ + 細分化メッシュの組み合わせが現実的に。

---

## Phase 4: 図表作成

| 出力 | 用途 | スクリプト |
|------|------|-----------|
| **Table 1** | モデル比較 (Accuracy, F1, IoU, AUC, Params, Time) | `src/evaluate.py` |
| **Table 2** | HP 探索結果 (Optuna best params) | `src/hparam_search.py` |
| **Fig 1** | Defect Probability Map (3D ヒートマップ) | `scripts/visualize_defects.py` |
| **Fig 2** | Training curves (Loss, F1 vs epoch) | `src/plot_training.py` |
| **Fig 3** | Confusion Matrix | `src/evaluate.py` |
| **Fig 4** | アブレーション (特徴量除去の影響) | 要実装 |
| **Fig 5** | データ量 vs 精度 (スケーリングカーブ) | 要実装 |

---

## Phase 5: アブレーションスタディ

**幾何特徴量の寄与定量化**:

| 実験 | 除去する特徴量 | 残る次元 |
|------|--------------|---------|
| Full | なし | 16 |
| w/o Normal | nx, ny, nz | 13 |
| w/o Curvature | k1, k2, H, K | 12 |
| w/o Geometry | Normal + Curvature | 9 |
| w/o Stress | s11, s22, s12, smises | 12 |
| Displacement only | ux, uy, uz のみ | 3 |

**メッシュ解像度の影響**:
- 50 mm / 25 mm / 10 mm で同一モデルを訓練 → F1 比較

---

## Phase 6: ロバスト性評価 (Sim-to-Real Gap)

| 実験 | 方法 |
|------|------|
| **ノイズ耐性** | ガウシアンノイズ (SNR 40, 20, 10 dB) 付加 → F1 劣化曲線 |
| **センサ欠落** | ランダムにノード特徴量を 0 化 (5%, 10%, 20%) |
| **温度変動** | 温度フィールドに ±10°C ランダム摂動 |
| **Data Augmentation** | 上記を訓練時にも適用 → ロバスト性向上 |

---

## 論文構成案

| Section | 内容 |
|---------|------|
| 1. Introduction | F8 事故 → CFRP/Al-HC SHM の必要性 |
| 2. Background | Lamb 波, GNN, サンドイッチ構造 |
| 3. Methodology | FEM モデル, グラフ構築, GNN アーキテクチャ |
| 4. Experiments | データセット, 4 GNN 比較, HP 探索, アブレーション |
| 5. Results | Table 1-2, Fig 1-5 |
| 6. Discussion | Sim-to-Real, 計算コスト, 限界 |
| 7. Conclusion | |

### Data Generation セクション記述メモ
- 学習: 81 サンプル, 検証: 20 サンプル (→ 拡張後は train/val/test 分割)
- 欠陥バリエーション: Small 10–30 mm 30%, Medium 30–80 40%, Large 80–150 25%, Critical 150–250 5%
- メッシュ: GLOBAL_SEED=50 mm（欠陥分解能 h≤D/2, docs/MESH_DEFECT_ANALYSIS.md）
- θ: 5–55°, z: 800–4200 mm
- ノード特徴量: 16 次元 (normal, curvature, displacement, temperature, stress, node_type)
- エッジ特徴量: 5 次元 (relative position, distance, normal angle)

### 比較表 (Table 1) 項目
- Model | Params | F1 | AUC | Precision | Recall | Inference (ms/graph)
