[← Home](Home)

# 開発ロードマップ — Development Roadmap

> 最終更新: 2026-02-28  
> **研究期間 2 年** を前提とした高目標版 → [2-Year-Goals](2-Year-Goals)

JAXA H3 ロケット CFRP/Al-Honeycomb フェアリングの GNN ベース構造ヘルスモニタリング (SHM) システム開発ロードマップ。

---

## 日本語概要

Phase 1 (データ生成) 完了。Phase 2 (ベンチマーク) が現在地。Phase 3 で 1,000→5,000 サンプル、Phase 7 でマルチクラス 4 クラス、Phase 8 で Sim-to-Real (DANN)、Phase 9 で FNO・PINN。用語は [用語集](Vocabulary) を参照。

---

## 全体概要

```
Phase 1  データ生成         ✅ 完了 (100 サンプル)
Phase 2  ベンチマーク学習   ← 現在地
Phase 3  データ拡張・大規模学習 (目標: 1,000 → 5,000 サンプル)
Phase 4  図表作成
Phase 5  アブレーション
Phase 6  ロバスト性評価
Phase 7  マルチクラス (debond / delam / impact / healthy)
Phase 8  Sim-to-Real (OGW + DANN)
Phase 9  高度手法 (FNO サロゲート, PINN 逆問題)
```

## 計算環境

| 環境 | スペック | 用途 |
|------|---------|------|
| **CPU** | pyenv miniconda3 / PyTorch 2.10 + PyG 2.7 | 前処理、プロトタイプ、100サンプル訓練 |
| **GPU** | **24 GB VRAM × 4枚** | 大規模訓練、HP探索、メッシュ細分化実験 |

### GPU スケーリング指針

| データ規模 | CPU 訓練時間 | GPU 1枚 | GPU 4枚 |
|-----------|-------------|---------|---------|
| 100 graphs (現状) | ~1 時間 | ~5 分 | — |
| 1,000 graphs | ~12 時間 | ~50 分 | ~15 分 |
| 5,000 graphs | ~3 日 | ~4 時間 | ~1 時間 |

※ GAT (625K params), 200 epochs, batch_size=4–32 での概算

---

## Phase 1: データ生成 ✅ 完了

**目的**: Abaqus FEM シミュレーションによる高忠実度データセット構築

| タスク | 状態 |
|--------|------|
| DOE 100 サンプル設計 | ✅ |
| Abaqus バッチ FEM 生成 | ✅ 100/100 |
| ODB → CSV 抽出 (変位・温度・応力) | ✅ |
| PyG グラフ変換 (train.pt / val.pt) | ✅ |
| データ品質検証 | ✅ |

### データ仕様

| 項目 | 値 |
|------|-----|
| グラフ数 | 101 (train 81 + val 20) |
| ノード/グラフ | 10,897 |
| エッジ/グラフ | 85,514 |
| ノード特徴量 | 16 次元 (法線, 曲率, 変位, 温度, 応力, ノード種別) |
| エッジ特徴量 | 5 次元 (相対位置, 距離, 法線角度) |
| クラス比 | defect 0.06% / healthy 99.94% |
| 欠陥サイズ | Small 30%, Medium 40%, Large 25%, Critical 5% |

---

## Phase 2: ベンチマーク学習 ← 現在地

**目的**: GNN 4 種 + 代替モデルを同一データで公平に比較

### 2a. GNN 4 種比較 (CPU で実施可能)

| モデル | パラメータ数 | 特徴 |
|--------|-------------|------|
| GCN | 61K | 軽量ベースライン |
| **GAT** | **625K** | エッジ特徴量活用、Attention による解釈性 |
| GIN | 127K | WL-test 等価の最大表現力 |
| SAGE | 112K | サンプリング集約、スケーラビリティ |

```bash
# 5-Fold CV で全モデル評価
for arch in gcn gat gin sage; do
  python src/train.py --arch $arch --cross_val 5 --epochs 200
done
```

### 2b. 代替モデル比較 (GPU 推奨)

| モデル | タイプ | 状態 |
|--------|-------|------|
| UV-Net (U-Net) | 2D CNN (円筒展開) | 実装済 |
| Point Transformer | 3D 点群 | 要実装 |
| **Graph Mamba** | State Space Model | 要実装 → [Cutting-Edge-ML](Cutting-Edge-ML) |

### 2c. ハイパーパラメータ探索 (GPU 4枚並列)

Optuna で 4 GPU に 1 trial ずつ割り当て、100 trials 並列探索。

| パラメータ | 探索範囲 |
|-----------|---------|
| hidden | 64, 128, 256 |
| layers | 3, 4, 6 |
| lr | 1e-4 – 1e-2 |
| dropout | 0.0 – 0.3 |
| focal_alpha | auto, 0.5, 0.75, 0.9 |
| batch_size | 4, 8, 16, 32 |

---

## Phase 3: データ拡張 → 大規模学習 (GPU)

**目的**: データ量・メッシュ解像度のスケーリングによる性能限界の把握

### 3a. データスケールアップ

| 規模 | サンプル数 | GPU 1枚 訓練時間 | 目的 | 2年目標 |
|------|-----------|-----------------|------|----------|
| 現状 | 100 | ~5 分 | パイプライン検証 | — |
| 中規模 | 500–1,000 | ~30 分 | 論文用ベンチマーク | **Year 1 末: 1,000** |
| 大規模 | 5,000+ | ~4 時間 | 汎化性能の限界評価 | **Year 2 末: 5,000** |

> **ボトルネック**: Abaqus FEM 生成が ~5 分/サンプル → FNO サロゲートで 100x 加速を目標

### 3b. メッシュ細分化実験

| メッシュサイズ | ノード数/グラフ | VRAM (batch=4) | 24GB GPU で |
|--------------|---------------|---------------|-----------|
| 50 mm (現状) | ~11K | ~200 MB | 余裕 |
| 25 mm | ~40K | ~800 MB | 余裕 |
| 10 mm | ~250K | ~5 GB | 対応可能 |

### 3c. マルチ GPU 訓練

PyG の `DataParallel` で 4 GPU 並列化。大規模データ + 細分化メッシュの組み合わせが現実的に。

---

## Phase 4: 図表作成

| 出力 | 用途 |
|------|------|
| **Table 1** | モデル比較 (F1, AUC, Precision, Recall, Params, Time) |
| **Table 2** | HP 探索結果 (Optuna best params) |
| **Fig 1** | Defect Probability Map (3D ヒートマップ) |
| **Fig 2** | Training curves (Loss, F1 vs epoch) |
| **Fig 3** | Confusion Matrix |
| **Fig 4** | アブレーション (特徴量除去の影響) |
| **Fig 5** | データ量 vs 精度 (スケーリングカーブ) |

---

## Phase 5: アブレーションスタディ

### 特徴量の寄与定量化

| 実験 | 除去する特徴量 | 残る次元 |
|------|--------------|---------|
| Full | なし | 16 |
| w/o Normal | nx, ny, nz | 13 |
| w/o Curvature | k1, k2, H, K | 12 |
| w/o Geometry | Normal + Curvature | 9 |
| w/o Stress | s11, s22, s12, smises | 12 |
| Displacement only | ux, uy, uz のみ | 3 |

### メッシュ解像度の影響

50 mm / 25 mm / 10 mm で同一モデルを訓練 → F1 比較

---

## Phase 6: ロバスト性評価 (Sim-to-Real Gap)

| 実験 | 方法 |
|------|------|
| **ノイズ耐性** | ガウシアンノイズ (SNR 40, 20, 10 dB) 付加 → F1 劣化曲線 |
| **センサ欠落** | ランダムにノード特徴量を 0 化 (5%, 10%, 20%) |
| **温度変動** | 温度フィールドに ±10°C ランダム摂動 |
| **Data Augmentation** | 上記を訓練時にも適用 → ロバスト性向上効果の検証 |

---

## Phase 7: マルチクラス分類 (目標)

**目的**: debonding のみから **debond / delam / impact / healthy** の 4 クラス分類へ拡張

| タスク | 状態 |
|--------|------|
| Phase A: healthy サンプル追加 | 未着手 |
| Phase B: デラミネーション FEM + ラベル | 未着手 |
| Phase C: 衝撃損傷 FEM + ラベル | 未着手 |
| Phase D: 4 クラス GNN 学習 | 未着手 |

詳細: [Multi-Class-Roadmap](Multi-Class-Roadmap)

---

## Phase 8: Sim-to-Real (OGW + DANN)

**目的**: FEM 学習モデルを実実験データ (Open Guided Waves) に転移

| タスク | 目標 |
|--------|------|
| OGW データ取得・前処理 | CFRP 平板、温度変動、オメガストリンガー剥離 |
| グラフ変換 | OGW センサ配置をグラフにマッピング |
| DANN | ドメイン識別器 + 勾配反転で FEM→OGW 適応 |
| 評価 | 転移後の精度劣化 < 15% |

---

## Phase 9: 高度手法 (FNO, PINN)

**目的**: データ生成加速・スパースセンサ対応

| タスク | 目標 |
|--------|------|
| FNO サロゲート | 欠陥マスク → 波動場の写像学習、100x データ生成加速 |
| PINN 逆問題 | スパースセンサ (20 点) から欠陥位置を物理制約で推定 |

---

## 論文構成案

| Section | 内容 |
|---------|------|
| 1. Introduction | F8 事故 → CFRP/Al-HC SHM の必要性 |
| 2. Background | Lamb 波, GNN, サンドイッチ構造 |
| 3. Methodology | FEM モデル, グラフ構築, GNN アーキテクチャ |
| 4. Experiments | データセット, 4 GNN 比較, HP 探索, アブレーション |
| 5. Results | Table 1-2, Fig 1-5 |
| 6. Discussion | Sim-to-Real, 計算コスト, スケーラビリティ |
| 7. Conclusion | |
