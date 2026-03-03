[← Home](Home) | [Architecture](Architecture) | [Two-Stage-Screening](Two-Stage-Screening)

# Physics-Residual Anomaly Detection (PRAD)

> **Status**: Stage 1+2 完了 — **ROC-AUC 0.992 / F1 0.775** (Stage 1)、**蒸留 R²=0.998** (Stage 2)
> **Date**: 2026-03-04
> **前提**: バイナリ GNN F1 > 0.8 達成済み
> **方針**: 既存コード (`models.py`, `train.py` 等) に一切干渉しない。全て新規ファイルで実装。

---

## 1. コンセプト — なぜ斬新か

従来の GNN-SHM は **「欠陥 vs 健全」の分類問題** として定式化している。
本手法は発想を根本から変え、**「健全な物理からのズレ」で欠陥を検出** する。

```
従来 (Classification):
  FEM特徴量 → GNN → softmax → {defect, healthy}
  問題: ラベル必須、クラス不均衡 (0.06%) に弱い

提案 (Physics-Residual):
  GNN が「この構造なら応力はこうなるはず」を学習 (self-supervised)
  → 実際の応力との残差 |σ_expected - σ_actual| で異常検出
  → ラベル不要、不均衡が無関係
```

### 1.1 文献上の空白地帯（2026年3月時点）

| 手法の組合せ | 既存論文 | 本研究の新規性 |
|---|---|---|
| Physics-Informed GraphMAE | **ゼロ** | 物理制約付き masked graph autoencoder |
| FNO → GNN Knowledge Distillation | **ゼロ** | FNO teacher が GNN に物理を蒸留 |
| Self-Supervised GNN for Composite SHM | **ゼロ** | FEM メッシュグラフでの self-supervised 学習 |
| Contrastive Learning on FEM Graphs | **ゼロ** | 健全/欠陥のグラフ対比学習 |

**関連する最近の重要論文**:
- Mechanics-Informed Autoencoder (Nature Comms, 2024) — "deploy-and-forget" だがグラフではない
- TPF-GNet (Scientific Reports, 2026) — power-flow 物理を GNN に埋込むが supervised
- GINO (NeurIPS 2023) — FNO+GNO の直列結合だが蒸留ではない
- GraphMAE (KDD 2022) — masked graph autoencoder だが物理制約なし

---

## 2. 3段階アーキテクチャ

```
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: PI-GraphMAE  (Self-Supervised Pre-training)        │
│                                                              │
│  FEM mesh graph のノード特徴量 (34次元) をランダムにマスク     │
│  → GNN encoder-decoder で復元                                │
│  → Loss = 復元誤差 + 物理制約 (応力平衡 ∇·σ ≈ 0)            │
│  → ラベル不要で「健全な物理法則」を学習                       │
│  ────────────────────────────────────────────────────────     │
│  入力: data.x (34d), マスク率 50%                            │
│  出力: 事前学習済み encoder                                   │
│  新規ファイル: src/prad/graphmae.py, src/prad/train_mae.py   │
└────────────────────────┬─────────────────────────────────────┘
                         │ encoder 重み
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2: FNO → GNN Knowledge Distillation                   │
│                                                              │
│  FNO (既存 models_fno.py) が「健全応力場」を予測 (teacher)    │
│  GNN encoder が FNO の表現を graph 上で再現 (student)         │
│  → GNN が「この構造ならこの応力のはず」を学習                 │
│  ────────────────────────────────────────────────────────     │
│  teacher: models_fno.py (read-only、重み凍結)                │
│  student: Stage 1 の encoder                                 │
│  新規ファイル: src/prad/distill_fno2gnn.py                   │
└────────────────────────┬─────────────────────────────────────┘
                         │ 物理を学んだ encoder
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3: Residual-Based Anomaly Scoring                     │
│                                                              │
│  推論時: GNN predicted features vs FEM actual features        │
│  → 残差 |predicted - actual| = ノードごとの異常スコア         │
│  → 閾値なしで欠陥位置・サイズ・重篤度が連続値で出る           │
│  → 残差パターンのクラスタリングで欠陥種の分離も可能           │
│  ────────────────────────────────────────────────────────     │
│  新規ファイル: src/prad/anomaly_score.py                     │
│  出力: ノードごとの異常スコア (0.0 ~ ∞)                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 各 Stage の詳細

### Stage 1: PI-GraphMAE (Physics-Informed Graph Masked Autoencoder)

**目的**: ラベルなしで「健全な構造の物理的振る舞い」を学習

```python
# 概念コード
class PIGraphMAE(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder    # GAT/SAGE ベース (既存と同アーキテクチャ)
        self.decoder = decoder    # 軽量 MLP
        self.mask_ratio = 0.5

    def forward(self, data):
        # 1. ランダムマスク (50% のノード特徴量を隠す)
        masked_x, mask = random_mask(data.x, self.mask_ratio)

        # 2. Encode → Decode (マスクされたノードを復元)
        z = self.encoder(masked_x, data.edge_index)
        x_recon = self.decoder(z)

        # 3. 損失 = 復元誤差 + 物理制約
        L_recon = cosine_error(x_recon[mask], data.x[mask])
        L_physics = stress_equilibrium_residual(x_recon, data.edge_index)
        loss = L_recon + lambda_phys * L_physics
        return loss
```

**物理制約 L_physics**:
```
∇·σ ≈ 0 (静的平衡)

離散近似: 各ノード i について
  Σ_j∈N(i) (σ_j - σ_i) / |r_j - r_i| ≈ 0

→ 復元された応力場が力学的に整合しているかを制約
→ 健全領域では L_physics ≈ 0 に収束
→ 欠陥領域では物理が破れるため L_physics > 0 にならざるを得ない
```

**マスク戦略**:

| 戦略 | マスク対象 | 狙い |
|---|---|---|
| Random | 全特徴量の 50% | ベースライン |
| Physics-aware | 応力+ひずみのみマスク | 物理的復元能力の強化 |
| Spatial block | 連続領域のノードをまとめてマスク | 欠陥に似た「情報欠落」への耐性 |

### Stage 2: FNO → GNN Distillation

**目的**: FNO が学習した「PDE 解演算子の知識」を GNN に転写

```python
# 概念コード
def distill_step(fno_teacher, gnn_student, fno_grid, graph_data):
    # FNO: 64×64 grid で健全応力場を予測 (frozen)
    with torch.no_grad():
        stress_fno = fno_teacher(fno_grid)  # (1, 64, 64)

    # grid → graph: FNO 予測を各ノード座標に補間
    stress_target = interpolate_grid_to_nodes(stress_fno, graph_data.pos)

    # GNN: graph 上で応力を予測
    stress_gnn = gnn_student(graph_data)

    # 蒸留損失
    L_distill = F.mse_loss(stress_gnn, stress_target)
    return L_distill
```

**ポイント**:
- `models_fno.py` の学習済み重みを **読み込むだけ** (書き換えない)
- FNO は 64×64 regular grid、GNN は ~15K irregular mesh → 解像度が異なる
- `interpolate_grid_to_nodes()` で FNO 予測をグラフノード位置にマッピング

### Stage 3: Residual Anomaly Scoring

**目的**: 「健全ならこうなるはず」と「実際の FEM 結果」の差で欠陥検出

```python
def anomaly_score(encoder, data):
    """
    encoder: Stage 1+2 で学習済み
    data: FEM 結果の PyG グラフ (欠陥あり or なし)
    """
    # encoder が「健全ならこうなるはず」を予測
    with torch.no_grad():
        z = encoder(data.x, data.edge_index)
        x_expected = decoder(z)

    # 残差 = 実際 - 予測
    residual = (data.x - x_expected).abs()

    # 応力関連次元 (dim 15-20) の残差を重視
    stress_residual = residual[:, 15:21].mean(dim=1)

    return stress_residual  # shape: (N_nodes,)
```

**異常スコアの解釈**:

| スコア | 意味 |
|---|---|
| ≈ 0 | 健全（予測通り） |
| 中程度 | 軽度の異常（小さい欠陥 or 境界効果） |
| 大 | 明確な欠陥（物理的に大きなズレ） |

**マルチクラスへの自然な拡張**:
```
残差ベクトル (34次元) のパターンが欠陥種で異なる:
  debonding  → 応力急落 + 変位増大
  impact     → 応力集中 + ひずみ増大
  delamination → せん断応力異常

→ 残差ベクトルの k-means or t-SNE で欠陥種を分離
→ ラベルなしでマルチクラスが可能
```

---

## 4. 実装計画 — 既存コードとの分離

### 4.1 ファイル構成

```
src/
├── models.py              # 既存 GNN (触らない)
├── models_fno.py          # 既存 FNO (読み込みのみ)
├── train.py               # 既存学習 (触らない)
├── train_fno.py           # 既存 FNO 学習 (触らない)
│
└── prad/                  # ★ 新規ディレクトリ (Physics-Residual AD)
    ├── __init__.py
    ├── graphmae.py        # Stage 1: PI-GraphMAE モデル定義
    ├── train_mae.py       # Stage 1: 自己教師あり事前学習
    ├── distill_fno2gnn.py # Stage 2: FNO→GNN 知識蒸留
    ├── distill_multitask.py # Stage 2: MAE再構成+応力予測の同時学習
    ├── finetune_distilled.py # 蒸留エンコーダの fine-tune + 比較評価
    ├── anomaly_score.py   # Stage 3: 残差ベース異常スコア
    ├── physics_loss.py    # 物理制約損失 (∇·σ ≈ 0)
    ├── diagnose.py        # 残差診断 (次元別分析)
    └── eval_prad.py       # 評価 (ROC, PR, t-SNE 可視化)
```

### 4.2 既存コードへの依存（読み込みのみ）

| 既存ファイル | 使い方 | 変更 |
|---|---|---|
| `models.py` | GAT/SAGE の encoder アーキテクチャを import | **なし** |
| `models_fno.py` | FNO2d の学習済み重みを load | **なし** |
| `csv_to_fno_grid.py` | grid 変換ロジックを再利用 | **なし** |
| `data/processed_*/*.pt` | PyG データを読み込み | **なし** |
| `data/fno_grids_*/*.npy` | FNO grid データを読み込み | **なし** |

### 4.3 実装順序

```
Week 1: Stage 1 (PI-GraphMAE) ✅ 完了
  ├─ graphmae.py: masked autoencoder 実装
  ├─ physics_loss.py: 応力平衡制約
  └─ train_mae.py: 健全データで事前学習
      → 達成: cos_sim=0.9987, ROC-AUC=0.992, F1=0.775

Week 2: Stage 2 (FNO Distillation) ✅ 完了
  ├─ distill_fno2gnn.py: grid→graph 補間 + 蒸留
  ├─ FNO teacher (val_rel_l2=0.022) → GNN student
  └─ 達成: R²=0.9984 (目標 0.90 を大幅達成)
  ★ 蒸留エンコーダの異常検出統合は改善なし（§6.5 参照）

Week 3: マルチクラス拡張 + 論文用図表
  ├─ 残差パターンの t-SNE / クラスタリング
  ├─ 8 欠陥タイプでの検出精度評価
  └─ 論文 Figure 作成
```

---

## 5. 期待される成果

### 5.1 技術的優位性

| 項目 | 従来 GNN 分類 | PRAD (提案) |
|---|---|---|
| ラベル依存 | 必須 | **不要** (self-supervised) |
| クラス不均衡 | 致命的 (F1=0 の原因) | **無関係** (異常検出) |
| マルチクラス | num_classes 変更 + 再学習 | **残差パターンで自動分離** |
| 解釈性 | ブラックボックス | **残差 = 物理的ズレ** (直感的) |
| FNO 活用 | スクリーニングのみ | **物理知識の蒸留** |
| 新規欠陥種 | 再学習必須 | **未知の欠陥も検出可能** (OOD detection) |

### 5.2 達成状況

| 項目 | 目標 | 実績 | Status |
|------|------|------|--------|
| 復元精度 | cos_sim > 0.95 | **0.9987** | ✅ |
| ROC-AUC | > 0.95 | **0.992** | ✅ |
| F1 | — | **0.775** | ✅ |
| FNO 蒸留 R² | > 0.90 | **0.9984** | ✅ |

### 5.3 論文上のインパクト

1. **Physics-Informed Graph Masked Autoencoder** — 手法として完全に新規
2. **FNO→GNN Knowledge Distillation** — neural operator と GNN の初の蒸留
3. **Self-Supervised SHM for Composites** — 複合材 SHM で初の self-supervised
4. **Unsupervised Multi-Class Defect Typing** — ラベルなしで欠陥種分類

→ **1本の論文で 4つの新規性** を主張可能

### 5.3 既存研究との両立

```
論文構成案:
  Section 4.1: Supervised GNN Baseline (既存, F1>0.8) ← 今のモデル
  Section 4.2: PRAD (提案手法)                        ← 新規
  Section 4.3: 比較 — PRAD vs Supervised vs FNO-only
  → 「supervised で十分良い性能を出した上で、さらに self-supervised で凌駕」
```

---

## 6. 実験結果（2026-03-04）

### 6.1 Stage 1 学習

```
Dataset:  processed_s12_czm_thermal_200_binary (200 samples)
Nodes:    15,206/graph × 200 graphs = 3.04M total
Defects:  0.61% of nodes (3,693 / 608,240 val nodes)
Encoder:  SAGEConv 4-layer, hidden=128
Decoder:  MLP 2-layer (128→128→34)
Training: 200 epochs, lr=1e-3, mask_ratio=0.5, lambda_physics=0.1
         val_loss=0.0011, cos_sim=0.9987
```

### 6.2 精度推移

| 段階 | ROC-AUC | PR-AUC | F1 | Precision | Recall | 改善施策 |
|------|---------|--------|----|-----------|--------|---------|
| v1 初回 (L1 scoring) | 0.416 | 0.005 | 0.013 | 0.006 | 0.421 | — |
| v2 Cosine scoring | 0.974 | 0.381 | 0.475 | 0.446 | 0.509 | L1→Cosine distance |
| **v3 + Graph smoothing** | **0.992** | **0.769** | **0.775** | **0.830** | **0.727** | グラフ空間スムージング |

### 6.3 スコアリング改善の詳細

**診断結果** (`diagnose.py`):
- Cosine distance: ROC-AUC=0.974 → L1 全次元 (0.437) より圧倒的
- dim 17 (s12) / dim 23 (le12): defect/healthy 比率 2.55x — 最も感度の高い次元
- 多くの次元は defect の方が残差が低い（anti-informative）

**グリッドサーチ結果** (alpha × smooth 設定):
```
alpha=1.0 (cosine only) + smooth_rounds=1, smooth_alpha=0.5 が最良
→ グラフスムージングで孤立 false positive を除去 → PR-AUC 2倍
```

**モデルバリエーション比較** (best scoring config):
```
mlp/mr50:         ROC=0.992  PR=0.769  F1=0.775  ← best
bottleneck/mr50:  ROC=0.977  PR=0.595  F1=0.632
mlp/mr70:         ROC=0.962  PR=0.246  F1=0.332
bottleneck/mr70:  ROC=0.971  PR=0.731  F1=0.755
```

### 6.4 主要な技術的知見

1. **Re-masking はMLPデコーダと非互換**: 初回実装で cos_sim=0.07 だった原因。MLP は各ノードを独立処理するため、re-masking でマスクノードの情報が完全消失
2. **Cosine distance >> L1**: 学習目的関数とスコアリング関数を一致させることが重要
3. **グラフスムージングの効果が甚大**: PR-AUC が 0.38→0.77 に倍増。欠陥は空間的に集中するため、近傍スコア平均で信号増幅 + ノイズ抑制
4. **シンプルなデコーダが最良**: ボトルネックや高マスク率は逆効果

### 6.5 Stage 2: FNO → GNN 蒸留

**FNO Teacher**: 392 epochs 学習済み、val_rel_l2=0.022
- 入力: (4, 64, 64) — z_norm, theta_norm, defect_mask, temp_norm
- 出力: (1, 64, 64) — smises (正規化: mean=21.26, std=23.24)

**蒸留**:
```
入力: FNO teacher (frozen) + MAE encoder (初期重み)
grid→graph 補間: FNO 64×64 grid → 15,206 不規則メッシュノード
  - 円筒座標変換: (x,y,z) → (θ, y_axial) → bilinear interpolation
  - 座標範囲: θ=[0°, 30°], y=[0, 10449mm]

結果: 100 epochs, 103秒
  Epoch 1:   loss=432.6  R²=0.752
  Epoch 40:  loss=5.14   R²=0.995
  Epoch 100: loss=2.10   R²=0.998  ← best
```

**蒸留エンコーダの異常検出への統合実験**:

| アプローチ | ROC-AUC | PR-AUC | F1 | 備考 |
|-----------|---------|--------|------|------|
| **Stage 1 only (baseline)** | **0.992** | **0.769** | **0.775** | **最良** |
| 蒸留エンコーダ直接差替 | 0.573 | 0.007 | 0.017 | エンコーダ表現が崩壊 |
| 蒸留 + decoder fine-tune (30ep) | 0.722 | 0.011 | 0.026 | 回復不十分 |
| 蒸留 + full fine-tune (10ep) | 0.964 | 0.193 | 0.352 | 回復途中 |
| マルチタスク (λ=0.1) | 0.958 | 0.196 | 0.275 | λ が大きすぎ |
| マルチタスク (λ=0.001, 200ep) | 0.977 | 0.718 | 0.698 | ベースラインに近いが未超過 |

**結論**: 蒸留エンコーダは応力予測に特化するため、34次元全体の再構成能力が崩壊。
マルチタスク学習（MAE再構成 + 応力予測の同時最適化）でも Stage 1 を超えられなかった。

**科学的示唆**:
1. **自己教師あり再構成は異常検出に自己完結的** — 再構成損失自体が「健全物理の学習」に最適化されている
2. **応力予測は直交的な知識** — 応力場予測(R²=0.998) と異常検出(F1=0.775) は異なる能力
3. **FNO 蒸留の価値**: 応力場予測タスク、転移学習、別ドメインへの適用では有効（R²=0.998）
4. **論文記述**: Stage 2 は「物理知識転移の成功例」+ 「異常検出には Stage 1 の再構成目的関数が最適」として報告

---

## 7. 関連ページ

| ページ | 内容 |
|--------|------|
| [Architecture](Architecture) | 全体パイプライン（FNO/GNN の位置づけ） |
| [Two-Stage-Screening](Two-Stage-Screening) | FNO スクリーニング設計 |
| [Cutting-Edge-ML](Cutting-Edge-ML) | 先端 ML 手法 |
| [Multi-Class-Roadmap](Multi-Class-Roadmap) | マルチクラス分類計画 |
| [Benchmark-Targets](Benchmark-Targets) | 精度目標 |
| [QML-Defect-Detection](QML-Defect-Detection) | 量子 ML 実験 |
