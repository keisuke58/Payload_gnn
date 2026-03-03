[← Home](Home) | [Batch-INP-Status](Batch-INP-Status) | [FEM-Realism-Roadmap](FEM-Realism-Roadmap)

# 2段階スクリーニング — FNO サロゲート + 高精度 FEM/GNN

> **Status**: 設計完了、実装フェーズ
> **Date**: 2026-03-03
> **目的**: 全 250 ケースの FEM ソルバー実行コストを 50-60% 削減

---

## 1. コンセプト

```
                         250 INP (全ケース)
                               │
                    ┌──────────▼──────────┐
         Stage 1    │  FNO Surrogate      │  ~0.1秒/ケース
                    │  (MC-Dropout UQ)    │  → μ(σ_mises), σ_uncertainty
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
         Screening  │  閾値近傍抽出        │  |μ - threshold| < k·σ
                    │  ~30-50% を選択      │  + σ大 = 不確実 → FEM必要
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
         Stage 2    │  Full FEM + GNN     │  ~5-30分/ケース
                    │  高精度再評価        │  → P(defect) 確率推定
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
         Optional   │  Quantum AE         │  二次加速 O(1/ε) vs O(1/ε²)
                    │  (将来研究)          │  → P の精度 ε=0.02
                    └─────────────────────┘
```

**論文上のストーリー**: 「FNO サロゲートによる事前スクリーニングで、GNN-SHM パイプラインの実用的スケーラビリティを実証」

---

## 2. 計算コスト削減効果

| | 全件 FEM | 2段階 |  削減率 |
|---|---:|---:|---:|
| CZM S12 (177件 × 5分) | 14.8 h | ~7 h | **53%** |
| C3D10 (73件 × 20分) | 24.3 h | ~10 h | **59%** |
| **合計** | **39.1 h** | **~17 h** | **~56%** |

### 2.1 なぜ 50-60% 削減できるか

FNO が高信頼で判定できるケース ≈ 全体の 50-70%:
- **明確に healthy**: 欠陥なし or 応力変化が閾値以下 → FEM 不要
- **明確に defect**: 大欠陥で応力パターンが顕著 → FEM で確認するまでもない
- **閾値近傍**: 小〜中欠陥で FNO の不確実性が高い → **FEM 必要**

---

## 3. Stage 1: FNO サロゲート

### 3.1 アーキテクチャ

既存 `src/models_fno.py` の `FNO2d`:
- SpectralConv2d × 4 層
- modes: 12×12, width: 32
- 入力: 4ch × 64×64 grid
- 出力: 1ch × 64×64 grid (応力場)

UQ 版 (`experiments/uq/fno_mcdropout_infer.py`):
- MC-Dropout (p=0.2) で T=20 回推論
- μ = 予測平均, σ = 予測標準偏差 (不確実性)

### 3.2 入出力設計

**入力 (4ch, 64×64 grid)**:

| ch | 内容 | ソース |
|---|---|---|
| 0 | z座標 (軸方向, 正規化) | nodes.csv: z |
| 1 | θ座標 (周方向, 正規化) | nodes.csv: theta |
| 2 | defect_mask (0/1) | nodes.csv: defect_label |
| 3 | 荷重条件 (差圧+温度, 正規化) | 温度×差圧の複合スカラー |

**出力 (1ch, 64×64 grid)**:

| ch | 内容 |
|---|---|
| 0 | von Mises 応力場 (σ_mises) |

### 3.3 学習データ

S12 CZM 200 サンプル → FNO 用 grid 変換:

```
nodes.csv (15,206 nodes × 34 features)
  → UV grid (64×64) via θ-z binning
  → 入力: [z_grid, θ_grid, defect_grid, load_grid]
  → 出力: [smises_grid]
```

変換スクリプト: `src/csv_to_fno_grid.py`

### 3.4 学習設定

| 項目 | 値 |
|---|---|
| Train/Val | 160/40 (既存分割と同じ) |
| Epochs | 500 |
| LR | 1e-3 → CosineAnnealing |
| Loss | Relative L2 |
| GPU | vancouver02 |
| 推定学習時間 | ~30分 |

---

## 4. Screening 判定ロジック

### 4.1 判定基準

```python
# FNO 推論結果
mu = fno_predict_mean(x)      # 応力場平均
sigma = fno_predict_std(x)    # MC-Dropout 不確実性

# 欠陥スコア: defect mask 近傍の応力変化量
score = (mu[defect_region] - mu[healthy_baseline]).mean()

# 閾値近傍の判定
threshold = calibrated_threshold  # S12 200 から校正
needs_fem = abs(score - threshold) < k * sigma.mean()
needs_fem |= sigma.mean() > sigma_cutoff  # 不確実性が高い
```

### 4.2 パラメータ

| パラメータ | 値 | 決め方 |
|---|---|---|
| k (閾値マージン) | 1.5–2.0 | S12 200 で CV |
| sigma_cutoff | 上位 30% | MC-Dropout 分布から |
| threshold | — | GNN F1 最適閾値から逆算 |

---

## 5. Stage 2: 高精度 FEM + GNN

Screening で選ばれたサブセット (~75–125件) のみ:

1. **Abaqus ソルバー実行** (frontale サーバー)
2. **ODB → CSV 抽出** (`extract_odb_results.py`)
3. **PyG グラフ構築** (`prepare_ml_data.py`)
4. **GNN 推論** (学習済み GAT, F1=0.855)
5. **P(defect) 出力** — ノードごとの欠陥確率

---

## 6. Optional: Quantum Amplitude Estimation

### 6.1 動機

Stage 2 の P(defect) 推定精度 ε を上げるには:
- 古典: O(1/ε²) サンプル必要 → ε=0.02 で N=4,612
- **量子 AE**: O(1/ε) → ε=0.02 で N≈50

### 6.2 現状

`experiments/quantum/ae_poc.py` で PoC 実装済み:
- 振幅エンコーディング (Ry ゲート)
- qiskit 未インストール → 古典フォールバック
- 結果: p_classical=0.436, N_required=4,612

### 6.3 将来

- IBM Quantum / Amazon Braket で実行
- QAOA (Quantum Approximate Optimization) との統合
- 論文の Future Work セクションに記載

---

## 7. 実装ロードマップ

```
Week 1: FNO 学習データ準備
  ├─ csv_to_fno_grid.py (CSV → 64×64 grid 変換)
  └─ S12 200 サンプルで FNO 学習 (vancouver02)

Week 2: Screening パイプライン
  ├─ MC-Dropout UQ 推論
  ├─ 閾値校正 (S12 200 の GNN 結果から)
  └─ サブセット選択ロジック

Week 3: バッチ実行
  ├─ 選択された ~100 件の FEM ソルバー実行
  ├─ ODB 抽出 + PyG 変換
  └─ GNN 推論 + 結果比較

Week 4: 検証・論文化
  ├─ 全件 FEM vs 2段階の精度比較
  ├─ 計算コスト削減率の定量評価
  └─ Wiki / 論文に結果まとめ
```

---

## 8. 既存コード

| ファイル | 内容 | 状態 |
|---|---|---|
| `src/models_fno.py` | FNO2d (114行) | ✅ 実装済み |
| `experiments/uq/fno_mcdropout_infer.py` | MC-Dropout UQ 推論 | ✅ 実装済み |
| `experiments/uq/two_stage_is.py` | 2段階 IS ロジック | ✅ PoC 済み |
| `experiments/quantum/ae_poc.py` | 量子 AE PoC | ✅ 古典版動作 |
| `src/csv_to_fno_grid.py` | CSV → FNO grid 変換 | **未作成** |
| `src/train_fno.py` | FNO 学習スクリプト | **未作成** |
| `src/screen_cases.py` | スクリーニング判定 | **未作成** |

---

## 9. 関連ページ

| ページ | 内容 |
|--------|------|
| [Batch-INP-Status](Batch-INP-Status) | 250 INP 生成状況 |
| [S12-CZM-Dataset](S12-CZM-Dataset) | FNO 学習データ (200サンプル) |
| [Benchmark-Targets](Benchmark-Targets) | GNN ベンチマーク目標 |
| [ML-Strategy](ML-Strategy) | ML 全体戦略 |
