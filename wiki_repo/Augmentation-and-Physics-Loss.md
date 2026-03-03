# Graph Augmentation + 物理インフォームド損失

> 実装日: 2026-03-03 | コミット: `56e6d69`

## 概要

GNN 学習パイプライン (`src/train.py`) に以下を追加:
1. **Graph Augmentation** — Feature Masking, Circumferential Flip
2. **物理インフォームド損失関数** — Smoothness, Stress Gradient, Connected Component
3. **データ拡張パイプライン** — `prepare_ml_data.py` の `--extra_inputs` 対応

全機能はデフォルト `0.0` で後方互換。有効化は CLI 引数で制御。

---

## 1. Graph Augmentation

### 1.1 Feature Masking (`--feature_mask`)

ランダムに特徴次元をゼロ化してロバスト性を向上。

```bash
python src/train.py --feature_mask 0.1  # 10%の確率で各要素をマスク
```

**効果**: 特定の特徴量への過度な依存を防ぎ、汎化性能を改善。

### 1.2 Circumferential Flip (`--augment_flip`)

フェアリングの軸対称性を利用した物理的に正しいデータ拡張。

```bash
python src/train.py --augment_flip 0.5  # 50%の確率で各バッチを反転
```

**反転する次元** (build_graph.py の34次元特徴量順序に基づく):

| Dim | 特徴量 | 操作 |
|-----|--------|------|
| 0, 2 | x, z 座標 | 符号反転 |
| 3, 5 | nx, nz 法線 | 符号反転 |
| 24, 26 | fiber_x, fiber_z | 符号反転 |
| 31 | circumferential angle θ | 符号反転 |

エッジ属性 (dx, dz) も同時に反転。

---

## 2. 物理インフォームド損失関数

### 2.1 Spatial Smoothness Loss (`--physics_lambda_smooth`)

隣接ノード間の P(defect) 差を抑制し、孤立した誤検出を削減。

$$L_{\text{smooth}} = \frac{1}{|E|} \sum_{(i,j) \in E} (p_i - p_j)^2$$

```bash
python src/train.py --physics_lambda_smooth 0.1
```

**物理的根拠**: 欠陥は空間的に連続した領域であり、単一ノードが孤立して欠陥と予測されることは物理的にありえない。

### 2.2 Stress Gradient Consistency (`--physics_lambda_stress`)

応力勾配が低い場所での欠陥予測にペナルティ。

$$L_{\text{stress}} = \text{mean}(p_i \cdot (1 - \|\nabla \sigma\|_{\text{norm},i}))$$

```bash
python src/train.py --physics_lambda_stress 0.05 --stress_dim 18
```

**物理的根拠**: デボンディング欠陥は局所的な応力集中を引き起こす。応力場が一様な領域で欠陥と予測することは物理法則に反する。`--stress_dim 18` は von Mises 応力 (smises) の特徴量インデックス。

### 2.3 Connected Component Penalty (`--physics_lambda_connected`)

近傍ノードも欠陥と予測されていない孤立ノードにペナルティ。

$$L_{\text{conn}} = \text{mean}(p_i \cdot (1 - \bar{p}_{\text{neighbors},i}))$$

```bash
python src/train.py --physics_lambda_connected 0.1
```

**物理的根拠**: 実際の欠陥は空間的にクラスタリングされる。孤立した1ノードの欠陥予測はノイズである可能性が高い。

---

## 3. 推奨ハイパーパラメータ

### ベースライン (augmentation のみ)
```bash
python src/train.py \
  --arch gat --hidden 128 --layers 4 \
  --loss focal --focal_gamma 3.0 \
  --residual --defect_weight 5.0 \
  --feature_mask 0.1 --augment_flip 0.5 \
  --drop_edge 0.1 --feature_noise 0.01
```

### フル設定 (augmentation + 物理損失)
```bash
python src/train.py \
  --arch gat --hidden 128 --layers 4 \
  --loss focal --focal_gamma 3.0 \
  --residual --defect_weight 5.0 \
  --feature_mask 0.1 --augment_flip 0.5 \
  --physics_lambda_smooth 0.1 \
  --physics_lambda_stress 0.05 \
  --physics_lambda_connected 0.1 \
  --cross_val 5 --epochs 200
```

### λ グリッドサーチ推奨範囲

| パラメータ | 推奨範囲 | 注意点 |
|-----------|---------|--------|
| `physics_lambda_smooth` | 0.01 ~ 0.5 | 大きすぎると全ノードが同じ予測に収束 |
| `physics_lambda_stress` | 0.01 ~ 0.1 | 正規化済みデータでは小さめに |
| `physics_lambda_connected` | 0.01 ~ 0.5 | smoothness と類似効果、片方で十分な場合あり |

---

## 4. TensorBoard モニタリング

物理損失は個別に TensorBoard に記録される:

- `physics/physics_smooth` — Smoothness Loss
- `physics/physics_stress` — Stress Gradient Loss
- `physics/physics_connected` — Connected Component Penalty

```bash
tensorboard --logdir runs/
```

---

## 5. データ拡張パイプライン

`prepare_ml_data.py` に `--extra_inputs` を追加し、複数バッチディレクトリの統合に対応。

```bash
python src/prepare_ml_data.py \
  --input abaqus_work/batch_s12_100_thermal \
  --extra_inputs abaqus_work/batch_s12_ext200 abaqus_work/batch_s12_ext100b \
  --output data/processed_s12_czm_thermal_500_binary
```

### データ拡充ロードマップ

| フェーズ | サンプル数 | DOE | Seed | 状態 |
|---------|-----------|-----|------|------|
| 初期 | 100 | doe_sector12_100.json | 42 | 完了 |
| Ext1 | +100 | doe_sector12_ext100.json | 2026 | 完了 |
| Ext2 | +200 | doe_sector12_ext200.json | 2027 | 実行中 |
| Ext3 | +100 | doe_sector12_ext100b.json | 2028 | 待機 |
| **合計** | **500** | | | |

---

## 6. 実験結果 (N=200, 5-fold CV)

> 実験日: 2026-03-03 | GPU: vancouver02 RTX 4090

### 6.1 条件比較

| 実験 | Mean F1 | Std | 改善 |
|------|---------|-----|------|
| Baseline (DW=5, Residual, aug/physics なし) | 0.6489 | 0.0522 | — |
| Augmentation のみ (mask=0.1, flip=0.5, drop=0.1, noise=0.01) | 0.6629 | 0.0416 | +2.2% |
| **Aug + 物理損失** (smooth=0.1, stress=0.05, connected=0.1) | **0.7208** | **0.0343** | **+11.1%** |

### 6.2 Fold 別結果

| Fold | Baseline | Aug Only | Aug + Physics |
|------|----------|----------|---------------|
| 0 | 0.5509 | 0.6456 | 0.6793 |
| 1 | 0.6623 | 0.6705 | 0.7415 |
| 2 | 0.7019 | 0.7005 | 0.7597 |
| 3 | 0.6486 | 0.5919 | 0.6798 |
| 4 | 0.6809 | 0.7059 | 0.7438 |

### 6.3 考察

- **物理損失が最も効果的**: +11.1% の F1 改善。全 fold で一貫して改善。
- **Augmentation 単体の効果は限定的**: +2.2%。N=200 では overfitting よりデータ不足が支配的。
- **分散の低下**: Physics loss で Std が 0.052 → 0.034 に低下 → 安定した予測。
- **参考**: 旧実行 (同パラメータ、別 fold 分割) では Mean F1=0.8239。fold 分割によるばらつきが大きい。

### 6.4 次のステップ

1. N=500 データセットで再実験 (Abaqus バッチ完了待ち)
2. λ グリッドサーチ (smooth, stress, connected の最適化)
3. 不確実性定量化 → [Uncertainty-Quantification](Uncertainty-Quantification)

---

## 7. 関連ファイル

| ファイル | 内容 |
|---------|------|
| `src/train.py` | Augmentation 関数 + 物理損失 + CLI 引数 |
| `src/prepare_ml_data.py` | `--extra_inputs` 対応 |
| `src/build_graph.py` L478-493 | 34次元特徴量の順序定義 |
