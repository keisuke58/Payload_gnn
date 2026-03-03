# 不確実性定量化 (Uncertainty Quantification)

> 実装日: 2026-03-03 | コミット: `d929d0c`

## 概要

GNN 欠陥予測の信頼度を定量化するモジュール (`src/uncertainty.py`)。
2つの相補的手法でエピステミック不確実性を推定:

1. **MC Dropout** — 単一モデルで T 回の確率的フォワードパス
2. **Deep Ensemble** — K 個の fold モデル間の予測分散
3. **Combined (MC × Ensemble)** — K×T サンプルで不確実性を分解

---

## 1. MC Dropout

### 原理

学習済みモデルの dropout を有効にしたまま T 回推論し、予測分散をエピステミック不確実性とする（Gal & Ghahramani, 2016）。

```python
from uncertainty import mc_dropout_predict

mean_prob, std_prob, entropy = mc_dropout_predict(
    model, x, edge_index, edge_attr, T=30
)
```

### 出力
| 変数 | 形状 | 説明 |
|------|------|------|
| `mean_prob` | (N,) | T 回の P(defect) 平均 |
| `std_prob` | (N,) | T 回の P(defect) 標準偏差 = エピステミック不確実性 |
| `entropy` | (N,) | 予測エントロピー H = −Σ p log p |

### CLI 使用例

```bash
# evaluate.py で MC Dropout 不確実性を計算
python src/evaluate.py \
  --model_path runs/best_model.pt \
  --data_dir data/processed_s12_czm_thermal_500_binary \
  --uncertainty --mc_T 30
```

---

## 2. Deep Ensemble

### 原理

5-fold CV で学習した K 個のモデルを全て使い、予測の分散をエピステミック不確実性とする。
MC Dropout と異なり、モデル初期値・学習データの多様性を反映。

```python
from uncertainty import ensemble_predict_with_uncertainty

mean_prob, std_prob, entropy = ensemble_predict_with_uncertainty(
    models, x, edge_index, edge_attr
)
```

### CLI 使用例

```bash
# ensemble_inference.py で Deep Ensemble 不確実性を計算
python scripts/ensemble_inference.py \
  --model_dir runs/5fold_models/ \
  --data_dir data/processed_s12_czm_thermal_500_binary \
  --uncertainty --mc_T 10
```

---

## 3. Combined (MC Dropout × Deep Ensemble)

### 原理

K 個のアンサンブルモデルそれぞれで T 回の MC Dropout → 合計 K×T サンプル。
不確実性を以下に分解:

- **Total uncertainty**: K×T サンプル全体の分散
- **Epistemic (モデル不確実性)**: アンサンブルメンバー平均値間の分散
- **Aleatoric 近似**: モデル内分散の平均

```python
from uncertainty import ensemble_mc_predict

mean_prob, total_std, epistemic_std, entropy = ensemble_mc_predict(
    models, x, edge_index, edge_attr, T=10
)
```

---

## 4. キャリブレーション指標

### 4.1 Expected Calibration Error (ECE)

予測確率と実際の陽性頻度の整合性を測定。低いほど良い。

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |acc_b - conf_b|$$

```python
from uncertainty import expected_calibration_error

ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(
    probs, targets, n_bins=10
)
```

### 4.2 不確実性品質メトリクス

```python
from uncertainty import uncertainty_quality_metrics

metrics = uncertainty_quality_metrics(probs, uncertainty, targets)
```

| メトリクス | 説明 | 良い値 |
|-----------|------|--------|
| `uncertainty_correct_mean` | 正解予測の平均不確実性 | 低い |
| `uncertainty_incorrect_mean` | 誤予測の平均不確実性 | 高い |
| `uncertainty_separation` | incorrect − correct の差 | 正で大きい |
| `uncertainty_auroc` | 不確実性をエラー検出器としたAUROC | > 0.7 |
| `ece` | Expected Calibration Error | < 0.05 |
| `accuracy_reject_X%` | 高不確実性 X% を棄却した後の精度 | 高い |

---

## 5. 可視化

### 5.1 不確実性マップ (3D)

4パネル: Ground Truth | P(defect) | Uncertainty | Error+Uncertainty

```python
from uncertainty import plot_uncertainty_map

plot_uncertainty_map(pos, probs, uncertainty, targets, "uncertainty_map.png")
```

### 5.2 キャリブレーション曲線

Reliability Diagram + 予測分布ヒストグラム

```python
from uncertainty import plot_calibration_curve

plot_calibration_curve(probs, targets, "calibration.png", n_bins=10)
```

---

## 6. 物理的意味と活用

### 6.1 不確実性が高い場所の解釈

| 高不確実性パターン | 物理的意味 | 対処 |
|-------------------|-----------|------|
| 欠陥境界部 | 欠陥→健全の遷移領域 | 正常（境界は本来不確定） |
| 低応力領域での高確信度 | 物理的に不自然な予測 | 物理損失で抑制済み |
| fold 間で予測が割れる | 学習データ不足の領域 | データ拡充で改善 |

### 6.2 Selective Prediction（選択的予測）

不確実性の高いノードを棄却し、信頼度の高い予測のみを出力:

```python
# 上位 10% の不確実性を持つノードを棄却
threshold = np.percentile(std_prob, 90)
confident_mask = std_prob <= threshold
confident_preds = (mean_prob[confident_mask] >= 0.5).astype(int)
```

これにより、precision を犠牲にせず recall を維持しつつ、偽陽性を削減。

---

## 7. 実験結果 (N=200, 5-Fold Ensemble)

> 実験日: 2026-03-04 | GPU: vancouver02 RTX 4090

### 7.1 アンサンブル推論パイプライン

| 手法 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| (A) Single model, t=0.50 | 0.8267 | 0.8771 | 0.8511 |
| (B) Ensemble (5-fold), t=0.50 | 0.7814 | 0.8576 | 0.8177 |
| (C) Ensemble, optimal t=0.881 | 0.9163 | 0.7709 | 0.8374 |
| (D) Ensemble, t @ Recall>=90% | 0.5975 | 0.9004 | 0.7183 |
| (E) Ensemble + PostProc, best F1 | 0.7754 | 0.8714 | **0.8206** |
| (F) Ensemble + PP, Recall>=90% | 0.6930 | 0.9004 | 0.7832 |

### 7.2 不確実性定量化メトリクス (Ensemble)

| メトリクス | 値 | 評価 |
|-----------|-----|------|
| **ECE** (Expected Calibration Error) | **0.0017** | 極めて良好 (< 0.05 が目標) |
| **Uncertainty AUROC** (エラー検出) | **0.9642** | 不確実性がエラーを正確に検出 |
| **Uncertainty separation** | 0.1406 | 誤予測の不確実性 > 正予測 |
| Accuracy (top 5% 棄却後) | 0.9997 | 高不確実性ノード除去で精度向上 |
| Accuracy (top 10% 棄却後) | 0.9997 | 同上 |
| Accuracy (top 20% 棄却後) | 0.9998 | 同上 |

### 7.3 考察

- **キャリブレーション (ECE=0.0017)**: 予測確率と実際の頻度がほぼ完全一致。閾値設定に信頼性がある。
- **エラー検出 (AUROC=0.9642)**: 不確実性が高いノードは実際に誤予測である確率が高い。安全クリティカルな応用に有用。
- **Selective Prediction**: 不確実性上位 20% を棄却するだけで精度 99.98% に到達 → SHM の「検査対象の優先順位付け」に直結。
- **Per-sample 分析**: 40 val サンプル中 35 サンプルが F1>0.7、ただし 5 サンプルは F1=0.0 (完全未検出) → これらのサンプルの不確実性が高ければ「検出困難」と正しく報告される。

---

## 8. 関連ファイル

| ファイル | 内容 |
|---------|------|
| `src/uncertainty.py` | MC Dropout, Deep Ensemble, Combined, ECE, 可視化 |
| `src/evaluate.py` | `--uncertainty`, `--mc_T` で MC Dropout 評価 |
| `scripts/ensemble_inference.py` | `--uncertainty`, `--mc_T` でアンサンブル不確実性 |
| `src/train.py` | BaseGNN の dropout 層（MC Dropout の前提） |

---

## 8. 参考文献

- Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML*.
- Lakshminarayanan, B. et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*.
- Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." *ICML*.
