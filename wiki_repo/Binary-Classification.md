[← Home](Home) | [S12 CZM Dataset](S12-CZM-Dataset) | [ML Strategy](ML-Strategy)

# Binary Classification (2クラス: Healthy vs Defect)

**Date**: 2026-03-03
**Status**: 非熱モデル学習完了 (F1=0.8686) / 熱有バッチ実行中

## 背景

8クラス多欠陥分類 (Val F1 = 0.25) から、**2クラス binary** (healthy=0, defect=1) に問題を簡略化。

**理由**:
- 8クラスでは欠陥クラスごとのサンプルが少なすぎる (0.01–0.15%/class)
- 「欠陥があるか否か」の検出が SHM の第一目標
- Binary 化で全欠陥ノードが1クラスに統合 → 学習シグナル強化

## データセット

**Source**: `abaqus_work/batch_s12_100/` → `data/processed_s12_czm_96_binary/`

| 項目 | 値 |
|------|------|
| グラフ数 | 96 (Train: 77, Val: 19) |
| ノード特徴量 | **34次元** (位置+法線+曲率+変位+温度+応力+ひずみ+繊維配向+積層+境界) |
| エッジ特徴量 | 5次元 |
| ノード/グラフ | 15,206 |
| エッジ/グラフ | 119,058 |
| Train defect 率 | 0.59% (6,947 / 1,170,862) |
| Val defect 率 | 0.48% (1,400 / 288,914) |

## 学習結果 (非熱, SAGE)

### 基本性能

| 指標 | argmax (t=0.50) | 最適閾値 (t=0.88) |
|------|----------------|-------------------|
| **F1** | 0.6731 | **0.8686** |
| Precision | - | 0.8548 |
| Recall | - | 0.8829 |
| AUC | 0.9997 | 0.9997 |

### 学習設定

| パラメータ | 値 |
|-----------|------|
| Architecture | SAGE |
| Hidden | 128 |
| Layers | 4 |
| Loss | Focal Loss (γ=2.0, per-class alpha) |
| LR | 1e-3 (Cosine Annealing) |
| Epochs | 200 (patience=30) |
| Batch size | 4 |

## エラー分析

### 欠陥タイプ別 F1 (threshold=0.88)

| タイプ | Val数 | 欠陥ノード | TP | FP | FN | 平均F1 | P(defect) | P(healthy) |
|--------|--------|-----------|-----|-----|-----|--------|-----------|------------|
| thermal_progression | 2 | 274 | 236 | 23 | 38 | **0.883** | 0.914 | 0.044 |
| delamination | 5 | 245 | 232 | 61 | 13 | 0.862 | 0.956 | 0.045 |
| debonding | 5 | 209 | 186 | 31 | 23 | 0.853 | 0.936 | 0.042 |
| impact | 3 | 241 | 227 | 63 | 14 | 0.849 | 0.964 | 0.049 |
| acoustic_fatigue | 2 | 299 | 247 | 19 | 52 | 0.819 | 0.917 | 0.045 |
| fod | 2 | 132 | 108 | 13 | 24 | **0.714** | 0.917 | 0.043 |

### 最も検出困難なグラフ

| Graph | タイプ | 欠陥ノード | F1 | TP/FP/FN | 原因 |
|-------|--------|-----------|------|----------|------|
| #8 | fod | 16 | **0.55** | 6/0/10 | 小欠陥、prob mean=0.85 が閾値0.88の直下 |
| #11 | acoustic_fatigue | 71 | 0.72 | 43/5/28 | FN多い |
| #12 | debonding | 14 | 0.78 | 9/0/5 | 小欠陥 |

### 確率分布

| ノード種別 | P10 | P25 | P50 | P75 | P90 |
|-----------|------|------|------|------|------|
| **Defect** (n=1,400) | 0.868 | 0.925 | 0.951 | 0.971 | 0.983 |
| **Healthy** (n=287,514) | - | - | - | - | 0.070 |

- Separation (mean_defect - mean_healthy): **0.890**
- FP率: 0.073% (210 / 287,514)
- Overlap: defect min=0.35 vs healthy P99.9=0.86

### 改善の方向性

| 施策 | 対象 | 期待効果 |
|------|------|---------|
| 熱荷重データで再学習 | 全タイプ | 温度+熱応力(3次元)に情報追加 → F1↑ |
| 差圧30kPaデータ | 全タイプ | 応力レベル6x → 欠陥の応力差拡大 |
| グラフ別適応閾値 | fod, 小欠陥 | 小欠陥で閾値を下げてFN削減 |
| FODサンプル増加 | fod | valに2グラフしかない → 過小評価リスク |
| acoustic_fatigue改善 | acoustic_fatigue | Recall 0.61 → boundary weight調整 |

## 進行中

### 熱有バッチ (CTE修正 + 差圧30kPa)
- Template: `Job-CZM-S12-Thermal.inp` (CTE: -0.3e-6/28e-6, 差圧: 30kPa)
- 実行: frontale02 (並列4ジョブ)
- 2ステップ: Step-1 (熱のみ) → Step-2 (熱+機械)
- 完了後: PyG変換 → binary変換 → vancouver02で学習 → 精度比較

## 関連ファイル

| ファイル | 用途 |
|---------|------|
| `scripts/convert_to_binary.py` | 8クラス → 2クラス変換 |
| `scripts/sweep_binary.sh` | 学習 + 閾値最適化 |
| `scripts/optimize_threshold.py` | 閾値スイープ F1 最大化 |
| `scripts/error_analysis.py` | 欠陥タイプ別エラー分析 |
| `src/prepare_ml_data.py` | CSV → PyG 前処理 (Job-S12-D* 対応) |
| `src/train.py` | GNN 学習 (binary 自動検出) |
