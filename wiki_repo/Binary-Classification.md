[← Home](Home) | [S12 CZM Dataset](S12-CZM-Dataset) | [ML Strategy](ML-Strategy)

# Binary Classification (2クラス: Healthy vs Defect)

**Date**: 2026-03-03
**Status**: 学習準備完了、vancouver02 で実行待ち

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

## 学習設定

```bash
# vancouver02 で実行
nohup bash scripts/sweep_binary.sh > sweep_binary.log 2>&1 &
```

| パラメータ | 値 |
|-----------|------|
| Architecture | SAGE |
| Hidden | 128 |
| Layers | 4 |
| Loss | Focal Loss (γ=2.0, per-class alpha) |
| LR | 1e-3 (Cosine Annealing) |
| Epochs | 200 (patience=30) |
| Batch size | 4 |

学習後に `optimize_threshold.py` で閾値最適化 (default 0.5 → 最適値探索)。

## 精度向上の戦略

| 手法 | 期待効果 | 状態 |
|------|---------|------|
| 8→2クラス化 | F1: 0.25 → 0.8+ | ✅ 完了 |
| 34次元特徴量 (旧20次元→) | 物理量充実 | ✅ 完了 |
| 閾値最適化 | F1 +0.03–0.10 | ✅ スクリプト準備済み |
| Architecture sweep | 最適モデル選定 | 🔄 次フェーズ |
| Boundary weight | 境界ノード重み付け | 🔄 次フェーズ |
| Residual connections | 深層GNN安定化 | 📋 計画中 |
| DropEdge / Feature noise | 正則化 | 📋 計画中 |

## 関連ファイル

| ファイル | 用途 |
|---------|------|
| `scripts/convert_to_binary.py` | 8クラス → 2クラス変換 |
| `scripts/sweep_binary.sh` | 学習 + 閾値最適化 |
| `scripts/optimize_threshold.py` | 閾値スイープ F1 最大化 |
| `src/prepare_ml_data.py` | CSV → PyG 前処理 (Job-S12-D* 対応) |
| `src/train.py` | GNN 学習 (binary 自動検出) |
