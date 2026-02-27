[← Home](Home)

# Dataset Format: データセット仕様

> **関連**: [Dataset-Comparison](Dataset-Comparison) — 欠陥あり vs 欠陥なし の可視化・比較統計  
> **欠陥生成**: [Defect-Generation-and-Labeling](Defect-Generation-and-Labeling) — 欠陥パラメータと defect_label 付与ルール

---

## 日本語概要

FEM (Abaqus) 出力の CSV 形式と PyTorch Geometric (PyG) の Data 形式を定義。**nodes.csv**: 座標、応力、温度、defect_label。**elements.csv**: 要素接続。**PyG Data**: ノード特徴 16–18 次元（法線、曲率、応力、温度）、エッジ特徴 5 次元。用語は [用語集](Vocabulary) を参照。

---

## 1. CSV フォーマット（FEM出力）

### nodes.csv

| 列名 | 型 | 範囲 | 説明 |
|------|-----|------|------|
| `node_id` | int | — | Abaqus ノードラベル |
| `x` | float | 0 ~ 5000 mm | X座標 |
| `y` | float | 0 ~ 5000 mm | Y座標 |
| `z` | float | 0 ~ 5000 mm | Z座標（軸方向） |
| `s11` | float | MPa | 応力 σ₁₁ |
| `s22` | float | MPa | 応力 σ₂₂ |
| `s12` | float | MPa | せん断応力 σ₁₂ |
| `dspss` | float | MPa | √(σ₁₁² + σ₂₂² + σ₁₂²) |
| `temperature` | float | 25 ~ 150 ℃ | 節点温度 |
| `thermal_smises` | float | ≥ 0 MPa | 熱応力 von Mises |
| `defect_label` | int | 0 or 1 | 0=健全, 1=欠陥 |

### elements.csv

| 列名 | 型 | 説明 |
|------|-----|------|
| `elem_id` | int | 要素ID |
| `etype` | str | 要素タイプ (S4RT, S3T, C3D8RT, ...) |
| `n1`~`n8` | int | ノード接続 (-1=未使用) |

## 2. PyTorch Geometric Data オブジェクト

### build_graph.py 出力（曲率対応版）

```python
Data(
    x       = [N, 18],      # ノード特徴量
    edge_index = [2, E],     # エッジ接続 (無向グラフ)
    edge_attr  = [E, 5-6],   # エッジ特徴量
    y       = [N],           # 欠陥ラベル (0/1)
    pos     = [N, 3],        # 座標 (x, y, z)
)
```

**ノード特徴量 x (18次元)**:

| Dim | Feature | 説明 |
|-----|---------|------|
| 0-2 | x, y, z | 位置座標 |
| 3-5 | nx, ny, nz | 表面法線ベクトル |
| 6 | κ₁ | 第1主曲率 |
| 7 | κ₂ | 第2主曲率 |
| 8 | H | 平均曲率 (κ₁+κ₂)/2 |
| 9 | K | ガウス曲率 κ₁·κ₂ |
| 10 | s11 | 応力 σ₁₁ |
| 11 | s22 | 応力 σ₂₂ |
| 12 | s12 | せん断応力 σ₁₂ |
| 13 | dspss | DSPSS |
| 14 | boundary | 境界ノード flag |
| 15 | loaded | 荷重ノード flag |
| 16 | temperature | 節点温度 |
| 17 | thermal_smises | 熱応力 Mises |

**エッジ特徴量 edge_attr (5-6次元)**:

| Dim | Feature |
|-----|---------|
| 0-2 | Δx, Δy, Δz (相対位置) |
| 3 | ユークリッド距離 |
| 4 | 法線間角度 |
| 5 | 測地線距離 (optional) |

### preprocess_fairing_data.py 出力（簡易版）

```python
Data(
    x       = [N, 12],      # ノード特徴量
    edge_index = [2, E],
    edge_attr  = [E, 4],
    y       = [N],
    pos     = [N, 3],
)
```

**ノード特徴量 x (12次元)**:

| Dim | Feature |
|-----|---------|
| 0-2 | x, y, z |
| 3 | s11 |
| 4 | s22 |
| 5 | s12 |
| 6 | ΔDSPSS (健全ベースライン差分) |
| 7-9 | node_type (one-hot: internal/boundary/loaded) |
| 10 | temperature |
| 11 | thermal_smises |

## 3. 正規化

**Z-score正規化** (feature-wise):

```python
x_norm = (x - mean) / std    # std < 1e-8 の場合 std=1.0
```

- 統計量は `norm_stats.pt` に保存
- 学習データの統計量で正規化し、推論時も同じ統計量を適用

## 4. データセット分割

| Split | 比率 | 用途 |
|-------|------|------|
| Train | 70% | 学習 |
| Validation | 15% | ハイパーパラメータ選択・Early Stopping |
| Test | 15% | 最終評価 |

- 層化分割（欠陥タイプ・サイズのバランス確保）
- OODテストセット: 未学習の欠陥サイズ・荷重条件

## 5. チェックポイント形式

```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'val_f1': float,
    'val_metrics': dict,
    'args': dict,
    'in_channels': int,       # 入力特徴量次元
    'edge_attr_dim': int,     # エッジ特徴量次元
}
```

## 6. 外部データセット

### Open Guided Waves (OGW)

- **Source**: [Zenodo Record 5105861](https://zenodo.org/records/5105861)
- **Format**: HDF5 (`coordinates.h5`, `data_z.h5`, `time.h5`)
- **用途**: Sim-to-Real Domain Adaptation のターゲットドメイン
- **構成**: 6種の励起条件 (16.5 / 50 / 100 / 200 / 300 kHz + Chirp)
