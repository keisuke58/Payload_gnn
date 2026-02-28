[← Home](Home)

# Dataset Format: データセット仕様

> **関連**: [Dataset-Comparison](Dataset-Comparison) — 欠陥あり vs 欠陥なし の可視化・比較統計  
> **欠陥生成**: [Defect-Generation-and-Labeling](Defect-Generation-and-Labeling) — 欠陥パラメータと defect_label 付与ルール

---

## 日本語概要

FEM (Abaqus) 出力の CSV 形式と PyTorch Geometric (PyG) の Data 形式を定義。**nodes.csv**: 座標、応力、温度、defect_label。**elements.csv**: 要素接続。**PyG Data**: ノード特徴 28 次元（法線、曲率、応力、温度、u_mag、ひずみ、繊維配向含む）、エッジ特徴 5 次元。用語は [用語集](Vocabulary) を参照。

---

## 1. CSV フォーマット（FEM出力）

### nodes.csv

| 列名 | 型 | 範囲 | 説明 |
|------|-----|------|------|
| `node_id` | int | — | Abaqus ノードラベル |
| `x` | float | 0 ~ 5000 mm | X座標 |
| `y` | float | 0 ~ 5000 mm | Y座標 |
| `z` | float | 0 ~ 5000 mm | Z座標 |
| `ux`, `uy`, `uz` | float | mm | 変位 |
| `s11` | float | MPa | 応力 σ₁₁ |
| `s22` | float | MPa | 応力 σ₂₂ |
| `s12` | float | MPa | せん断応力 σ₁₂ |
| `dspss` / `smises` | float | MPa | von Mises 応力 |
| `u_mag` | float | mm | 変位の大きさ √(ux²+uy²+uz²) |
| `temp` | float | 25 ~ 150 ℃ | 節点温度 |
| `thermal_smises` | float | ≥ 0 MPa | 熱応力 von Mises (ODB に含まれる場合) |
| `le11`, `le22`, `le12` | float | — | ひずみ (LE, ODB に LE 出力時) |
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
    x       = [N, 29],      # ノード特徴量
    edge_index = [2, E],     # エッジ接続 (無向グラフ)
    edge_attr  = [E, 5-6],   # エッジ特徴量
    y       = [N],           # 欠陥ラベル (0/1)
    pos     = [N, 3],        # 座標 (x, y, z)
)
```

**ノード特徴量 x (34次元)**:

| Dim | Feature | 説明 |
|-----|---------|------|
| 0-2 | x, y, z | 位置座標 (mm) |
| 3-5 | nx, ny, nz | 表面法線。曲面の向き、ガイド波モード変換に影響 |
| 6-9 | κ₁, κ₂, H, K | 曲率。波の集束・発散、局所形状 |
| 10-12 | ux, uy, uz | 変位ベクトル (mm)。欠陥で局所増大 |
| 13 | u_mag | 変位の大きさ \|u\|。1次元にまとめた変位強度 |
| 14 | temp | 節点温度 (°C)。熱荷重・CTE 不整合 |
| 15-18 | s11, s22, s12, smises | 応力 (MPa)。欠陥で荷重伝達断絶 |
| 19 | principal_stress_sum | 主応力和 σ₁+σ₂ = s11+s22 (MPa) |
| 20 | thermal_smises | 熱応力 von Mises (ODB に含まれる場合) |
| 21-23 | le11, le22, le12 | ひずみ (LE)。損傷と相関が強い |
| 24-26 | fiber_circum_x,y,z | 繊維配向（周方向単位ベクトル）。CFRP 異方性 |
| 27-30 | layup_0, layup_45, layup_minus45, layup_90 | 積層角度 [0,45,-45,90]° (rad) |
| 31 | circum_angle | 周方向角度 θ。0° 積層の局所方向 |
| 32-33 | boundary, loaded | 境界・荷重ノード flag |

詳細・追加候補: [Node-Features](Node-Features)

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
