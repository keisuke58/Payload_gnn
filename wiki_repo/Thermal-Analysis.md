[← Home](Home)

# Thermal Analysis: 熱解析統合

## 概要

JAXA文献値に基づく熱環境データをAbaqus FEMモデルに統合し、熱応力をGNNの入力特徴量として追加。CFDは行わず、文献値ベースの境界条件を使用する方針。

## 設計判断

| 判断項目 | 選択 | 理由 |
|---------|------|------|
| CFD vs 文献値 | **文献値** | プロジェクト主目的（GNN-SHM）とずれる。コスパ悪い |
| 逐次 vs 完全連成 | **逐次（2ステップ）** | 熱→構造の一方向連成で十分 |
| 定常 vs 非定常 | **定常（ピーク温度）** | 最悪条件の熱応力を捕捉。SHM学習には十分 |
| GNN特徴量 | **+2次元** | temperature + thermal_smises |

## 熱環境パラメータ

```python
T_REF          = 25.0   # ℃  基準温度（応力フリー）
T_OUTER_SKIN   = 150.0  # ℃  外板表面ピーク
T_INNER_SKIN   = 50.0   # ℃  内板表面
```

### 温度分布

```
        R_OUTER (150℃)
        ┃ Outer Skin ┃  ← 全ノード均一 T_OUTER
        ┃════════════┃
        ┃            ┃
        ┃  Core      ┃  ← 外面: T_OUTER / 内面: T_INNER
        ┃  (gradient)┃     動径位置で2分割
        ┃            ┃
        ┃════════════┃
        ┃ Inner Skin ┃  ← 全ノード均一 T_INNER
        R_INNER (50℃)
```

## CFRP 熱物性 (T300/914)

| Property | Value | Unit |
|----------|-------|------|
| Density | 1.58e-9 | t/mm³ |
| Conductivity (fiber) | 7.0e-3 | mW/(mm·℃) |
| Conductivity (transverse) | 0.8e-3 | mW/(mm·℃) |
| Specific Heat | 9.0e+8 | mm²/(s²·℃) |
| CTE (fiber) | **-0.3e-6** | /℃ |
| CTE (transverse) | **28.0e-6** | /℃ |

### 温度依存弾性率

| T (℃) | E1 (MPa) | E2 (MPa) | ν12 | G12 (MPa) |
|--------|----------|----------|-----|-----------|
| 25 | 135,000 | 10,000 | 0.30 | 5,000 |
| 100 | 133,000 | 9,000 | 0.30 | 4,500 |
| 150 | 130,000 | 7,000 | 0.29 | 3,800 |
| 200 | 125,000 | 5,500 | 0.28 | 3,200 |

> E2（マトリクス支配）は150℃で30%低下 — 剥離に直結

## ハニカムコア熱物性 (Al-5052)

| Property | Value | Unit |
|----------|-------|------|
| Density | 4.8e-11 | t/mm³ |
| Conductivity | 1.5e-3 | mW/(mm·℃) |
| Specific Heat | 9.0e+8 | mm²/(s²·℃) |
| CTE | **23.0e-6** | /℃ |

## CTE不整合と熱応力

CFRP横方向CTE (28e-6) と Al CTE (23e-6) の差:

```
ΔαΔT ≈ (28 - 23) × 10⁻⁶ × (150 - 25) = 6.25 × 10⁻⁴
σ_thermal ≈ E₂ × ΔαΔT ≈ 10,000 × 6.25e-4 ≈ 6.25 MPa
```

→ 界面に ~6 MPa のせん断応力が発生（物理的に妥当）

## Abaqus 実装

### 要素タイプ（連成温度-変位対応）

| 変更前 | 変更後 | 用途 |
|--------|--------|------|
| S4R | **S4RT** | 外板・内板（シェル） |
| S3 | **S3T** | 三角形シェル |
| C3D8R | **C3D8RT** | コア（ソリッド） |
| C3D6 | **C3D6T** | ウェッジ |
| C3D4 | **C3D4T** | テトラ |

### 解析ステップ

```
Initial  →  Thermal (CoupledTempDisp, Steady)  →  Load (Static)
  T_REF      温度場適用 → 熱応力発生              機械荷重追加
  25℃        NT, S, E, U, COORD 出力              S, E, U, NT, COORD 出力
```

### Field Output

Thermalステップ・Loadステップ両方で以下を出力:
- `S` — 応力テンソル
- `E` — ひずみテンソル
- `U` — 変位
- `NT` — 節点温度
- `COORD` — 節点座標

## CSV出力 → GNN連携

`export_results_csv()` が ODB から以下をCSVに出力:

| 列名 | ソース | 説明 |
|------|--------|------|
| node_id | — | ノードID |
| x, y, z | Load step COORD | 座標 |
| s11, s22, s12 | Load step S | 機械+熱 複合応力 |
| dspss | Load step S | 応力マグニチュード |
| **temperature** | **Thermal step NT** | **節点温度** |
| **thermal_smises** | **Thermal step S (Mises)** | **熱応力のみのMises** |
| defect_label | 外部入力 | 0=健全, 1=欠陥 |

## GNN特徴量への統合

### 後方互換性

- `temperature`/`thermal_smises` 列がCSVにあれば読み込み、なければスキップ
- 旧データ (熱なし): 16次元 / 10次元のまま動作
- 新データ (熱あり): 18次元 / 12次元

### 正規化

Z-score正規化がfeature-wiseで適用されるため、温度(25-150℃)と応力(0-1000+ MPa)のスケール差は自動的に処理される。

## 検証チェックリスト

- [ ] Abaqus実行: 外板150℃、内板50℃、コア勾配を確認
- [ ] CSV出力: temperature列(25-150℃), thermal_smises列(非負)
- [ ] GNN入力: `data.x.shape` = (N, 18) or (N, 12)
- [ ] 後方互換: 旧CSV → (N, 16) or (N, 10) のまま動作
- [ ] 学習比較: 熱特徴量あり/なしでF1スコア比較
