[← Home](Home) | [C3D10-Batch-Generation](C3D10-Batch-Generation) | [S12-CZM-Dataset](S12-CZM-Dataset)

# バッチ INP 生成状況

> **Status**: INP 生成完了、ソルバー実行中
> **Date**: 2026-03-03
> **総 INP 数**: 250 ファイル / 5.1 GB

---

## 1. 概要

`generate_czm_sector12.py` (CZM S12) および `generate_realistic_dataset.py` (C3D10) から、
5 つの DOE 設計を統合して **250 サンプル分の INP ファイル** を生成完了。
現在 Abaqus ソルバーによる検証実行を進めている。

### 1.1 DOE 構成

| DOE ファイル | サンプル数 | モデル | 欠陥タイプ |
|---|:---:|---|---|
| `doe_c3d10_mech_100.json` | 105 | C3D10 ソリッドコア | debonding×100 + healthy×5 |
| `doe_multitype_100.json` | 100 | CZM S12 セクター | debonding×40 + impact×30 + FOD×30 |
| `doe_multitype_30.json` | 30 | CZM S12 セクター | debonding×12 + impact×9 + FOD×9 |
| `doe_phase1.json` | 20 | C3D10 ソリッドコア | debonding×20 |
| `doe_extended_test.json` | 14 | CZM S12 / C3D10 混合 | delam×2 + acoustic×3 + thermal×1 + FOD×2 + debonding×3 + impact×2 + inner_debond×1 |
| **合計** | **269** (DOE定義) | | |

> **注**: DOE 間でジョブ名が重複するため、実ファイル数は 250。

---

## 2. 生成済み INP ファイル

### 2.1 タイプ別サマリ

| 欠陥タイプ | INP 数 | CZM S12 (<15MB) | C3D10 中 (15-50MB) | C3D10 大 (>50MB) | 合計サイズ |
|---|:---:|:---:|:---:|:---:|---:|
| H3_Debond | 161 | 101 | 53 | 7 | 2.8 GB |
| H3_FOD | 41 | 39 | 0 | 2 | 509 MB |
| H3_Impact | 38 | 36 | 0 | 2 | 779 MB |
| H3_Thermal | 4 | 0 | 0 | 4 | 273 MB |
| H3_InnerDebond | 2 | 0 | 0 | 2 | 438 MB |
| H3_Acoustic | 2 | 0 | 0 | 2 | 145 MB |
| H3_Delam | 1 | 0 | 0 | 1 | 73 MB |
| H3_Healthy | 1 | 1 | 0 | 0 | 1.7 MB |
| **合計** | **250** | **177** | **53** | **20** | **5.1 GB** |

### 2.2 モデル種別

| モデル | 特徴 | INP サイズ目安 | ノード数 |
|---|---|---|---|
| **CZM S12** | 1/12セクター (30°), COH3D8接着層, ~65K nodes | 7–12 MB | ~15,000 (OuterSkin) |
| **C3D10 中** | C3D10 tet コア, Barrel のみ, ~170K lines | 15–50 MB | ~100K |
| **C3D10 大** | C3D10 tet コア, フルモデル, ~1.4M lines | 50–384 MB | ~366K (OuterSkin) |

---

## 3. ソルバー実行状況

### 3.1 完了済み

| ジョブ | モデル | ステータス | ODB サイズ |
|---|---|:---:|---:|
| H3_Debond_0127 | CZM S12 | ✅ 完了 | 26 MB |
| H3_Debond_0128 | CZM S12 | ✅ 完了 | 26 MB |
| H3_Debond_0129 | CZM S12 | ✅ 完了 | 26 MB |
| H3_Debond_0130 | CZM S12 | ✅ 完了 | 26 MB |
| H3_Debond_0131 | CZM S12 | ✅ 完了 | 26 MB |
| H3_Debond_0257 | CZM S12 | ✅ 完了 | 26 MB |
| H3_Healthy_0000 | CZM S12 | ✅ 完了 | 6.5 MB |
| H3_Ideal_0000 | — | ✅ 完了 | 26 MB |
| H3_Ideal_50mm_0000 | — | ✅ 完了 | 6.5 MB |

### 3.2 ODB 存在 (ステータス未確認)

| ジョブ | ODB サイズ | 備考 |
|---|---:|---|
| H3_Acoustic_0000 | 216 MB | |
| H3_Debond_0004 | 56 MB | C3D10 |
| H3_Debond_0011 | 49 MB | C3D10 |
| H3_Impact_0010 | 660 MB | 非常に大きい |
| H3_InnerDebond_0009 | 254 MB | |
| H3_Thermal_0012 | 49 MB | C3D10 |

### 3.3 実行残数

| カテゴリ | INP 数 | 完了 ODB | 残り |
|---|:---:|:---:|:---:|
| H3_Debond | 161 | 8 | 153 |
| H3_FOD | 41 | 0 | 41 |
| H3_Impact | 38 | 1 | 37 |
| 他 (Thermal, Acoustic, etc.) | 10 | 4 | 6 |
| **合計** | **250** | **13** | **237** |

---

## 4. 欠陥パラメータ例

### 4.1 Debonding

```json
{
  "defect_type": "debonding",
  "theta_deg": 52.94,
  "z_center": 3414.83,
  "radius": 60.97
}
```

### 4.2 Impact

```json
{
  "defect_type": "impact",
  "theta_deg": 31.86,
  "z_center": 2135.13,
  "radius": 68.73,
  "damage_ratio": 0.49
}
```

### 4.3 FOD (Foreign Object Damage)

```json
{
  "defect_type": "fod",
  "theta_deg": 11.08,
  "z_center": 3396.72,
  "radius": 68.73
}
```

### 4.4 Extended Types

```json
{
  "defect_type": "delamination",
  "theta_deg": 39.33,
  "z_center": 2203.85,
  "radius": 68.73,
  "delam_depth": 0.25
}
```

---

## 5. 生成パイプライン

```
generate_doe.py
  → doe_c3d10_mech_100.json     (105 samples)
  → doe_multitype_100.json      (100 samples)
  → doe_multitype_30.json       ( 30 samples)
  → doe_phase1.json             ( 20 samples)
  → doe_extended_test.json      ( 14 samples)

generate_czm_sector12.py (Abaqus CAE)  → H3_*.inp (CZM S12)
generate_realistic_dataset.py          → H3_*.inp (C3D10)

[INP 生成完了 ← いまここ]

Abaqus solver                          → H3_*.odb
extract_odb_results.py                 → results/nodes.csv
prepare_ml_data.py                     → data/processed_*/
train.py (vancouver02)                 → GNN 学習
```

---

## 6. 次のステップ

1. **ソルバー実行**: CZM S12 モデル (177 件) を frontale サーバーでバッチ実行
2. **C3D10 モデル**: 73 件の中〜大モデルは frontale (93GB RAM) で順次実行
3. **ODB 抽出**: 完了分から `extract_odb_results.py` で CSV 変換
4. **品質チェック**: 物理量 (変位, 応力, 温度) の妥当性確認
5. **PyG 変換**: `prepare_ml_data.py` で学習データ構築
6. **vancouver02 転送**: GPU サーバーへデータ転送 → GNN 学習

### 6.1 サーバー割り当て案

| サーバー | モデル | 予定数 | メモリ |
|---|---|:---:|---|
| frontale01 | CZM S12 | ~60 | 93 GB |
| frontale02 | CZM S12 + C3D10 | ~60 | 93 GB |
| frontale04 | CZM S12 | ~60 | 93 GB |
| marinos03 | C3D10 大 | ~20 | 93 GB |
| localhost (fifawc) | CZM S12 のみ | ~40 | 62 GB |

### 6.2 時間見積もり

| モデル | 時間/サンプル | 数 | 合計 (逐次) | 合計 (4並列) |
|---|---:|:---:|---:|---:|
| CZM S12 | ~5 分 | 177 | ~15 時間 | ~4 時間 |
| C3D10 中 | ~15 分 | 53 | ~13 時間 | ~3 時間 |
| C3D10 大 | ~30 分 | 20 | ~10 時間 | ~3 時間 |

---

## 7. 関連ページ

| ページ | 内容 |
|--------|------|
| [S12-CZM-Dataset](S12-CZM-Dataset) | 既存 S12 CZM 96 サンプルデータセット |
| [C3D10-Batch-Generation](C3D10-Batch-Generation) | C3D10 バッチ生成計画 |
| [Ground-Truth-FEM](Ground-Truth-FEM) | Ground Truth モデル (~90% リアリズム) |
| [Extended-Defect-Types](Extended-Defect-Types) | 7 欠陥タイプの定義 |
| [FEM-Realism-Roadmap](FEM-Realism-Roadmap) | リアリズム向上ロードマップ |
| [Defect-Generation-and-Labeling](Defect-Generation-and-Labeling) | 欠陥生成ロジック |
