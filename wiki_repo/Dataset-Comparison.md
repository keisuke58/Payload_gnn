[← Home](Home)

# データセット比較: 欠陥あり vs 欠陥なし

> 欠陥アリデータと欠陥なしデータの可視化・比較統計量のまとめ  
> 最終更新: 2026-02-28

---

## 1. 概要

| 項目 | 値 |
|------|-----|
| **健全サンプル** | 1 (healthy_baseline) |
| **欠陥サンプル** | 20 (sample_0000 ～ sample_0019) |
| **DOE 計画** | [doe_phase1.json](doe_phase1.json) |
| **欠陥計画** | [docs/DEFECT_PLAN.md](../docs/DEFECT_PLAN.md) |

---

## 2. データ形式の違い

本データセットには **2 種類の抽出形式** が混在しています。

### 2.1 形式 A: 応力形式 (Stress)

| サンプル | ノード数 | カラム |
|----------|----------|--------|
| healthy_baseline | 22,220 | node_id, x, y, z, s11, s22, s12, dspss, defect_label |
| sample_0001 | 22,220 | 同上 |

- **抽出元**: 複数パート (Skin_Outer, Skin_Inner, Core) の統合
- **用途**: GNN 学習用（応力・欠陥ラベル付き）

### 2.2 形式 B: 変位・温度形式 (Displacement)

| サンプル | ノード数 | カラム |
|----------|----------|--------|
| sample_0000, 0002～0019 | 745 | node_id, x, y, z, ux, uy, uz, temp |

- **抽出元**: 外スキンのみ (Part-OuterSkin)
- **用途**: 現在の Abaqus バッチ抽出パイプライン

---

## 3. 比較統計量

### 3.1 ノード数

| カテゴリ | ノード数 | 備考 |
|----------|----------|------|
| **Healthy** | 22,220 | 形式 A |
| **Defect (形式 A)** | 22,220 | sample_0001 のみ |
| **Defect (形式 B)** | 745 ± 0 | sample_0000, 0002～0019 |

![Node count comparison](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/figures/dataset_comparison/node_count_comparison.png)

### 3.2 応力統計 (形式 A: healthy vs sample_0001)

| 指標 | healthy_baseline | sample_0001 (欠陥) |
|------|------------------|---------------------|
| **n_nodes** | 22,220 | 22,220 |
| **n_defect** | 0 | 77 |
| **dspss mean** | 30.78 MPa | 30.81 MPa |
| **dspss std** | 31.32 MPa | 31.54 MPa |

欠陥サンプルでは **77 ノード** が欠陥ラベル 1 を持つ。応力平均は健全とほぼ同等だが、標準偏差がわずかに増加。

### 3.3 欠陥パラメータ分布 (DOE Phase 1)

| パラメータ | 範囲 | 備考 |
|------------|------|------|
| **θ (deg)** | 5 ～ 55 | 周方向位置 |
| **z_center (mm)** | 800 ～ 4,200 | 軸方向中心 |
| **radius (mm)** | 10 ～ 250 | 欠陥半径（層化: Small/Medium/Large/Critical） |

| サンプル | θ (deg) | z (mm) | r (mm) | サイズ階層 |
|----------|---------|--------|--------|------------|
| sample_0000 | 38.1 | 895 | 17 | Small |
| sample_0001 | 10.4 | 1,021 | 29 | Small |
| sample_0007 | 26.4 | 1,781 | 73 | Medium |
| sample_0014 | 22.1 | 1,874 | 93 | Large |
| sample_0019 | 31.9 | 2,386 | 179 | Critical |

---

## 4. 可視化

### 4.1 空間分布 (x, y, z)

健全と欠陥サンプルの座標分布比較。

![Spatial comparison](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/figures/dataset_comparison/spatial_comparison.png)

### 4.2 応力分布 (DSPSS)

健全 vs 欠陥サンプル (sample_0001) の等価応力 (DSPSS) 分布。

![Stress comparison](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/figures/dataset_comparison/stress_comparison.png)

### 4.3 欠陥の空間配置 (sample_0001)

欠陥ラベル 1 のノードが x-z 平面にどのように分布するか。

![Defect spatial](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/figures/dataset_comparison/defect_spatial_sample_0001.png)

### 4.4 ジオメトリ (r-z 断面)

円筒座標 r = √(x²+y²) と z の関係。カラーマップは応力 (dspss) または温度 (temp)。

**健全ベースライン**

![r-z healthy](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/figures/dataset_comparison/r_z_healthy_baseline.png)

**欠陥サンプル (sample_0000)**

![r-z sample](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/figures/dataset_comparison/r_z_sample_0000.png)

---

## 5. 統計サマリー (JSON)

```json
{
  "n_healthy": 1,
  "n_defect": 20,
  "healthy_n_nodes": 22220,
  "defect_n_nodes_mean": 1818.75,
  "defect_n_nodes_std": 4680.37
}
```

- `defect_n_nodes_mean` は形式 A/B 混在の平均（形式 B が 745 で多数のため低め）
- 形式別では: 形式 A = 22,220, 形式 B = 745

---

## 6. 再現方法

```bash
# 比較解析スクリプト実行
python scripts/analyze_dataset_comparison.py --output_dir figures/dataset_comparison

# 出力
#   figures/dataset_comparison/comparison_stats.csv
#   figures/dataset_comparison/summary.json
#   figures/dataset_comparison/*.png
```

---

## 7. 関連ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| [Dataset-Format.md](Dataset-Format.md) | データセット仕様・CSV フォーマット |
| [docs/DEFECT_PLAN.md](../docs/DEFECT_PLAN.md) | 欠陥パラメータ計画 |
| [dataset_output/README.md](dataset_output/README.md) | データディレクトリ説明 |
