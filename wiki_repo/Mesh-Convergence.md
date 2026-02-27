# メッシュ収束チェック

> **目的**: 欠陥検出精度に対するメッシュサイズの影響を定量評価する
> 固定欠陥 (θ=30°, z=2500 mm, r=50 mm) に対し、要素サイズ h を変化させて比較

---

## 収束チェック結果

![Mesh Convergence Check](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/mesh_convergence.png)

---

## 数値データ

| h [mm] | ノード数 | 欠陥ノード数 | h/D 比 | ステータス |
|-------:|---------:|------:|------:|:----------|
| 100 | ~2,668 | ~0 | 1.0 | 未生成 (推定値) |
| 50 | 10,897 | 2 | 0.5 | OK |
| 25 | 42,892 | 12 | 0.25 | OK |
| 12 | 193,587 | 56 | 0.12 | OK |
| 10 | ~277,146 | ~90 | 0.1 | 失敗 (数値特異点) |

> **h/D**: 要素サイズ / 欠陥直径 (D = 2r = 100 mm)

---

## べき乗則フィッティング

| 量 | フィット結果 | 理論値 | 備考 |
|:---|:---|:---|:---|
| 総ノード数 N | N ∝ h⁻²·⁰² | h⁻² | 2Dメッシュの理論通り |
| 欠陥ノード数 N_d | N_d ∝ h⁻²·³³ | h⁻² | 局所細分化の影響でやや急峻 |

---

## 解像度品質の基準: h/D 比

| h/D 範囲 | 品質 | 欠陥ノード数 (r=50mm) | 判定 |
|:---|:---|:---|:---|
| < 0.25 | Excellent | 12+ ノード | GNN学習に十分 |
| 0.25 – 0.5 | Good | 3–12 ノード | 最低限の検出可能 |
| ≥ 0.5 | Poor | 0–2 ノード | 検出不可能 |

---

## 考察

### h=50mm (現行メッシュ)
- 欠陥ノードが **2個のみ** — r=50mm の欠陥でも h/D=0.5 でギリギリ
- Small tier (r < 50mm) の欠陥は検出不可能
- 大規模データセット生成の **計算速度** を優先した妥協

### h=25mm (理想メッシュ)
- 欠陥ノード **12個** — GNN学習に十分な解像度
- ノード数は 50mm の約4倍 (42,892 vs 10,897)
- Medium tier 以上の欠陥を確実に検出

### h=12mm (高精度メッシュ)
- 欠陥ノード **56個** — 欠陥形状を高精度に捉える
- ノード数は 50mm の約18倍 (193,587)
- 計算コストが大きいが、Ground Truth 生成に有用

### h=10mm (失敗)
- Abaqus ソルバーで **数値特異点 (numerical singularity)** が発生
- 要素が細かすぎてコア-スキン界面の要素品質が低下した可能性
- 要素アスペクト比の改善 or ソルバー設定の調整が必要

### h=100mm (未実行)
- 推定ノード数 ~2,700 — 欠陥検出は不可能
- 参考のためのデータポイントとして将来生成可能

---

## 結論

| 用途 | 推奨メッシュ | 理由 |
|:---|:---|:---|
| **大規模データセット** (100+ samples) | h=50mm | 計算速度重視、r≥100mm の欠陥は検出可能 |
| **高品質データセット** | h=25mm | h/D < 0.25 で Medium 以上の欠陥を確実に検出 |
| **Ground Truth** | h=12mm | 最高精度、少数サンプルの検証用 |

---

## 再現スクリプト

```bash
# 収束チェックグラフの生成
python scripts/plot_convergence.py

# 個別メッシュの FEM 実行
python src/run_batch.py --doe doe_ideal.json --output_dir dataset_output_ideal         # 25mm
python src/run_batch.py --doe doe_ideal_12mm.json --output_dir dataset_output_ideal_12mm # 12mm
python src/run_batch.py --doe doe_ideal_50mm.json --output_dir dataset_output_ideal_50mm # 50mm (参照)
```

---

**関連ページ**: [Dataset-Format](Dataset-Format) | [Mesh-Defect-Resolution](Mesh-Defect-Resolution) | [Dataset-Generation-Status](Dataset-Generation-Status)
