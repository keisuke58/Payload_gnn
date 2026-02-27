# 健全ベースライン (Healthy Baseline) 検証チェックリスト

> **欠陥挿入前に全項目 PASS であることを確認すること。**
> 健全データの精度が GNN 学習の成否を決める。

## 検証の重要性

- デボンディング検出は **健全 vs 欠陥** の差分で学習する
- 健全ベースラインに誤りがあると、全サンプルで系統的バイアスが発生
- 欠陥なしの精度 (Specificity) が極めて重要

## 検証項目一覧 (20項目)

| カテゴリ | 項目 | 内容 |
|----------|------|------|
| **Files** | nodes_csv | nodes.csv が存在 |
| | elements_csv | elements.csv が存在 |
| **Geometry** | Z_range | z 範囲が 0–HEIGHT (Barrel-only または Full) |
| | R_range | 半径が R_INNER–R_OUTER |
| | Theta_sector | 1/6 セクション (θ=0–60°) |
| **Integrity** | No_NaN | NaN なし |
| | No_Inf | Inf なし |
| | No_duplicate_nodes | node_id 重複なし |
| | Element_nodes_exist | 全要素の節点が nodes.csv に存在 |
| **Labels** | All_zero | defect_label が全て 0 |
| **Physics** | Stress_magnitude | 応力が妥当範囲 (MPa) |
| | DSPSS_range | 変位が妥当範囲 |
| **Metadata** | n_nodes | メタデータと実数一致 |
| | n_elements | メタデータと実数一致 |
| | n_defect_zero | n_defect_nodes=0 |
| | defect_type | defect_type=healthy |
| **Parts** | Skin_Outer | 外スキン節点あり |
| | Skin_Inner | 内スキン節点あり |
| | Core | コア節点あり |
| **Preprocess** | PyG_convert | PyG Data 変換成功 |

## 実行方法

```bash
# 検証実行
python scripts/validate_healthy_baseline.py

# レポート保存
python scripts/validate_healthy_baseline.py --report figures/healthy_validation_report.txt

# 厳格モード (1つでも FAIL で exit 1)
python scripts/validate_healthy_baseline.py --strict
```

## 最新検証結果 (2026-02-28)

```
Result: 20/20 passed - ALL PASS
```

- データ: `dataset_output/healthy_baseline`
- モード: Barrel-only (z=0–5000 mm)
- 節点: 22,220 (Skin_Outer 5,555, Skin_Inner 5,555, Core 11,110)
- 要素: 16,200

## デボンディング挿入前の確認フロー

1. `validate_healthy_baseline.py` を実行 → **全項目 PASS**
2. `visualize_fairing_h3_check.py` で形状確認
3. `visualize_fairing_3d.py` で 3D 確認
4. 上記完了後に `run_batch.py` でデボンディングサンプル生成
