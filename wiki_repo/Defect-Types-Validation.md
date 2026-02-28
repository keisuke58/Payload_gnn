[← Home](Home)

# 欠陥タイプ検証 — Defect Types Validation

> 最終更新: 2026-02-28  
> 拡張欠陥タイプ（7種）の整合性を検証する手順とチェック項目。

---

## 1. 概要

`scripts/validate_defect_types.py` により、以下を検証する:

- **Consistency**: DEFECT_TYPE_MAP / DEFECT_TYPES / DEFECT_TYPE_NAMES の一貫性
- **DOE**: 全7種で DOE が正常生成されるか
- **Params**: TYPE_PARAM_RANGES の定義
- **Extract**: extract_odb が新パラメータを metadata に書き出すか
- **Dataset**: 既存データセットで verify_dataset_integrity が通るか（オプション）

---

## 2. 実行方法

```bash
# 全チェック（データセット検証含む、dataset_multitype_100 がある場合）
python scripts/validate_defect_types.py

# 高速実行（データセット検証スキップ、CI 向け）
python scripts/validate_defect_types.py --skip-dataset

# レポート保存
python scripts/validate_defect_types.py --report validation_report.txt
```

---

## 3. チェック項目

| チェック | 内容 |
|----------|------|
| Consistency.DEFECT_TYPE_MAP | extract_odb の DEFECT_TYPE_MAP と generate_doe の DEFECT_TYPES が一致 |
| Consistency.DEFECT_TYPE_NAMES | train.py の DEFECT_TYPE_NAMES が healthy + 7 types |
| DOE.Generation | generate_doe が全タイプで正常にサンプル生成 |
| DOE.Params | 各サンプルに theta_deg, z_center, radius が存在 |
| DOE.Type_coverage | 全7種が少なくとも1件ずつ DOE に含まれる |
| Params.TYPE_PARAM_RANGES | fod, impact, delamination, acoustic_fatigue の型別パラメータ定義 |
| Extract.Metadata_params | delam_depth, fatigue_severity を metadata に出力 |
| Dataset.Multiclass_verify | verify_dataset_integrity が multi-class defect_label で通過 |

---

## 4. verify_dataset_integrity の multi-class 対応

`scripts/verify_dataset_integrity.py` は `defect_label` が 0–7 の multi-class をサポート:

- **0**: healthy
- **1–7**: 各欠陥タイプ

欠陥ノード数は `(defect_label != 0).sum()` で算出。

---

## 5. 関連

| ページ | 内容 |
|--------|------|
| [Extended-Defect-Types](Extended-Defect-Types) | 拡張欠陥タイプ一覧 |
| [Defect-Physics-Validation](Defect-Physics-Validation) | 物理量検証・可視化・数値分析 |
| [Defect-Occurrence-Probability-and-Dataset-Ratio](Defect-Occurrence-Probability-and-Dataset-Ratio) | 発生確率・データセット割合 |
| [Defect-Generation-and-Labeling](Defect-Generation-and-Labeling) | 欠陥生成 |
| [Dataset-Format](Dataset-Format) | データ形式 |
