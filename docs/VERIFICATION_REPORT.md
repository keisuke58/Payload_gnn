# 動作確認レポート — 2026-02-28

## 実施した改善

### 1. 熱荷重の INP パッチ
- `scripts/patch_inp_thermal.py` を追加
- `writeInput` 後に INP をパッチし、`*Initial Conditions, type=TEMPERATURE` と `*Temperature` (Step-1) を注入
- 初期温度 20°C、Step-1 で外板 120°C・内板 20°C を適用

### 2. 座標系の修正
- **Abaqus Revolve**: Y=軸方向、XZ=半径方向
- パーティション: `XZPLANE` で y=constant の平面を使用
- BC (Fix_Bottom): `pointOn[0][1] < 1` (y 方向)
- `is_face_in_defect_zone`: 軸方向に y、半径に sqrt(x²+z²) を使用
- `extract_odb_results._is_node_in_defect`: 同様の座標系に修正

### 3. パッチ正規表現の修正
- `*Static` 行の正規表現を修正し、温度ブロックの誤挿入を防止

## 結果サマリ

| 項目 | 結果 |
|------|------|
| 解析完了 | ✅ 正常終了 (ANALYSIS COMPLETE) |
| 初期温度 (IC) | ✅ 20°C が適用されている |
| Step-1 温度 | ✅ 外板 120°C が ODB に正しく反映 |
| 変位 (U) | ✅ 熱応力による非ゼロ変位を確認 |

## 検証実施結果 (2026-02-28)

`scripts/verify_odb_thermal.py` により ODB を直接検証：

```
=== Displacement (U) ===
  Ux: min=-1.016e+00 max=4.568e+00 mean=1.829e+00
  Uy: min=-2.611e-01 max=3.418e+00 mean=9.341e-01
  Uz: min=-4.469e-01 max=2.310e+00 mean=1.161e+00

=== Temperature (TEMP) ===
  min=120.00 C  max=120.00 C  mean=120.00 C
  [OK] Outer skin ~120C applied
```

- **温度**: 外板 Outer Skin に 120°C が正しく適用されている
- **変位**: 熱応力による変形が得られている（Ux/Uy/Uz いずれも非ゼロ）

## nodes.csv の温度列について

- **対応済み**: `extract_odb_results` が NT11（節点温度）を参照するよう修正
- **結果**: nodes.csv に temp=120°C が正しく出力される

## 残課題

1. **Tie 除外**: 剥離の物理的再現は未実装（Contact/CZM の検討が必要）

## 実行コマンド

```bash
# 欠陥あり
abaqus cae noGUI=src/generate_fairing_dataset.py -- --param_file defect_params.json --job_name Job-Verify-Defect

# ODB 検証（温度・変位の確認）
abaqus python scripts/verify_odb_thermal.py --odb Job-Verify-Defect.odb

# ODB 抽出
abaqus python src/extract_odb_results.py --odb Job-Verify-Defect.odb --output verify_defect --defect_json defect_params.json
```
