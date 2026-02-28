# 拡張欠陥タイプ — Extended Defect Types

> 7種類の欠陥モードを FEM で再現。2026-02-28 実装。

---

## 1. 実装済み欠陥一覧

| ID | 名称 | 界面/発生要因 | FEM モデル化 |
|----|------|----------------|--------------|
| 0 | **healthy** | 欠陥なし | 欠陥挿入なし |
| 1 | **debonding** | 外スキン-コア | 剛性低下 (CFRP 1%) |
| 2 | **fod** | コア内異物 | 剛性増加 (5–20倍) |
| 3 | **impact** | 衝撃損傷 | マトリクス劣化 + コア圧潰 |
| 4 | **delamination** | 積層内層間剥離 | せん断剛性低下 (delam_depth) |
| 5 | **inner_debond** | 内スキン-コア | 内スキン剛性低下 |
| 6 | **thermal_progression** | CTE 不整合 | 剛性低下 + 熱膨張率変化 |
| 7 | **acoustic_fatigue** | 音響疲労 (147–148 dB) | 疲労による剛性低下 |

---

## 2. 各欠陥のパラメータ

| 欠陥 | 追加パラメータ | 範囲 |
|------|----------------|------|
| debonding | — | — |
| fod | stiffness_factor | 5.0–20.0 |
| impact | damage_ratio | 0.1–0.5 |
| delamination | delam_depth | 0.2–0.8 (影響を受ける積層比) |
| inner_debond | — | — |
| thermal_progression | — | — |
| acoustic_fatigue | fatigue_severity | 0.2–0.5 (残留剛性) |

---

## 3. DOE 生成

```bash
# 全7種の欠陥を含む DOE (デフォルト割合)
python src/generate_doe.py --n_samples 70 --output doe_extended.json

# 特定タイプのみ
python src/generate_doe.py --n_samples 30 --defect_types delamination inner_debond thermal_progression acoustic_fatigue --output doe_new4.json
```

デフォルト割合: debonding 25%, fod 15%, impact 15%, delamination 15%, inner_debond 10%, thermal_progression 10%, acoustic_fatigue 10%

---

## 4. 学術的根拠

全欠陥の FEM モデル化は **docs/DEFECT_MODELS_ACADEMIC.md** に文献・数値根拠を記載。CFRP 専門家（教授レベル）によるレビューに耐える記述とする。

## 5. 関連ファイル

- `src/generate_fairing_dataset.py` — 欠陥材料・セクション・割り当て（文献参照付き）
- `src/generate_doe.py` — DOE パラメータ生成
- `src/extract_odb_results.py` — DEFECT_TYPE_MAP
- `src/train.py` — DEFECT_TYPE_NAMES
