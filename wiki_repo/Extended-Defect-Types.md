[← Home](Home)

# 拡張欠陥タイプ — Extended Defect Types

> 最終更新: 2026-02-28  
> 7種類の欠陥モードを FEM で再現。**CFRP 専門家（教授レベル）による学術的妥当性**を確保。  
> 学術的根拠: プロジェクトルート `docs/DEFECT_MODELS_ACADEMIC.md`  
> 検証: [Defect-Types-Validation](Defect-Types-Validation)

---

## 1. 実装済み欠陥一覧（学術文献準拠）

| ID | 名称 | 界面/発生要因 | FEM モデル化 | 主な文献 |
|----|------|----------------|--------------|----------|
| 0 | **healthy** | 欠陥なし | 欠陥挿入なし | — |
| 1 | **debonding** | 外スキン-コア | 剛性 1%（荷重伝達喪失） | NASA NTRS 20160005994 |
| 2 | **fod** | コア内異物 | 剛性 5–20倍（硬質異物） | MDPI Appl. Sci. 2024 |
| 3 | **impact** | 衝撃損傷 (BVID) | マトリクス劣化 + コア圧潰 | Composites Part B 2017, ASTM D7136 |
| 4 | **delamination** | 積層内層間剥離 | せん断剛性低下 | Compos. Sci. Technol. 2006, MDPI Materials 2019 |
| 5 | **inner_debond** | 内スキン-コア | 内スキン剛性低下 | NASA NTRS, DEFECT_PLAN |
| 6 | **thermal_progression** | CTE 不整合 | 剛性低下 + CTE 8e-6 | Composites Part B 2018 |
| 7 | **acoustic_fatigue** | 音響疲労 (147–148 dB) | 疲労による剛性低下 | UTIAS Acoustic Fatigue 2019 |

---

## 2. 各欠陥のパラメータ（学術的根拠）

| 欠陥 | 追加パラメータ | 範囲 | 根拠 |
|------|----------------|------|------|
| debonding | — | — | 円形パッチ（製造不良典型）。界面剥離で荷重伝達ほぼ喪失。 |
| fod | stiffness_factor | 5.0–20.0 | 金属系 FOD の局所剛性増加。CTE 12e-6。 |
| impact | damage_ratio | 0.1–0.5 | BVID 残留剛性。CAI 強度 30–50% 低下に整合。 |
| delamination | delam_depth | 0.2–0.8 | 層間剥離の影響を受ける積層比。有効せん断剛性低下。 |
| inner_debond | — | — | debonding と同様の力学。内界面に適用。 |
| thermal_progression | — | — | CFRP (−0.3) vs Al (23) ×10⁻⁶/°C の CTE 不整合。 |
| acoustic_fatigue | fatigue_severity | 0.2–0.5 | 147–148 dB による界面疲労。残留剛性で表現。 |

---

## 3. DOE 生成

```bash
# 全7種の欠陥を含む DOE
python src/generate_doe.py --n_samples 70 --output doe_extended.json

# 新規4種のみ
python src/generate_doe.py --n_samples 20 \
  --defect_types delamination inner_debond thermal_progression acoustic_fatigue \
  --output doe_new4.json
```

デフォルト割合: debonding 25%, fod 15%, impact 15%, delamination 15%, inner_debond 10%, thermal_progression 10%, acoustic_fatigue 10%

---

## 4. 検証

```bash
# 拡張欠陥タイプの整合性検証
python scripts/validate_defect_types.py

# 高速（データセット検証スキップ）
python scripts/validate_defect_types.py --skip-dataset
```

詳細は [Defect-Types-Validation](Defect-Types-Validation) を参照。

---

## 5. 学術的妥当性（CFRP 専門家向け）

全欠陥モデルは **等価剛性法（Equivalent Stiffness Reduction）** に基づき、以下で採用されている手法と整合する:

- **NASA NTRS**: Face Sheet/Core Disbond、曲面板の座屈解析
- **Composites Part A/B, Compos. Sci. Technol.**: マトリクス割れ・層間剥離の剛性低下
- **ASTM D7136/D7137**: BVID・CAI 試験に基づく残留剛性
- **UTIAS, MDPI**: 音響疲労、FOD 特性

詳細な文献・数値根拠は **docs/DEFECT_MODELS_ACADEMIC.md** を参照。教授レベルでのレビューに耐える記述とする。

---

## 6. 関連

| ページ | 内容 |
|--------|------|
| [Defect-Generation-and-Labeling](Defect-Generation-and-Labeling) | 欠陥生成・ラベル付け |
| [Defect-Types-Validation](Defect-Types-Validation) | 整合性検証 |
| [Defect-Physics-Validation](Defect-Physics-Validation) | 物理量検証・可視化 |
| [Multi-Class-Roadmap](Multi-Class-Roadmap) | マルチクラス分類ロードマップ |
| docs/DEFECT_MODELS_ACADEMIC.md | 学術的根拠・文献一覧 |
