# 欠陥モデルの学術的根拠 — Academic Justification for Defect Models

> **対象読者**: CFRP 複合材・航空宇宙構造の専門家（教授レベル）  
> **目的**: 全欠陥タイプの FEM モデル化が学術文献・産業基準に基づくことを示す。  
> **参照**: NASA NTRS, Composites Science and Technology, ASTM, JAXA H3 調査

---

## 1. 総論

本プロジェクトの欠陥モデルは、**等価剛性法（Equivalent Stiffness Reduction）** に基づく。CZM（Cohesive Zone Model）や VCCT による厳密な破壊力学解析の代わりに、損傷ゾーンを**劣化した材料定数**で表現する。この手法は以下で広く採用されている:

- NASA NTRS 20160005994: Face Sheet/Core Disbond Growth
- ASTM D7136/D7136M: BVID 試験
- 航空宇宙 FEM 研究におけるグローバル-ローカル解析

**限界**: 破壊進展・モードミクスティの詳細は CZM で扱うべき。本モデルは **SHM 用 ML データ生成** を目的とし、欠陥ゾーンの応力・変位パターンを物理的に妥当な範囲で再現する。

---

## 2. 各欠陥タイプの学術的根拠

### 2.1 debonding（外スキン-コア界面剥離）

| 項目 | 値 | 根拠 |
|------|-----|------|
| **剛性低下** | E, G を 1% に | 界面剥離により荷重伝達がほぼ喪失。NASA NTRS 20160005994 では disbond 領域を接触のみ（Tie 解除）でモデル化。等価剛性法では「荷重を伝えない」≈ 極小剛性。 |
| **形状** | 円形パッチ | 製造不良の典型的モデル化。MDPI Appl. Sci. 2020, Honeycomb Sandwich with Face/Core Debonding。 |
| **文献** | NASA NTRS 20160005994, NTRS 20150019391, Composites Part B (2018) S0263822318322347 |

**F8 事故**: JAXA 製造検査で PSS 4個全てに想定を超える剥離を確認。製造時接着不良が有力要因。

---

### 2.2 fod（Foreign Object Debris / 異物混入）

| 項目 | 値 | 根拠 |
|------|-----|------|
| **剛性増加** | コア E, G を 5–20 倍 | 硬質異物（金属片、工具片）がハニカムセル内に混入した場合、局所剛性が著増。Appl. Sci. 2024 "Finite Element-Inspired Method to Characterize FOD in Carbon Fiber Composites" で FOD の剛性・形状の影響を解析。 |
| **CTE** | 12×10⁻⁶/°C | 金属系異物の典型値。Al 23, 鋼 11–12。コアとの CTE 差で熱応力が生じうる。 |
| **密度** | 200 kg/m³ | ハニカム 50 kg/m³ に対し、金属異物は 4 倍程度。 |
| **文献** | MDPI Appl. Sci. 2024 16(3):1459, Int. J. Impact Eng. (2005) |

---

### 2.3 impact（衝撃損傷 / BVID）

| 項目 | 値 | 根拠 |
|------|-----|------|
| **スキン** | E1×0.7, E2×dr, G×dr (dr=0.1–0.5) | BVID では繊維は部分的に保持、マトリクスは著しく劣化。Composites Part B (2017) "Failure mechanisms and damage evolution of laminated composites under CAI" でマトリクス破壊・層間剥離が主損傷と報告。 |
| **コア** | E3×0.1, G×0.1 | ハニカムの圧潰（cell buckling）。Composites Part A (2019) "Damage behavior of honeycomb sandwich panel under low-velocity impact" でコア圧潰が観察。 |
| **damage_ratio** | 0.1–0.5 | 残留剛性率。BVID 試験（ASTM D7136）で CAI 強度が 30–50% 低下する事例に整合。 |
| **文献** | Composites Part B 2017, Composites Part A 2019, ASTM D7136/D7137 |

---

### 2.4 delamination（積層内層間剥離）

| 項目 | 値 | 根拠 |
|------|-----|------|
| **せん断剛性** | G12, G13, G23 × (1−depth) | 層間剥離により interlaminar shear の負荷経路が断たれる。Composites Science and Technology (2006) "Effective interlaminar shear moduli in composites containing transverse ply cracks" で有効せん断剛性の低下を解析。 |
| **E2** | 0.3 + 0.5×(1−depth) | マトリクス割れに伴う横弾性係数低下。Multi-directional stiffness degradation (Composites Part A 2001)。 |
| **delam_depth** | 0.2–0.8 | 影響を受ける積層の割合。MDPI Materials 2019 "Relationship Between Matrix Cracking and Delamination in CFRP Cross-Ply Laminates" で層間剥離とマトリクス割れの関係を報告。 |
| **文献** | Compos. Sci. Technol. 2006, Compos. Part A 2001, MDPI Materials 2019 12(23):3990 |

---

### 2.5 inner_debond（内スキン-コア接着不良）

| 項目 | 値 | 根拠 |
|------|-----|------|
| **モデル** | debonding と同様（剛性 1%） | 界面は異なるが力学は同一。内スキン-コア界面も OoA 接着で形成。DEFECT_PLAN で「次点」として明記。 |
| **荷重経路** | 内圧・クランプ拘束 | 内スキンは外圧に対する反力、クランプ付近の応力伝達に寄与。剥離で荷重経路が断たれる。 |
| **文献** | NASA NTRS 20160005994（両界面の disbond を扱う）, DEFECT_PLAN.md |

---

### 2.6 thermal_progression（熱応力による進展）

| 項目 | 値 | 根拠 |
|------|-----|------|
| **剛性** | E, G × 0.05 | CTE 不整合（CFRP ≈ −0.3×10⁻⁶ vs Al ≈ 23×10⁻⁶/°C）により界面に熱応力が蓄積。繰返し熱サイクルで界面き裂が進展。 |
| **CTE** | 8×10⁻⁶/°C | 界面が「開いた」状態の等価表現。健全 CFRP 2×10⁻⁶ より高く、熱負荷でより大きな変形を生じさせる。 |
| **文献** | "Interfacial toughening and bending performance of CFRP/aluminum-honeycomb sandwich", "Numerical fracture analysis for disbonded honeycomb core sandwich" (Composites Part B 2018) |

---

### 2.7 acoustic_fatigue（音響疲労）

| 項目 | 値 | 根拠 |
|------|-----|------|
| **負荷** | 打ち上げ 147–148 dB | H3 フェアリング設計荷重。高 SPL による振動応答が界面に疲労損傷を蓄積。 |
| **剛性** | E×(0.2+0.5×(1−sev)), G×sev | 疲労による界面弱化。UTIAS "Acoustic Fatigue Research for Honeycomb Sandwich with Impact Damage" で、衝撃損傷が音響疲労寿命を著しく短縮（35–40% 強度低下）と報告。 |
| **fatigue_severity** | 0.2–0.5 | 残留剛性。サイクル数・SPL に依存。 |
| **文献** | UTIAS Acoustic Fatigue 2019, Compos. Part A 2004 "Acoustic emission based tensile characteristics of sandwich composites" |

---

## 3. 材料定数・座標系

| 定数 | CFRP 健全 | 単位 | 根拠 |
|------|-----------|------|------|
| E1 | 160 GPa | MPa | Toray T1000G クラス |
| E2 | 10 GPa | MPa | 同上 |
| G12, G13 | 5 GPa | MPa | 同上 |
| G23 | 3 GPa | MPa | 同上 |
| α | 2×10⁻⁶ | /°C | 繊維方向 |
| コア E3 | 1000 MPa | MPa | Al ハニカム out-of-plane |
| コア G13, G23 | 400, 240 MPa | MPa | せん断 |

---

## 4. 検証・限界

- **検証済み**: 熱荷重（外板 120°C）による応力・変位が物理的に妥当（verify_odb_thermal.py）。
- **限界**: 破壊進展・疲労き裂進展は含まない。静的等価剛性のみ。
- **今後の拡張**: CZM による進展解析、Explicit 動解析による衝撃再現。

---

## 5. 参考文献一覧

1. NASA NTRS 20160005994: Face Sheet/Core Disbond Growth in Honeycomb Sandwich Panels
2. NASA NTRS 20150019391: Effect of Facesheet-Core Disbonds on Buckling Load
3. NASA NTRS 20180006522: 3D Simulations for Mode I Delamination
4. Composites Part B 2018, S0263822318322347: Numerical fracture analysis for disbonded honeycomb
5. Composites Science and Technology 2006: Effective interlaminar shear moduli with transverse cracks
6. Composites Part A 2001: Multi-directional stiffness degradation by matrix cracking
7. MDPI Materials 2019 12(23):3990: Matrix cracking and delamination in CFRP cross-ply
8. MDPI Appl. Sci. 2024 16(3):1459: FOD characterization in carbon fiber composites
9. UTIAS 2019: Acoustic fatigue for honeycomb sandwich with impact damage
10. ASTM D7136/D7137: BVID and CAI testing
11. JAXA H3 F8 調査報告
12. KHI ANSWERS: H3 フェアリング製造技術
