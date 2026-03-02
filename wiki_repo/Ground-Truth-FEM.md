[← Home](Home) | [FEM-Realism-Roadmap](FEM-Realism-Roadmap) | [S12-CZM-Dataset](S12-CZM-Dataset)

# Ground Truth FEM モデル — 推論検証用 ~90% リアリズム

> **作成日**: 2026-03-03
> **ファイル**: `src/generate_ground_truth.py` (2892行)
> **ベース**: `generate_czm_sector12.py` (CZM 1/12セクタ) + 4つの物理強化

---

## 1. 目的

GNN 学習後の**推論精度検証**に使う「物理的に最も正確な単一モデル」。
学習データ (S12 CZM, ~75%) よりも高い忠実度で、モデルの汎化性能を測定する。

---

## 2. 追加した物理 (+15% リアリズム)

| # | 項目 | ベース (S12 CZM) | Ground Truth | 効果 |
|---|------|------------------|--------------|------|
| 1 | **Cp(z) 空力圧力** | 全域 0 kPa | 10ゾーン分布, q∞=35kPa × Cp(z) | +5% |
| 2 | **ダブラー** | なし | z=1500,2500 に 16-ply (2mm) | +5% |
| 3 | **製造残留応力** | なし | ΔT=-155°C 硬化冷却ステップ | +3% |
| 4 | **オジーブ補剛** | なし | θ=15° 縦ストリンガー 1本 | +2% |
| 5 | **z依存外スキン温度** | 120°C 均一 | 100-221°C (10ゾーン) | (含む) |

### 2.1 Cp(z) 空力圧力分布

Modified Newtonian + H3 風洞データに基づく圧力係数:

```
z (mm)     Cp         P (MPa)    備考
──────────────────────────────────────────
0          -0.100     -0.0035    バレル底部 (負圧=吸引)
2600       -0.022     -0.0008    バレル中間
5000       +0.050     +0.0018    バレル-オジーブ接合部
7000       +0.356     +0.0125    オジーブ中間
9000       +0.699     +0.0245    オジーブ上部
10400      +1.000     +0.0350    ノーズ先端 (淀み点)
```

差圧 30 kPa (バレル内面) はそのまま維持。

### 2.2 ダブラー (16-ply 補強)

| ゾーン | z 範囲 | 備考 |
|--------|--------|------|
| z=1500 ±50mm | 1450–1550 mm | アクセスドア相当フレーム |
| z=2500 ±50mm | 2450–2550 mm | HVAC 相当フレーム |

通常 8-ply (1mm) → 16-ply [45/0/-45/90]₂ₛ (2mm)。Inner/Outer 両スキンに適用。

### 2.3 製造残留応力

```
オートクレーブ硬化温度: 175°C
室温: 20°C
ΔT = -155°C

CTE ミスマッチによる残留ひずみ:
  CFRP 繊維:  -0.3e-6 × (-155) = +0.047‰ (微小膨張)
  CFRP 横方:  28e-6 × (-155)   = -4.34‰  (大きな収縮)
  Al HC:      23e-6 × (-155)   = -3.57‰  (収縮)
```

### 2.4 オジーブストリンガー

```
位置: θ = 15° (セクタ中央)
範囲: z = 5000–9900 mm (バレル上端→ノーズ手前)
寸法: 高さ 30mm × 厚さ 2mm
材料: CFRP_FRAME (準等方性 70GPa)
接合: OuterSkin に Tie 拘束
```

---

## 3. 解析ステップ構成

ベース (2ステップ) から **4ステップ** に拡張:

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────────┐
│ Initial  │───→│ Step-Cure│───→│ Step-Thermal │───→│Step-Mechanical│
│ T=175°C  │    │ T→20°C   │    │ z依存加熱     │    │ Cp(z)+差圧+3G │
│ (全パーツ) │    │ (残留応力) │    │ outer:100-221│    │               │
└──────────┘    └──────────┘    └──────────────┘    └───────────────┘
```

| ステップ | 目的 | 温度条件 | 機械荷重 |
|---------|------|---------|---------|
| Initial | BC設定 | 全パーツ 175°C | — |
| Step-Cure | 硬化冷却 (残留応力生成) | 全パーツ → 20°C | — |
| Step-Thermal | 運用加熱 | Outer: z依存, Inner: 20°C, Core: 70°C | — |
| Step-Mechanical | 全荷重 | (維持) | Cp(z) + 差圧 30kPa + 3G |

---

## 4. リアリズムスコア更新

```
                以前(Tie)    S12 CZM     +熱+差圧     Ground Truth
               ━━━━━░░░    ━━━━━━━━░░  ━━━━━━━━━░░  ━━━━━━━━━━━━━░░
構造再現度:      ~30%         ~60%        ~60%         ~72%  ← ダブラー+ストリンガー
荷重再現度:      ~20%         ~45%        ~65%         ~82%  ← Cp(z)+残留応力
接着物理:        ~10%         ~70%        ~70%         ~70%
──────────────────────────────────────────────────────────────────
総合:            ~25%         ~60%        ~75%         ~90%  ← Ground Truth
```

### 残り ~10% (beyond scope)

| 項目 | 必要技術 | 備考 |
|------|---------|------|
| 音響荷重 (~148 dB) | ランダム圧力場 | Explicit 解析必要 |
| 分離衝撃 (1500G) | Explicit + 接触 | 衝撃解析 |
| 過渡熱 (0→60s) | 熱伝導解析 | 連成必要 |
| ボルト接合 | 点結合 | パラメトリック |

→ 論文で "beyond scope of static analysis" と明記可能。

---

## 5. 使い方

```bash
cd abaqus_work

# Healthy ground truth (INP のみ)
abaqus cae noGUI=../src/generate_ground_truth.py -- --job Job-GT-H --healthy --no_run

# Debonding defect
abaqus cae noGUI=../src/generate_ground_truth.py -- --job Job-GT-D001 \
  --defect '{"defect_type":"debonding","theta_deg":15,"z_center":3000,"radius":200}'

# ODB 抽出 (既存スクリプトそのまま使える)
abaqus python ../src/extract_odb_results.py Job-GT-H/results
```

### extract_odb_results.py との互換性

| Step | 抽出される応力 | 対応フィールド |
|------|--------------|---------------|
| Step-Thermal | 熱応力 (残留+運用加熱) | `thermal_smises` |
| Step-Mechanical | 全荷重応力 | `smises`, `s11`, `s22`, `s12` |

**注意**: Step-Cure の残留応力は Step-Thermal の初期状態に含まれるため、個別抽出は不要。

---

## 6. パーツ構成

```
InnerSkin (shell, 8-ply / 16-ply doubler)
  ↓ Tie
AdhesiveInner (COH3D8, 0.2mm)
  ↓ Tie
Core (C3D10, 37.6mm)
  ↓ Tie
AdhesiveOuter (COH3D8, 0.2mm)
  ↓ Tie
OuterSkin (shell, 8-ply / 16-ply doubler)
  ↓ Tie
Stringer (shell, 2mm, ogive only)   ← NEW

+ Ring Frame × 9 (Tie → InnerSkin)
```

---

## 7. 検証計画

1. **INP 生成**: `--no_run` で構文チェック
2. **1 healthy** を frontale02 でソルバ実行
3. **ODB 抽出**: CSV のカラム数・ノード数確認
4. **残留応力確認**: Step-Cure 後の Mises 応力分布
5. **Cp(z) 確認**: Step-Mechanical の圧力分布可視化
6. **1 debonding** で推論テスト
