[<- Home](Home)

# Realistic H3 Fairing FEM Model

> **Status**: Phase 1 & Phase 2 COMPLETED
> **Date**: 2026-03-01

## 1. Overview

[JAXA-Fairing-Specs Section 3.1](JAXA-Fairing-Specs) に基づき、開口部・リングフレーム・Tie 拘束を含むリアリスティック FEM モデルを構築。
既存の `generate_fairing_dataset.py`（ML 用簡易モデル）とは独立した高忠実度モデル。

| 項目 | 仕様 |
|------|------|
| スクリプト | `src/generate_realistic_fairing.py` |
| ソルバー | Abaqus/Standard 2024 |
| ジオメトリ | 1/6 セクタ (60°), バレル 5000mm + タンジェントオジーブ 5400mm |
| 構造 | CFRP/Al-HC サンドイッチ (Face 1.0mm + Core 38mm) |
| 拘束 | Tie (InnerSkin ↔ Core ↔ OuterSkin + RingFrames) |
| 荷重 | 熱勾配 (Outer 120°C, Inner 20°C, Core 70°C) + **Max Q 圧力 (50 kPa)** |
| BC | 底面 (y=0) 固定 |

## 2. Phase 構成

### Phase 1: AccessDoor + Ring Frames
- **開口部**: AccessDoor φ1,300mm (z=1500, θ=30°)
- **リングフレーム**: 6 本 (z=500, 2500, 3000, 3500, 4000, 4500)
  - z=1000, 1500, 2000 はAccessDoor と干渉するためスキップ
- **メッシュ**: 152,752 nodes / 131,133 elements

### Phase 2: All Openings + Ring Frames
- **開口部**: 5 個
  - AccessDoor φ1,300mm (z=1500, θ=30°)
  - HVAC Door φ400mm (z=2500, θ=20°)
  - RF Window φ400mm (z=4000, θ=40°)
  - Vent Hole ×2 φ100mm (z=300, θ=15° / θ=45°)
- **リングフレーム**: 4 本 (z=500, 3000, 3500, 4500)
- **メッシュ**: 237,284 nodes / 198,567 elements

### Phase 3 (Future): Doublers + Defect Integration
- 開口周辺ダブラー (16 ply CFRP)
- 既存欠陥生成ロジック統合
- 対称境界条件精密化

## 3. Key Technical Decisions

### 3.1 Opening Implementation — Void Section Method
開口部は DatumPlane パーティション (4 面/開口) で境界を作成し、開口内部に**ボイドセクション** (E=1 MPa, t=0.01mm) を割り当て。

- Un-sectioned faces は Abaqus メッシュ生成でスキップされない → 全要素にプロパティ付与が必要
- ボイド材料 (E=1 MPa vs CFRP E=160,000 MPa) で剛性比 ~10⁻⁸ → 構造的に「穴」と等価
- 開口領域はアイソトロピック → MaterialOrientation 不要でエラー回避

### 3.2 Ring Frame Modeling
- 各フレーム: 独立 ShellRevolve パーツ (r_inner → r_outer, 60° 弧)
- 高さ 50mm, 厚さ 3mm, CFRP アイソトロピック簡略化
- InnerSkin に **Tie** 拘束 (positionToleranceMethod=COMPUTED)
- 開口部と干渉するフレームは自動スキップ

### 3.3 Multi-Resolution Mesh (5-Tier Seeding)
| Tier | 領域 | シードサイズ | 備考 |
|------|------|------------|------|
| 1 | グローバル | 25 mm | S4R / C3D8R |
| 2 | リングフレーム | 15 mm | 各フレーム個別 |
| 3 | 開口周辺 | 10 mm | `seedEdgeBySize`, margin 可変 |
| 4 | 欠陥ゾーン | 10 mm | 欠陥中心 ± (r_def + 150mm) |
| **4b** | **パーティション境界** | **4 mm** | **境界 ± 30mm, `constraint=FINER`** |

> Tier 4b はパーティション境界付近のスライバー要素防止のための局所リファイン。
> 詳細: [[Mesh-Defect-Resolution]]

## 4. Results

### 4.1 Numerical Summary

| 指標 | Phase 1 | Phase 2 |
|------|---------|---------|
| von Mises 中央値 | 0.39 MPa | 4.46 MPa |
| von Mises 95%ile | 17.0 MPa | 17.3 MPa |
| von Mises 最大 | 93.7 MPa | 94.6 MPa |
| 変位 平均 | 2.81 mm | 3.03 mm |
| 変位 最大 | 29.6 mm | 27.4 mm |
| 温度範囲 | 0–120°C | 0–120°C |
| ODB サイズ | 44 MB | 66 MB |

### 4.2 von Mises Stress Comparison

![Stress Comparison](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/realistic_fairing/stress_comparison.png)

- 開口縁部で明確な応力集中（赤色帯）
- Phase 2 は複数開口の相互作用で全体的に応力中央値が上昇 (0.39 → 4.46 MPa)
- 最大応力は両者ほぼ同等 (~94 MPa) — AccessDoor 縁が支配的

### 4.3 Displacement

![Displacement Comparison](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/realistic_fairing/displacement_comparison.png)

![Displacement Profile](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/realistic_fairing/displacement_profile.png)

- ノーズ先端で最大変位 (~30mm) — 熱膨張の累積
- Phase 2 はバレル中央部の変位がやや大きい（複数開口による剛性低下）
- リングフレーム位置で局所的な変位抑制効果

### 4.4 Temperature Distribution

![Temperature](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/realistic_fairing/temperature_comparison.png)

- 外皮 120°C / 内皮 20°C の熱勾配が正しく適用
- 開口部 (void) は温度 0°C（膨張ゼロ）

### 4.5 Stress Distribution

![Stress Histogram](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/realistic_fairing/stress_histogram.png)

- Phase 1: 大部分の節点が低応力 (中央値 0.39 MPa)、開口縁のみ高応力
- Phase 2: 応力分布がより広範 (中央値 4.46 MPa)、複数開口の影響

### 4.6 Opening Detail Views (Phase 2)

| AccessDoor φ1300 | HVAC Door φ400 | RF Window φ400 |
|:-:|:-:|:-:|
| ![AccessDoor](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/realistic_fairing/detail_AccessDoor.png) | ![HVAC](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/realistic_fairing/detail_HVAC_Door.png) | ![RF](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/realistic_fairing/detail_RF_Window.png) |

- 各開口周辺で応力集中が確認可能
- 緑破線 = 開口境界（近似矩形パーティション）

## 5. Physical Validation

| チェック項目 | 結果 | 根拠 |
|---|---|---|
| 応力集中 | 開口縁で上昇 | 理論値 SCF ≈ 3 と整合 |
| 熱変形パターン | ノーズ先端最大 | 自由端の累積膨張 |
| フレーム補剛 | 変位プロファイルに段差 | Tie 拘束の効果 |
| 温度勾配 | Inner→Core→Outer 単調増加 | 熱伝達整合性 OK |
| 最大応力レベル | ~94 MPa | CFRP T1000G 許容応力 (~700 MPa) の 13% → 安全 |

## 6. Usage

```bash
# Phase 1: AccessDoor + 6 Ring Frames
cd abaqus_work
abaqus cae noGUI=../src/generate_realistic_fairing.py -- --job Job-Realistic-P1 --phase 1 --no_run
python ../scripts/patch_inp_thermal.py Job-Realistic-P1.inp
abaqus job=Job-Realistic-P1 input=Job-Realistic-P1.inp cpus=4

# Phase 2: All Openings + 4 Ring Frames
abaqus cae noGUI=../src/generate_realistic_fairing.py -- --job Job-Realistic-P2 --phase 2 --no_run
python ../scripts/patch_inp_thermal.py Job-Realistic-P2.inp
abaqus job=Job-Realistic-P2 input=Job-Realistic-P2.inp cpus=4

# Extract results
abaqus python ../src/extract_odb_results.py --odb Job-Realistic-P1.odb --output ../dataset_realistic/phase1
```
