[← Home](Home)

# ガイド波（ラム波）シミュレーション — Guided Wave Simulation

> 最終更新: 2026-03-06 (Step 6 30° フェアリングセクタ検証完了)
> **Phase 4/5 準備**: アクティブ SHM のための Abaqus/Explicit 動解析

---

## 概要

PZT センサーによるガイド波（Lamb 波）伝播を Abaqus/Explicit でシミュレーション。
A₀ モード（屈曲波）の伝播・散乱を利用してデボンディング欠陥を検出する。

### なぜガイド波か

| 手法 | SHM 適合性 | データ価値 | 計算コスト | 学術インパクト |
|------|-----------|-----------|-----------|--------------|
| **ガイド波（採用）** | ◎ PZT実装可能 | ◎ 時系列+空間 | ○ 数分/ケース | ◎ 最先端テーマ |
| フェアリング分離動解析 | △ 既知事象 | ○ 動荷重応答 | × 大規模 | ○ |

---

## Step 1: 平板サンドイッチパネル検証 ✅ 完了

### モデル仕様

| パラメータ | 値 |
|-----------|-----|
| パネルサイズ | 300 × 300 mm |
| CFRP スキン | 1 mm × 2 (Inner + Outer), [45/0/-45/90]s |
| Al-HC コア | 38 mm, C3D8R (EXPLICIT) |
| 結合 | Tie 拘束 (Skin ↔ Core) |
| メッシュサイズ | 3.0 mm (≈ λ/8 at 50 kHz) |
| 要素数 | ~153,000 ノード, ~140,000 要素 |
| ソルバー | Abaqus/Explicit (mass scaling なし) |
| 解析時間 | 0.5 ms (500 μs) |
| 計算時間 | ~12 分/ケース (4 CPU, frontale04) |

### 励振条件

| パラメータ | 値 |
|-----------|-----|
| 波形 | Hanning 窓トーンバースト |
| 周波数 | 50 kHz |
| サイクル数 | 5 (T_burst = 100 μs) |
| 方向 | Z (面外, A₀ モード励起) |
| 位置 | パネル中心 (0, 0) |
| 力 | 1.0 N 集中荷重 |

```
A(t) = 0.5 × (1 - cos(2πt/T_burst)) × sin(2πft)    (0 ≤ t ≤ T_burst)
A(t) = 0                                              (t > T_burst)
```

### 欠陥モデル

| パラメータ | 値 |
|-----------|-----|
| 欠陥タイプ | デボンディング (Tie 除去) |
| 位置 | x = 80 mm, y = 0 mm |
| 半径 | 25 mm |
| 実装 | Outer Skin ↔ Core 間の Tie 拘束を欠陥ゾーンで除去 |

### センサー配置 (v2)

| センサー | X 位置 | 備考 |
|----------|--------|------|
| Sensor 0 | 0 mm | 励振点 |
| Sensor 1 | 30 mm | 欠陥ゾーン手前 |
| Sensor 2 | 60 mm | 欠陥境界付近 (55mm) |
| Sensor 3 | 90 mm | 欠陥ゾーン内 |
| Sensor 4 | 120 mm | 欠陥ゾーン外側 |

---

## 結果

### 波形比較 (Healthy vs Debonding)

<img src="https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/guided_wave/gw_comparison.png" width="100%">

**観測結果 (5センサー, v2):**
- Sensor 0 (励振点): Healthy / Debond はほぼ同一
- Sensor 1 (x=30mm): 直接波は類似、後続の散乱波で微小な差異
- **Sensor 2 (x=60mm)**: 欠陥境界付近で波形が変化し始める
- **Sensor 3 (x=90mm)**: 欠陥ゾーン中心。最大の波形差異
- Sensor 4 (x=120mm): 欠陥透過後、反射波の影響で位相がずれる

### 散乱波 (差分信号)

<img src="https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/guided_wave/gw_difference.png" width="100%">

Defect - Healthy の差分信号。Sensor 2-3 (欠陥ゾーン 55-105mm) で最大の散乱波。

### ヒルベルト変換エンベロープ解析

<table>
<tr><th>Healthy</th><th>Debonding</th></tr>
<tr>
<td><img src="https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/guided_wave/gw_envelope_healthy.png" width="100%"></td>
<td><img src="https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/guided_wave/gw_envelope_debond.png" width="100%"></td>
</tr>
</table>

赤: Hilbert エンベロープ、緑: ピーク時刻、橙: 初期到達 (5% 閾値)

### 波面伝播スナップショット

<img src="https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/guided_wave/gw_snapshots_v2.png" width="100%">

上段: Healthy (同心円状の波面), 下段: Debonding (欠陥ゾーン付近で波面の乱れ)
緑破線円 = 欠陥ゾーン (x=80mm, r=25mm)

### 波面伝播アニメーション

<img src="https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/guided_wave/gw_wave_comparison_v2.gif" width="100%">

左: Healthy, 右: Debonding (緑破線 = 欠陥ゾーン)。
80フレーム (0 ~ 684 μs)。欠陥ゾーンで波面が散乱・反射する様子が確認可能。

### 群速度

ヒルベルト変換エンベロープ + 参照ベース法 (sensor 0 → 各センサーの初期到達時間)

| モデル | Method 3 平均 | Mindlin 理論 | 偏差 | 備考 |
|--------|-------------|-------------|------|------|
| Healthy | **1089 m/s** | ~1911 m/s | 29.7% | 0→1: 1067, 0→2: 1196 m/s |
| Debond | **1161 m/s** | ~1911 m/s | 25.1% | 0→1: 1112, 0→2: 983, 0→3: 1085 m/s |

> **理論値について**: Mindlin 厚板理論 (E_eff = 45.8 GPa for QI layup) の位相速度 ~1911 m/s。
> FEM 測定値は 3D 効果・近接場・分散効果により ~57-60% に低下。物理的に妥当な範囲。

---

## スクリプト一覧

| ファイル | 実行環境 | 用途 |
|----------|---------|------|
| `src/generate_guided_wave.py` | Abaqus CAE | モデル生成 (Healthy + Defect) |
| `scripts/extract_gw_history.py` | Abaqus Python | センサー時刻歴抽出 |
| `scripts/extract_gw_field.py` | Abaqus Python | フィールド出力抽出 (アニメーション用) |
| `scripts/plot_gw_comparison.py` | Python (matplotlib) | 波形比較プロット |
| `scripts/plot_gw_animation.py` | Python (matplotlib) | 波面アニメーション GIF |
| `scripts/plot_gw_sweep.py` | Python (matplotlib) | 欠陥パラメータスイープ比較 |
| `scripts/plot_gw_curved_comparison.py` | Python (matplotlib) | 平板 vs 曲面比較 |
| `scripts/batch_gw_sweep.sh` | Bash | 欠陥スイープバッチ実行 |
| `scripts/batch_gw_freq_sweep.sh` | Bash | 周波数スイープバッチ実行 |
| `scripts/plot_gw_freq_noise.py` | Python (matplotlib) | 周波数スイープ + ノイズロバストネス解析 |

### 使い方

```bash
# モデル生成 (Healthy)
abaqus cae noGUI=src/generate_guided_wave.py -- --job Job-GW-Healthy --no_run

# モデル生成 (Debonding)
abaqus cae noGUI=src/generate_guided_wave.py -- --job Job-GW-Debond \
    --defect '{"x_center":80,"y_center":0,"radius":25}' --no_run

# 時刻歴抽出
abaqus python scripts/extract_gw_history.py Job-GW-Healthy.odb Job-GW-Debond.odb

# フィールド出力抽出
abaqus python scripts/extract_gw_field.py Job-GW-Healthy.odb Job-GW-Debond.odb

# 波形プロット
python scripts/plot_gw_comparison.py abaqus_work/Job-GW-Healthy_sensors.csv \
    abaqus_work/Job-GW-Debond_sensors.csv

# アニメーション生成
python scripts/plot_gw_animation.py abaqus_work/Job-GW-Healthy abaqus_work/Job-GW-Debond
```

---

## 技術的ポイント

### Abaqus/Explicit 固有の設定
- `ExplicitDynamicsStep` — プロジェクト初の Explicit 使用
- `elemLibrary=EXPLICIT` (S4R, C3D8R)
- **Mass scaling なし** — 波速歪み防止
- CFL: dt ≈ 3mm / 1e7 mm/s = 3e-7 s → ~1,700 ステップ

### メッシュ要件 (λ/8 ルール)
50 kHz の A₀ モード: λ ≈ 31 mm → h ≤ 3.9 mm → seed = 3.0 mm

### デボンディング実装
Tie 拘束の選択的除去。欠陥ゾーンのフェース:
1. Datum Plane で Outer Skin + Core を 4 面パーティション
2. 欠陥内フェースを判定 (`_point_in_defect()`)
3. 健全フェースのみ Tie 拘束 → 欠陥部は自由表面（デボンディング）

---

## Step 2: 欠陥パラメータスイープ ✅ 完了

基準ケース (D1: r=25, x=80) に加え、欠陥サイズと位置を変化させた 4 ケースを実行。

### DOE

| ケース | 半径 (mm) | 位置 x (mm) | 備考 |
|--------|-----------|-------------|------|
| **D1** (Debond-v2) | 25 | 80 | 基準ケース |
| **D2** (r15) | 15 | 80 | 小さい欠陥 |
| **D3** (r40) | 40 | 80 | 大きい欠陥 |
| **D4** (near) | 25 | 40 | 励振点に近い |
| **D5** (edge) | 25 | 120 | パネル端 |

### Damage Index (DI)

DI = RMS(scattered) / RMS(incident) — 各センサーでの散乱波強度

<img src="https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/guided_wave/gw_sweep_damage_index.png" width="100%">

| ケース | S0 | S1 | S2 | S3 | S4 | 平均DI |
|--------|-----|-----|-----|-----|-----|--------|
| D4 (x=40, 近い) | 0.093 | 1.424 | 1.549 | 1.606 | 1.460 | **1.226** |
| D3 (r=40, 大) | 0.042 | 0.214 | 1.118 | 1.267 | 2.140 | 0.956 |
| D1 (r=25, 基準) | 0.037 | 0.170 | 1.287 | 1.814 | 1.366 | 0.935 |
| D2 (r=15, 小) | 0.061 | 0.179 | 0.295 | 1.134 | 1.611 | 0.656 |
| D5 (x=120, 端) | 0.023 | 0.078 | 0.157 | 0.424 | 1.088 | 0.354 |

**考察:**
- **位置依存性**: 励振点に近い欠陥 (D4) ほど散乱が大きい (DI = 1.226 vs 0.354)
- **サイズ依存性**: r=15→25→40 で DI は 0.656→0.935→0.956（非線形的な飽和傾向）
- **センサー依存性**: 欠陥ゾーンに近いセンサー (S2-S3) で最大の DI

### 波形比較

<img src="https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/guided_wave/gw_sweep_waveforms.png" width="100%">

### 散乱波比較

<img src="https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/guided_wave/gw_sweep_scattered.png" width="100%">

---

## Step 3: 曲面パネル検証 ✅ 完了

H3 フェアリングのバレル部（R=2600mm）をモデル化し、曲率がガイド波に与える影響を検証。

### 曲面モデル仕様

| パラメータ | 値 |
|-----------|--------|
| 幾何 | 円筒セクタ (BaseShellRevolve / BaseSolidRevolve) |
| 半径 | 2600 mm (内スキン) |
| セクター角 | 7.17° (弧長 330 mm) |
| 高さ | 300 mm (軸方向) |
| サンドイッチ構成 | 平板と同一 (1/38/1 mm) |
| メッシュ | 177,760 ノード (平板比 17%増) |
| 計算時間 | ~12 分/ケース |
| 欠陥 | y=150, theta=3.6°, r=25 mm (弧距離~80mm) |

### 技術的ポイント

**材料方位:**
- コア: 円筒座標系 (CSYS-1=R, CSYS-2=θ, CSYS-3=Y)
- スキン: 直交座標系 (Y=軸方向, axis=AXIS_3 で法線に投影)

**励振:** 放射方向力（円筒座標系 cf1）— A₀ モード励起

**センサー:** 弧に沿って 0, 30, 60, 90, 120 mm 配置、U_r = U1·cos(θ) + U3·sin(θ) で放射方向変位を抽出

### 平板 vs 曲面: Damage Index 比較

<img src="https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/guided_wave/gw_flat_vs_curved_di.png" width="100%">

| センサー | S0 | S1 | S2 | S3 | S4 | 平均DI |
|----------|-----|-----|-----|-----|-----|--------|
| 平板 | 0.037 | 0.170 | 1.287 | 1.814 | 1.366 | 0.935 |
| 曲面 (R=2600) | 0.446 | 1.048 | 1.361 | 1.138 | 0.749 | **0.948** |

**考察:**
- 平均 DI はほぼ同等 (0.935 vs 0.948) → 曲面でもデボンディング検出能力は維持
- 曲面は散乱が **均等に分散** — 曲率による波面の閉じ込め効果
- 平板は **欠陥通過後のセンサー (S3-S4) に集中** — 直進波の散乱

### 波形比較 (正規化)

<img src="https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/guided_wave/gw_flat_vs_curved_healthy.png" width="100%">

### 散乱波比較

<img src="https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/guided_wave/gw_flat_vs_curved_scattered.png" width="100%">

---

## Step 4: ノイズロバストネス検証 ✅ 完了

既存のセンサーデータにガウスノイズ（SNR = 40〜5 dB）を重畳し、検出限界を定量化。

### 手法
- 既存 CSV データ（D1, D2, D4, D5）にモンテカルロ法（20 試行/点）でガウスノイズ注入
- SNR = signal RMS / noise RMS (dB)
- 閾値: DI = 0.2 を検出限界とする

### DI vs SNR

<img src="https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/guided_wave/gw_noise_robustness.png" width="100%">

| ケース | Clean | 40dB | 30dB | 25dB | 20dB | 15dB | 10dB | 5dB |
|--------|-------|------|------|------|------|------|------|-----|
| D1 (r=25, x=80) | 0.935 | 0.935 | 0.940 | 0.948 | 0.967 | 1.003 | 1.066 | 1.166 |
| D2 (r=15, x=80) | 0.656 | 0.656 | 0.660 | 0.669 | 0.690 | 0.736 | 0.826 | 0.971 |
| D4 (r=25, x=40) | 1.226 | 1.226 | 1.228 | 1.232 | 1.241 | 1.258 | 1.288 | 1.333 |
| D5 (r=25, x=120) | 0.354 | 0.355 | 0.363 | 0.378 | 0.410 | 0.474 | 0.598 | 0.794 |

**考察:**
- **全ケースで SNR 5 dB まで DI > 0.2** → 検出閾値を大きく上回り、極めて堅牢
- ノイズ増加で DI は上昇（ノイズ RMS が散乱信号を底上げ）
- 最も困難なケース（D5: 端の欠陥）でも SNR 5dB で DI=0.794
- **実用含意**: PZT の典型的 SNR は 20〜40 dB → 十分な検出マージン

---

## Step 5: 周波数スイープ ✅ 完了

25 / 50 / 75 / 100 kHz で健全 + D1欠陥（r=25mm, x=80mm）を各周波数で実行。

### 実行条件

| パラメータ | 値 |
|-----------|-----|
| 周波数 | 25, 50, 75, 100 kHz |
| ケース数 | 8（健全4 + D1欠陥4） |
| メッシュ調整 | λ/8 ルールで自動調整（高周波 → 細メッシュ） |
| 実行サーバー | frontale04 (4 CPU) |
| 総計算時間 | ~2.5 時間 |

### DI vs 周波数

| 周波数 | Mean DI | S0 | S1 | S2 | S3 | S4 |
|--------|---------|------|------|------|------|------|
| 25 kHz | 0.935 | 0.086 | 0.572 | 1.270 | 1.567 | 1.180 |
| 50 kHz | 0.935 | 0.037 | 0.170 | 1.287 | 1.814 | 1.366 |
| 75 kHz | 1.070 | 0.104 | 0.584 | 1.560 | 1.211 | 1.890 |
| 100 kHz | **1.257** | 0.087 | 0.998 | 1.303 | 1.775 | 2.123 |

### 知見

- **全周波数で DI >> 0.2**（検出閾値の 4.7〜6.3 倍）→ 周波数選択に関わらず検出可能
- **高周波数ほど DI 上昇**: 100 kHz で最大（DI=1.257）、短波長が欠陥散乱に有利
- **S0（加振点）は常に低 DI**: 入射波が支配的で散乱成分が相対的に小さい
- **S3-S4（欠陥直上〜遠方）で最大 DI**: 欠陥位置 x=80mm に近い S3 が高感度
- **推奨検出帯域**: 75〜100 kHz（DI > 1.0、十分なマージン）

![DI vs Frequency](images/guided_wave/gw_freq_sweep_di.png)

---

## Step 6: 30° フェアリングセクタ検証 ✅ 完了

H3 フェアリングの 30° バレルセクタ（z=500–2500 mm, R=2600 mm）でガイド波モデルを実行。
CZM によるデボンディング欠陥、10 センサ（クロス配置）、リングフレーム、開口部を含む。

### モデル仕様

| パラメータ | 値 |
|-----------|-----|
| セクター角 | 30° (1/12 円周) |
| 軸方向 | z = 500–2500 mm |
| 半径 | 2600 mm (内スキン) |
| ノード数 | 約 63,000 |
| 解析時間 | 3.92 ms |
| 計算時間 | 約 47 分/ケース (4 CPU) |

### Damage Index (H3 健全 vs D3 欠陥)

| センサー | arc [mm] | DI |
|----------|----------|-----|
| S0 | 0 | 1.462 |
| S1 | 30 | 1.173 |
| S2 | 90 | 1.209 |
| S3 | 120 | 1.189 |
| S4–S7 | 60 | 1.31–1.37 |
| S8 | 60 | 1.279 |
| **平均** | — | **1.299** |

**考察:**
- 全センサで DI > 1.0 → 散乱波 RMS が入射波 RMS を上回り、欠陥検出に十分
- 平板 D3 (r=40 mm) の Mean DI 0.956 より高く、フェアリング曲率下でも検出感度は維持
- クロス配置センサ（周方向 5 + 軸方向 5）により欠陥位置の空間分解が可能

![Fairing DI](images/guided_wave/gw_fairing_di.png)

---

## 今後の計画

- [x] ヒルベルト変換による群速度の正確な測定
- [x] 複数欠陥パターン（サイズ・位置のバリエーション）
- [x] 曲面（フェアリング形状）への拡張
- [x] 30° フェアリングセクタ（CZM・リングフレーム・開口部）検証
- [x] ノイズロバストネス検証（SNR 5dB まで検出可能を確認）
- [x] 周波数スイープ（25〜100 kHz、全帯域で DI >> 0.2、推奨 75〜100 kHz）
- [ ] GNN 入力データとしての時系列特徴量設計
- [ ] PZT ネットワークのトポロジ最適化
