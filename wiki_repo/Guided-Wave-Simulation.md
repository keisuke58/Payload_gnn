[← Home](Home)

# ガイド波（ラム波）シミュレーション — Guided Wave Simulation

> 最終更新: 2026-03-05
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

## 今後の計画

- [ ] ヒルベルト変換による群速度の正確な測定
- [ ] 曲面（フェアリング形状）への拡張
- [ ] 複数欠陥パターン（サイズ・位置のバリエーション）
- [ ] GNN 入力データとしての時系列特徴量設計
- [ ] PZT ネットワークのトポロジ最適化
