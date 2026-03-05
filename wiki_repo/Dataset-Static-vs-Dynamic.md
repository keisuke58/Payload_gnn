[← Home](Home)

# 静解析 vs 動解析 データセット評価

> 静解析と動解析は別基準で評価。**現状は動解析メイン**。

---

## 1. 分類

| 種別 | ソルバ | 主なデータ | 品質指標 |
|------|--------|------------|----------|
| **静解析** | Abaqus/Standard (StaticStep) | nodes.csv, elements.csv | 変位・温度・応力 |
| **動解析** | Abaqus/Explicit (ExplicitDynamicsStep) | *_sensors.csv (時刻歴) | センサ数・波形・SNR |

---

## 2. 静解析データセット

### 2.1 対象

| データセット | 生成元 | 備考 |
|-------------|--------|------|
| dataset_output | generate_fairing_dataset.py | 60°セクタ、熱負荷 |
| dataset_realistic_25mm_100 | generate_realistic_dataset.py | 25mm メッシュ |
| batch_s12_100_thermal | generate_czm_sector12.py | CZM 1/12 セクタ、熱+機械 |

### 2.2 評価基準（静解析専用）

- 変位 (ux, uy, uz) が非ゼロ
- 温度 > 10°C（熱解析含む場合）
- nodes.csv / elements.csv 存在
- PyG 変換可能

**検証**: `scripts/verify_dataset_quality.py --data_dir <dir>`

---

## 3. 動解析データセット（メイン）

### 3.1 対象

| データセット | 生成元 | 備考 |
|-------------|--------|------|
| abaqus_work/gw_fairing_dataset | generate_gw_fairing.py + extract_gw_history | GW フェアリング 30° |
| Job-GW-Fair-* | 同上 | 本番 DOE 用 |
| Job-GW-Curved-*, Job-GW-Freq* | generate_guided_wave.py | 平板/曲面板、周波数掃引 |

### 3.2 評価基準（動解析専用）

- *_sensors.csv 存在
- センサ数 ≥ 5（最大10、欠損あり得る）
- データ行数 ≥ 100（十分な時刻歴）
- 振幅が非ゼロ（波形が検出されている）
- Step-Wave 由来であること

**検証**: `scripts/verify_gw_dataset_quality.py`

### 3.3 動解析スコア（案）

| 項目 | 配点 | 基準 |
|------|------|------|
| CSV 存在 | 20 | 全ジョブで _sensors.csv あり |
| センサ数 | 20 | 平均 ≥ 8 |
| 時刻歴長 | 20 | 平均 ≥ 500 行 |
| 信号品質 | 20 | 振幅 RMS > 閾値 |
| PyG 変換 | 20 | build_gw_graph 成功 |
| **合計** | **100** | |

---

## 4. 現状（動解析メイン）

- **本番 GW フェアリング**: 未生成（`batch_generate_gw_dataset.sh all` 待ち）
- **フェアリングテスト**: Job-GW-Fair-Test-H3, -D3（各9センサ、3922行）→ **100/100**
- **全 GW 混在**: 旧モデル含め 20 件 → 91/100

**本番データ作成前のチェック**（100点でOK）:
```bash
python scripts/verify_gw_dataset_quality.py --data_dir abaqus_work --filter fairing
# → Estimated score: 100/100 なら batch_generate_gw_dataset.sh all 実行可
```

**検証コマンド**:
```bash
# 静解析
python scripts/verify_dataset_quality.py --data_dir dataset_output

# 動解析（フェアリングのみ＝本番前チェック）
python scripts/verify_gw_dataset_quality.py --data_dir abaqus_work --filter fairing

# 動解析（全 GW データ）
python scripts/verify_gw_dataset_quality.py --data_dir abaqus_work
```

---

## 5. 関連

| ページ | 内容 |
|--------|------|
| [**静解析・動解析融合ランキング**](Static-Dynamic-Fusion-Ranking) | データ量・実現可能性・効果の総合おすすめ |
| [Dataset-Perfect-Score](Dataset-Perfect-Score) | 静解析向け 100/120 点基準 |
| [GW-GNN-Pipeline-Design](GW-GNN-Pipeline-Design) | 動解析パイプライン設計 |
| [Guided-Wave-Simulation](Guided-Wave-Simulation) | GW シミュレーション概要 |
