# GW → GNN パイプライン設計

センサ時刻歴 CSV からグラフ構築・ML 前処理への流れを定義する。

## 1. データフロー概要

```
extract_gw_history.py 出力
  ↓
{job_name}_sensors.csv  (time_s, sensor_0_Ur, sensor_1_Ur, ...)
  ↓
build_gw_graph.py      (CSV → PyG Data)
  ↓
prepare_gw_ml_data.py  (データセットスキャン → train/val split)
  ↓
train.py               (既存 GNN 学習、または GW 専用 train_gw.py)
```

## 2. 入力形式

### 2.1 センサ CSV

- **パス**: `abaqus_work/gw_fairing_dataset/*_sensors.csv`
- **形式**:
  - 1行目: `time_s,sensor_0_Ur,sensor_1_Ur,...`
  - 2行目: `# x_mm` または `# arc_mm` + 位置値
  - 3行目以降: 時刻歴データ
- **センサ数**: 最大10（欠損あり得る: メッシュ境界等でスキップ）

### 2.2 ラベル

- **DOE JSON** (`doe_gw_fairing.json`): 各サンプルの defect_params
- **Healthy**: `Job-GW-Fair-Healthy` → ラベル 0
- **Defect**: `Job-GW-Fair-0000` など → ラベル 1（二値分類）
- 将来: 欠陥位置・サイズの回帰

## 3. グラフ構造

### 3.1 ノード

- **数**: センサ数（9〜10）
- **特徴量** (候補):
  - 時系列統計: 最大振幅、RMS、ピーク到達時刻
  - 周波数特徴: FFT 係数（50kHz 付近のパワー）
  - エンベロープ: Hilbert 変換
  - 位置: CSV 2行目の x_mm / arc_mm（正規化）

### 3.2 エッジ

- **接続** (候補):
  - **k-NN**: 位置（arc, z）に基づく近傍
  - **全結合**: 10 ノード程度なら許容
  - **固定トポロジ**: 5 circumferential + 5 axial の格子

### 3.3 グラフレベルラベル

- `y`: 0 (healthy) / 1 (defect)
- 将来: `y_reg` で欠陥パラメータ

## 4. モジュール設計

### 4.1 build_gw_graph.py

```python
def load_sensor_csv(csv_path) -> (times, sensor_data, positions)
def extract_time_features(times, sensor_data) -> np.ndarray  # (n_sensors, n_features)
def build_gw_graph(csv_path, label, positions=None) -> Data
```

- `Data.x`: (n_sensors, n_features)
- `Data.edge_index`: (2, n_edges)
- `Data.y`: スカラー (0 or 1)
- `Data.pos`: (n_sensors, 2)  # arc_mm, z 正規化

### 4.2 prepare_gw_ml_data.py

```python
def collect_gw_samples(csv_dir, doe_path, include_augmented=True) -> [(csv_path, label), ...]
def prepare_gw_dataset(input_dir, output_dir, doe_path, val_ratio=0.2, include_augmented=True)
```

- サンプル収集: `*_sensors.csv` をスキャン
- Healthy: `Job-GW-Fair-Healthy_sensors.csv`, `Job-GW-Fair-Healthy-A*.csv`（augmentation）
- Defect: `Job-GW-Fair-0000_sensors.csv` など
- `--no_augmented`: Healthy-A* を除外

### 4.3 train_gw.py ✅ 実装済

- グラフ分類（graph-level）: GNN encoder + `global_mean_pool` + MLP
- クラスバランス: Focal Loss / `--class_weight` / `--weighted_sampler`
- アーキテクチャ: GCN, GAT, GIN, SAGE

```bash
python src/prepare_gw_ml_data.py --input abaqus_work/gw_fairing_dataset --output data/processed_gw_100
python src/train_gw.py --data_dir data/processed_gw_100 --arch gat --epochs 200
```

## 5. 実装状況

1. ✅ `build_gw_graph.py`: CSV 読み込み + 時間特徴抽出 + グラフ構築
2. ✅ `prepare_gw_ml_data.py`: Healthy-A* 対応、train/val 保存
3. ✅ `train_gw.py`: graph-level 2 値分類、クラスバランス対応

## 6. 参照

- `scripts/extract_gw_history.py`: 出力形式
- `scripts/verify_gw_extract.py`: 整合性検証
- `src/build_graph.py`: 既存 FEM グラフ構築（参考）
