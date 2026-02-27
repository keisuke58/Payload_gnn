[← Home](Home)

# Dataset Generation Status

**Status**: In Progress (32/100 品質検証済み)
**Date**: 2026-02-28

## 生成中のデータ

**H3 フェアリングのデボンディング欠陥 FEM データ**（100 サンプル）

| 項目 | 内容 |
|------|------|
| **出力先** | `dataset_output/` |
| **メッシュ** | GLOBAL_SEED = 50 mm |
| **欠陥** | 外スキン-コア界面の円形デボンディング |
| **サンプル数** | 100（欠陥ありのみ） |
| **熱荷重** | 初期 20°C、Step-1 外板 120°C（熱パッチ適用済み） |

### 欠陥サイズ階層

| 階層 | 半径 (mm) | 割合 |
|------|-----------|------|
| Small | 20–50 | 30% |
| Medium | 50–80 | 40% |
| Large | 80–150 | 25% |
| Critical | 150–250 | 5% |

### 各サンプルの出力

- `nodes.csv` — 座標 (x,y,z)、変位 (ux,uy,uz)、**温度 (NT11)**、**defect_label**
- `elements.csv` — 要素接続、Mises 応力
- `metadata.csv` — theta_deg, z_center, radius, n_defect_nodes

### 進捗

| 状態 | 件数 | 備考 |
|------|------|------|
| **品質検証済み** | 32/100 | 変位・温度ともに正しく抽出 |
| **未完了** | 68/100 | ODB 未生成または再実行待ち |

### 品質検証

```bash
python scripts/verify_dataset_quality.py   # データセット品質スコア
python scripts/verify_odb_thermal.py       # ODB 熱・変位の個別検証
```

### 残り 68 サンプルの再実行

```bash
python scripts/re_run_thermal_only.py --start 33 --end 100   # 熱パッチ + Abaqus + 抽出
# または
python src/run_batch.py --doe doe_100.json --output_dir dataset_output --force
```
