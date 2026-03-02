[<- Home](Home) | [Realistic-Fairing-FEM](Realistic-Fairing-FEM) | [Dataset-Generation-Status](Dataset-Generation-Status)

# C3D10 + 機械荷重 バッチ生成計画

> **Status**: Phase 1 テスト完了、Phase 3 バッチ生成待ち
> **Date**: 2026-03-01
> **Dataset**: `dataset_c3d10_mech_105` (100 debonding + 5 healthy)

---

## 1. 背景と目的

前回ベンチマーク (25 サンプル, 熱荷重のみ) は全モデル defect F1 = 0.000。
原因: データ不足 + 極端なクラス不均衡 + 熱のみでは欠陥シグナルが弱い。

**目標**: C3D10 + 機械荷重モデルで 105 サンプルを生成し、GNN が欠陥を検出できるか確認する。

| 比較項目 | 旧データセット | 新データセット |
|---------|-------------|-------------|
| サンプル数 | 25 | 105 (100 defect + 5 healthy) |
| コアメッシュ | S4R シェルのみ | **C3D10** 四面体ソリッド |
| 荷重 | 熱のみ | 熱 + **差圧 5kPa + 重力 3G** |
| ノード数/サンプル | ~23k | **~366k** |
| 欠陥タイプ | debonding のみ | debonding のみ |
| defect 比率 | ~5% | ~6.5% |

---

## 2. Phase 1 テスト結果

### 2.1 実行概要

| 項目 | 値 |
|------|-----|
| サーバー | frontale02 (93GB RAM, 24 core) |
| CAE 生成時間 | 74 秒 |
| ソルバー時間 | 14 分 |
| 抽出時間 | 3 分 |
| **合計** | **18 分/サンプル** |
| Core mesh | 839,319 nodes / 506,143 elements (C3D10) |
| Total mesh | 1,491,965 nodes / 1,156,722 elements |
| Memory | 40 GB |

### 2.2 物理量検証

| チェック項目 | 結果 | 判定 |
|---|---|---|
| ノード数 (OuterSkin) | 366,497 | OK |
| `u_mag` max | 22.2 mm | OK — 機械荷重の効果あり |
| `temp` 範囲 | 100–221°C | OK — z 依存プロファイル |
| `smises` max | 445 MPa | OK — 熱+機械応力 |
| `thermal_smises` max | 89.7 MPa | OK — 熱のみ (Step-1) |
| smises ≠ thermal_smises | 差: 平均 5.4, 最大 315 MPa | OK — 2ステップ動作確認 |
| `defect_label=1` | 24,050 ノード (6.56%) | OK |
| グラフ構築 | 34次元ノード / 5次元エッジ | OK |

### 2.3 グラフ仕様

```
Nodes:      366,497
Edges:    2,924,726
x shape:  [366497, 34]   (34-dim node features)
edge_attr: [2924726, 5]  (5-dim edge features)
avg degree: 8.0
```

---

## 3. Core メッシュ修正

### 3.1 問題

`generate_realistic_dataset.py` で Core (38mm 厚ソリッド) の C3D10 メッシュが 0 ノードを生成。

**原因**: 開口部/欠陥ゾーンのメッシュリファインメント (10mm, 4mm) が Core にも適用され、38mm 厚のソリッドに対して細かすぎるシードが入り、メッシャーが破壊。

### 3.2 修正

```python
# Before: all_skin_insts = (inst_inner, inst_core, inst_outer)
# After:  opening_insts = (inst_inner, inst_outer)  # Core excluded

# Opening refinement: skins only
for inst in opening_insts:
    assembly.seedEdgeBySize(edges=edges, size=opening_seed, ...)

# Defect refinement: skins only
for inst in skin_only_insts:
    assembly.seedEdgeBySize(edges=edges, size=defect_seed, ...)
```

- Core は **global_seed (25mm) のみ** で C3D10 メッシュ成功
- 欠陥ゾーンのパーティションも Core から除外 (debonding は Core セクション変更不要)

---

## 4. バッチ生成計画

### 4.1 DOE

```
doe_c3d10_mech_100.json
  105 samples = 100 debonding + 5 healthy
  seed: 2026
  theta: [5°, 55°]
  z: [800, 4200] mm
  size tiers: Small/Medium/Large/Critical
```

### 4.2 サーバー割り当て

| サーバー | サンプル範囲 | 数 | 作業 Dir |
|---------|------------|---|---------|
| frontale01 | 0–26 | 27 | `abaqus_work_f01` |
| frontale02 | 27–53 | 27 | `abaqus_work_f02` |
| frontale04 | 54–79 | 26 | `abaqus_work_f04` |
| marinos03 | 80–104 | 25 | `abaqus_work_m03` |

localhost は除外 (62GB RAM では C3D10 で OOM リスク)。

### 4.3 時間見積もり

- 1 サンプル: ~18 分 (CAE 1分 + ソルバー 14分 + 抽出 3分)
- 27 サンプル/サーバー: ~8 時間
- **全体: ~8-9 時間 (4 サーバー並列)**

### 4.4 実行コマンド

```bash
# 全サーバー起動
bash scripts/dispatch_c3d10.sh

# 進捗確認
bash scripts/dispatch_c3d10.sh status

# 緊急停止
bash scripts/dispatch_c3d10.sh kill
```

---

## 5. 後処理 (バッチ完了後)

### 5.1 品質チェック

```bash
# 成功数確認
find dataset_c3d10_mech_105 -name nodes.csv | wc -l

# 物理量スポットチェック
python scripts/verify_dataset_quality.py --input dataset_c3d10_mech_105
```

### 5.2 PyG データ構築

```bash
python src/prepare_ml_data.py \
  --input dataset_c3d10_mech_105 \
  --output data/processed_c3d10_mech \
  --val_ratio 0.2 --no_geodesic
```

### 5.3 vancouver02 へ転送

```bash
scp data/processed_c3d10_mech/{train.pt,val.pt,norm_stats.pt} \
  vancouver02:~/Payload2026/data/processed_c3d10_mech/
```

---

## 6. GNN ベンチマーク計画 (vancouver02)

### 6.1 4 アーキテクチャ比較

```bash
for arch in gcn gat gin sage; do
  python src/train.py --arch $arch --loss focal --focal_gamma 3.0 \
    --epochs 200 --batch_size 4 --hidden 128 --layers 4 \
    --data_dir data/processed_c3d10_mech \
    --run_name benchmark_c3d10_${arch}
done
```

### 6.2 不均衡対策の比較

| 手法 | 概要 |
|------|------|
| full_graph + focal loss | ベースライン |
| defect_centric subgraph | `--sampler defect_centric --subgraph_hops 4` |
| weighted CE | `--loss weighted_ce` |

### 6.3 成功基準

| 指標 | 前回 (thermal, 25 samples) | 今回目標 |
|------|--------------------------|---------|
| defect F1 | 0.000 | **> 0.0** (非ゼロ) |
| val AUC | 0.70 | > 0.75 |
| Macro F1 | 0.000 | > 0.10 |

**最低限の成功**: defect F1 > 0 が出れば「機械荷重が効いている」ことの証拠。

---

## 7. `run_batch.py` 改善

### 7.1 `memory=40gb` 追加

旧データセットの 14/100 失敗はソルバー OOM が主因。`--memory` 引数 (デフォルト 40gb) を追加して解決。

```python
# src/run_batch.py L186
['abaqus', 'job=' + job_name, 'input=' + job_name + '.inp',
 'cpus=%d' % n_cpus, 'memory=%s' % memory]
```

---

## 8. 関連ページ

| ページ | 内容 |
|--------|------|
| [Realistic-Fairing-FEM](Realistic-Fairing-FEM) | FEM モデル詳細 (C3D10, 2ステップ解析) |
| [FEM-Realism-Roadmap](FEM-Realism-Roadmap) | リアリズム向上ロードマップ |
| [Dataset-Generation-Status](Dataset-Generation-Status) | 旧データセット生成状況 |
| [Node-Features](Node-Features) | 34 次元ノード特徴量の定義 |
| [Benchmark-Targets](Benchmark-Targets) | GNN ベンチマーク目標 |
