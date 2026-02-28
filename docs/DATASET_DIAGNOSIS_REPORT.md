# データセット診断レポート

**日付**: 2026-02-28  
**目的**: 物理データ（変位・応力・温度）がゼロ、GNN F1=0.0 の原因解明

---

## 1. 現状サマリ

| データセット | ノード数 | 物理データ | stress列 | 備考 |
|-------------|----------|------------|----------|------|
| **dataset_output** | 10,897 | sample_0000 のみ全ゼロ、他は非ゼロ | ✓ | 正常データあり |
| **dataset_output_100** | 10,897 | **全サンプルゼロ** | ✗ | ML パイプラインのデフォルト入力 |
| **dataset_output_25mm_400** | 42,892 | **全サンプルゼロ** | ✓ | メッシュのみ同一、物理量ゼロ |

---

## 2. 詳細確認結果

### 2.1 dataset_output（100サンプル）

- **sample_0000**: ux=uy=uz=0, temp=20.0, s11=s22=s12=smises=0 → **ODB抽出失敗**
- **sample_0001〜0099**: ux,uy,uz, temp=120, stress 非ゼロ → **正常**

```text
sample_0000: non-zero physics rows: 0
sample_0001: non-zero physics rows: 10897
sample_0050: non-zero physics rows: 10897
```

### 2.2 dataset_output_100（ML デフォルト入力）

- **全サンプル**: ux=uy=uz=0, temp=0.0
- **カラム**: `node_id,x,y,z,ux,uy,uz,temp,defect_label` → **stress 列なし**
- `extract_odb_results.py` の出力形式と**不一致**（本来は s11,s22,s12,smises を含む）

### 2.3 dataset_output_25mm_400（42,892ノード）

- **全サンプル**: ux=uy=uz=0, temp=0.0, s11=s22=s12=smises=0
- カラム形式は `extract_odb_results` と一致
- メッシュ座標は全サンプル同一、変わるのは `defect_label` のみ

---

## 3. 原因分析

### 3.1 根本原因: ODB からの物理量抽出が失敗している

1. **Abaqus 解析が正しく完了していない**
   - 熱負荷（*Temperature）が INP に含まれていない
   - `patch_inp_thermal.py` が適用されていない、または INP の形式が想定と異なる
   - 解析が即終了し、変位・応力・温度がゼロのまま

2. **ODB に結果が書き込まれていない**
   - Step-1 の Field Output に U, S, NT が無い
   - 解析失敗時は ODB が空または不完全

3. **extract_odb_results の実行環境・呼び出しの不整合**
   - `dataset_output_100` の nodes.csv に stress 列が無い
   - `extract_odb_results.py` は常に stress を出力するため、**別経路で nodes.csv が生成されている**可能性

### 3.2 F1=0.0 の理由

GNN のノード特徴量は以下に依存:

```python
# build_graph.py: ノード特徴 (20次元)
[x, y, z, nx, ny, nz, k1, k2, H, K, ux, uy, uz, temp, s11, s22, s12, smises, boundary, loaded]
```

- **ux, uy, uz, temp, s11, s22, s12, smises** が全てゼロ
- メッシュ座標 (x,y,z) と curvature は全サンプル同一
- 変化するのは **defect_label** のみ（教師ラベル）

→ 入力に欠損・欠如が多く、モデルは物理的なシグナルを学習できないため F1=0.0 になる。

---

## 4. 推奨対応

### 4.1 即時対応: 正常データを使う

```bash
# dataset_output を使用（sample_0001〜0099 に物理データあり）
python src/prepare_ml_data.py --input dataset_output --output data/processed_ok
python src/train.py --data_dir data/processed_ok --output_dir runs/retrain
```

**注意**: sample_0000 はゼロのため、除外するか再抽出が必要。

### 4.2 ODB 抽出の検証

```bash
# 1. ODB の内容確認
abaqus python scripts/verify_odb_thermal.py --odb abaqus_work/H3_Debond_0001.odb

# 2. 手動で ODB から再抽出
python scripts/extract_existing_odbs.py --doe doe_100.json --output_dir dataset_output_100
```

### 4.3 INP の thermal パッチ確認

```bash
# patch 適用の有無を確認
python scripts/patch_inp_thermal.py abaqus_work/H3_Debond_0001.inp
# "Patched" と出るか、"No patch needed" かで判断
```

### 4.4 dataset_output_100 の再生成

- `dataset_output_100` の nodes.csv は `extract_odb_results` の出力形式と一致していない
- 再バッチ実行で ODB 抽出を正しく行うか、`extract_existing_odbs.py` で既存 ODB から再抽出することを推奨

---

## 5. チェックリスト

- [ ] `dataset_output` の sample_0000 を除外して ML 学習
- [ ] `abaqus_work/*.odb` の存在と中身を確認
- [ ] `verify_odb_thermal.py` で U, NT が非ゼロか確認
- [ ] `patch_inp_thermal.py` が INP に適用されているか確認
- [ ] `dataset_output_100` を `extract_existing_odbs.py` で再抽出
- [ ] `prepare_ml_data.py` のデフォルト入力を `dataset_output` に変更（暫定）

---

## 6. 25mm データセット修正（2026-02-28 対応済み）

### 実施した修正

1. **patch_inp_thermal.py**
   - Part-Core-1 の熱負荷を追加（外板120°C、内板20°C、コア70°C）
   - Initial Conditions に Part-Core-1 を追加
   - Node Output の NT 追加を U,RF 順にも対応

2. **extract_odb_results.py**
   - `--strict` オプション: 物理量が全てゼロの場合に失敗（ODB の thermal 欠損を検出）

3. **run_batch.py**
   - `--strict`: 抽出時に物理量ゼロを検出して失敗
   - `--keep_odb`: デバッグ用に ODB を残す

4. **regenerate_25mm.sh**
   - 25mm データセットの再生成スクリプト
   - `--extract-only`: 既存 ODB から再抽出のみ

### 25mm 再生成コマンド

```bash
# 全 400 サンプル再生成（Abaqus 実行が必要、数時間）
bash scripts/regenerate_25mm.sh

# 最初の 5 サンプルのみ（動作確認用）
bash scripts/regenerate_25mm.sh --start=0 --end=5

# ODB が既にある場合の再抽出のみ
bash scripts/regenerate_25mm.sh --extract-only
```
