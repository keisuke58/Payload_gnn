# データセットディレクトリ命名規則

## 規則

`dataset_output_<mesh>mm_<n>`

| 項目 | 例 | 説明 |
|------|-----|------|
| mesh | 50 | GLOBAL_SEED (mm) |
| n | 100 | サンプル数 |

例: `dataset_output_50mm_100` — メッシュ 50 mm、100 サンプル

## 既存データ

| ディレクトリ | メッシュ | サンプル数 |
|--------------|----------|------------|
| `dataset_output/` | 200 mm (旧) | 既存 |
| `dataset_output_100/` | 50 mm | 100 (生成中) |

## INP ファイル

Abaqus は job submit 時に `abaqus_work/<job_name>.inp` を生成する。
`--keep_inp` を付けると各サンプル dir に `model.inp` をコピーする。

```bash
python src/run_batch.py --doe doe_100_50mm.json --keep_inp
```

## 実行例

```bash
# デフォルト出力: dataset_output_50mm_100
python src/run_batch.py --doe doe_100_50mm.json

# 明示指定
python src/run_batch.py --doe doe_100_50mm.json --output_dir dataset_output_50mm_100
```
