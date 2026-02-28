# データセット完璧化手順

## 実施済み

1. **温度抽出**: `extract_odb_results` が NT11（節点温度）を参照 → temp=120°C が nodes.csv に出力
2. **熱パッチ**: `patch_inp_thermal.py` が *Static 複数行形式に対応、Step-1 に *Temperature を挿入
3. **32/100 サンプル**: 変位・温度ともに正しく抽出済み

## 残り 68 サンプルを完了する手順

### 方法 A: 既存 ODB からの抽出（ODB が存在する場合）

```bash
python scripts/extract_existing_odbs.py --doe doe_100.json --start 0 --end 100
```

### 方法 B: INP パッチ + Abaqus 再実行 + 抽出

```bash
# 1サンプルずつ順次実行（キューが空いてから）
python scripts/re_run_thermal_only.py --doe doe_100.json --start 33 --end 100
```

### 方法 C: フル再生成（CAE から）

```bash
python src/run_batch.py --doe doe_100.json --output_dir dataset_output --force --start 33 --end 100
```

## 品質確認

```bash
python scripts/verify_dataset_quality.py
# 期待: Good (both): 100, Estimated score: 100/100
```

## 検証

```bash
# 1サンプルの ODB 検証
abaqus python scripts/verify_odb_thermal.py --odb abaqus_work/H3_Debond_0000.odb
# 期待: Ux/Uy/Uz 非ゼロ, Temperature 120°C
```
