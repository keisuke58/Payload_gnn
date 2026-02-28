# データセット 80点 達成手順

## 現状

- **達成**: 32/100 サンプルが変位・温度ともに正しく抽出済み（temp=120°C, ux≠0）
- **残り**: 68 サンプルは `re_run_thermal_only.py` で順次処理

## 実施済みの修正

1. **`run_batch.py`**
   - `--force`: 既存サンプルを上書きして再生成
   - `PROJECT_ROOT` を env と `--project_root` で渡す
   - ODB 待機ループ（最大 30 秒）を追加

2. **`generate_fairing_dataset.py`**
   - `--project_root` 引数でパッチスクリプトのパスを指定
   - 熱パッチ（Initial Conditions, *Temperature, NT）を確実に適用

3. **`extract_odb_results.py`**
   - フレーム 0 のステップに対するフォールバック処理を追加

4. **`patch_inp_thermal.py`**
   - NT（節点温度）を Node Output に追加（条件を `RF, U, NT` に修正）

## 80点達成の実行手順

Abaqus ライセンスが利用可能な状態で、以下を実行してください。

```bash
cd /home/nishioka/Payload2026

# 全 100 サンプルを熱パッチ付きで再生成（数時間かかります）
python src/run_batch.py --doe doe_100.json --output_dir dataset_output --force

# または範囲指定（例: 0〜10 でテスト）
python src/run_batch.py --doe doe_100.json --output_dir dataset_output --force --start 0 --end 10
```

## 検証

```bash
# 1サンプルの ODB を検証
abaqus python scripts/verify_odb_thermal.py --odb abaqus_work/H3_Debond_0000.odb

# 期待される出力:
#   Ux/Uy/Uz: 非ゼロ
#   Temperature: min=120 max=120 (外板)

# nodes.csv の確認
head -5 dataset_output/sample_0000/nodes.csv
# temp 列が 120 付近、ux/uy/uz が非ゼロであること
```

## 注意

- ライセンス不足時は `ERROR 1A00003A: No more licenses available` が出ます
- 並列実行は避け、1 ジョブずつ実行してください
- 1 サンプルあたり約 5〜10 分想定（100 サンプルで約 8〜16 時間）
