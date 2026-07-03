# CLAUDE.md — Project Instructions for Claude Code

## Project
JAXA H3 ロケット CFRP/Al-Honeycomb フェアリングの **GNN-SHM**（構造ヘルスモニタリング）研究プロジェクト。
2025年 H3 F8 事故で顕在化した CFRP/Al ハニカム界面の**デボンディング（skin-core 剥離）欠陥検出**を、
FEM（Abaqus）で生成した応答データ上の **Graph Neural Network** で実現する。

研究は単一の検出タスクを超えて、複数の研究ラインに拡張済み（下記「Research Lines」参照）。

## Communication / Working Style
- **日本語で会話する**
- まず1つ生成して確認してから丁寧に進める（バッチ生成や大量変更の前に1サンプルで検証）
- 図・ラベルは**英語**で作成する（`Fig. 1`, `Figure 1` 等）

## Git / PR Rules
- PR の body に "Generated with Claude Code" を**書かない**
- コミットメッセージは日本語OK、`Co-Authored-By` は付ける
- 開発ブランチ指定がある場合はそれに従う（指定外ブランチへ push しない）

## Environment
- **Abaqus 2024**（埋め込み Python 3.10.5）— FEM 生成・ODB 抽出に使用
- **PyTorch 2.x / PyG 2.x**（pyenv miniconda3, pip 管理）— GNN 学習に使用
- GPU: 24GB × 4
- Abaqus 実行は `abaqus_work/` から: `abaqus cae noGUI=...`
- ODB 抽出は: `abaqus python src/extract_odb_results.py`
- クラスタ: PBS/Torque（`qsub`、`scripts/dispatch_parallel.sh` 等）。`Job-*.o<id>` / `.e<id>` は再生成可能ログ（gitignore 済み）

### 依存関係インストール
```bash
pip install -r requirements-gnn.txt        # メイン（GNN 学習スタック = core + torch + PyG）
pip install -r requirements-quantum.txt    # 任意（Qiskit, 量子 GNN）
pip install -r requirements-fem-alt.txt    # 任意（FEniCS / JAX-FEM 代替 FEM）
```
`pyproject.toml` の optional-extras（`gnn` / `quantum` / `fem` / `dev`）でも同等。

## Primary Pipeline（静的 GNN 欠陥検出）
```
generate_doe.py → generate_fairing_dataset.py (Abaqus CAE) / run_batch.py
  → extract_odb_results.py (Abaqus Python)
  → build_graph.py → prepare_ml_data.py
  → train.py → evaluate.py / predict_api.py
```

### よく使うコマンド
```bash
# 軽量サニティチェック（Abaqus/GPU/学習なし — H3 spec 検証 + pytest）
./scripts/reproduce_core.sh        # = make reproduce

# テストのみ / spec 検証のみ
make test                          # pytest tests/ -q
make validate                      # python scripts/validate_h3_specs.py

# 学習（必ず src/ から実行）
cd src && python train.py --arch gat --data_dir ../dataset/processed --epochs 200 --cross_val 5

# 推論 API
MODEL_CHECKPOINT=runs/<run>/best_model.pt uvicorn predict_api:app --port 8000

# 学習結果の比較
python scripts/compare_model_results.py
```

`train.py` の主な引数: `--arch {gcn,gat,gatv2,gin,sage,lgsta,meshgnn,gps,multiscale,transolver}`,
`--task {node_cls,multitask}`, `--loss {weighted_ce,focal}`, `--sampler {full_graph,defect_centric}`,
`--cross_val N`, `--multi_gpu`, 物理正則化 `--physics_lambda_*`。

## Code Conventions（重要）
1. **学習スクリプトは `src/` から実行する** — import は bare（`from models import ...`）。
   pytest は `pyproject.toml` で `pythonpath=["src"]` を設定済み。
2. **コミットしないもの**: `runs/`, `dataset_*/`, `dataset_output*/`, `data/`, `*.odb`,
   `doe_*.json`, `figures/`, LaTeX/Abaqus/クラスタの scratch（`.gitignore` 参照）。
3. **Abaqus/FEM バッチは重い** — 軽量 reproduce では実行しない。
4. **pre-commit**: ruff（`--fix`）+ 各種フック。`src|scripts|tests` 対象。
   ```bash
   pip install pre-commit && pre-commit install
   ```
5. **CI**（`.github/workflows/ci.yml`）: CPU PyTorch + core deps で H3 spec 検証 + pytest。
6. **Wiki 同期**: `.github/workflows/sync-wiki.yml` が `wiki_repo/` を GitHub Wiki に反映。

## Data — 34-dim Node Features（静的解析）
位置・幾何(10) + 変位(4) + 温度(1) + 応力(5) + 熱応力(1) + ひずみ(3) + 繊維配向(3) + 積層構成(5) + 境界フラグ(2) = **34次元**
（GW センサグラフは別スキーマ。`build_gw_graph.py` 参照）

## Key Directories
- `src/` — メインコード（生成・抽出・グラフ構築・学習・モデル定義）
  - `src/prad/` — PI-GraphMAE 自己教師あり事前学習 + 蒸留（PRAD ライン）
  - `src/vt/` — H3 Virtual Twin（6DOF 飛行 orchestrator, 推進・空力・空力加熱）
- `scripts/` — 検証・可視化・解析・論文図・バッチ dispatch（200+ ファイル）
- `docs/` — 設計/検証ドキュメント（`ARCHITECTURE.md` が pipeline 概要）
- `wiki_repo/` — GitHub Wiki ソース（`images/` 含む）
- `dataset_*/`, `data/processed_*/` — データセット & PyG 前処理済み（gitignore）
- `runs/` — 学習ログ（TensorBoard, checkpoints, gitignore）
- `results/` — 公開可能な評価成果物（図・JSON・HTML ダッシュボード）
- `experiments/` — `quantum/`, `uq/` 実験
- `papers/` — 文献 & 参照リポジトリ clone（`papers/repos/` は内部 gitignore）
- `portfolio/` — 半導体企業別ポートフォリオ資料（本研究と独立）

## Research Lines（src/ 内の主要サブ研究）
| ライン | 主なエントリ | 目的 |
|--------|------------|------|
| **静的 GNN 検出** | `train.py`, `models.py`, `build_graph.py` | 曲率対応グラフ + GAT/GCN/GIN/SAGE で node 単位欠陥検出 |
| **FEM 生成** | `generate_fairing_dataset.py`, `generate_realistic_dataset.py`, `run_batch.py` | Abaqus H3 フェアリング + 熱荷重/デボンド欠陥 |
| **Guided-Wave (GW)** | `generate_guided_wave.py`, `generate_gw_fairing.py`, `build_gw_graph.py`, `train_gw.py`, `train_gw_stgnn.py` | 弾性波センサ時刻歴 → グラフ → SHM 分類 |
| **2段/3段 SHM** | `fairing_stage2.py`, `run_two_stage_pipeline.py` | Stage-1 検出 → Stage-2 特性同定 → Stage-3 予後（flight clearance） |
| **ドメイン適応 / sim2real** | `domain_adapt.py`, `payload_da_gw.py`, `prototype_dann.py`, `scripts/*ogw*` | FEM→実測ギャップ補正、OGW（Open Guided Waves）conformal 検出 |
| **PRAD（自己教師+蒸留）** | `src/prad/train_mae.py`, `src/prad/distill_fno2gnn.py`, `src/prad/finetune_distilled.py` | PI-GraphMAE 事前学習 → FNO 物理知識を GNN へ蒸留 |
| **代理モデル（surrogate）** | `train_fno*.py`, `models_fno*.py`, `prototype_deeponet.py`, `prototype_pinn.py`, `mifno_gw.py` | FNO / DeepONet / PINN で FEM 高速化 |
| **UQ（不確かさ定量化）** | `pce_driver.py`, `pce_advanced.py`, `reliability_analysis.py`, `uncertainty.py` | PCE / 信頼性解析 |
| **基盤モデル** | `anomalygfm_shm.py`, `chronos_shm.py`, `benchmark_foundation.py`, `gpn_shm.py` | AnomalyGFM / Chronos-2 / Poseidon・DPOT ベンチ |
| **フェアリング分離動力学** | `generate_fairing_separation.py`, `build_separation_graph.py`, `extract_separation_results.py` | Abaqus/Explicit 分離機構（破断ボルト・火工品） |
| **Virtual Twin** | `src/vt/orchestrator.py` | T-0→SECO 全フェーズ 6DOF 飛行統合シミュレーション |
| **量子** | `models_quantum.py`, `train_quantum.py`, `train_quantum_node.py` | 量子 GNN プロトタイプ（任意依存） |

## Docs Map
- `docs/ARCHITECTURE.md` — pipeline データフロー + モジュール層
- `docs/ZENODO.md`, `CITATION.cff`, `.zenodo.json` — アーカイブ / 引用
- `AGENTS.md` — AI エージェント向け簡易エントリ
- `README.md` — 公開向け概要 + Quick Start
- `wiki_repo/Home.md` — フル索引・ステータス・ナビゲーション
- `ROADMAP.md`, `LITERATURE_REVIEW.md`, `MESHGRAPHNET_VARIANTS.md`, `TEMPERATURE_ROBUSTNESS.md` ほか — 研究メモ
