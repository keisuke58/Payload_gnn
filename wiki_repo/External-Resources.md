[← Home](Home)

# External Resources — Open-Source Tools & Public Datasets

> **GNN-SHM プロジェクトに活用可能なオープンソースツール・公開データセットの調査結果**
> 最終更新: 2026-03-07

関連ページ: [Literature-Review](Literature-Review) · [Dataset-Survey](Dataset-Survey) · [Dataset-Comparison](Dataset-Comparison) · [Guided-Wave-Simulation](Guided-Wave-Simulation)

## 導入ステータス

| リソース | 状態 | 場所 | 備考 |
|---------|------|------|------|
| **DINS-SHM (Zenodo)** | ✅ 取得・展開済 | `data/external/dins_shm/` | 903MB, 等方性+複合材 waveguide |
| **CONCEPT** | ✅ **取得済** | `data/external/concept/DATASET_PLATEUN01/` | .mat形式, 18条件 (H7+D11) |
| **NASA CompDam_DGD** | ✅ **Abaqus 2024 検証済** | `external/CompDam_DGD/` | コンパイル+テスト完了 |
| **Open Guided Waves** | ✅ 取得済 | `data/external/` | 既存 |
| **NASA Composites** | ✅ 取得済 | `data/external/NASA_Composites/` | 既存 (4.3GB) |

---

## 導入推奨度

| Tier | 意味 | 基準 |
|------|------|------|
| **S** | 即導入推奨 | 本プロジェクトの手法・対象と直接一致 |
| **A** | 強く推奨 | 手法が近く、比較・ベンチマークに有用 |
| **B** | 参考推奨 | 方法論やデータ形式が参考になる |
| **C** | 周辺参考 | 間接的に参考になるツール・データ |

---

## 1. SHM・ガイド波関連 (Tier S/A)

### 1.1 DINS-SHM — Deep Inverse NDE via SHM [Tier S]

| 項目 | 内容 |
|------|------|
| **GitHub** | [mahindrautela/DINS-SHM](https://github.com/mahindrautela/DINS-SHM) |
| **データセット** | [Zenodo DOI: 10.5281/zenodo.13844147](https://zenodo.org/records/13844147) |
| **手法** | ガイド波 + スペクトル有限要素 (SFEM) → CNN/LSTM で損傷検出・位置特定 |
| **データ形式** | 時系列波形 + CWT 時間-周波数画像 |
| **ノイズ対応** | 複数レベルのガウスノイズで汚染したデータを含む |
| **フレームワーク** | TensorFlow 2.0, Jupyter Notebook |

**本プロジェクトへの導入ポイント:**
- ガイド波ベースの損傷検出という点で **最も手法が近い**
- SFEM シミュレーションデータ → 本プロジェクトの Abaqus Explicit GW データとの **比較ベンチマーク** に最適
- CNN/LSTM 結果 vs GNN 結果の定量比較が可能
- 複合材 waveguide での検証実績あり

**取得済データ構造 (2026-03-07):**
```
data/external/dins_shm/
├── 1_SDE_isotropicDL.zip        (651 MB) — 等方性 waveguide データ
├── 2_SDE_compositeDL.zip        (211 MB) — 複合材 waveguide データ
├── isotropic/                   ← 展開済
│   ├── 0_DataSet/               — Ax/Flex × D/UD (各2500)
│   ├── 3_CWT/                   — CWT画像 (D/UD/WithNoise)
│   └── 4_TestSet/               — 100/150kHz テストセット
└── composite/                   ← 展開済 (本PJに直接関連)
    ├── 0_DataSet/
    │   ├── Ax2500Comp_D.txt     — 軸方向波 × 損傷 (2500 × 8192 time steps)
    │   ├── Ax2500Comp_UD.txt    — 軸方向波 × 健全 (2500 × 8192)
    │   ├── Flex2500Comp_D.txt   — 曲げ波 × 損傷 (2500 × 8192)
    │   └── Flex2500Comp_UD.txt  — 曲げ波 × 健全 (2500 × 8192)
    ├── 2_Labels/
    │   └── labels.csv           — (位置[0.1~0.85], サイズ[0.005~0.1]) × 2500
    └── 3_CWT/                   — CWT 時間-周波数画像
```

**データ仕様:**
- 各サンプル: 8,192 time steps の波形 (CSV カンマ区切り)
- 損傷ラベル: 位置 50段階 (0.1~0.85) × サイズ 50段階 (0.005~0.1) = 2,500 組合せ
- 値域: ~10^-13 ~ 10^-2 (変位応答)

---

### 1.2 CONCEPT — Composite Lamb Wave Dataset [Tier S]

| 項目 | 内容 |
|------|------|
| **GitHub** | [shm-unesp/DATASET_PLATEUN01](https://github.com/shm-unesp/DATASET_PLATEUN01) |
| **概要** | 複合材プレートのラム波計測データ (健全 + 損傷状態) |
| **計測** | PZT センサアレイ、pitch-catch 方式 |
| **欠陥** | デラミネーション (層間剥離) |
| **提供元** | UNESP Ilha Solteira SHM Lab |

**本プロジェクトへの導入ポイント:**
- **実験データ** によるシミュレーション結果の妥当性検証
- PZT pitch-catch は本プロジェクトのセンサ構成と **完全一致**
- 複合材 + ラム波 + デラミネーション = プロジェクトの3要素が揃う
- Sim-to-Real ギャップの定量評価に活用可能

**取得済データ構造 (2026-03-07):**
```
data/external/concept/DATASET_PLATEUN01/data/
├── EXPERIMENT_INFO.mat  — 実験パラメータ (PZT数, Fs=5MHz, 250kHz, etc.)
├── H_T0.mat ~ H_T60.mat — 健全状態 × 7温度 (0~60°C, 10°C刻み)
└── D1.mat ~ D11.mat      — 損傷状態 × 11段階 (パテ面積増加)
```

**データ形式:** 各 .mat ファイルに `data: (1000, 4, 100)` — 1000 time points × 4 PZT channels × 100 measurements

**ライセンス:** CC-BY-NC-SA。引用必須: Ferreira et al. (2024) MSSP, da Silva et al. (2020) JIMSS 等。

---

### 1.3 ConvLSTM for SHM/NDT [Tier A]

| 項目 | 内容 |
|------|------|
| **GitHub** | [xaviergoby/ConvLSTM-Computer-Vision-for-SHM-and-NDT](https://github.com/xaviergoby/ConvLSTM-Computer-Vision-for-Structural-Health-Monitoring-SHM-and-NonDestructive-Testing-NDT) |
| **Stars** | 47 |
| **手法** | ConvLSTM + ガイド波 + PZT センサ |
| **言語** | Python |

**本プロジェクトへの導入ポイント:**
- 時系列ガイド波データの LSTM 処理手法が参考になる
- GNN (空間的) vs ConvLSTM (時空間的) の比較論文の題材に
- PZT データ前処理パイプラインの参考実装

---

### 1.4 WaveDL — Wave-Based Inverse Problems [Tier A]

| 項目 | 内容 |
|------|------|
| **GitHub** | [ductho-le/WaveDL](https://github.com/ductho-le/WaveDL) |
| **Stars** | 43 |
| **手法** | 波動伝播逆問題の DL フレームワーク (弾性波・超音波) |
| **応用** | 材料特性同定、欠陥検出、地震波解析 |

**本プロジェクトへの導入ポイント:**
- 波動逆問題としての定式化 → Physics-Residual Anomaly Detection との理論的接点
- 材料特性同定モジュールは CFRP 異方性パラメータ推定に応用可能

---

## 2. 複合材 FEM ツール (Tier S/A)

### 2.1 NASA CompDam_DGD — Composite Damage Model [Tier S]

| 項目 | 内容 |
|------|------|
| **GitHub** | [nasa/CompDam_DGD](https://github.com/nasa/CompDam_DGD) |
| **種別** | Abaqus/Explicit VUMAT サブルーチン |
| **対応 Abaqus** | 2021 (2024でも互換の可能性あり) |
| **要素** | C3D8R (主), C3D8, C3D6, COH3D8 等 |
| **コンパイラ** | Intel Fortran 11.1+ |

**対応損傷モード:**

| 損傷モード | 手法 | 本プロジェクトとの関連 |
|-----------|------|----------------------|
| **マトリクス引張/圧縮** | DGD + B-K 混合モード則 | CFRP スキンの樹脂クラック |
| **繊維引張破断** | 最大ひずみ + バイリニア軟化 | スキン破壊の上限評価 |
| **繊維キンク (座屈)** | FKT (Fiber Kinking Theory) | 圧縮荷重下の CFRP 破壊 |
| **界面摩擦** | Alfano-Sacco モデル | デボンディング後の接触挙動 |
| **疲労** | Cohesive fatigue model | 音響疲労下の剥離進展 |

**本プロジェクトへの導入ポイント:**
- 現行の CZM (COH3D8) モデルの **損傷進展の精度向上**
- NASA 品質の VUMAT → 論文での信頼性向上
- 疲労モデルは将来の **音響疲労下デボンディング進展** シミュレーションに必須
- `examples/` ディレクトリにテストケース付属 → 即座に検証可能

**導入方法:**
```bash
git clone https://github.com/nasa/CompDam_DGD.git external/CompDam_DGD
cd external/CompDam_DGD
# Intel Fortran でコンパイル
abaqus make library=for/CompDam_DGD.for
# テスト実行
abaqus job=tests/test_matrix_tension double=both
```

**検証結果 (2026-03-07):**
- ✅ Abaqus 2024 + Intel Fortran 2021.9.0 でコンパイル成功
- ✅ `test_C3D8R_elastic_fiberTension` テスト正常完了 (`THE ANALYSIS HAS COMPLETED SUCCESSFULLY`)
- ⚠️ `double=both` (倍精度) 必須
- 📂 配置先: `external/CompDam_DGD/`

---

### 2.2 OpenAeroStruct [Tier B]

| 項目 | 内容 |
|------|------|
| **GitHub** | [mdolab/OpenAeroStruct](https://github.com/mdolab/OpenAeroStruct) |
| **手法** | 空力 + 構造 連成最適化 (VLM + FEM Beam) |
| **フレームワーク** | OpenMDAO |

**本プロジェクトへの導入ポイント:**
- フェアリング空力荷重の簡易推定ツールとして活用
- Max Q 時の圧力分布を得るための補助ツール

---

## 3. SHM 公開データセット集 (Tier A/B)

### 3.1 ベンチマークデータセット一覧

| データセット | 構造 | 計測 | 欠陥タイプ | 本PJとの親和性 |
|-------------|------|------|-----------|--------------|
| **[CONCEPT](https://github.com/shm-unesp/DATASET_PLATEUN01)** | 複合材プレート | ラム波 PZT | デラミネーション | **★★★** |
| **[DINS-SHM (Zenodo)](https://zenodo.org/records/13844147)** | 等方性/複合材 waveguide | SFEM シミュレーション | 損傷位置・サイズ | **★★★** |
| **[Open Guided Waves](https://zenodo.org/records/5105861)** | CFRP プレート | ガイド波実験 | ステッカー損傷 | **★★★** |
| **[BERT](https://github.com/shm-unesp/DATASET_BOLTEDBEAM)** | Al ボルト結合梁 | 加速度 | ボルト緩み | ★★ |
| **[MAGNOLIA](https://github.com/shm-unesp)** | 磁気弾性系 | 振動 | 剛性変化 | ★ |
| **[Impact-Events](https://github.com/Smart-Objects/Impact-Events-Dataset)** | プラスチック薄板 | PZT + IoT | 衝撃位置 | ★★ |

### 3.2 産業データセット (Tier B/C)

| データセット | 概要 | 提供元 |
|-------------|------|--------|
| **C-MAPSS** | 航空エンジン劣化シミュレーション | NASA |
| **CWRU Bearing** | ベアリング振動 + 故障診断 | Case Western Reserve U. |
| **FEMTO Bearing** | Run-to-failure 加速度データ | FEMTO-ST |

参考: [awesome-industrial-datasets](https://github.com/jonathanwvd/awesome-industrial-datasets)

---

## 4. 軌道・飛行シミュレーション (Tier C)

| リポジトリ | 概要 | Stars | 言語 |
|-----------|------|-------|------|
| **[RocketPy](https://github.com/RocketPy-Team/RocketPy)** | 6-DOF 高性能ロケット軌道シミュレータ | 1.5k+ | Python |
| **[OpenRocket](https://github.com/openrocket/openrocket)** | モデルロケット設計・シミュレータ | 1k+ | Java |
| **[MAPLEAF](https://github.com/henrystoldt/MAPLEAF)** | 6-DOF フレームワーク | 200+ | Python |
| **[CamPyRoS](https://github.com/cuspaceflight/CamPyRoS)** | Cambridge Python Rocketry Simulator | — | Python |

**本プロジェクトへの導入ポイント:**
- フェアリング分離時の荷重条件推定 (RocketPy の 6-DOF 解析)
- Max Q の高度・時刻推定 → 熱・圧力荷重条件の設定根拠

---

## 5. 導入優先度ロードマップ

### Phase 1: 即時導入 — ✅ 完了 (2026-03-07)

| # | ツール/データ | 状態 | 結果 |
|---|-------------|------|------|
| 1 | **DINS-SHM (Zenodo)** | ✅ 取得・展開済 | 等方性+複合材 10,000サンプル, 8192 time steps |
| 2 | **CONCEPT** | ✅ 取得済 | .mat形式, 18条件 (H×7温度 + D×11段階) |
| 3 | **NASA CompDam_DGD** | ✅ Abaqus 2024 検証済 | コンパイル+テスト成功 |

### Phase 2: 中期導入 (1-2 months)

| # | ツール/データ | アクション | 期待効果 |
|---|-------------|----------|---------|
| 4 | **CompDam_DGD VUMAT** | 既存 CZM モデルに統合 | CFRP 損傷モデル精度向上 |
| 5 | **DINS-SHM ベンチマーク** | GW 波形比較スクリプト作成 | GNN vs CNN/LSTM 比較 |
| 6 | **CONCEPT 検証** | 実験データ vs Abaqus FEM 比較 | Sim-to-Real ギャップ定量化 |

### Phase 3: 長期活用 (3+ months)

| # | ツール/データ | アクション | 期待効果 |
|---|-------------|----------|---------|
| 6 | **WaveDL** | 波動逆問題定式化の参考 | PRAD 手法の理論強化 |
| 7 | **RocketPy** | 飛行荷重条件の補助推定 | 荷重条件の根拠強化 |

---

## 6. ベンチマーク比較結果 (2026-03-07)

スクリプト: `scripts/benchmark_external_datasets.py`

### Fig. B1: Guided Wave Waveform Comparison
3データセットの波形を横並び比較。本PJ (Abaqus Explicit) vs DINS-SHM (SFEM) vs CONCEPT (実験)。

![Waveform Comparison](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/external_benchmark/01_waveform_comparison.png)

### Fig. B2: Frequency Content Comparison
各データセットの PSD 比較。本PJは50kHz帯にピーク、CONCEPTは250kHz、DINS-SHMは正規化周波数。

![Frequency Comparison](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/external_benchmark/02_frequency_comparison.png)

### Fig. B3: CONCEPT — Temperature Sensitivity
温度変化（0-60°C）によるラム波信号の変動。40°C以上でRMSが顕著に増加 → 温度補償の重要性を示す。

![CONCEPT Temperature](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/external_benchmark/03_concept_temperature.png)

### Fig. B4: DINS-SHM — Damage Sensitivity Map
損傷位置・サイズに対する Damage Index (RMS比)。サイズ増加に伴いDIが単調増加 → サイズ推定の可能性。

![DINS Damage Map](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/external_benchmark/04_dins_damage_map.png)

### Fig. B5: Cross-Dataset Summary
3データセットの仕様を横断比較した一覧表。

![Cross Dataset Summary](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/external_benchmark/05_cross_dataset_summary.png)

**主な知見:**
- 本PJの Abaqus Explicit データは DINS-SHM (SFEM) と同オーダーの波動応答 → シミュレーション手法は妥当
- CONCEPT の実験データは温度依存性を明確に示しており、本PJの熱応力モデリングの重要性を裏付け
- DINS-SHM の DI マップは損傷サイズとの強い相関を示し、GNN によるサイズ推定の実現可能性を支持

---

## 7. GitHub SHM コミュニティ

| トピック | URL | リポジトリ数 |
|---------|-----|------------|
| structural-health-monitoring | [GitHub Topics](https://github.com/topics/structural-health-monitoring) | 100+ |
| predictive-maintenance | [GitHub Topics](https://github.com/topics/predictive-maintenance) | 300+ |
| aerospace | [GitHub Topics](https://github.com/topics/aerospace) | 500+ |
| non-destructive-testing | [GitHub Topics](https://github.com/topics/non-destructive-testing) | 50+ |
