# 文献調査: GW FNO サロゲート・Multi-Defect SHM

GW (Guided Wave) + FNO (Fourier Neural Operator) サロゲートモデル、
Multi-Defect FEM、データ拡張に関する関連文献の調査結果。

**調査日**: 2026-03-11
**調査方法**: Semantic Scholar API + 知識ベース + Gemini 補完

---

## 1. FNO / Neural Operator for GW Surrogate

波動伝播 FEM のサロゲートとして Neural Operator を使う研究。

| # | 論文 | 年 | 出典 | 引用 | GitHub |
|---|------|-----|------|------|--------|
| 1 | Li et al. "FNO for Parametric PDEs" | 2021 | ICLR | 3000+ | [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) |
| 2 | Rai & Mitra "DeepONet for Lamb Wave in Composite Laminates" | 2023 | Composite Structures | - | なし（DeepXDE で再現可能） |
| 3 | Lehmann et al. "MIFNO for 3D elastodynamics" | 2024 | J. Computational Physics | 17 | [lehmannfa/MIFNO](https://github.com/lehmannfa/MIFNO) + HEMEW3D データセット |
| 4 | Wang et al. "Transfer Learning FNO for Wave Equations" | 2024 | IEEE TGRS | 17 | - |
| 5 | Gao et al. "Dual-Stage FNO for Ultrasonic GW Corrosion Imaging" | 2025 | IUS | 0 | 未公開 |
| 6 | Li et al. "B-FNO for parametric acoustic wave" | 2025 | Engineering Computations | 4 | - |
| 7 | Li et al. "GINO: Geometry-Informed Neural Operator" | 2023 | NeurIPS | - | [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) |
| 8 | Wen et al. "U-FNO" | 2022 | Advances in Water Resources | - | neuraloperator 内 |
| 9 | Yang et al. "Neural Operator for Elastic Wave Simulation" | 2023 | arXiv | - | - |
| 10 | Song et al. "PI-FNO for Wave Equations" | 2023 | arXiv | - | - |

### 特に重要

- **Rai & Mitra (2023)**: 複合積層板の Lamb 波伝播を DeepONet で代替。FEM 比 10^4 倍高速。我々のプロジェクトと最も直接的に関連。
- **Gao et al. (2025)**: FNO + 超音波 GW の最新。スペクトルバイアス問題を Dual-Stage で解決。
- **Lehmann et al. (2024) MIFNO**: 3万件の3D弾性波シミュレーションデータセット (HEMEW3D) が公開。コード・データ共にオープン。
- **Li et al. (2023) GINO**: 3D 曲面形状（フェアリング等）に対応した Neural Operator。neuraloperator ライブラリに統合。

---

## 2. Multi-Defect SHM

複数欠陥の同時検出・特性評価に関する研究。

| # | 論文 | 年 | 出典 | 引用 | 備考 |
|---|------|-----|------|------|------|
| 1 | Shao et al. "Lamb wave multi-damage dataset + multi-task DL" | 2023 | SHM Journal | 9 | マルチタスク DL で複数損傷定量化 |
| 2 | Chen et al. "Lamb wave cross-domain damage identification for CFRP" | 2026 | MSSP | 0 | CFRP + Lamb 波 + 転移学習の最新 |
| 3 | Miorelli et al. "Simultaneous Detection of Multiple Defects" | 2024 | NDT&E Int. | - | エンコーダ共有型マルチタスク |
| 4 | Zhao et al. "GNN-Based Damage Detection in SHM" | 2023 | Smart Mat. & Struct. | - | GNN + SHM（我々と同じ枠組み） |
| 5 | Rautela et al. "Multiple Damage in Composites Using Lamb Waves + ML" | 2022 | Composite Structures | - | 複合材 + Lamb 波 + 複数欠陥 |
| 6 | De Fenza et al. "Multi-Damage Identification Using GW + DL" | 2023 | MSSP | - | CNN で損傷数・位置・サイズ推定 |
| 7 | Shao et al. "Multi-Task Damage ID for Composite Stiffened Plate" | 2025 | IWSHM | 0 | 最新の複合構造向け |

### データセット

- [shm-unesp/DATASET_PLATEUN01](https://github.com/shm-unesp/DATASET_PLATEUN01) — Lamb 波実験データセット

---

## 3. Data Augmentation for SHM / NDT

限定的な FEM データからの学習データ拡張手法。

| # | 論文 | 年 | 出典 | 備考 |
|---|------|-----|------|------|
| 1 | Sony et al. "Data Augmentation for DL-Based SHM: A Review" | 2023 | Sensors | 包括的レビュー |
| 2 | Liao et al. "Physics-Informed Data Augmentation for GW" | 2023 | SHM Journal | 物理制約付き拡張（Born 近似） |
| 3 | Fan et al. "GAN for SHM Data Augmentation" | 2023 | Eng. Structures | GAN ベース信号生成 |
| 4 | Melville et al. "Transfer Learning for GW Signals" | 2022 | MSSP | FEM → 実験への転移学習 |
| 5 | Abdeljaber et al. "Mixup/CutMix for SHM with Limited Data" | 2022-23 | MSSP | Mixup を振動ベース SHM に適用 |
| 6 | Sawant et al. "Few-Shot Learning for GW Damage Detection" | 2023 | NDT&E Int. | メタ学習フレームワーク |

### 我々の実装

- **Mixup** (Born 近似に基づく線形重畳): `src/dataset_fno_gw_augmented.py`
- **Noise injection** (SNR 20-40 dB): センサーノイズをモデル化
- **Amplitude scaling**: 線形弾性による振幅スケーリング
- **Multi-defect FEM**: 1 FEM に 18-45 欠陥を配置 → N サンプル生成

---

## 4. DL Surrogate for FEM Simulation

FEM を機械学習で代替するサロゲートモデル全般。

| # | 論文 | 年 | 出典 | GitHub |
|---|------|-----|------|--------|
| 1 | Pfaff et al. "MeshGraphNets" (DeepMind) | 2021 | ICLR | [deepmind-research/meshgraphnets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) |
| 2 | Li & Wang "Surrogate for FEM of Elastic Wave in Composites" | 2023 | AIP Advances | なし |
| 3 | Shukla et al. "PINN for Lamb Wave + Damage Detection" | 2020-23 | JCP / CMAME | - |
| 4 | You et al. "Operator Learning for Composite Materials" | 2023 | CMAME | - |
| 5 | Liang et al. "DL Surrogate for Fast FEA" | 2022 | CMAME | - |
| 6 | Wang & Perdikaris "Neural Operator for Long-Time Integration" | 2023 | Nature MI | - |

### PyG 版 MeshGraphNets

- TensorFlow 公式版に加え、PyTorch Geometric 版がコミュニティで多数公開
- 我々の PyG 環境 (2.7.0) で直接利用可能

---

## 5. ベンチマークデータセット & ライブラリ

| 名称 | 種類 | URL | 備考 |
|------|------|-----|------|
| **neuraloperator** | PyTorch ライブラリ | [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) | FNO, GINO, DeepONet 統合 |
| **OpenGuidedWaves** | GW 実験データ | [openguidedwaves.de](https://openguidedwaves.de/) | CFRP 板 PZT アレイ計測 |
| **HEMEW3D** | 3D 弾性波 FEM | lehmannfa/MIFNO | 3万件シミュレーション |
| **DATASET_PLATEUN01** | Lamb 波実験 | [shm-unesp](https://github.com/shm-unesp/DATASET_PLATEUN01) | 板構造 |
| **Long_Term_Guided_Waves** | GW 処理スクリプト | [SmartDATA-Lab](https://github.com/SmartDATA-Lab/Long_Term_Guided_Waves) | OpenGW 用 Python |
| **LANL SHM Dataset** | 振動 SHM | LANL 公式ポータル | ベンチマーク代表格 |
| **NASA Prognostics** | CFRP 疲労 | NASA Ames | AE 信号含む |

---

## 6. 我々のアプローチの位置づけ

### 新規性

1. **Multi-defect FEM → FNO サロゲート**: 1 FEM に 18-45 欠陥を配置し、局所センサーパッチで個別欠陥データを抽出。既存研究では single-defect FEM が主流。
2. **CFRP/Al-Honeycomb サンドイッチ構造**: フェアリング固有のサンドイッチ構造（CZM interface + 7種欠陥タイプ）は未踏。
3. **GNN + FNO ハイブリッド**: 静的 GNN (デボンディング検出) + 動的 FNO (GW サロゲート) の二段階アプローチ。
4. **Physics-informed augmentation**: Born 近似に基づく Mixup + 線形弾性スケーリング。

### 関連研究との比較

| 手法 | Rai & Mitra (2023) | Gao et al. (2025) | 我々 |
|------|--------------------|--------------------|------|
| Operator | DeepONet | Dual-Stage FNO | FNO1d |
| 構造 | CFRP laminate | Metal plate | CFRP/Al sandwich |
| 欠陥 | Single | Corrosion | Multi (7 types) |
| Data augmentation | なし | なし | Mixup + Noise + Scale |
| Multi-defect FEM | なし | なし | 18-45 defects/FEM |

---

## 7. 今後の調査課題

- [ ] Rai & Mitra (2023) の再現実装（DeepONet ベースライン比較）
- [ ] GINO を曲面フェアリングに適用する実験
- [ ] OpenGuidedWaves データでの FNO 検証
- [ ] PI-FNO (物理制約付き) による少数データ学習の検証
- [ ] GAN ベースのデータ拡張検討（Wave-GAN）
