# Literature: Dynamic Analysis + GNN-SHM for Composites

Collected: 2026-03-08

---

## A. CFRP + GNN Damage Detection (Core)

### 1. PIGMind — Physics-informed GNN for CFRP damage localization
- **Title**: Interpretable and physics-informed graph neural modeling for intelligent damage localization in CFRP composites
- **Journal**: Composites Part B, 2026 (published 2026-02-04)
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S1359836826000909
- **PDF**: 未取得（ScienceDirect有料）
- **Abstract/Method**:
  - PIGMind = Physics-Informed Graph Modeling with Individualized Dynamics
  - 物理由来のエネルギー指標をグラフ構築に埋め込み、隣接行列が超音波ガイド波の方向依存性を明示的に表現
  - CFRP板に8センサアレイを配置した実験で検証
  - **平均位置推定誤差 10.18mm**（グラフベース手法比で7.8%改善）
  - 異方性・ノイズ条件下でも安定した性能
- **Relevance**: 我々のアプローチ（FEA物理量→ノード特徴量→GNN）に最も近い。物理情報のグラフ構造への組み込み方が直接参考になる

### 2. CFRP-former — Baseline-free Lamb wave + GCN + Transformer
- **Title**: Baseline-free assisted Lamb wave-based damage detection in CFRP composites using graph convolutional networks and Transformer models
- **Journal**: Composites Science and Technology, 2024 (published 2024-11-07)
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S026322412402044X
- **PDF**: 未取得（ScienceDirect有料）
- **Abstract/Method**:
  - CFRP-former = 3モジュール構成: GCNN + MLP + Transformer Encoder
  - Lamb波信号を時空間グラフとして構築し入力
  - GCNN: センサ-損傷間のトポロジカル関係をモデリング、マルチスケール空間特徴抽出
  - MLP: 次元削減（後段の情報保持）
  - Transformer Encoder: マルチヘッドアテンションで時系列パターンを捕捉
  - **ベースラインフリー**（健全時の基準信号不要）→ 実運用で大きな利点
  - シミュレーション事前学習 + ドメイン差異ベースの動的ファインチューニング（過学習抑制）
  - Hilbert畳み込みモジュールでクラス間分離・クラス内凝集を改善
- **Relevance**: 動的波動データ×GCN×Transformerの具体手法。動解析結果をグラフにマッピングする際の参考

### 3. g-SDDL — Vibration data → GCN structural damage detection
- **Title**: Structural damage detection framework based on graph convolutional network directly using vibration data
- **Journal**: Structures, 2022
- **URL**: https://www.researchgate.net/publication/358399533
- **PDF**: 未取得（ResearchGateで要リクエスト）
- **Abstract/Method**:
  - g-SDDL = Graph-based Structural Damage Detection and Localization
  - センサ配置をグラフ構造として振動応答を**直接**GCN入力（手動特徴量抽出不要）
  - グラフ理論で空間データ、畳み込みで時系列データを同時処理
  - 損傷検出・位置推定・定量化の3タスクで**精度 >90%**
  - 複数損傷シナリオ: 各損傷に対応するGCNモデルをスタック
  - 低い時間計算量、複雑な前処理不要
- **Relevance**: 動解析シミュレーション結果をGNNに食わせる発想の原点。我々の approach に直接つながる

---

## B. Spatio-Temporal GNN (Dynamic Response Modeling)

### 4. GNSS — Spatiotemporal GNN for structural response
- **Title**: Graph Neural Network-Based Spatiotemporal Structural Response Modeling in Buildings
- **Authors**: Fangyu Liu, Yongjia Xu, Junlin Li, Linbing Wang
- **Journal**: Journal of Computing in Civil Engineering, Vol 39(2), 2025
- **DOI**: 10.1061/JCCEE5.CPENG-6229
- **URL**: https://ascelibrary.org/doi/abs/10.1061/JCCEE5.CPENG-6229
- **PDF**: 未取得（ASCE有料）
- **Abstract/Method**:
  - GNSS = Graph Network-based Structure Simulator
  - 6階建物全体をグラフとして表現（ノード=質量、エッジ=柱・梁）
  - 構造部材の空間位置・接続関係 + 時系列構造データの時間相関を同時にモデリング
  - Encoder → Processor → Decoder の3段構成
  - 4つのバリエーション（GNSS-NE, GNSS-N2E, GNSS-NUEU, GNSS-Full）で特徴統合・アーキテクチャを比較
- **Relevance**: フェアリングの動解析タイムステップデータを扱う際のアーキテクチャ参考。構造→グラフのマッピング手法が直接応用可能

### 5. TPS-GNN — Temporal periodicity + spatial distribution
- **Title**: TPS-GNN: Predictive model for bridge health monitoring data based on temporal periodicity and global spatial distribution characteristics
- **Authors**: Yu Liu, Lianzhen Zhang, Jiyu Xin, Sijie Peng
- **Journal**: Structural Health Monitoring, 2025
- **DOI**: 10.1177/14759217251330971
- **URL**: https://journals.sagepub.com/doi/10.1177/14759217251330971
- **PDF**: 未取得（SAGE有料）
- **Abstract/Method**:
  - 課題: 従来の多層GNNは over-smoothing でセンサ間の影響を区別できない。RNNは長期依存性・周期パターンの捕捉が弱い
  - 空間: 異なるホップ近傍の表現を直接利用し、局所・大域の空間依存性を正確に捕捉（over-smoothing 回避）
  - 時間: 日次・週次・季節の周期情報を位置エンコーディングとしてフロー データに統合
  - current model と historical model で連続的・周期的時系列データを別々に処理
  - **実橋梁2データセットで既存SOTAを最大78%改善**
- **Relevance**: 動解析の周期的応答（固有振動数近傍など）をGNNで捉えるアイデアの参考

### 6. LGSTA-GNN — Local-Global Spatiotemporal Attention
- **Title**: LGSTA-GNN: A Local-Global Spatiotemporal Attention Graph Neural Network for Bridge Structural Damage Detection
- **Journal**: Buildings (MDPI), 16(2), 348, 2025
- **URL**: https://www.mdpi.com/2075-5309/16/2/348
- **PDF**: 未取得（MDPI OA だがダウンロード403）
- **Abstract/Method**:
  - マルチスケール時間-周波数特徴抽出モジュール
  - Local graph feature extraction: グラフ畳み込みで固有の空間関係をモデリング
  - Global graph attention: 構造的に情報量の多いノードを適応的に強調し、センサ間依存性を捕捉
  - 精度・適合率・再現率・F1全てで従来のGNN variants / DL手法を上回る
  - t-SNE可視化と混同行列で識別能力の向上を検証
- **Relevance**: Local-Global の二段階アテンション構造は、フェアリングの局所（欠陥周辺）と大域（全体モード形状）の両方を捕捉する設計に参考になる

---

## C. Review Papers

### 7. ML for SHM of Aerospace Structures (2025 Review)
- **Title**: Machine Learning for Structural Health Monitoring of Aerospace Structures: A Review
- **Journal**: Sensors (MDPI), 25(19), 6136, 2025
- **URL**: https://www.mdpi.com/1424-8220/25/19/6136
- **Key**: 航空宇宙に特化した包括的レビュー。Supervised/unsupervised/hybrid全カバー。
- **PDF**: `MDPI_ML_SHM_Aerospace_2025.pdf` ✅

### 8. Deep Learning for SHM — Data, Algorithms, Applications (2023 Review)
- **Title**: Deep Learning for Structural Health Monitoring: Data, Algorithms, Applications, Challenges, and Trends
- **Journal**: Sensors (MDPI), 23(21), 8824, 2023
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC10650096/
- **Key**: DL×SHM の包括的レビュー。動的データの扱い方の整理に有用。
- **PDF**: 未取得（PMCページがHTML返却）

### 9. SHM in Composite Structures — Comprehensive Review (2022)
- **Title**: Structural Health Monitoring in Composite Structures: A Comprehensive Review
- **Journal**: Sensors (MDPI), 2022
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC8747674/
- **Key**: 複合材SHMの全体像。デラミネーション、繊維破断、衝撃損傷。
- **PDF**: 未取得

---

## D. Related / Bonus

### 10. Mechanics-informed Autoencoder (Nature Communications, 2024)
- **Title**: Mechanics-informed autoencoder enables automated detection and localization of unforeseen structural damage
- **Journal**: Nature Communications, 2024
- **DOI**: 10.1038/s41467-024-52501-4
- **URL**: https://www.nature.com/articles/s41467-024-52501-4
- **Key**: 力学の事前知識をAutoencoderに組み込み、未知の損傷パターンも検出。Physics-informed の別アプローチとして参考。
- **PDF**: `NatureComm_Mechanics_Informed_AE_2024.pdf` ✅

### 11. DL-based damage detection in composite laminates (Frontiers, 2024)
- **Title**: Damage detection of composite laminates based on deep learnings
- **Journal**: Frontiers in Physics, 2024
- **DOI**: 10.3389/fphy.2024.1456236
- **URL**: https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2024.1456236/full
- **Key**: 複合材積層板のDL損傷検出。表面ひずみ場からの検出。
- **PDF**: `Frontiers_DL_Composite_Laminates_2024.pdf` ✅

---

## Downloaded PDFs (in this directory)

| File | Paper | Status |
|------|-------|--------|
| `MDPI_ML_SHM_Aerospace_2025.pdf` | #7 ML for SHM Aerospace Review | ✅ |
| `NatureComm_Mechanics_Informed_AE_2024.pdf` | #10 Mechanics-informed AE | ✅ |
| `Frontiers_DL_Composite_Laminates_2024.pdf` | #11 DL Composite Laminates | ✅ |

### Not downloaded (paywalled)
- #1 PIGMind (Composites Part B) — ScienceDirect
- #2 CFRP-former (Comp Sci Tech) — ScienceDirect
- #3 Vibration GCN (Structures) — Elsevier
- #4 Spatiotemporal GNN (ASCE) — ASCE Library
- #5 TPS-GNN (SHMS) — SAGE
- #6 LGSTA-GNN (Buildings) — MDPI (403)

---

## Recommended Reading Order
1. **#1 PIGMind** — 最もプロジェクトに近い（CFRP × Physics-informed GNN）
2. **#2 CFRP-former** — 動的データ(Lamb波) × GCN の具体手法
3. **#7 Review** ✅ — 航空宇宙SHMの全体マップ（PDF取得済み）
4. **#10 NatComm** ✅ — Physics-informed の別解（PDF取得済み）
5. **#4 Spatiotemporal GNN** — 動解析タイムステップデータへの拡張
