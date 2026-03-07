[← Home](Home) · [2-Year-Goals](2-Year-Goals) · [Roadmap-2028](Roadmap-2028)

# Structural Foundation Model (SFM) — Phase 3 詳細設計

> 最終更新: 2026-03-08
> **構造力学版の汎用基盤モデル — 未知の構造体にゼロショットで損傷検出**

---

## 日本語概要

構造力学における Physics Foundation Model を構築する。多種の構造体（フェアリング、平板、円筒殻等）で事前学習し、**学習していない構造体にもゼロショットで損傷検出**を実現する。GPhyT（流体力学版 Foundation Model）の成功に触発されつつ、構造力学・SHM に特化した設計とする。構造力学版はまだ世界的に空白地帯であり、先駆者となることを目指す。

---

## 1. なぜ Foundation Model か

### 1.1 現状の限界

| 問題 | 現在のアプローチ | Foundation Model |
|------|----------------|------------------|
| 新構造体への適用 | FEM再生成 + GNN再学習 (数週間) | **ゼロショット or 5-shot (数分)** |
| データ効率 | 構造体ごとに数千サンプル必要 | 事前学習の知識を転移 |
| 物理法則の活用 | 損失関数に手動で追加 | **自動的に物理を学習** |
| スケーラビリティ | 小規模モデル (100K params) | 大規模 (10M–200M params) |

### 1.2 先行事例と空白地帯

| Foundation Model | ドメイン | 状態 |
|-----------------|---------|------|
| [GPhyT](https://arxiv.org/abs/2509.13805) | 流体力学・一般物理 | 発表済 (2025) |
| [Walrus](https://techxplore.com/news/2026-01-foundation-ai-physics-words-scientific.html) | マルチフィジックス | 発表済 (2026) |
| AlphaFold | タンパク質構造 | 発表済 (2024) |
| **構造力学 / SHM** | **構造損傷検出** | **★ 空白地帯 ★** |

---

## 2. アーキテクチャ設計

### 2.1 全体構成: Graph Transformer + Physics Heads

```
Input: Multi-Structure Graph
  │
  ├── Node Features (34–50 dim)
  │     Position, Displacement, Stress, Strain,
  │     Temperature, Material Properties, ...
  │
  ├── Edge Features (5–8 dim)
  │     Relative position, Distance, Normal angle,
  │     Material interface flag, ...
  │
  └── Global Features (10 dim)
        Structure type, Loading condition, Scale, ...

  ↓
┌─────────────────────────────────┐
│  Patch-based Graph Tokenizer   │
│  (METIS partitioning → tokens) │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│  Graph Transformer Backbone    │
│  12 layers, d=768, 12 heads    │
│  + Graph Positional Encoding   │
│    (Laplacian eigenvectors)    │
│  + Structure-Type Embedding    │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│  Task-Specific Heads           │
│  ├── Damage Detection (cls)    │
│  ├── Damage Localization (seg) │
│  ├── Stress Prediction (reg)   │
│  ├── Remaining Life (reg)      │
│  └── Anomaly Score (unsup)     │
└─────────────────────────────────┘
```

### 2.2 Patch-based Graph Tokenizer

大規模グラフ（120K+ nodes）を直接 Transformer に入力するのはメモリ不足。METIS グラフ分割でパッチ化:

| パラメータ | 値 |
|-----------|-----|
| パッチサイズ | ~500 nodes/patch |
| パッチ数 | ~240 patches/graph (120K nodes) |
| パッチ特徴量 | Set Transformer で集約 → 768 dim |
| パッチ間エッジ | 隣接パッチ間に edge を定義 |

### 2.3 Graph Positional Encoding

標準的な PE（正弦波等）はグラフには不適。代わりに:

| 手法 | 説明 |
|------|------|
| Laplacian Eigenvectors | グラフラプラシアンの固有ベクトル上位 k 個 |
| Random Walk PE | ランダムウォーク遷移確率 |
| Structure-Type Embedding | 構造体タイプ（フェアリング=0, 平板=1, ...）を学習可能な埋め込み |

### 2.4 モデルサイズ

| バリアント | Layers | Dim | Heads | Params | GPU Memory |
|-----------|--------|-----|-------|--------|------------|
| SFM-Small | 6 | 384 | 6 | ~10M | ~4 GB |
| SFM-Base | 12 | 768 | 12 | ~50M | ~16 GB |
| SFM-Large | 24 | 1024 | 16 | ~200M | ~48 GB (2×4090) |

---

## 3. 事前学習

### 3.1 Stage 1: Masked Node Prediction (MNP)

[PI-GraphMAE](Physics-Residual-Anomaly-Detection) の拡張版:

| 項目 | 仕様 |
|------|------|
| マスク率 | 50% のノード特徴量をマスク |
| 損失 | MSE (マスクされたノードの特徴量復元) |
| データ | 全28K samples（全構造体） |
| 学習 | 4×RTX 4090, ~3日 (SFM-Base) |
| 目的 | 構造力学の汎用的な特徴表現を獲得 |

### 3.2 Stage 2: Physics-Informed Contrastive Learning (PICL)

物理法則を埋め込む対照学習:

```
Positive pairs:
  - 同じ構造体の異なるメッシュ解像度
  - 同じ欠陥パターンの異なる荷重条件

Negative pairs:
  - 異なる欠陥タイプ
  - 健全 vs 損傷

Physics Loss:
  L_phys = ‖∇·σ + f‖² + ‖ε - ½(∇u + ∇uᵀ)‖²
```

| 項目 | 仕様 |
|------|------|
| 対照損失 | InfoNCE (温度 τ=0.07) |
| 物理損失 | 応力平衡 + ひずみ-変位適合 |
| 損失合計 | L = L_InfoNCE + 0.1 × L_phys |
| 学習 | 4×RTX 4090, ~2日 |

### 3.3 Stage 3: Multi-Task Fine-Tuning

| タスク | 損失 | 重み |
|--------|------|------|
| Damage Detection | Focal Loss | 1.0 |
| Damage Localization | Dice + CE | 0.5 |
| Stress Prediction | MSE | 0.3 |
| Remaining Life | Huber Loss | 0.2 |

---

## 4. 学習データ

### 4.1 データ構成

| 構造体 | サンプル数 | 欠陥タイプ | ソース |
|--------|-----------|-----------|--------|
| H3フェアリング (CFRP/Al-HC) | 10,000 | 7タイプ | Abaqus + FNO |
| CFRP平板 | 5,000 | Debond, Delam, Impact | Abaqus + JAX-FEM |
| 円筒殻 (金属/CFRP) | 5,000 | 腐食, 亀裂, Debond | Abaqus |
| スティフナ付きパネル | 3,000 | Debond, 亀裂 | Abaqus |
| 外部データセット | 5,000 | 各種 | OGW, NASA, DINS-SHM |
| **合計** | **28,000** | — | — |

### 4.2 特徴量の統一

異なる構造体間で特徴量を統一するための正規化:

| 特徴量 | 正規化方法 |
|--------|-----------|
| 座標 | 構造体サイズで正規化 → [0, 1] |
| 変位 | 最大変位で正規化 |
| 応力 | 材料の降伏応力で正規化 |
| 温度 | (T - T_ref) / ΔT_max |
| 材料特性 | 新規追加: E, ν, ρ, α (正規化) |

---

## 5. Zero-Shot / Few-Shot 評価

### 5.1 評価プロトコル

| 設定 | 説明 |
|------|------|
| **Zero-Shot** | 学習データに含まれない構造体で直接推論 |
| **5-Shot** | 5サンプルでファインチューン後に推論 |
| **20-Shot** | 20サンプルでファインチューン後に推論 |
| **Full** | 全データでファインチューン（上限性能） |

### 5.2 評価用構造体（Hold-Out）

| 構造体 | データソース | 備考 |
|--------|------------|------|
| 衛星パネル (Al-HC) | 自作 FEM (100 samples) | 学習に含めない |
| 航空機翼構造 (CFRP) | NASA CompDam (50 samples) | 形式変換して使用 |
| 圧力容器 (金属) | 公開データ (50 samples) | 完全に異なるドメイン |

### 5.3 目標性能

| 設定 | F1 (Damage) | ROC-AUC | 備考 |
|------|-------------|---------|------|
| Zero-Shot | > 0.65 | > 0.85 | 論文のメインクレーム |
| 5-Shot | > 0.75 | > 0.90 | 実用的なフォールバック |
| 20-Shot | > 0.85 | > 0.95 | 既存手法を大幅に凌駕 |
| Full | > 0.92 | > 0.98 | 上限性能 |

---

## 6. スケーリング則

### 6.1 検証実験

| 変数 | 範囲 | 測定値 |
|------|------|--------|
| モデルサイズ | 10M, 50M, 200M | Val Loss |
| データ量 | 1K, 5K, 15K, 28K | Val Loss |
| 構造体数 | 1, 2, 3, 5 | Zero-Shot F1 |

### 6.2 期待される結果

```
Val Loss ∝ N^{-α}  (N = データ量)
Val Loss ∝ P^{-β}  (P = パラメータ数)

α, β を推定 → 構造力学における Neural Scaling Law を初めて実証
```

---

## 7. 論文構成（Nature Computational Science 向け）

| Section | 内容 |
|---------|------|
| Abstract | 構造力学版Foundation Model、Zero-Shot損傷検出 |
| 1. Introduction | Foundation Modelの成功（LLM, AlphaFold）→ 構造力学は空白 |
| 2. SFM Architecture | Graph Transformer + Physics-Informed Pre-training |
| 3. StructuralBench Dataset | 28K samples, 5構造体, 7欠陥タイプ |
| 4. Scaling Laws | データ/モデルサイズ vs 性能のPower Law |
| 5. Zero-Shot Generalization | 未知構造体での損傷検出（F1 > 0.65） |
| 6. Ablation | 物理損失の効果、構造体数の効果 |
| 7. Discussion | 限界、今後の展望、社会的インパクト |
| Methods | 詳細な実装、学習手順、評価プロトコル |

---

## 8. 実装ロードマップ

| 月 | タスク | 成果物 |
|----|--------|--------|
| Month 9 | Graph Transformer実装 + 小規模テスト | コードベース |
| Month 10 | Tokenizer + PE 実装 | — |
| Month 11 | Stage 1 MNP 事前学習 | 事前学習済みモデル |
| Month 12 | Stage 2 PICL 学習 | 物理埋込みモデル |
| Month 13 | Multi-Task Fine-Tuning | タスク特化モデル |
| Month 14 | Zero-Shot / Few-Shot 評価 | 性能数値 |
| Month 15 | スケーリング則実験 | Power Law パラメータ |
| Month 16 | 論文執筆 + 投稿 | Nature Comp. Sci. 投稿 |

---

## 9. 関連ページ

| ページ | 内容 |
|--------|------|
| [2-Year-Goals](2-Year-Goals) | 全体目標 |
| [Roadmap-2028](Roadmap-2028) | 詳細ロードマップ |
| [Physics-Residual-Anomaly-Detection](Physics-Residual-Anomaly-Detection) | PI-GraphMAE (Stage 1 のベース) |
| [Cutting-Edge-ML](Cutting-Edge-ML) | Graph Transformer, Graph Mamba 等 |
| [Architecture](Architecture) | 現行パイプライン |
| [Two-Stage-Screening](Two-Stage-Screening) | FNO サロゲート (データ生成加速) |

---

## 10. 実装済み Foundation Model コンポーネント (2026-03)

| コンポーネント | ファイル | 状態 | 結果 |
|---------------|---------|------|------|
| Chronos-Bolt (GW異常検知) | `src/chronos_shm.py` | 完了 | 100% detection (102件, FP=4) |
| AnomalyGFM (グラフ異常検知) | `src/anomalygfm_shm.py` | 完了 | AUROC=0.75 (fine-tuned) |
| **GPN (Bayesian GNN)** | `src/gpn_shm.py` | **完了** | **AUROC=0.999, AUPRC=0.956** |
| MeshGraphNet (FEMサロゲート) | `src/physicsnemo_surrogate.py` | **完了** | R²=0.993–0.999 (推論テスト) |
| **サロゲートデータ拡張** | `scripts/augment_with_surrogate.py` | **完了** | 500件合成, 品質検証済み |
| **GPN + 拡張データ** | `src/gpn_shm.py` | **完了** | **AUROC=0.998, AUPRC=0.951** (1:1比率) |
| 静的-動的ハイブリッド融合 | `src/fuse_static_dynamic.py` | 完了 | 34→66 dim (PCA 99.8% variance) |
| Chronos埋め込み抽出 | `src/extract_chronos_embeddings.py` | 完了 | 102 samples processed |
| 統合実行スクリプト | `scripts/run_all_fm.sh` | 完了 | — |
| **Cross-Structure汎化テスト** | `src/cross_structure_eval.py` | **完了** | 下記参照 |

### Cross-Structure Generalization Results (Fairing → CompDam Flat Plate)

| Setting | AUROC | Notes |
|---------|:-----:|-------|
| Zero-shot | 0.24 | Transfer fails; epistemic uncertainty ~998 (OOD correctly detected) |
| 5-shot | **0.74** | Dramatic improvement with just 5 labeled nodes |
| 20-shot | **0.75** | Marginal gain over 5-shot |
| 50-shot | **0.78** | Steady improvement |

**Key findings:**
- Direct zero-shot transfer across different structures/defect types fails
- GPN epistemic uncertainty correctly identifies OOD data (~998 vs ~0.01 in-domain)
- Few-shot fine-tuning recovers significant performance (0.24 → 0.78)
- **Motivates Foundation Model pre-training** for true cross-structure generalization

---

## 11. 最新文献調査 (2024–2026)

→ **[Advanced SHM Literature Survey (2024–2026)](Advanced-SHM-Literature-2024-2026)**

### 次期実装候補（文献調査より）

| 優先度 | 手法 | 論文 | インパクト |
|--------|------|------|-----------|
| ~~P1~~ | ~~Graph Posterior Network (GPN)~~ | Stadler+, NeurIPS 2021 | **実装済み: AUROC=0.999** |
| P2 | CWT特徴量融合 | Xu+, SHM 2025 | Lamb波の周波数特徴をGNNに追加 |
| P3 | BNN推論層 | Cho+, 2025 | サロゲートモデルの信頼区間 |
| ~~P4~~ | ~~物理エッジ重み~~ | TPF-GNet, 2026 | **実装済み: PhysicsGPN AUROC=0.997** |
| P5 | Transformer MAE | Benfenati+, 2024 | 事前学習のバックボーン強化 |

---

## 12. 参考文献

| 論文 | 関連 |
|------|------|
| [GPhyT (2025)](https://arxiv.org/abs/2509.13805) | Physics Foundation Model の先行事例 |
| [Walrus / Polymathic AI (2026)](https://techxplore.com/news/2026-01-foundation-ai-physics-words-scientific.html) | マルチドメイン物理学習 |
| Rampášek et al. (2022) GPS | Graph Transformer のベース |
| He et al. (2022) GraphMAE | Masked Graph Autoencoder |
| Kaplan et al. (2020) Scaling Laws | Neural Scaling Law の原論文 |
| [Benfenati et al. (2024)](https://arxiv.org/abs/2404.02944) | Foundation Models for SHM |
| [Stadler et al. (2021)](https://arxiv.org/abs/2110.14012) | Graph Posterior Network (Bayesian GNN) |
| [Xu et al. (2025)](https://journals.sagepub.com/doi/10.1177/14759217241311942) | CFRP Deep Transfer Learning (CWT + Lamb Wave) |
| [Cho et al. (2025)](https://arxiv.org/abs/2512.03115) | BNN for Digital Twin SHM |
| [Song et al. (2026)](https://journals.sagepub.com/doi/10.1177/14759217241305050) | Physics-Guided NN for Lamb Wave |
