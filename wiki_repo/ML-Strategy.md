[← Home](Home)

# ML Strategy: JAXA H3 Fairing SHM (2026)

> **日本語概要**: シミュレーション検証 (Phase 3) から実験的 PoC (Phase 4) への移行が目標。GNN/FNO は FEM データで有望だが、Sim-to-Real ギャップが実機適用の最大リスク。ベンチマーク → DANN によるドメイン適応 → Physics-Informed 学習の順で推進。**超最先端 ML** (Graph Mamba, Equivariant GNN) は [Cutting-Edge-ML](Cutting-Edge-ML) を参照。

---

## 1. Executive Summary
The goal is to transition from **simulated validation (Phase 3)** to **experimental proof-of-concept (Phase 4)**. Current GNN/FNO models show promise on FEM data, but the "Sim-to-Real" gap remains the primary risk for deployment on the H3 rocket.

## 2. Current Model Portfolio

### 2a. GNN ベンチマーク結果 (2026-03-01)

**データ**: `processed_realistic_25mm_27` (25 サンプル, 8 クラス, ~120K ノード/グラフ)
**環境**: 3 サーバー並列 (vancouver01/02: RTX 4090 x4, stuttgart02: RTX 3090 x4)
**損失関数**: Focal Loss (γ=2, per-class alpha)

| Model | Params | GPU Mem | Best Val F1 | Note |
| :--- | :--- | :--- | :--- | :--- |
| **SAGE** | 117K | ~1 GB | **0.2496** | full_graph, hidden=128 |
| GCN | 64K | ~1 GB | 0.1937 | full_graph, hidden=128 |
| GAT | 168K | ~3.4 GB | 0.0563 | defect_centric, hidden=64 |
| GAT | 634K | OOM | — | full_graph, hidden=128 → 24GB 超過 |

**現状の課題**:
- 全モデルとも **healthy のみ検出** (F1≈0.97)、欠陥クラスは F1=0.000
- 原因: 極端なクラス不均衡 (healthy 99.58% vs 欠陥 0.01–0.15%) + データ 25 サンプル
- **次のステップ**: データ 100 サンプルでの再学習、Optuna HP 探索

### 2b. 代替モデル

| Model | Type | Strengths | Weaknesses | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Point Transformer** | 3D Point Cloud | Handles complex curvature well; Permutation invariant. | Computationally heavy ($O(N^2)$ attention); Slow inference. | Implemented (`models_point.py`) |
| **FNO (Fourier Neural Operator)** | Operator Learning | **Resolution invariant**; Extremely fast inference (FFT). | Struggles with irregular boundaries/meshes; complex to adapt to non-grid data. | Implemented (`models_fno.py`) |
| **UV-Net (U-Net)** | 2D CNN | Mature architecture; fast; easy to interpret. | Requires UV mapping (distortion); loses 3D geometric context. | Implemented (`run_uv_2d.py`) |
| **Graph Mamba** | State Space Model | O(N) スケール; 長距離依存に強い | 密な隣接行列が必要; PyG 統合未完 | プロトタイプ済 |
| **E(3)-Equivariant GNN** | 等変 GNN | 物理的整合性; データ効率 | 球面調和関数への変換が必要 | プロトタイプ済 |

## 3. Strategic Roadmap (Next 6 Months)

### Phase 3.5: Rigorous Benchmarking (Immediate)
*   **Action**: Establish a unified `BenchmarkDataset` class.
*   **Metrics**:
    *   **Accuracy/F1**: Defect detection rate.
    *   **IoU (Intersection over Union)**: Localization precision.
    *   **Inference Time (ms)**: Critical for real-time monitoring during launch.
*   **Goal**: Downselect to 1 primary architecture for Phase 4.

### Phase 4: Sim-to-Real Domain Adaptation (Critical)
*   **Problem**: FEM data is noise-free. Real sensor data (Open Guided Waves / JAXA tests) has noise, coupling variability, and environmental effects.
*   **Strategy**:
    1.  **Data Augmentation**: Inject Gaussian noise, sensor dropout, and time-warping into FEM training data.
    2.  **Domain Adversarial Training (DANN)**: Train a feature extractor that cannot distinguish between FEM and Experiment domains, while correctly classifying defects.
    3.  **CycleGAN**: Translate "Clean FEM Wavefields" $\leftrightarrow$ "Noisy Experimental Wavefields".

### Phase 5: Physics-Informed Learning (Research)
*   **Concept**: Constrain the model with the **Elastic Wave Equation**.
*   **Implementation**: Add a residual loss term $L_{physics} = || \nabla \cdot \sigma - \rho \ddot{u} ||^2$.
*   **Benefit**: Reduces data requirements; ensures physical consistency of predictions.

## 4. Cutting-Edge ML (2024–2025 SOTA)

| 手法 | 利点 | 参照 |
|------|------|------|
| **Graph Mamba** | 長距離依存・over-squashing 解消、O(N) スケール | [Cutting-Edge-ML](Cutting-Edge-ML) |
| **E(3)-Equivariant GNN** | 回転・並進不変、物理的整合性、データ効率 | 同上 |
| **FNO** | 解像度非依存、データ生成 1000x 加速 | 同上 |
| **PINN** | 物理制約、スパースセンサ逆問題 | 同上 |

## 5. 学習基盤 (完了)

| 機能 | 状態 |
|------|------|
| TensorBoard ロギング (loss, F1, AUC, precision, recall, lr, hparams) | ✅ |
| Focal Loss (per-class alpha, gamma 調整) | ✅ |
| DDP マルチ GPU 対応 (torchrun) | ✅ |
| DefectCentric サブグラフサンプラー | ✅ |
| CSV + TensorBoard 二重ロギング | ✅ |

## 6. Immediate Action Items
1.  [x] **GNN Benchmark**: SAGE/GCN/GAT を Focal Loss で比較 → SAGE が最良
2.  [ ] **データ拡充**: 100 サンプルで再学習 → クラス不均衡の影響を評価
3.  [ ] **Optuna HP 探索**: 4 GPU 並列で 100 trials
4.  [ ] **Graph Mamba 統合**: プロトタイプを train.py パイプラインに統合
5.  [ ] **Synthesize Noise**: ロバスト性テスト用ノイズ付きテストセット作成
