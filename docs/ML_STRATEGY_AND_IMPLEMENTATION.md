# Advanced ML Strategy & Implementation Plan for H3 Fairing SHM

**Project**: Payload Fairing Defect Detection using Geometric Deep Learning & Physics-Informed AI  
**Target**: JAXA H3 Rocket (CFRP/Aluminum Honeycomb Sandwich Structure)

---

## 1. Academic Context & Problem Formulation

### 1.1 The Challenge: "Curse of Dimensionality" & "Geometric Distortion"
Traditional Computer Vision (CNNs) operates on Euclidean grids (images). However, aerospace structures like the H3 Fairing are **non-Euclidean 2-Manifolds** embedded in 3D space.
*   **Metric Distortion**: Projecting a curved Ogive nose cone onto a 2D plane (UV mapping) introduces area and angle distortions, degrading defect localization accuracy.
*   **Anisotropy**: CFRP laminates have direction-dependent wave velocities ($c(\theta)$). Standard isotropic convolution kernels cannot capture this physics.
*   **Multi-Scale Physics**: Guided waves ($\lambda \approx 1-5$ cm) propagate over a huge structure ($L \approx 10$ m). Discretizing this requires millions of degrees of freedom (DOFs), making standard FEM computationally prohibitive for real-time monitoring.

### 1.2 Our Solution: Physics-Geometric AI
We propose a multi-modal AI framework that respects the underlying physics and geometry of the fairing.

---

## 2. Methodology: Theoretical Foundations

### 2.1 Geometry-Aware Graph Neural Networks (GA-GNN)
**Concept**: Instead of pixels, we operate on the **Mesh Graph** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.
*   **Manifold Learning**: We define convolution on the manifold surface using Message Passing Neural Networks (MPNN).
    $$ h_i^{(l+1)} = \gamma^{(l)} \left( h_i^{(l)}, \square_{j \in \mathcal{N}(i)} \, \phi^{(l)}(h_i^{(l)}, h_j^{(l)}, e_{ij}) \right) $$
*   **Curvature-Adaptive Attention**: To handle the Ogive shape, we inject **Principal Curvatures** ($\kappa_1, \kappa_2$) and **Surface Normals** ($\mathbf{n}$) into the edge features $e_{ij}$. This allows the network to distinguish between "wave focusing due to geometry" and "scattering due to defects."
*   **Anisotropic Aggregation**: Edge weights are modulated by the angle between the edge vector $\mathbf{r}_{ij}$ and the fiber orientation $\mathbf{d}_{fiber}$, mimicking the physical wave velocity profile.

### 2.2 Fourier Neural Operators (FNO)
**Concept**: Learning the **Solution Operator** of the Wave Equation, not just a specific solution.
*   **Operator Learning**: We learn a mapping $G_\theta: a \to u$, where $a$ is the defect parameter field (e.g., stiffness distribution) and $u$ is the wave field response.
*   **Resolution Invariance**: FNOs learn in the frequency domain (Fourier space).
    $$ (K u)(x) = \mathcal{F}^{-1} (R \cdot \mathcal{F}(u))(x) $$
    This allows training on coarse FEM meshes and evaluating on dense experimental data without retraining (Zero-Shot Super-Resolution).
*   **Application**: Rapid generation of "Digital Twin" wavefields to augment training data.

### 2.3 Physics-Informed Neural Networks (PINNs)
**Concept**: Embedding the **Elastodynamic Wave Equation** directly into the Loss Function.
*   **Governing Equation**:
    $$ \rho \ddot{\mathbf{u}} = \nabla \cdot \boldsymbol{\sigma} + \mathbf{f} $$
*   **Loss Function**:
    $$ \mathcal{L} = \mathcal{L}_{data} + \lambda_{PDE} || \rho \frac{\partial^2 \mathbf{u}}{\partial t^2} - \nabla \cdot (\mathbf{C} : \nabla \mathbf{u}) ||^2 $$
*   **Inverse Problem**: We treat the stiffness tensor $\mathbf{C}(x)$ as a trainable parameter. By minimizing the physics loss against sparse sensor data, the network "discovers" the debonded region (where stiffness $\mathbf{C}$ drops) automatically.

---

## 3. Implementation Plan (Roadmap)

### Phase 1: Baseline Geometric Learning ✅
**Goal**: Establish the superiority of Mesh-based learning over 2D projections.
*   [x] **Data Generation**: Abaqus scripting for H3 Fairing (Cylinder + Ogive).
*   [x] **Graph Construction**: `build_graph.py` incorporating ($x,y,z, n_x,n_y,n_z, \kappa_1, \kappa_2$).
*   [x] **Model Implementation**: `src/models.py` (GCN, GAT, GIN, SAGE) + `src/models_point.py` (PointTransformer)
*   [x] **Training**: `src/train.py` (Focal Loss, 5-Fold CV, Early Stopping)
*   [x] **UV Baseline**: `src/run_uv_2d.py` (UV Mapping + U-Net)

### Phase 2: Operator Learning for Surrogate Modeling ✅ (Prototype)
**Goal**: Accelerate data generation by 1000x using Neural Operators.
*   [x] **FNO**: `src/models_fno.py` + `src/prototype_fno.py` (2D合成データ, IoU ~0.75, 超解像検証済み)
*   [x] **DeepONet**: `src/prototype_deeponet.py` (Branch-Trunk構成, 解像度非依存)
*   [ ] **Production**: Abaqus FEMデータでの本格学習

### Phase 3: Physics-Informed Refinement ✅ (Prototype)
**Goal**: Pinpoint defects with sparse sensors using PINNs.
*   [x] **PINN**: `src/prototype_pinn.py` (1D波動方程式の逆問題, 欠陥領域の波速低下推定)
*   [ ] **Production**: 2D/3D曲面への拡張, FNO事前学習との連携

### Phase 4: Sim-to-Real Domain Adaptation ✅ (Prototype)
**Goal**: Bridge the gap between Abaqus and Real H3 Data.
*   [x] **DANN**: `src/prototype_dann.py` (Gradient Reversal Layer, プログレッシブλスケジュール)
    *   Feature Extractor (共有) + Label Predictor + Domain Discriminator
    *   Source-onlyベースラインとの比較機能付き
*   [ ] **Production**: Abaqus H3 (Source) → Open Guided Waves (Target) での実データ適応.

### Phase 5: Advanced Architectures ✅ (Prototype)
**Goal**: Explore next-generation architectures for scalability and physical correctness.
*   [x] **Graph Mamba**: `src/prototype_graph_mamba.py` (Selective SSM, O(N) linear complexity)
    *   Graph Transformer の O(N²) に対し線形計算量でスケーラブル
    *   大規模メッシュ (>50K nodes) での長距離波動伝播パターン捕捉
*   [x] **E(3)-Equivariant GNN**: `src/prototype_equivariant_gnn.py` (EGNN, Satorras et al. 2021)
    *   回転・並進に対する同変性を理論保証
    *   SE(3) 同変性の数値検証付き
*   [ ] **Production**: PyG Data統合, e3nn ライブラリ活用, 実データベンチマーク

---

## 4. Repository Structure for "Perfect GitHub"

```
Payload2026/
├── docs/
│   ├── ML_STRATEGY_AND_IMPLEMENTATION.md  <-- (This Document)
│   ├── LITERATURE_REVIEW.md               <-- (Competitor Analysis)
│   └── API_REFERENCE.md                   <-- (Code Documentation)
├── src/
│   ├── models/
│   │   ├── gnn_layers.py       # Custom Anisotropic Message Passing
│   │   ├── fno_3d.py           # Fourier Neural Operator
│   │   └── pinns.py            # Physics-Informed Loss Modules
│   ├── geometry/
│   │   ├── curvature.py        # Discrete Differential Geometry (DDG)
│   │   └── meshing.py          # Adaptive Remeshing
│   └── pipelines/
│       ├── train_surrogate.py  # Phase 2
│       └── inverse_solver.py   # Phase 3
├── data/
│   └── raw/                    # Abaqus ODBs
└── README.md                   # Professional Landing Page
```

## 5. References
1.  **Li, Z., et al.** "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021*.
2.  **Raissi, M., et al.** "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems." *Journal of Computational Physics (2019)*.
3.  **Bronstein, M. M., et al.** "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges." *arXiv:2104.13478 (2021)*.
