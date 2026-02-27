[← Home](Home)

# Literature Review: Guided Wave SHM for Composite Structures

> 最終更新: 2026-02-28

---

## 日本語版（概要）

本ドキュメントは、CFRP ペイロードフェアリングの構造ヘルスモニタリング（SHM）に Graph Neural Networks（GNN）と有限要素法（FEM）を適用するうえで関連する主要文献をまとめたものである。

### 主なトピック

1. **Open Guided Waves (OGW) データセット** — 誘導波 SHM のベンチマーク用プラットフォーム。CFRP 平板、温度変動データ、オメガストリンガー剥離データを含む。
2. **GNN による波ベース SHM** — 深層学習を用いた逆問題（損傷位置推定）の解法。
3. **フェアリング構造の FEM モデリング** — ハニカムサンドイッチ、シェル要素、グローバル・ローカル解析の戦略。
4. **航空宇宙分野の GNN アプローチ** — スパースセンサ vs 高密度メッシュグラフ、振動モード vs 誘導波の比較。
5. **本研究の新規性** — 幾何情報付きグラフ構築、高密度メッシュグラフ、物理に基づく異方性、UV マッピング vs 多様体学習のベンチマーク。

---

## English Version

This document summarizes key literature relevant to applying Graph Neural Networks (GNNs) and Finite Element Method (FEM) for Structural Health Monitoring (SHM) of CFRP Payload Fairings.

## 1. Open Guided Waves (OGW) Dataset & Benchmarks

The Open Guided Waves platform is the primary data source for benchmarking guided wave-based SHM algorithms.

*   **Moll, J., et al. (2019). "Open Guided Waves: Online Platform for Ultrasonic Guided Wave Measurements." *Structural Health Monitoring*, 18(6), 1903–1914.**
    *   **Significance**: Introduces the OGW platform.
    *   **Dataset 1 (Carbon Fiber Composite Plate)**:
        *   **Specimen**: Quasi-isotropic CFRP plate ($500 \times 500 \times 2$ mm), layup $[45/0/-45/90]_{s}$.
        *   **Sensors**: 12 piezoelectric transducers (PWAS) for SHM (pitch-catch).
        *   **Damage**: Reversible "pseudo-defect" (attached aluminum mass) to simulate local stiffness change without permanent damage.
        *   **Relevance**: Provides a clean baseline for training GNNs on experimental wave propagation data in flat CFRP.

*   **Moll, J., et al. (2019). "Temperature affected guided wave propagation in a composite plate complementing the Open Guided Waves Platform." *Scientific Data*, 6, 191.**
    *   **Significance**: Extends Dataset 1 with temperature variations ($20^\circ$C to $60^\circ$C).
    *   **Relevance**: Crucial for developing robust GNNs that can distinguish between environmental effects (global stiffness change) and damage (local scattering). Fairings experience extreme thermal gradients.

*   **Dataset with Omega Stringer (Zenodo Record 5105861)**
    *   **Specimen**: CFRP plate with a bonded omega stringer.
    *   **Damage Types**:
        *   Stringer debonding (critical for fairings).
        *   Impact damage.
    *   **Relevance**: This is the closest public proxy to a payload fairing's skin-stringer or skin-core structure. The "debonding" defect perfectly mimics facesheet-core separation in honeycomb sandwiches.

## 2. GNNs for Wave-Based SHM

*   **Rautela, M., et al. (2021). "Deep Inverse Neural Surrogates for SHM."** (Repo: `DINS-SHM`)
    *   **Concept**: Solves the inverse problem (localization) using deep learning on wave data.
    *   **Relevance**: Demonstrates how to map time-series wave data from sparse sensors to damage coordinates.

## 3. FEM Modeling of Fairing Structures

*   **NASA (Various Technical Reports)**
    *   **Key Insight**: Payload fairings are typically honeycomb sandwich structures (e.g., Aluminum honeycomb core with CFRP facesheets).
    *   **Modeling Strategy**:
        *   **Shell Elements**: Conventional shell elements (S4R) for facesheets.
        *   **Solid/Continuum Shell**: For the honeycomb core to capture transverse shear and crushing.
        *   **Global-Local Approach**: Global shell model for buckling loads $\rightarrow$ Local solid model for defect analysis.

## 4. Recent GNN Approaches in Aerospace (Competitor Analysis)

Recent studies have begun applying Graph Neural Networks to aerospace structures, but significant gaps remain regarding curved, anisotropic sandwich structures.

*   **"Real-Time Damage Detection and Localization on Aerospace Structures" (2024)**
    *   **Method**: Uses GNNs on a graph where nodes are strain sensors and edges represent spatial proximity.
    *   **Limitation**: Relies on a **sparse sensor network**. The graph connectivity is artificial (sensor-to-sensor), not physical (material continuity).
    *   **Contrast**: Our approach uses the **entire FEM mesh (or high-density point cloud)** as the graph. This allows the GNN to learn the actual wave propagation physics through the continuum, rather than just correlating sparse sensor outputs.

*   **"Semi-supervised vibration-based structural health monitoring via deep graph neural network"**
    *   **Method**: Combines 1D-CNN for time-series vibration data with a Transformer-based Graph Neural Network for spatial correlation.
    *   **Limitation**: Primarily focused on global vibration modes (low frequency) rather than guided waves (high frequency, local scattering).
    *   **Contrast**: We target **guided waves (50-300 kHz)** which are sensitive to small, local defects like skin-core debonding, whereas vibration modes are often insensitive to such local damage in large stiff structures.

*   **"G-Twin: Graph neural network-based digital twin" (2025)**
    *   **Method**: Reconstructs full stress fields from sparse sensor data using GNNs for offshore wind turbines.
    *   **Relevance**: Similar concept of "Mesh-to-Graph" reconstruction.
    *   **Contrast**: Our work introduces **Geometry-Aware Node Features** (Surface Normals, Curvature, Fiber Orientation) specifically to handle the **mode conversion** that occurs in cylindrical/ogive payload fairings, which is not a primary concern in beam/plate-like wind turbine structures.

## 5. Novelty of Our Research (Selling Points)

Based on the review above, our research introduces the following unique contributions:

1.  **Geometry-Aware Graph Construction**: Unlike standard GNNs that only use $(x,y,z)$ coordinates, we explicitly encode **Surface Normal Vectors** and **Principal Curvatures** into the node features. This is critical for payload fairings where curvature causes wave focusing/defocusing and mode conversion.
2.  **High-Density Mesh Graph**: We treat the structure as a continuous manifold (thousands of nodes) rather than a sparse sensor graph (dozens of nodes). This enables high-resolution defect localization.
3.  **Physics-Informed Anisotropy**: We incorporate the **Fiber Orientation Vector** at each node to account for the direction-dependent wave velocity in CFRP, which standard isotropic GNNs fail to capture.
4.  **Systematic Benchmark**: We provide the first direct comparison of **UV-Mapping (2D CNN)** vs. **Manifold Learning (GNN / Point Transformer)** for defect detection on complex aerospace shells.

## Action Plan for Research
1.  **Validation**: Use OGW Dataset 1 (Flat Plate) to validate the "Wave-to-Graph" conversion pipeline.
2.  **Extension**: Use OGW Stringer Dataset to test debonding detection.
3.  **Application**: Train the validated GNN architecture on the synthetic Abaqus data (Cylindrical Fairing) generated in this project.
