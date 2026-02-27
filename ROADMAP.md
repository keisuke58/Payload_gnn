# Payload Fairing SHM Development Roadmap

This roadmap outlines the development of a Graph Neural Network (GNN) based Structural Health Monitoring (SHM) system for JAXA Epsilon/H3 class payload fairings, focusing on guided wave propagation in CFRP honeycomb sandwich structures.

## Phase 1: High-Fidelity Data Generation (Current Focus)
**Goal**: Create a comprehensive dataset of guided wave propagation in curved honeycomb sandwich panels using FEM (Abaqus) and surrogate experimental data (Open Guided Waves).

*   [x] **Open Guided Waves Dataset**:
    *   Download and inspect `OGW_CFRP_Stringer_Wavefield_Intact.zip`.
    *   Develop parser for HDF5 wavefield data.
*   [ ] **Abaqus Simulation (Fairing Model)**:
    *   **Geometry**: 1/6th cylindrical shell (Radius ~1.3m for Epsilon S / ~2.7m for H3).
    *   **Material**:
        *   Skin: CFRP (Quasi-isotropic or Cross-ply).
        *   Core: Aluminum Honeycomb (Homogenized or detailed shell-element representation).
    *   **Physics**: Dynamic Explicit (Guided Lamb Waves, 50-300 kHz).
    *   **Defects**: Face sheet debonding (separation from core).
*   [ ] **Data Formatting**:
    *   Convert FEM outputs (ODB) to Graph format (Nodes, Edges, Features).

## Phase 2: Graph Construction & Feature Engineering
**Goal**: Represent the continuous curved shell structure as a graph suitable for GNNs, preserving geometric information.

*   **Graph Topology**:
    *   Nodes: FEM mesh nodes or sensor locations.
    *   Edges: k-nearest neighbors (k-NN) or Delaunay triangulation based on geodesic distance.
*   **Node Features**:
    *   $(x, y, z)$ coordinates (normalized).
    *   Surface normals $(n_x, n_y, n_z)$ to encode curvature.
    *   Material orientation vectors (critical for CFRP anisotropy).
*   **Edge Features**:
    *   Geodesic distance.
    *   Relative angle between surface normals (local curvature).

## Phase 3: GNN Architecture Development
**Goal**: Develop a Physics-Informed GNN to detect and localize debonding.

*   **Baseline Model**: GraphSAGE or GAT (Graph Attention Network) for transductive learning.
*   **Curvature-Awareness**: Implement **StructGNN**-inspired architecture or Coordinate-based MLPs (like PointNet++) integrated into message passing.
*   **Physics-Informed Loss**:
    *   Incorporate wave equation constraints or dispersion relation consistency.
*   **Input**: Time-series wave signals at sensor nodes.
*   **Output**: Probability map of defect presence on the fairing surface.

## Phase 4: Sim-to-Real Domain Adaptation
**Goal**: Bridge the gap between idealized FEM data and real-world experimental noise/attenuation.

*   **Domain Adaptation (DA)**:
    *   Use **Transfer Component Analysis (TCA)** or Adversarial Domain Adaptation.
    *   **Source Domain**: Abaqus FEM data (CFRP Honeycomb).
    *   **Target Domain**: Open Guided Waves data (CFRP Stringer) as a proxy for real experiments.
*   **Validation**:
    *   Train on FEM, test on OGW dataset.
    *   Quantify performance drop and recover using unlabeled target data.

## Phase 5: Deployment & Visualization
*   **Visualization**: 3D interactive mapping of damage probabilities on the fairing mesh.
*   **Inference**: Real-time processing of sensor array data.
