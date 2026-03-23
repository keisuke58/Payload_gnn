# Graph Neural Network-based Structural Health Monitoring for CFRP/Al-Honeycomb Fairing: Multi-domain Defect Detection and Separation Anomaly Identification

## Authors
K. Nishioka et al.

---

## Abstract (draft)

We present a Graph Neural Network (GNN) framework for structural health monitoring (SHM) of CFRP/Al-honeycomb sandwich fairings used in the JAXA H3 launch vehicle. The framework addresses two complementary SHM tasks: (1) multi-class defect detection from guided wave (GW) and thermo-mechanical FEM simulations, and (2) bolt-stuck anomaly identification during fairing separation dynamics. A Graph Attention Network (GAT) with 34-dimensional node features and 5-dimensional edge features achieves AUC=0.992 on 5-class defect classification (560 graphs, 42k nodes/graph) and AUC=0.999 on separation anomaly detection (15 DOE cases). Cross-dataset validation on two public benchmark datasets (OGW #3 wavefield, NASA CFRP fatigue) confirms architecture generalizability. Transfer learning within the same structural domain improves F1 by 181%, while cross-domain transfer shows negligible benefit, suggesting domain-specific pretraining is critical.

**Keywords:** GNN, SHM, CFRP, composite fairing, defect detection, anomaly detection, transfer learning

---

## 1. Introduction
- H3 rocket fairing: CFRP/Al-HC sandwich, debonding risk
- SHM motivation: launch-critical structure, inspection cost
- GNN advantage over CNN for irregular FEM meshes
- Contributions:
  1. 34-dim physics-informed node features (curvature, fiber orientation, stress)
  2. Multi-task framework: defect detection + separation anomaly
  3. Cross-dataset validation on 4 datasets
  4. Transfer learning analysis (same-domain vs cross-domain)

## 2. Methodology

### 2.1 FEM Simulation Pipeline
- **GW-SHM**: Abaqus/Explicit, guided wave propagation at 50-200 kHz
  - 7 defect types: debonding, FOD, impact, delamination, inner_debond, thermal, acoustic fatigue
  - DOE: stratified LHS, 4 size categories (50-400mm)
  - Field output optimization: 190 GB → 2.1 GB (100x reduction)
- **Fairing Separation**: Abaqus/Explicit, CONN3D2 connector damage
  - Normal + 14 anomaly cases (1-12 stuck bolts, spring stiffness 300-8000 N/mm)
  - Opening spring forces via CLOAD

### 2.2 Graph Construction
- Curvature-aware graph from FEM mesh (build_graph.py)
- **Node features (34-dim)**:
  Position(3) + Normal(3) + Curvature(4) + Displacement(4) + Temperature(1) + Stress(5) + Thermal stress(1) + Strain(3) + Fiber orientation(3) + Layup(4) + Circumferential angle(1) + Boundary type(2)
- **Edge features (5-dim)**:
  Relative position(3) + Euclidean distance(1) + Normal angle(1)
- Physics-informed edge weight: SED-based (PIGMind-style)

### 2.3 GNN Architectures
- **GAT** (Graph Attention Network): multi-head attention, edge-aware
- **GCN** (Graph Convolutional Network): spectral-based
- **GraphSAGE**: sampling + aggregation
- **GIN** (Graph Isomorphism Network): WL-test powerful
- Common: 3 layers, hidden=64, focal loss (γ=2), cosine LR schedule

### 2.4 Training Strategy
- Focal loss for class imbalance (defect: 0.09-3% of nodes)
- Augmentation: DropEdge, FeatureNoise, circumferential flip
- Transfer learning: pretrained on GW-SHM → fine-tuned on separation

## 3. Experiments

### 3.1 Datasets

| Dataset | Source | Nodes/Graph | Graphs | Classes | Features |
|---------|--------|-------------|--------|---------|----------|
| GW-SHM (700v2) | Own FEM | 42k | 700 | 5 | 34-dim |
| Separation | Own FEM | 34k | 15 | 2 | 34-dim |
| OGW #3 | Public (Zenodo) | 23k | 2 | 2 | 12-dim |
| NASA CFRP | Public (NASA) | 12 | 84 | 2 | 8-dim |

### 3.2 Architecture Comparison (Table 1)

| Architecture | AUC | OptF1 | Dataset |
|-------------|-----|-------|---------|
| **SAGE** | 0.995 | **0.788** | 700v2 5-class |
| GAT | 0.992 | 0.758 | 700v2 5-class |
| GCN | — | 0.756 | 700v2 5-class |
| GIN | — | 0.737 | 700v2 5-class |

### 3.3 Cross-dataset Validation (Table 2)

| Dataset | AUC | OptF1 |
|---------|-----|-------|
| GW-SHM (700v2) | 0.992 | 0.758 |
| Separation (15 DOE) | 0.999 | 0.730 |
| OGW #3 wavefield | 1.000 | 1.000 |
| NASA CFRP fatigue | 0.924 | 0.800 |

### 3.4 Transfer Learning Analysis (Table 3)

| Method | Target F1 | Δ vs Baseline |
|--------|-----------|---------------|
| Baseline (random init) | 0.317 | — |
| Same-domain pretrain (GW→Sep) | 0.730 | **+130%** |
| Cross-domain pretrain (OGW3→GW) | 0.284 | -10% |
| MMD domain adaptation | 0.319 | +0.6% |

### 3.5 Prediction Localization (Figure X)
- GAT P(defect): defect nodes = 0.98, healthy nodes = 0.31
- Concentration ratio: **3.2x** on defect regions
- TP/FP/FN overlay visualization

### 3.6 Data Scaling (Figure X)
- Separation: 3 graphs → 15 graphs: F1 0.260 → 0.730 (+181%)
- Diminishing returns expected beyond ~50 graphs

## 4. Discussion
- Same-domain transfer >> cross-domain transfer
- Feature space mismatch (34-dim FEM vs 12-dim wavefield) limits cross-domain utility
- GAT attention is relatively uniform; prediction probability is a better localization metric
- SAGE slightly outperforms GAT on 5-class (0.788 vs 0.758)
- Fairing separation: CONN3D2 connector damage is a viable Explicit alternative to MODEL CHANGE

## 5. Conclusion
- GNN-SHM framework for H3 CFRP fairing: multi-defect + separation anomaly
- AUC > 0.99 on both tasks with transfer learning
- Cross-dataset validation confirms architecture generalizability
- Same-domain pretraining is critical; cross-domain shows negative transfer
- Future: 100+ GW samples, real sensor data validation, online SHM deployment

---

## Figures List
1. fig_fem_model_overview.png — Pipeline schematic
2. fig_overview_6panel.png — Separation dynamics (6 panels)
3. fig_prediction_localization.png — GAT defect localization (4 samples)
4. fig_gnn_comprehensive.png — GNN results summary
5. fig_spatial_displacement.png — Separation displacement field
6. fig_separation_stress.png — Stress comparison Normal vs Stuck
7. fairing_3d_viewer.html — Interactive 3D viewer (supplementary)
8. Architecture comparison bar chart (to generate)
9. Data scaling curve (to generate)
10. ROC curves per architecture (to generate)

## Tables List
1. Architecture comparison (SAGE/GAT/GCN/GIN)
2. Cross-dataset validation (4 datasets)
3. Transfer learning analysis (4 methods)
4. Dataset summary
5. Node feature description (34-dim)
