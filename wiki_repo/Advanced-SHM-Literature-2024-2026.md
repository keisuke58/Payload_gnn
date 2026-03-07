[← Home](Home) · [Literature-Review](Literature-Review) · [Foundation-Model](Foundation-Model)

# Advanced SHM Literature Survey (2024–2026)

> Last updated: 2026-03-07
> **GNN × Bayesian × Foundation Model × Physics-Informed: Recent advances for composite SHM**

---

## Overview

This page summarizes recent (2024–2026) literature relevant to our GNN-based SHM system for CFRP/Al-Honeycomb fairing debonding detection. Focus areas:

1. **Bayesian GNN / Uncertainty Quantification** — Trustworthy damage detection with confidence
2. **Foundation Models for SHM** — Pre-trained models, zero-shot, transfer learning
3. **GNN for Aerospace SHM** — Graph-based damage detection and localization
4. **Physics-Informed GNN** — Embedding physical laws into graph architectures
5. **Deep Transfer Learning for CFRP** — Cross-structure Lamb wave knowledge transfer
6. **Multi-Modal Fusion** — Combining FEA, sensor, and time-series data

---

## 1. Bayesian GNN / Uncertainty Quantification

### 1.1 Graph Posterior Network (GPN)

| Item | Detail |
|------|--------|
| **Paper** | Stadler et al., "Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification" |
| **Venue** | NeurIPS 2021 ([arXiv:2110.14012](https://arxiv.org/abs/2110.14012)) |
| **Code** | [github.com/stadlmax/Graph-Posterior-Network](https://github.com/stadlmax/Graph-Posterior-Network) |

**Architecture:**
```
Node Features → Normalizing Flow → Dirichlet Pseudo-Counts
  → PPR (Personalized PageRank) Message Passing
  → Aggregated Posterior → Class Probabilities + Uncertainty
```

**Key Contributions:**
- Three axioms characterizing expected uncertainty behavior in homophilic graphs
- Explicit Bayesian posterior updates via Dirichlet distributions on graph nodes
- Separates **aleatoric** (data noise) and **epistemic** (model ignorance) uncertainty
- Outperforms MC Dropout, Deep Ensembles, and DPN on OOD detection

**Relevance to Our Project:**
- **HIGH PRIORITY** — Direct replacement for our GAT classifier
- Each node outputs: P(defect class) + uncertainty estimate
- OOD detection → identifies novel/unseen defect types automatically
- Complements existing `src/uncertainty.py` (MC Dropout + Ensemble) with a principled Bayesian alternative
- PPR message passing naturally respects graph topology (mesh connectivity)

### 1.2 Real-Time SHM with Bayesian Neural Networks

| Item | Detail |
|------|--------|
| **Paper** | Cho et al., "Real-Time SHM with BNN: Distinguishing Aleatoric and Epistemic Uncertainty for Digital Twin Frameworks" |
| **Venue** | arXiv:2512.03115 (2025) |

**Methodology:**
- PCA dimensionality reduction + BNN with Hamiltonian Monte Carlo (HMC) inference
- Maps sparse strain gauge data → full-field strain distribution on CFRP specimens
- Separates aleatoric (sensor noise) vs epistemic (model limitation) uncertainty

**Key Results:**
- Strain field reconstruction R² > 0.9 on CFRP with varying crack lengths
- Real-time uncertainty maps even near crack-induced singularities

**Relevance:**
- Apply BNN layer on MeshGraphNet surrogate output → confidence intervals on predicted FEA fields
- Digital twin integration: uncertainty-aware structural assessment

### 1.3 Bayesian Framework for Composite SHM under Limited Data

| Item | Detail |
|------|--------|
| **Paper** | Ferreira et al., "Bayesian data-driven framework for structural health monitoring of composite structures under limited experimental data" |
| **Venue** | Structural Health Monitoring (2025), [DOI:10.1177/14759217241236801](https://journals.sagepub.com/doi/10.1177/14759217241236801) |

**Methodology:**
- Bayesian inference (MCMC) for FEM model updating from limited experimental data
- Neural network surrogate to accelerate MCMC sampling
- Compensates for data scarcity via principled uncertainty propagation

**Relevance:**
- Directly applicable to our 100-sample FEM dataset situation
- FEM parameter posterior → guides data augmentation strategy
- NN surrogate acceleration aligns with our MeshGraphNet approach

### 1.4 Bayesian Network Review for SHM

| Item | Detail |
|------|--------|
| **Paper** | "Bayesian Network in SHM: Theoretical Background and Applications Review" |
| **Venue** | Sensors (2025), [MDPI](https://www.mdpi.com/1424-8220/25/12/3577) |

Comprehensive review covering BN applications in SHM: damage identification, reliability assessment, remaining useful life prediction. Highlights the gap in BN+GNN integration for SHM.

---

## 2. Foundation Models for SHM

### 2.1 Transformer MAE Foundation Model for SHM

| Item | Detail |
|------|--------|
| **Paper** | Benfenati et al., "Foundation Models for Structural Health Monitoring" |
| **Venue** | arXiv:2404.02944 (2024, updated 2025) |

**Architecture:**
```
Vibration Signals → Tokenization → Masked AutoEncoder (Transformer)
  → Self-supervised Pre-training on Multiple Bridges
  → Fine-tune for: Anomaly Detection / Traffic Load Estimation
```

**Key Results:**

| Task | Foundation Model | Best Baseline | Improvement |
|------|-----------------|---------------|-------------|
| Anomaly Detection | 99.9% accuracy (15 windows) | 95.0% PCA (120 windows) | +4.9%, 8x faster |
| Traffic Load (light) | R²=0.97 | R²=0.91 (RF) | +0.06 |
| Traffic Load (heavy) | R²=0.90 | R²=0.84 (RF) | +0.06 |

**Additional Contributions:**
- Knowledge Distillation: compress large Transformer → edge device deployment
- Cross-structure generalization demonstrated across 3 viaducts

**Relevance:**
- Validates our masked feature reconstruction pre-training approach (`pretrain_foundation.py`)
- Motivates upgrade to Transformer backbone for stronger representations
- KD strategy applicable when deploying to embedded SHM systems

### 2.2 Our Current FM Implementation (for reference)

| Component | Current Status | Next Step |
|-----------|---------------|-----------|
| Chronos-Bolt | Zero-shot GW anomaly: 100% defect detection | Integrate CWT features |
| AnomalyGFM | Zero-shot graph anomaly: AUROC=0.54 (no domain pre-training) | Domain-specific fine-tuning |
| MeshGraphNet | Surrogate training in progress | Use for data augmentation |
| Masked Pre-training | `pretrain_foundation.py` implemented | Scale up to Transformer MAE |

---

## 3. GNN for Aerospace SHM

### 3.1 Real-Time Damage Detection on Aerospace Structures Using GNN

| Item | Detail |
|------|--------|
| **Paper** | "Real-Time Damage Detection and Localization on Aerospace Structures Using GNN" |
| **Venue** | MDPI Journal of Sensor and Actuator Networks (August 2025), [Link](https://www.mdpi.com/2224-2708/14/5/89) |

**Key Contributions:**
- Two GNN architectures: binary damage detection + spatial probability damage localization
- Composite aerospace structures as target domain
- Damage localization outputs probability distribution over structure

**Relevance:**
- Same domain (composite aerospace) — our work extends to curved sandwich structures
- Spatial probability output → complement our per-node classification with continuous probability maps

### 3.2 GNN-Based Probabilistic Graphical Model for Damage Detection

| Item | Detail |
|------|--------|
| **Paper** | Li et al., "Structural Damage Detection Using GNN-Based Probabilistic Graphical Models" |
| **Venue** | SSRN (2025, under review), [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5143083) |

**Methodology:**
- GNN trained for probabilistic graphical model inference on structural graphs
- Replaces conventional Belief Propagation and MCMC for damage localization
- Studies graph properties that influence size generalization

**Key Results:**
- Superior localization accuracy vs Belief Propagation and MCMC
- Higher computational efficiency (enables real-time)
- Generalizes to structures larger than training data

**Relevance:**
- Probabilistic damage propagation modeling → estimate spatial extent of debonding
- Size generalization → train on small meshes, deploy on full-scale fairing

---

## 4. Physics-Informed Approaches

### 4.1 Physics-Guided NN for Lamb Wave SHM (Boundary Reflection Elimination)

| Item | Detail |
|------|--------|
| **Paper** | Song et al., "Physics-guided neural network for structural health monitoring with Lamb waves through boundary reflection elimination" |
| **Venue** | SHM Journal (2026), [DOI:10.1177/14759217241305050](https://journals.sagepub.com/doi/10.1177/14759217241305050) |

**Methodology:**
- Multiscale Spatiotemporal (MSST) Fusion Network
- Eliminates boundary reflections from Lamb wave signals
- Enables precise Time-of-Flight (ToF) extraction of scattered waves from damage

**Key Result:**
- Enlarges detection area by removing boundary interference
- More precise damage localization in complex structures

**Relevance:**
- Pre-processing for our GW sensor data before Chronos embedding
- Fairing has strong boundary reflections from ring frames and openings
- Could significantly improve Chronos embedding quality

### 4.2 Temporal Power Flow Graph Network (TPF-GNet)

| Item | Detail |
|------|--------|
| **Paper** | "Research on structural damage identification based on temporal power flow graph network" |
| **Venue** | Scientific Reports (2026), [Link](https://www.nature.com/articles/s41598-026-37356-7) |

**Architecture:**
```
Sensors → Graph Nodes
Structural Connectivity → Graph Edges
  + Learnable Equivalent Stiffness (edge weight)
  + Learnable Damping (edge weight)
  → Energy flow simulation through GNN
  → Damage identification via energy anomaly
```

**Relevance:**
- **Directly applicable** to our edge features
- Current edge_attr: [dx, dy, dz, distance, normal_angle]
- Add: learnable stiffness + damping parameters → physics-informed edges
- Energy flow paradigm naturally captures wave propagation in composites

### 4.3 Physics-Informed GNN Conserving Momentum

| Item | Detail |
|------|--------|
| **Paper** | "A physics-informed graph neural network conserving linear and angular momentum for dynamical systems" |
| **Venue** | Nature Communications (2025), [Link](https://www.nature.com/articles/s41467-025-67802-5) |

**Relevance:**
- Demonstrates how to hard-code conservation laws (momentum) into GNN message passing
- Could enforce stress equilibrium (∇·σ + f = 0) in our surrogate model

---

## 5. Deep Transfer Learning for CFRP

### 5.1 Cross-Workpiece Deep Transfer Learning (CWTL) for CFRP Delamination

| Item | Detail |
|------|--------|
| **Paper** | Xu et al., "Deep transfer learning for delamination damage in CFRP composite materials" |
| **Venue** | SHM Journal (2025), [DOI:10.1177/14759217241311942](https://journals.sagepub.com/doi/10.1177/14759217241311942) |

**Methodology:**
```
Lamb Wave Signals → CWT (Continuous Wavelet Transform) → Time-Frequency Images
  → Pre-trained CNN (ResNet/VGG) → Feature Extraction
  → Domain Adaptation (MMD loss) → Cross-Workpiece Transfer
  → Damage Classification (Healthy / Delaminated)
```

**Key Contributions:**
- CWT converts 1D Lamb wave signals to 2D time-frequency representations
- Transfer learning across different CFRP structures without retraining
- Addresses data scarcity by transferring from data-rich to data-poor structures

**Relevance:**
- **CWT feature extraction** complements our Chronos embedding approach
- Chronos: captures temporal patterns in raw signal → global anomaly
- CWT: captures frequency content changes → mode-specific damage signatures
- Could fuse both: `x_node = [x_static(34) | x_chronos(32) | x_cwt(16)]`

### 5.2 Hybrid Deep Transfer Learning for Delamination in Composite Laminates

| Item | Detail |
|------|--------|
| **Paper** | "A Hybrid Deep Transfer Learning Framework for Delamination Identification in Composite Laminates" |
| **Venue** | Sensors (2025), [Link](https://www.mdpi.com/1424-8220/25/3/826) |

**Relevance:**
- Hybrid CNN+LSTM for temporal + spatial features
- Transfer from simulation to experimental data (sim-to-real gap)

---

## 6. Multi-Modal Fusion

### 6.1 GNN-Based Multimodal Sensor Fusion

| Item | Detail |
|------|--------|
| **Paper** | "Graph neural network-based multimodal sensor fusion" |
| **Venue** | SPIE (2025), [Link](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13682/1368205/Graph-neural-network-based-multimodal-sensor-fusion-for-robust-autonomous/10.1117/12.3073578.full) |

**Fusion Strategies:**
- **Early fusion**: Concatenate raw features before GNN
- **Intermediate fusion**: Modality-specific encoders → attention/GNN layers → combined
- **Late fusion**: Separate branches → decision-level combination

**Relevance:**
- Our current approach (fuse_static_dynamic.py) uses early fusion (concatenation)
- Consider upgrading to intermediate fusion with cross-attention between static FEA and dynamic GW features

---

## 7. Synthesis: Actionable Insights for Our Project

### 7.1 Priority Implementation Roadmap

| Priority | Technique | Source | Impact | Effort |
|----------|-----------|--------|--------|--------|
| **P1** | Graph Posterior Network (GPN) | §1.1 | Bayesian uncertainty on every node. OOD defect detection | Medium |
| **P2** | CWT Feature Fusion | §5.1 | Complement Chronos with frequency-domain features | Medium |
| **P3** | BNN Layer on Surrogate | §1.2 | Confidence intervals on MeshGraphNet predictions | Low |
| **P4** | Physics-Informed Edge Weights | §4.2 | Learnable stiffness/damping in edge_attr | Low |
| **P5** | Transformer MAE Pre-training | §2.1 | Stronger foundation model backbone | High |
| **P6** | Cross-Attention Fusion | §6.1 | Better static-dynamic feature interaction | Medium |

### 7.2 Architecture Evolution

```
Current Pipeline (2026-03):
  FEM Mesh → PyG Graph (34-dim nodes) → GAT/GCN → Node Classification

Proposed Next-Gen Pipeline:
  FEM Mesh → PyG Graph (34-dim nodes)
    + Chronos Embedding (32-dim, IDW interpolated)
    + CWT Features (16-dim, per-sensor)
    = 82-dim node features
    ↓
  Graph Posterior Network (GPN)
    + Physics-Informed Edge Weights (stiffness, damping)
    + Normalizing Flow → Dirichlet Posterior
    ↓
  Output: P(defect_class) + σ_aleatoric + σ_epistemic
    + Spatial Probability Map
    + OOD Flag (novel defect type)
```

### 7.3 Key Gaps in Literature (Our Novelty)

| Gap | Our Contribution |
|-----|------------------|
| No Bayesian GNN for composite SHM | First GPN application to CFRP debonding |
| No multi-modal (FEA+GW) graph fusion | Static-dynamic hybrid with IDW interpolation |
| No FM for structural graph data | Masked pre-training on high-density mesh graphs |
| No curved sandwich GNN | Curvature-aware features for cylindrical fairing |

---

## 8. References (BibTeX)

```bibtex
@inproceedings{stadler2021gpn,
  title={Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification},
  author={Stadler, Maximilian and Charpentier, Bertrand and Geisler, Simon and Z{\"u}gner, Daniel and G{\"u}nnemann, Stephan},
  booktitle={NeurIPS},
  year={2021}
}

@article{benfenati2024foundation,
  title={Foundation Models for Structural Health Monitoring},
  author={Benfenati, Luca and others},
  journal={arXiv:2404.02944},
  year={2024}
}

@article{cho2025realtime,
  title={Real-Time SHM with Bayesian Neural Networks},
  author={Cho, Hanbin and others},
  journal={arXiv:2512.03115},
  year={2025}
}

@article{xu2025transfer,
  title={Deep transfer learning for delamination damage in CFRP composite materials},
  author={Xu, Zhuo jun and Li, Hao and Yu, Jian bo and Jingwen, Yu},
  journal={Structural Health Monitoring},
  year={2025},
  doi={10.1177/14759217241311942}
}

@article{song2026physics,
  title={Physics-guided neural network for SHM with Lamb waves through boundary reflection elimination},
  author={Song, Yang and Shan, Shengbo and Zhang, Yuanman and Cheng, Li},
  journal={Structural Health Monitoring},
  year={2026},
  doi={10.1177/14759217241305050}
}

@article{li2025gnn,
  title={Structural Damage Detection Using GNN-Based Probabilistic Graphical Models},
  author={Li, Teng and Wu, Stephen and Huang, Yong and Li, Hui},
  journal={SSRN},
  year={2025}
}

@article{tpfgnet2026,
  title={Research on structural damage identification based on temporal power flow graph network},
  author={various},
  journal={Scientific Reports},
  year={2026}
}

@article{ferreira2025bayesian,
  title={Bayesian data-driven framework for SHM of composite structures under limited experimental data},
  author={Ferreira, Leonardo and others},
  journal={Structural Health Monitoring},
  year={2025},
  doi={10.1177/14759217241236801}
}
```

---

## Related Pages

| Page | Content |
|------|---------|
| [Literature-Review](Literature-Review) | Original literature review (OGW, FEM, GNN basics) |
| [Foundation-Model](Foundation-Model) | SFM Phase 3 detailed design |
| [Uncertainty-Quantification](Uncertainty-Quantification) | Current MC Dropout + Ensemble implementation |
| [Physics-Residual-Anomaly-Detection](Physics-Residual-Anomaly-Detection) | PI-GraphMAE approach |
| [Static-Dynamic-Fusion-Literature](Static-Dynamic-Fusion-Literature) | FEA + sensor fusion references |
| [Cutting-Edge-ML](Cutting-Edge-ML) | Graph Transformer, Graph Mamba, etc. |
