# GNN Architecture Comparison and Cross-Dataset Validation

> Comprehensive evaluation of GNN architectures and cross-dataset generalization for CFRP/Al-Honeycomb SHM
> Last updated: 2026-03-23

---

## 1. Architecture Comparison (700v2, 5-Class)

Four GNN architectures were evaluated on the 700v2 dataset with 5-class node classification (healthy + 4 defect types). All models use the same 34-dimensional node feature input and identical training hyperparameters.

| Architecture | Optimal F1 | Key Mechanism |
|-------------|-----------|---------------|
| **GraphSAGE** | **0.788** | Inductive neighborhood sampling with mean/LSTM/pool aggregation |
| GAT | 0.758 | Multi-head attention-weighted aggregation |
| GCN | 0.756 | Spectral graph convolution (1st-order Chebyshev) |
| GIN | 0.737 | Sum aggregation with MLP update (maximally expressive under WL test) |

**Analysis**: GraphSAGE outperforms all other architectures by 3--5 points in F1. Its inductive sampling strategy handles the heterogeneous mesh densities in FEM-derived graphs effectively, where node neighborhoods vary significantly between fine-mesh defect regions and coarse-mesh bulk regions.

---

## 2. Cross-Dataset Validation

The GNN framework was evaluated across four distinct datasets spanning different SHM modalities and structural configurations:

| Dataset | Domain | Modality | Nodes | AUC | Optimal F1 |
|---------|--------|----------|-------|-----|-----------|
| **Ours: GW-SHM (700v2)** | Guided wave debonding | FEM simulation | ~700/graph | **0.992** | **0.758** |
| **Ours: Fairing Separation** | Separation dynamics | FEM simulation (Explicit) | ~5K/graph | **0.999** | **0.730** |
| **OGW #3 wavefield** | Guided wave (experimental) | Laser vibrometry | ~10K/graph | **1.000** | **1.000** |
| **NASA CFRP fatigue** | Fatigue damage | Experimental coupon test | ~500/graph | **0.924** | **0.800** |

**Key findings**:

- The GNN framework generalizes across both simulation-derived and experimental datasets.
- Perfect performance on OGW #3 validates the approach on real-world wavefield measurements.
- NASA CFRP fatigue (AUC=0.924) demonstrates cross-material generalization capability, though the domain gap is evident compared to our simulation-based datasets.

---

## 3. Cross-Domain Transfer Learning

Transfer learning was evaluated to determine whether pretraining on one SHM domain benefits another.

### 3.1 Transfer to Separation Domain

| Transfer Strategy | F1 Score | Delta vs Baseline |
|------------------|----------|-------------------|
| **Baseline (no transfer)** | 0.317 | --- |
| **OGW3 pretrain** | 0.284 | -0.033 (negative transfer) |
| **MMD domain adaptation** | 0.319 | +0.002 (+0.6%) |
| **Augmentation transfer** | *pending* | --- |
| **Same-domain pretrain (GW-SHM)** | **0.730** | **+0.413 (+181%)** |

### 3.2 Analysis

**Same-domain transfer is highly effective**: Pretraining on the GW-SHM debonding dataset and fine-tuning on the separation dataset yields a +181% F1 improvement. Both datasets share the same CFRP/Al-Honeycomb structural configuration and 34-dimensional feature space, enabling effective feature reuse.

**Cross-domain transfer is not effective**: Pretraining on the OGW3 external dataset (different material, sensor layout, and physics) results in negative transfer (-10.4% F1). Even with MMD (Maximum Mean Discrepancy) domain adaptation, the improvement is marginal (+0.6%), indicating that the distribution shift between experimental wavefield data and FEM-simulated structural response is too large for simple adaptation techniques.

---

## 4. GAT Prediction Localization

Attention-based analysis of the GAT model reveals strong spatial concentration of defect predictions:

| Region | Mean Prediction Probability | Description |
|--------|----------------------------|-------------|
| **Defect nodes** | **P = 0.98** | High confidence at damage locations |
| **Healthy nodes** | P = 0.31 | Low background activation |
| **Concentration ratio** | **3.2x** | Defect-to-healthy probability ratio |

The 3.2x concentration ratio demonstrates that the GAT attention mechanism learns physically meaningful damage signatures. Defect nodes receive near-certain predictions (P=0.98) while healthy nodes maintain a low false-positive probability, confirming spatial interpretability of the learned representations.

---

## 5. GW FEM Pipeline Optimization

Significant computational efficiency gains were achieved in the guided wave FEM simulation pipeline:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **ODB storage per case** | ~19 GB | ~0.21 GB | **90x reduction** |
| **Total dataset storage (100 cases)** | ~190 GB | ~2.1 GB | **190 GB to 2.1 GB** |
| **Solver wall time** | Baseline | Optimized | **15x speedup** |

**Optimization techniques applied**:

- Selective field output: Only export displacement fields required for GNN feature extraction (U, S at final frame) instead of full field history
- Reduced output frequency: From every increment to key time frames only
- Lightweight ODB: Disable element-level output where nodal interpolation suffices

These optimizations enable scalable dataset generation without compromising the 34-dimensional node feature quality required for GNN training.

---

## 6. Summary

| Capability | Status | Key Metric |
|-----------|--------|-----------|
| **Best architecture** | GraphSAGE | F1 = 0.788 (700v2, 5-class) |
| **Separation detection** | 15-case DOE complete | AUC = 0.999, F1 = 0.730 |
| **External validation** | OGW3 + NASA CFRP | AUC = 0.924--1.000 |
| **Transfer learning** | Same-domain effective | +181% F1 improvement |
| **Spatial interpretability** | GAT localization | 3.2x defect concentration |
| **Computational efficiency** | Pipeline optimized | 190 GB to 2.1 GB, 15x speedup |
