# H3 Fairing Dynamic Analysis Surrogate Learning Pipeline

This document outlines the design and implementation plan for the **Dynamic Analysis Surrogate Model** using Fourier Neural Operators (FNO) and DeepONet, aimed at accelerating defect detection and Structural Health Monitoring (SHM) for the H3 Payload Fairing.

## 1. Introduction

Traditional SHM relies on computationally expensive Finite Element Analysis (FEA) simulations (e.g., Abaqus/Explicit) to predict guided wave propagation in composite structures. To enable real-time defect detection and probabilistic risk assessment, we propose a surrogate modeling approach that learns the mapping from defect configurations to wave field responses.

### Objectives
- **Speed**: Accelerate forward simulations from hours to milliseconds.
- **Resolution Independence**: Use FNO to train on coarse mesh data and infer on fine grids (super-resolution).
- **Parametric Flexibility**: Use DeepONet to predict wave fields for arbitrary defect locations and sizes without retraining.

## 2. Pipeline Architecture

### 2.1 Data Generation (High-Fidelity Physics)
- **Source**: Abaqus/Explicit (`src/generate_guided_wave.py`)
- **Physics**: Lamb wave propagation in CFRP/Aluminum honeycomb sandwich panels.
- **Defect Modeling**:
  - Debonding (Cohesive Zone Model / Tie Constraint release)
  - Impact Damage (Stiffness reduction)
  - Foreign Object Debris (Mass addition)
- **Output**:
  - Full-field displacement history (ODB files)
  - Sensor history output (`scripts/extract_gw_history.py` -> `*_sensors.csv`)

### 2.2 Data Preprocessing
- **Wave Field Normalization**: Standardize amplitude to [-1, 1] or Z-score.
- **Grid Interpolation**: Convert unstructured FEA mesh data to regular grids for FNO input (if using full-field training).
- **Sensor Layout Encoding**: For DeepONet, encode sensor positions and readings as branch inputs.

### 2.3 Surrogate Models

#### A. Fourier Neural Operator (FNO)
- **Role**: Global wave field prediction & Super-resolution.
- **Input**:
  - Defect mask / Stiffness map ($a(x)$)
  - Initial wave excitation
- **Output**:
  - Full wave field snapshot at time $T$ ($u(x, T)$)
- **Key Feature**: Resolution invariance allows training on 64x64 grids and testing on 128x128 or higher.
- **Status**: Prototype implemented in `src/models_fno.py` and `src/prototype_fno.py`.

#### B. Deep Operator Network (DeepONet)
- **Role**: Sensor-based field reconstruction.
- **Input**:
  - **Branch**: Sparse sensor readings $\{u(x_i, t)\}_{i=1}^m$
  - **Trunk**: Query coordinates $(x, y, z)$
- **Output**: Wave amplitude at query location.
- **Use Case**: Reconstructing the full wave field from experimental sensor data to visualize potential defects.
- **Status**: Prototype implemented in `src/prototype_deeponet.py`.

### 2.4 Uncertainty Quantification (UQ)
- **Method**: Bayesian FNO / MC Dropout.
- **Goal**: Estimate confidence intervals for defect predictions.
- **Implementation**: `experiments/uq/` contains scripts for MC Dropout and Importance Sampling.

## 3. Research & References

1.  **Li et al. (2020)**, "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021.
    - *Relevance*: Foundation for resolution-invariant operator learning.
2.  **Lu et al. (2021)**, "Learning Nonlinear Operators via DeepONet based on the Universal Approximation Theorem", Nature Machine Intelligence.
    - *Relevance*: Basis for sensor-to-field mapping.
3.  **Recent Applications (2022-2024)**:
    - *Seismic Wave Propagation*: FNO outperforms CNNs in capturing global wave phenomena.
    - *Composite Delamination*: DeepONet used for real-time delamination growth prediction.

## 4. Implementation Plan

### Phase 1: Prototype & Synthetic Data (Current)
- [x] Implement FNO2d architecture (`src/models_fno.py`)
- [x] Implement DeepONet architecture (`src/prototype_deeponet.py`)
- [x] Verify resolution invariance on synthetic wave data (`src/prototype_fno.py`)
- [x] Initial UQ experiments (`experiments/uq/`)

### Phase 2: Abaqus Integration (Next Steps)
- [ ] Generate large-scale dataset (1000+ cases) using `src/generate_gw_fairing.py`.
- [ ] Develop `src/train_gw_surrogate.py` to train FNO on ODB-extracted fields.
- [ ] Validate DeepONet for sensor sparse reconstruction.

### Phase 3: Real-World Validation
- [ ] Transfer learning from simulation to experimental data (if available).
- [ ] Integrate into "Digital Twin" dashboard for real-time monitoring.

## 5. Directory Structure
```
src/
  ├── generate_guided_wave.py    # Abaqus simulation script
  ├── models_fno.py              # FNO implementation
  ├── prototype_deeponet.py      # DeepONet implementation
  ├── train_fno.py               # Training script (generic)
  └── ...
scripts/
  ├── extract_gw_history.py      # Data extraction
  └── ...
experiments/
  ├── uq/                        # Uncertainty Quantification
  └── quantum/                   # Quantum algos (future work)
docs/
  └── WIKI.md                    # This document
```
