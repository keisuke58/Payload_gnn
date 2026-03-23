# Fairing Separation Dynamics — Abaqus/Explicit Model

> H3 Fairing separation dynamics FEM + GNN anomaly detection
> Last updated: 2026-03-23

---

## 1. Background and Motivation

### 1.1 H3 Flight 8 Incident (2025/12)

JAXA investigation report (2025/12/25) revealed anomalous vibration during separation:

| Item | Normal | F8 Measured |
|------|--------|-------------|
| **Separation vibration frequency** | ~18 Hz | **6--7 Hz** |
| **Vibration duration** | ~0.1 s | **~1.5 s** |
| **Outcome** | Normal separation | PSS failure, satellite detachment |

### 1.2 Separation Mechanism

H3 fairing separation occurs in three stages:

1. **Horizontal separation**: Frangible bolt (V-notch) x 72 --- simultaneous fracture
2. **Vertical separation**: Pyro-cord cutting along the vertical seam
3. **Opening**: Spring + hinge rotation to deploy two half-shells

Hypothesis: anomalous 6--7 Hz low-frequency vibration occurs under fault conditions (bolt misfire, asymmetric spring force, etc.). This FEM model validates the hypothesis.

---

## 2. FEM Model Specification

### 2.1 Model Overview

| Item | Value |
|------|-------|
| **Solver** | Abaqus/Explicit |
| **Symmetry** | 1/4 model (90 deg x 2 sectors) |
| **Sectors** | Q1: theta=0 deg--90 deg, Q2: theta=90 deg--180 deg |
| **Seam** | theta=90 deg (Q1--Q2 contact plane) |
| **Symmetry BC** | XSYMM at theta=0 deg (Q1 end), XSYMM at theta=180 deg (Q2 end) |
| **DOF estimate** | ~50K (full model ~2M DOF at 1/4 cost) |

### 2.2 Geometry Parameters

| Parameter | Value | Note |
|-----------|-------|------|
| `RADIUS` | 2,600 mm | Inner skin radius |
| `CORE_T` | 38 mm | Al honeycomb core thickness |
| `FACE_T` | 1.0 mm | CFRP skin thickness (each face) |
| `BARREL_Z_MAX` | 5,000 mm | Barrel height |
| `FAIRING_DIAMETER` | 5,200 mm | Full diameter |
| `ADAPTER_HEIGHT` | 100 mm | Adapter ring height |

### 2.3 Part Configuration (7 parts)

| Part | Element Type | Material | Description |
|------|-------------|----------|-------------|
| Q1-InnerSkin | S4R (shell) | Mat-CFRP | Inner skin (theta=0 deg--90 deg) |
| Q1-Core | C3D8R (solid) | Mat-Honeycomb | Honeycomb core |
| Q1-OuterSkin | S4R (shell) | Mat-CFRP | Outer skin |
| Q2-InnerSkin | S4R (shell) | Mat-CFRP | Inner skin (theta=90 deg--180 deg) |
| Q2-Core | C3D8R (solid) | Mat-Honeycomb | Honeycomb core |
| Q2-OuterSkin | S4R (shell) | Mat-CFRP | Outer skin |
| Adapter | S4R (shell) | Mat-Al7075 | Adapter ring |

### 2.4 Material Properties

| Material | E / E_ij (GPa) | nu | rho (tonne/mm^3) | Damping alpha |
|----------|----------------|-----|-------------------|---------------|
| **CFRP** | 160/10/10 (Engineering Constants) | 0.3 | 1.6e-9 | 100 |
| **Honeycomb** | 1.0/0.01/0.01 | 0.001 | 5e-11 | 100 |
| **Al7075** | 71.7 | 0.33 | 2.81e-9 | 100 |
| **Steel** | 200 | 0.3 | 7.85e-9 | --- |

### 2.5 Constraints and Loads

| Type | Description |
|------|-------------|
| **Tie (4x)** | Q1/Q2 x Inner/Outer skin to core bonding |
| **BC: Adapter** | ENCASTRE (all DOFs fixed) |
| **BC: Symmetry** | XSYMM at theta=0 deg (Q1), XSYMM at theta=180 deg (Q2) |
| **Gravity** | 29,430 mm/s^2 (3G) in -Y direction |

### 2.6 Steps

| Step | Duration | Mass Scaling | Description |
|------|----------|-------------|-------------|
| **Step-Preload** | 5 ms | factor=0.0001 | Gravity + differential pressure preload |
| **Step-Separation** | 200 ms | --- | Separation dynamics |

### 2.7 Field Output

- **Field**: U, V, A, RF, S, LE, STATUS (2 ms interval = 500 Hz)
- **History**: ALLKE, ALLSE, ALLAE, ALLIE, ETOTAL

---

## 3. CONN3D2 Separation Mechanism (v2)

### 3.1 Design Rationale

`*MODEL CHANGE, REMOVE` is **not available in Abaqus/Explicit** (Standard only). The v2 model uses **Connector Damage** via `CONN3D2` elements as the separation mechanism.

### 3.2 Connector Architecture

Two connector behaviors are defined:

| Behavior | Elasticity (N/mm) | Damage | Purpose |
|----------|--------------------|--------|---------|
| **CONN-BEH-RELEASE** | 1.0e6 per component | Force-based initiation (100 N) + energy evolution (0.001) | Bolts/pyro that **break** at separation |
| **CONN-BEH-STUCK** | 1.0e8 per component | None (no damage keywords) | Bolts that **remain locked** (fault condition) |

**Key implementation details:**

1. **CONN3D2 elements** connect node pairs across:
   - Seam (Q1 <-> Q2 at theta=90 deg) --- represents pyro-cord
   - Bottom flange (Q1/Q2 InnerSkin <-> Adapter) --- represents frangible bolts

2. **Release connectors**: Low damage initiation threshold (100 N per component) with near-zero energy evolution (0.001). Under the opening CLOAD forces, these connectors break almost instantly at the start of Step-Separation.

3. **Stuck connectors**: 100x higher stiffness (1e8 vs 1e6 N/mm), no `*Connector Damage` keywords. These remain rigid throughout the simulation, simulating bolt misfire.

4. **Opening forces**: `*CLOAD` applied at bottom nodes of Q1/Q2 InnerSkin in the z-direction (opposing for each half-shell), representing the spring-driven deployment force.

### 3.3 INP Post-Processing

The separation mechanism is injected into the INP file via `add_separation_to_inp()`:

```
Block A (before *End Assembly):
  *Element, type=CONN3D2, elset=CONN-RELEASE
  *Element, type=CONN3D2, elset=CONN-STUCK
  *Connector Section assignments

Block M (after *End Assembly):
  *Connector Behavior, name=CONN-BEH-RELEASE
    *Connector Elasticity + *Connector Damage Initiation/Evolution
  *Connector Behavior, name=CONN-BEH-STUCK
    *Connector Elasticity (no damage)

Block B (inside Step-Separation):
  *Cload (opening spring forces)
```

### 3.4 Fault Parameterization

| Parameter | CLI Flag | Default | Range |
|-----------|----------|---------|-------|
| Number of stuck bolts | `--n_stuck_bolts N` | 0 | 0--12 |
| Stuck bolt positions | `--stuck_bolt_indices` | Evenly spaced | Arbitrary list |
| Spring stiffness | `--spring_stiffness` | 500 N/mm | 300--8000 N/mm |
| Spring preload force | `SPRING_PRELOAD` | 5,000 N | Constant |

---

## 4. Results (v2)

### 4.1 Separation Response Summary

Three cases completed with v2 CONN3D2 mechanism:

| Case | Description | Q1 u_max (mm) | Q2 u_max (mm) | Q1 sigma_max (MPa) | Q2 sigma_max (MPa) |
|------|-------------|---------------|---------------|---------------------|---------------------|
| **Normal** | All bolts released | 6.34 | 6.13 | 168 | 166 |
| **Stuck3** | 3 bolts stuck | 2.75 | 6.13 | 187 | 165 |
| **Stuck6** | 6 bolts stuck | 2.75 | 6.13 | 187 | 165 |

**Key observations:**

- **Normal case**: Symmetric response between Q1 and Q2 as expected (both halves release cleanly)
- **Stuck cases**: Q1 displacement is constrained (2.75 mm vs 6.34 mm) while Q2 remains unaffected
- **Stress concentration**: Stuck bolts cause ~11% increase in Q1 peak Mises stress (187 vs 168 MPa)
- **Stuck3 vs Stuck6**: Nearly identical response --- the constraint is dominated by a few key bolt locations

### 4.2 Extracted Data

Results are stored in `results/separation/`:

| File Pattern | Content |
|-------------|---------|
| `Sep-v2-{case}_time_history.csv` | Per-frame max displacement, stress, velocity per instance |
| `Sep-v2-{case}_snapshot_t{T}.csv` | Nodal fields at key time frames (5, 10, 50, 100, 150, 200 ms) |
| `Sep-v2-{case}_energy.csv` | Kinetic, strain, total energy history |
| `Sep-v2-{case}_nodes.csv` | Full nodal data for graph construction (34-dim features) |
| `Sep-v2-{case}_elements.csv` | Element connectivity for edge construction |

---

## 5. GNN Anomaly Detection

### 5.1 Feature Compatibility

The separation graph uses the same **34-dimensional node feature** layout as the GW-SHM pretrained model:

```
Position/geometry (10): x, y, z, nx, ny, nz, k1, k2, H, K
Displacement (4):       ux, uy, uz, u_mag
Temperature (1):        temp
Stress (5):             s11, s22, s12, smises, principal_stress_sum
Thermal stress (1):     thermal_smises
Strain (3):             le11, le22, le12
Fiber orientation (3):  fiber_x, fiber_y, fiber_z
Layup (5):              layup_0, layup_45, layup_m45, layup_90, circum_angle
Boundary flags (2):     is_boundary, is_loaded
```

### 5.2 Anomaly Labeling Strategy

- **Normal case** (`Sep-v2-Normal`): All nodes labeled 0 (healthy)
- **Stuck bolt cases**: Nodes with **stress ratio > 2x** compared to Normal case are labeled 1 (anomaly)
  - The stress ratio is computed per-node: `sigma_stuck / sigma_normal`
  - Threshold of 2x captures the stress concentration pattern around stuck bolt locations

### 5.3 Transfer Learning Results

#### Initial 3-Case Baseline

| Metric | Value | Note |
|--------|-------|------|
| **Pretrained model** | GAT (GW-SHM) | Trained on guided wave debonding data |
| **Fine-tuning** | Separation dataset | 3 graphs (Normal, Stuck3, Stuck6) |
| **AUC** | **0.979** | Near-perfect ranking of anomalous nodes |
| **Optimal F1** | **0.260** | Low due to extreme class imbalance (3 graphs only) |
| **Training data** | 3 graphs | Minimal --- DOE campaign improved this significantly |

#### 15-Case DOE Results (Updated 2026-03-23)

The completed DOE campaign with 15 cases dramatically improved detection performance:

| Metric | 3-Case Baseline | 15-Case DOE | Improvement |
|--------|----------------|-------------|-------------|
| **AUC** | 0.979 | **0.999** | +0.020 |
| **Optimal F1** | 0.260 | **0.730** | +0.470 (+181%) |

The 15-case DOE confirms that diverse fault parameterization (stuck bolt count + spring stiffness variation) is essential for robust anomaly detection. The near-perfect AUC of 0.999 indicates excellent ranking capability, while the F1 improvement from 0.260 to 0.730 demonstrates the critical role of training set diversity.

### 5.4 Cross-Domain Transfer Analysis

Transfer learning was evaluated between the GW-SHM domain (debonding detection) and the separation dynamics domain:

| Condition | F1 Score | Note |
|-----------|----------|------|
| **Baseline (no transfer)** | 0.317 | Train from scratch on separation data |
| **Same-domain pretrain** | **0.730** | GW-SHM pretrain + fine-tune on separation (+181%) |
| **Cross-domain (OGW3 pretrain)** | 0.284 | Negative transfer observed |
| **MMD adaptation** | 0.319 | Marginal improvement (+0.6%) |

**Key finding**: Same-domain pretraining (GW-SHM to separation) is highly effective (+181% F1), but cross-domain transfer from external datasets is not effective due to significant distribution shift.

### 5.5 Architecture Comparison (700v2 5-Class)

Four GNN architectures were evaluated on the 700v2 dataset with 5-class classification:

| Architecture | Optimal F1 | Note |
|-------------|-----------|------|
| **GraphSAGE** | **0.788** | Best overall performance |
| GAT | 0.758 | Attention-based aggregation |
| GCN | 0.756 | Spectral convolution baseline |
| GIN | 0.737 | Maximally expressive (WL-test) |

GraphSAGE achieves the best performance, likely due to its inductive sampling strategy being well-suited for the heterogeneous mesh structures in fairing FEM models.

### 5.6 GAT Prediction Localization

Attention-based prediction analysis on defective regions reveals strong spatial concentration:

| Region | Mean Prediction Probability |
|--------|----------------------------|
| **Defect nodes** | **P = 0.98** |
| **Healthy nodes** | P = 0.31 |
| **Concentration ratio** | **3.2x** |

The 3.2x concentration ratio confirms that GAT attention heads learn physically meaningful damage signatures, validating the spatial interpretability of the GNN approach.

### 5.7 Processed Data

PyG datasets are stored in `data/processed_separation/`:

| File | Content |
|------|---------|
| `separation_dataset.pt` | Full dataset (3 graphs) |
| `train.pt` / `val.pt` | Train/validation split |
| `norm_stats.pt` | Feature normalization statistics |

---

## 6. DOE Campaign (In Progress)

### 6.1 Parameter Space

12 additional cases expanding the fault parameter space:

| Parameter | Values |
|-----------|--------|
| **Stuck bolts** | 1, 2, 4, 5, 8, 9, 12 |
| **Spring stiffness** (N/mm) | 300, 800, 2000, 8000 |

Cross-combination of stuck bolt count and spring stiffness produces a diverse training set.

### 6.2 Dataset Scale

| Phase | Cases | Status |
|-------|-------|--------|
| v2 initial | 3 (Normal, Stuck3, Stuck6) | Complete |
| DOE expansion | 12 (stuck bolts + spring variation) | In progress |
| **Total** | **15 graphs** | --- |

### 6.3 Expected Impact

- **F1 improvement**: More positive samples will improve precision/recall balance
- **Generalization**: Diverse fault conditions prevent overfitting to specific stuck bolt patterns
- **Spring stiffness sensitivity**: Enables analysis of deployment force asymmetry effects

---

## 7. Pipeline

```
generate_fairing_separation.py (Abaqus CAE)
  |  INP generation (geometry, materials, steps)
  v
add_separation_to_inp() (INP post-processing)
  |  Inject CONN3D2 connectors + damage behaviors + CLOAD
  v
qsub Abaqus/Explicit (frontale01-04)
  |  Step-Preload (5 ms) + Step-Separation (200 ms)
  v
extract_separation_results.py --graph (Abaqus Python)
  |  ODB -> nodes CSV (34-dim) + elements CSV + time history
  v
build_separation_graph.py (PyTorch Geometric)
  |  CSV -> PyG Data objects with anomaly labels
  v
train.py --pretrained (fine-tune GNN)
  |  Transfer learning from GW-SHM pretrained GAT
  v
Anomaly detection (AUC, F1, node-level predictions)
```

---

## 8. Usage

### 8.1 INP Generation + Analysis (qsub)

```bash
# Normal separation
qsub -v JOB_NAME=Sep-v2-Normal scripts/qsub_fairing_sep_run.sh

# Stuck bolt fault (3 bolts)
qsub -v JOB_NAME=Sep-v2-Stuck3,EXTRA_ARGS="--n_stuck_bolts 3" \
  scripts/qsub_fairing_sep_run.sh

# Stuck bolt + spring variation
qsub -v JOB_NAME=Sep-v2-Stuck3-K800,EXTRA_ARGS="--n_stuck_bolts 3 --spring_stiffness 800" \
  scripts/qsub_fairing_sep_run.sh
```

### 8.2 Result Extraction

```bash
# Time history + snapshots
abaqus python src/extract_separation_results.py --odb abaqus_work/Sep-v2-Normal.odb

# Graph-mode: full nodal fields + element connectivity for GNN
abaqus python src/extract_separation_results.py --odb abaqus_work/Sep-v2-Normal.odb --graph
```

### 8.3 Graph Construction

```bash
python src/build_separation_graph.py \
    --results_dir results/separation \
    --output_dir data/processed_separation \
    --cases Sep-v2-Normal Sep-v2-Stuck3 Sep-v2-Stuck6
```

### 8.4 INP Generation Only (Local)

```bash
LD_PRELOAD=/home/nishioka/libfake_x11.so \
  abaqus cae noGUI=src/generate_fairing_separation.py -- --job Sep-Test --no_run
```

---

## 9. References

| Reference | Content |
|-----------|---------|
| JAXA H3-8 Investigation Report (2025/12/25) | In-flight separation vibration measurements |
| KTH thesis (Fairing separation FE) | Abaqus beam element removal for separation modeling |
| Kawasaki H3 fairing development | Frangible bolt (V-notch) mechanism |

---

## 10. File Index

| File | Description |
|------|-------------|
| `src/generate_fairing_separation.py` | Main Abaqus CAE script (geometry + INP post-processing) |
| `src/extract_separation_results.py` | ODB result extraction (time history + graph mode) |
| `src/build_separation_graph.py` | CSV to PyG Data conversion with anomaly labeling |
| `scripts/qsub_fairing_sep.sh` | PBS job script (INP generation + analysis) |
| `scripts/qsub_fairing_sep_run.sh` | PBS job script (re-generate + analysis) |
| `scripts/qsub_extract_sep_graph.sh` | PBS job script (graph-mode ODB extraction) |
| `results/separation/Sep-v2-*.csv` | Extracted results (v2 CONN3D2 mechanism) |
| `data/processed_separation/` | PyG processed datasets |
| `abaqus_work/Sep-*.inp` | Generated INP files |
| `abaqus_work/Sep-*.odb` | Analysis result ODB files |
