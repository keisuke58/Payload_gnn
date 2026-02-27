[← Home](Home)

# Research Report: GNN-Based SHM for Payload Fairings

## 1. Target Structure Specification: JAXA Epsilon/H3 Fairing

### 1.1 Material Composition
Research confirms that modern payload fairings for Epsilon (Enhanced/S) and H3 launch vehicles are **Carbon Fiber Reinforced Polymer (CFRP) Sandwich Structures**.
*   **Face Sheets**: CFRP (Carbon Fiber/Epoxy). High specific stiffness.
*   **Core**: Aluminum Honeycomb. High crushing strength and stiffness-to-weight ratio.
*   **Manufacturing**: Often Out-of-Autoclave (OOA) or filament wound (for monolithic parts), but fairings are typically large sandwich shells made in halves. Kawasaki Heavy Industries (KHI) and Beyond Gravity (formerly RUAG) are key manufacturers.

### 1.2 Geometric Parameters (Estimates for Simulation)
*   **Diameter**:
    *   Epsilon: ~2.6m (Max diameter).
    *   H3: ~5.2m (Standard fairing).
*   **Structure**:
    *   Cylindrical section + Ogive nose.
    *   Construction: Two half-shells (Clamshell) held by separation bolts.

**Implication for Modeling**:
*   The simulation **MUST** model a sandwich structure (Shell-Solid-Shell or Homogenized Shell).
*   **Anisotropy**: CFRP face sheets are highly anisotropic. Wave velocity depends on propagation direction relative to fibers.
*   **Curvature**: The cylindrical geometry causes mode conversion and focusing/defocusing of guided waves, unlike flat plates.

---

## 2. Physics of Guided Waves in Honeycomb Sandwich

### 2.1 Propagation Characteristics
*   **Leaky Waves**: In sandwich structures, wave energy often leaks from the skin into the core and the opposite skin.
*   **High Attenuation**: The honeycomb core (and adhesive layer) acts as a damper. High-frequency waves (>100 kHz) attenuate rapidly compared to monolithic plates.
*   **Modes**:
    *   **Global Modes**: Flexural waves of the entire sandwich (low frequency).
    *   **Skin Modes**: Rayleigh-like waves confined to the skin (high frequency). **This is preferred for detecting skin-core debonding.**

### 2.2 Defect Signature (Debonding)
*   When the skin separates from the core (debonding), the local boundary condition changes from "supported" to "free".
*   **Trapped Energy**: Waves entering the debonded region often get "trapped," causing amplitude amplification and ringing (standing waves) within the debonded skin.
*   **Velocity Change**: Phase velocity decreases in the debonded region (thin plate vs. sandwich).

---

## 3. GNN Methodology for Curved Shells

### 3.1 Graph Construction (Mesh-to-Graph)
Standard CNNs fail on curved surfaces due to distortion in 2D projection. GNNs operate directly on the non-Euclidean manifold.
*   **Nodes**: Finite Element nodes or sensor locations.
*   **Edges**: Connected if geodesic distance < threshold $R$, or k-Nearest Neighbors.

### 3.2 Feature Engineering for Curvature
To make the GNN "geometry-aware," node features must include:
1.  **Coordinates**: $(x, y, z)$ (Global position).
2.  **Surface Normals**: $(n_x, n_y, n_z)$. The change in normal vectors between neighbors encodes **local curvature**.
    *   *Insight from StructGNN*: Explicitly encoding structural properties (stiffness, mass) into the graph improves physics compliance.
3.  **Material Direction**: Vector defining the fiber orientation $(v_1, v_2, v_3)$ at each node.

### 3.3 Architecture Candidates
*   **StructGNN / Physics-GNN**: Architectures that mimic the mass-stiffness matrix operations.
*   **GraphSAGE / GAT**: Standard inductive frameworks. Good for learning local wave scattering patterns.
*   **Kernel GNN**: specialized for continuous manifolds.

---

## 4. Sim-to-Real Domain Adaptation

### 4.1 The Problem
FEM data is "clean" and idealized. Real experimental data has:
*   Sensor coupling variability.
*   Environmental noise.
*   Manufacturing variability (thickness variations, resin rich zones).

### 4.2 Strategy: Transfer Learning
*   **Source Domain**: Abaqus Simulation (Rich labels, infinite data).
*   **Target Domain**: Open Guided Waves Dataset (Limited labels, real physics) / Future Epsilon Test Data.
*   **Method**: **Domain Adversarial Neural Networks (DANN)** or **Transfer Component Analysis (TCA)**.
    *   Align the feature distributions of FEM and Experiment in a latent space.
    *   Train the classifier on FEM features, ensuring they are indistinguishable from Experimental features.

---

## 5. Next Steps (Actionable)

1.  **Refine Abaqus Model**: Switch to "Composite Layup" tool in Abaqus to model Skin-Core-Skin sandwich explicitly.
2.  **Frequency Selection**: Target 50-100 kHz range where "Skin Modes" begin to dominate but attenuation is manageable.
3.  **GNN Prototype**: Build a simple GraphSAGE model using $(x, y, z, \text{time\_series})$ as input to predict defect location.
