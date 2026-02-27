# Literature Review: Guided Wave SHM for Composite Structures

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

## Action Plan for Research
1.  **Validation**: Use OGW Dataset 1 (Flat Plate) to validate the "Wave-to-Graph" conversion pipeline.
2.  **Extension**: Use OGW Stringer Dataset to test debonding detection.
3.  **Application**: Train the validated GNN architecture on the synthetic Abaqus data (Cylindrical Fairing) generated in this project.
