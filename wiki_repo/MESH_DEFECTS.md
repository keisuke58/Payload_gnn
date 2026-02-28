# Mesh and Defect Visualization

This page visualizes the mesh structures used in the simulation and the characteristics of defects.

## Mesh Structures

We use different mesh sizes for convergence analysis and model training.

### 25mm Mesh (Standard Coarse)
Used for large-scale dataset generation (training data).
- **Approx. Nodes**: ~43,000
- **Element Type**: S4R (Quadrilateral) + S3 (Triangular)
- **Usage**: Main training dataset for GNNs.

![25mm Element Distribution](images/mesh/25mm/01_element_types.png)
![25mm Wireframe](images/mesh/25mm/02_mesh_wireframe.png)

### 12mm Mesh (Fine)
Used for high-fidelity validation and convergence checks.
- **Approx. Nodes**: ~194,000
- **Usage**: Validation baseline, high-frequency defect resolution.

![12mm Element Distribution](images/mesh/12mm/01_element_types.png)
![12mm Wireframe](images/mesh/12mm/02_mesh_wireframe.png)

### 10mm Mesh (Ultra-Fine)
Used for extreme convergence testing.
- **Status**: Data generation pending / In progress.
- **Target Nodes**: ~280,000+

## Defect Characteristics

We simulate debonding defects (separation between face sheet and core) using Gaussian perturbations in the displacement/stress fields.

### Defect Signatures
Visual comparison of Healthy vs. Defective states.

#### Displacement Magnitude
Defects cause local bulging (increased displacement) under internal pressure.
![Displacement Comparison](images/defects/displacement_comparison.png)

#### Von Mises Stress
Defects cause stress concentrations at the edges of the debonded area.
![Stress Comparison](images/defects/stress_comparison.png)

### 3D Defect Visualization
A 3D view of the defect signature (residual displacement).
![3D Defect](images/defects/defect_3d_residual.png)
