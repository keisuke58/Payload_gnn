#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate visualizations for Wiki:
1. Mesh structures (25mm, 12mm)
2. Defect visualizations (using visualize_defects.py)
3. Generate a Markdown report for the Wiki.
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_REPO = os.path.join(PROJECT_ROOT, "wiki_repo")
IMAGES_DIR = os.path.join(WIKI_REPO, "images")
MESH_IMAGES_DIR = os.path.join(IMAGES_DIR, "mesh")
DEFECT_IMAGES_DIR = os.path.join(IMAGES_DIR, "defects")

os.makedirs(MESH_IMAGES_DIR, exist_ok=True)
os.makedirs(DEFECT_IMAGES_DIR, exist_ok=True)

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def generate_mesh_visualizations():
    print("Generating mesh visualizations...")
    
    # 25mm Mesh
    data_25mm = os.path.join(PROJECT_ROOT, "dataset_output_25mm_400", "sample_0100")
    if os.path.exists(data_25mm):
        output_25mm = os.path.join(MESH_IMAGES_DIR, "25mm")
        os.makedirs(output_25mm, exist_ok=True)
        cmd = f"python scripts/visualize_mesh_structure.py --data {data_25mm} --output {output_25mm}"
        run_command(cmd)
    else:
        print(f"Warning: 25mm data not found at {data_25mm}")

    # 12mm Mesh
    data_12mm = os.path.join(PROJECT_ROOT, "dataset_output_ideal_12mm", "sample_0000")
    if os.path.exists(data_12mm):
        output_12mm = os.path.join(MESH_IMAGES_DIR, "12mm")
        os.makedirs(output_12mm, exist_ok=True)
        cmd = f"python scripts/visualize_mesh_structure.py --data {data_12mm} --output {output_12mm}"
        run_command(cmd)
    else:
        print(f"Warning: 12mm data not found at {data_12mm}")

    # 10mm Mesh (Placeholder / Check)
    data_10mm = os.path.join(PROJECT_ROOT, "dataset_output_ideal_10mm", "sample_0000")
    if os.path.exists(os.path.join(data_10mm, "nodes.csv")):
        output_10mm = os.path.join(MESH_IMAGES_DIR, "10mm")
        os.makedirs(output_10mm, exist_ok=True)
        cmd = f"python scripts/visualize_mesh_structure.py --data {data_10mm} --output {output_10mm}"
        run_command(cmd)
    else:
        print(f"Note: 10mm data not found/empty at {data_10mm}. Skipping.")

def generate_defect_visualizations():
    print("Generating defect visualizations...")
    # visualize_defects.py defaults to wiki_repo/images/defects, so we just run it
    cmd = "python scripts/visualize_defects.py"
    run_command(cmd)

def update_wiki():
    print("Updating Wiki...")
    wiki_file = os.path.join(WIKI_REPO, "MESH_DEFECTS.md")
    
    content = """# Mesh and Defect Visualization

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
"""
    
    with open(wiki_file, "w") as f:
        f.write(content)
    
    print(f"Wiki page generated at: {wiki_file}")

    # Append to main WIKI.md if not already linked
    main_wiki = os.path.join(PROJECT_ROOT, "WIKI.md")
    if os.path.exists(main_wiki):
        with open(main_wiki, "r") as f:
            main_content = f.read()
        
        if "MESH_DEFECTS.md" not in main_content:
            with open(main_wiki, "a") as f:
                f.write("\n\n## Visualizations\n- [Mesh and Defect Visualization](wiki_repo/MESH_DEFECTS.md)\n")
            print("Linked to main WIKI.md")

if __name__ == "__main__":
    generate_mesh_visualizations()
    generate_defect_visualizations()
    update_wiki()
