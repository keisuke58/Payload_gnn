[← Home](Home)

# ML Strategy: JAXA H3 Fairing SHM (2026)

## 1. Executive Summary
The goal is to transition from **simulated validation (Phase 3)** to **experimental proof-of-concept (Phase 4)**. Current GNN/FNO models show promise on FEM data, but the "Sim-to-Real" gap remains the primary risk for deployment on the H3 rocket.

## 2. Current Model Portfolio
| Model | Type | Strengths | Weaknesses | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Point Transformer** | 3D Point Cloud | Handles complex curvature well; Permutation invariant. | Computationally heavy ($O(N^2)$ attention); Slow inference. | Implemented (`models_point.py`) |
| **FNO (Fourier Neural Operator)** | Operator Learning | **Resolution invariant**; Extremely fast inference (FFT). | struggles with irregular boundaries/meshes; complex to adapt to non-grid data. | Implemented (`models_fno.py`) |
| **UV-Net (U-Net)** | 2D CNN | Mature architecture; fast; easy to interpret. | Requires UV mapping (distortion); loses 3D geometric context. | Implemented (`run_uv_2d.py`) |

## 3. Strategic Roadmap (Next 6 Months)

### Phase 3.5: Rigorous Benchmarking (Immediate)
*   **Action**: Establish a unified `BenchmarkDataset` class.
*   **Metrics**:
    *   **Accuracy/F1**: Defect detection rate.
    *   **IoU (Intersection over Union)**: Localization precision.
    *   **Inference Time (ms)**: Critical for real-time monitoring during launch.
*   **Goal**: Downselect to 1 primary architecture for Phase 4.

### Phase 4: Sim-to-Real Domain Adaptation (Critical)
*   **Problem**: FEM data is noise-free. Real sensor data (Open Guided Waves / JAXA tests) has noise, coupling variability, and environmental effects.
*   **Strategy**:
    1.  **Data Augmentation**: Inject Gaussian noise, sensor dropout, and time-warping into FEM training data.
    2.  **Domain Adversarial Training (DANN)**: Train a feature extractor that cannot distinguish between FEM and Experiment domains, while correctly classifying defects.
    3.  **CycleGAN**: Translate "Clean FEM Wavefields" $\leftrightarrow$ "Noisy Experimental Wavefields".

### Phase 5: Physics-Informed Learning (Research)
*   **Concept**: Constrain the model with the **Elastic Wave Equation**.
*   **Implementation**: Add a residual loss term $L_{physics} = || \nabla \cdot \sigma - \rho \ddot{u} ||^2$.
*   **Benefit**: Reduces data requirements; ensures physical consistency of predictions.

## 4. Immediate Action Items
1.  [ ] **Run Benchmark**: Compare UV-Net vs PointTrans vs FNO on `Job-H3-Fairing-Test` dataset.
2.  [ ] **Synthesize Noise**: Create a "Noisy Test Set" to evaluate robustness.
3.  [ ] **Deploy to Wiki**: Document findings in this Wiki.
