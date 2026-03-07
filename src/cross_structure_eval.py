#!/usr/bin/env python3
"""Cross-Structure Generalization Evaluation.

Apply fairing-trained GPN to CompDam flat plate data (zero-shot / few-shot).
Tests whether damage detection transfers across different structural geometries.

Workflow:
  1. Load CompDam flat plate node data → construct 34-dim features (aligned with fairing)
  2. Build k-NN graph with edge features
  3. Load fairing-trained GPN checkpoint
  4. Zero-shot: apply directly (no fine-tuning)
  5. Few-shot: fine-tune on small subset of flat plate data
  6. Report AUROC, AUPRC, visualize damage map

Usage:
    python src/cross_structure_eval.py --mode zero-shot
    python src/cross_structure_eval.py --mode few-shot --n_shot 20
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.spatial import KDTree
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 34-dim feature layout (same as build_graph.py build_curvature_graph):
# [0:3]  pos (x,y,z)
# [3:6]  normal (nx,ny,nz)
# [6:10] curvature (k1,k2,H,K)
# [10:13] displacement (ux,uy,uz)
# [13]   u_mag
# [14]   temperature
# [15:19] stress (s11,s22,s12,mises)
# [19]   principal_stress_sum
# [20]   thermal_smises
# [21:24] strain (le11,le22,le12)
# [24:27] fiber orientation (fx,fy,fz)
# [27:31] layup angles (0,45,-45,90 in rad)
# [31]   circum_angle
# [32:34] node_type (boundary, loaded)


def load_compdam_as_pyg(
    node_csv: Path,
    elem_csv: Path,
    k_neighbors: int = 12,
) -> Data:
    """Convert CompDam flat plate data to PyG Data with 34-dim features."""
    df = pd.read_csv(node_csv)
    df_elem = pd.read_csv(elem_csv)

    n_nodes = len(df)
    print(f"Loading CompDam: {n_nodes} nodes, {len(df_elem)} elements")

    # ── Position (3 dims) ──
    # Normalize to similar scale as fairing (fairing coords ~0-2000mm, plate ~0-100mm)
    coords = df[['x', 'y', 'z']].values.astype(np.float32)
    # Scale to match fairing range: plate 100mm → fairing ~2000mm scale
    pos = torch.tensor(coords, dtype=torch.float32)

    # ── Normal (3 dims) ──
    # For flat plate: compute from element faces
    normals = compute_flat_plate_normals(coords, df_elem)
    normal_tensor = torch.tensor(normals, dtype=torch.float32)

    # ── Curvature (4 dims) ──
    # Flat plate: k1 ≈ k2 ≈ H ≈ K ≈ 0
    curvature = torch.zeros(n_nodes, 4, dtype=torch.float32)

    # ── Displacement (3 + 1 dims) ──
    disp = torch.tensor(df[['U1', 'U2', 'U3']].values, dtype=torch.float32)
    u_mag = torch.tensor(df['Umag'].values, dtype=torch.float32).unsqueeze(1)

    # ── Temperature (1 dim) ──
    # CompDam impact: no thermal → set to reference temp
    # Fairing data: temp ~100-221 K. Use 150 K as neutral value
    temp = torch.full((n_nodes, 1), 150.0, dtype=torch.float32)

    # ── Stress (4 dims): s11, s22, s12, mises ──
    s11 = torch.tensor(df['S11'].values, dtype=torch.float32).unsqueeze(1)
    s22 = torch.tensor(df['S22'].values, dtype=torch.float32).unsqueeze(1)
    s12 = torch.tensor(df['S12'].values, dtype=torch.float32).unsqueeze(1)
    mises = torch.tensor(df['Mises'].values, dtype=torch.float32).unsqueeze(1)
    stress = torch.cat([s11, s22, s12, mises], dim=1)

    # ── Principal stress sum (1 dim) ──
    principal_sum = (s11 + s22)  # (n, 1)

    # ── Thermal von Mises (1 dim) ──
    thermal_mises = torch.zeros(n_nodes, 1, dtype=torch.float32)

    # ── Strain (3 dims): approximate from stress using IM7-8552 properties ──
    # E1=171420, E2=9080, nu12=0.32, G12=5290
    E1, E2, nu12, G12 = 171420.0, 9080.0, 0.32, 5290.0
    le11 = (s11 / E1 - nu12 * s22 / E1).squeeze()  # simplified
    le22 = (-nu12 * s11 / E1 + s22 / E2).squeeze()
    le12 = (s12 / (2 * G12)).squeeze()
    strain = torch.stack([le11, le22, le12], dim=1)

    # ── Fiber orientation (3 dims) ──
    # Flat plate: fiber direction depends on ply angle
    # Use average over [45/0/-45/90]s layup → approximately (1,0,0) for 0-deg dominant
    fiber = torch.zeros(n_nodes, 3, dtype=torch.float32)
    fiber[:, 0] = 1.0  # x-direction (0-deg ply)

    # ── Layup angles (4 dims) ──
    # Same [45/0/-45/90]s layup as fairing
    layup_deg = [0.0, 45.0, -45.0, 90.0]
    layup_rad = torch.tensor([math.radians(a) for a in layup_deg], dtype=torch.float32)
    layup = layup_rad.unsqueeze(0).expand(n_nodes, 4)

    # ── Circumferential angle (1 dim) ──
    # Flat plate: no circumferential direction → set to 0
    circum_angle = torch.zeros(n_nodes, 1, dtype=torch.float32)

    # ── Node type (2 dims): boundary, loaded ──
    x_arr = coords[:, 0]
    y_arr = coords[:, 1]
    plate_size = 100.0
    tol = 2.0  # element size
    is_boundary = ((x_arr < tol) | (x_arr > plate_size - tol) |
                   (y_arr < tol) | (y_arr > plate_size - tol)).astype(np.float32)
    # Impact zone: center of plate (radius ~15mm from center)
    cx, cy = plate_size / 2, plate_size / 2
    dist_from_center = np.sqrt((x_arr - cx)**2 + (y_arr - cy)**2)
    is_loaded = (dist_from_center < 15.0).astype(np.float32)
    node_type = torch.tensor(np.stack([is_boundary, is_loaded], axis=1), dtype=torch.float32)

    # ── Assemble 34-dim features ──
    x_features = torch.cat([
        pos,             # 3: position
        normal_tensor,   # 3: normal
        curvature,       # 4: k1, k2, H, K
        disp,            # 3: ux, uy, uz
        u_mag,           # 1: |u|
        temp,            # 1: temperature
        stress,          # 4: s11, s22, s12, mises
        principal_sum,   # 1: σ_sum
        thermal_mises,   # 1: thermal mises
        strain,          # 3: le11, le22, le12
        fiber,           # 3: fiber direction
        layup,           # 4: layup angles
        circum_angle,    # 1: circum angle
        node_type,       # 2: boundary, loaded
    ], dim=1)

    print(f"Feature shape: {x_features.shape} (expected: [N, 34])")
    assert x_features.shape[1] == 34, f"Expected 34 features, got {x_features.shape[1]}"

    # ── Damage labels ──
    y = torch.tensor(df['damage_label'].values, dtype=torch.long)
    n_damaged = (y > 0).sum().item()
    print(f"Damage labels: {n_damaged}/{n_nodes} nodes damaged ({100*n_damaged/n_nodes:.2f}%)")

    # ── Build k-NN graph (using scipy KDTree) ──
    print(f"Building k-NN graph (k={k_neighbors})...")
    tree = KDTree(coords)
    _, indices = tree.query(coords, k=k_neighbors + 1)  # +1 includes self
    src_list, dst_list = [], []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self
            src_list.append(i)
            dst_list.append(j)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Edge features: dx, dy, dz, euclidean_dist, curvature_diff
    row, col = edge_index
    dx = pos[col, 0] - pos[row, 0]
    dy = pos[col, 1] - pos[row, 1]
    dz = pos[col, 2] - pos[row, 2]
    dist = torch.sqrt(dx**2 + dy**2 + dz**2)
    # Curvature diff: 0 for flat plate
    curv_diff = torch.zeros_like(dist)
    edge_attr = torch.stack([dx, dy, dz, dist, curv_diff], dim=1)

    print(f"Graph: {n_nodes} nodes, {edge_index.shape[1]} edges")

    data = Data(
        x=x_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        pos=pos,
    )
    return data


def compute_flat_plate_normals(coords, df_elem):
    """Compute approximate surface normals for flat plate nodes."""
    n_nodes = len(coords)
    normals = np.zeros((n_nodes, 3), dtype=np.float32)

    # For C3D8R hex elements, top/bottom face normals are ±z
    # Determine which face each node is closest to
    z_vals = coords[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_mid = (z_min + z_max) / 2

    for i in range(n_nodes):
        if z_vals[i] > z_mid:
            normals[i] = [0, 0, 1]   # top face: +z
        else:
            normals[i] = [0, 0, -1]  # bottom face: -z

    return normals


def load_gpn_model(checkpoint_path, device):
    """Load trained GPN model from checkpoint."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from gpn_shm import GPNModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = GPNModel(
        n_features=config.get("n_features", 34),
        n_classes=config.get("n_classes", 2),
        dim_hidden=config["dim_hidden"],
        dim_latent=config["dim_latent"],
        radial_layers=config["radial_layers"],
        appnp_K=config["appnp_K"],
        appnp_alpha=config["appnp_alpha"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded GPN: {n_params:,} params, trained AUROC={ckpt.get('best_auroc', 'N/A')}")
    return model, config


def normalize_features(data, norm_stats=None):
    """Normalize features using fairing data statistics or per-feature."""
    if norm_stats is not None:
        # Use fairing normalization
        mean = norm_stats["mean"]
        std = norm_stats["std"]
        data.x = (data.x - mean) / (std + 1e-8)
    else:
        # Per-feature standardization
        mean = data.x.mean(dim=0, keepdim=True)
        std = data.x.std(dim=0, keepdim=True)
        data.x = (data.x - mean) / (std + 1e-8)
    return data


@torch.no_grad()
def zero_shot_eval(model, data, device):
    """Zero-shot evaluation: apply fairing-trained GPN directly."""
    from gpn_shm import subsample_graph

    model.eval()
    data_sub = subsample_graph(data, max_nodes=8000)
    data_sub = data_sub.to(device)

    out = model(data_sub.x, data_sub.edge_index)
    alpha = out["alpha"]
    alpha0 = alpha.sum(dim=-1)

    # P(defect) = alpha[:,1] / alpha0
    p_defect = (alpha[:, 1] / alpha0).cpu().numpy()
    # Epistemic uncertainty = num_classes / alpha0
    epistemic = (alpha.shape[1] / alpha0).cpu().numpy()

    y_true = data_sub.y.cpu().numpy()
    y_true_binary = (y_true > 0).astype(int)

    # Metrics
    if y_true_binary.sum() > 0 and y_true_binary.sum() < len(y_true_binary):
        auroc = roc_auc_score(y_true_binary, p_defect)
        auprc = average_precision_score(y_true_binary, p_defect)
    else:
        auroc = auprc = float('nan')

    # Optimal F1
    best_f1 = 0
    for t in np.linspace(0.01, 0.99, 100):
        pred = (p_defect > t).astype(int)
        if pred.sum() > 0:
            f1 = f1_score(y_true_binary, pred)
            best_f1 = max(best_f1, f1)

    print(f"\n{'='*60}")
    print("Zero-Shot Cross-Structure Results")
    print(f"{'='*60}")
    print(f"  Source: Fairing (CFRP/Al-HC, debonding)")
    print(f"  Target: Flat plate (CFRP, impact damage)")
    print(f"  Nodes: {len(y_true)} (subsampled)")
    print(f"  Damaged: {y_true_binary.sum()}/{len(y_true)} ({100*y_true_binary.mean():.2f}%)")
    print(f"  AUROC:  {auroc:.4f}")
    print(f"  AUPRC:  {auprc:.4f}")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Epistemic uncertainty: mean={epistemic.mean():.4f}, "
          f"damaged={epistemic[y_true_binary==1].mean():.4f}, "
          f"healthy={epistemic[y_true_binary==0].mean():.4f}")

    return {
        'auroc': auroc, 'auprc': auprc, 'f1': best_f1,
        'p_defect': p_defect, 'epistemic': epistemic,
        'y_true': y_true_binary, 'pos': data_sub.pos.cpu().numpy(),
    }


def few_shot_eval(model, data, device, n_shot=20, epochs=30, lr=1e-3):
    """Few-shot fine-tuning on a small subset of target domain data."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from gpn_shm import subsample_graph, gpn_loss

    data_sub = subsample_graph(data, max_nodes=8000)

    # Split: n_shot labeled nodes for fine-tuning, rest for evaluation
    n_nodes = data_sub.x.shape[0]
    y = data_sub.y
    y_binary = (y > 0).long()

    # Sample n_shot nodes (balanced if possible)
    defect_idx = (y_binary == 1).nonzero(as_tuple=True)[0]
    healthy_idx = (y_binary == 0).nonzero(as_tuple=True)[0]

    n_defect_shot = min(n_shot // 2, len(defect_idx))
    n_healthy_shot = n_shot - n_defect_shot

    perm_d = defect_idx[torch.randperm(len(defect_idx))[:n_defect_shot]]
    perm_h = healthy_idx[torch.randperm(len(healthy_idx))[:n_healthy_shot]]
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[perm_d] = True
    train_mask[perm_h] = True
    eval_mask = ~train_mask

    print(f"\nFew-shot: {n_shot} labeled nodes "
          f"(defect={n_defect_shot}, healthy={n_healthy_shot})")

    # Fine-tune
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_dev = data_sub.to(device)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(data_dev.x, data_dev.edge_index)
        alpha = out["alpha"]
        loss = gpn_loss(alpha[train_mask], y_binary[train_mask].to(device))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{epochs}, loss={loss.item():.4f}")

    # Evaluate on remaining nodes
    model.eval()
    with torch.no_grad():
        out = model(data_dev.x, data_dev.edge_index)
        alpha = out["alpha"]
        alpha0 = alpha.sum(dim=-1)
        p_defect = (alpha[:, 1] / alpha0).cpu().numpy()

    y_eval = y_binary[eval_mask].numpy()
    p_eval = p_defect[eval_mask.numpy()]

    if y_eval.sum() > 0 and y_eval.sum() < len(y_eval):
        auroc = roc_auc_score(y_eval, p_eval)
        auprc = average_precision_score(y_eval, p_eval)
    else:
        auroc = auprc = float('nan')

    best_f1 = 0
    for t in np.linspace(0.01, 0.99, 100):
        pred = (p_eval > t).astype(int)
        if pred.sum() > 0:
            f1 = f1_score(y_eval, pred)
            best_f1 = max(best_f1, f1)

    print(f"\n{'='*60}")
    print(f"{n_shot}-Shot Cross-Structure Results")
    print(f"{'='*60}")
    print(f"  AUROC:  {auroc:.4f}")
    print(f"  AUPRC:  {auprc:.4f}")
    print(f"  Best F1: {best_f1:.4f}")

    return {'auroc': auroc, 'auprc': auprc, 'f1': best_f1}


def visualize_results(results, out_dir):
    """Visualize cross-structure evaluation results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = results['pos']
    y_true = results['y_true']
    p_defect = results['p_defect']
    epistemic = results['epistemic']

    # Use x-y plane (flat plate top view)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Ground truth damage map
    ax = axes[0]
    colors = ['#2196F3' if y == 0 else '#F44336' for y in y_true]
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=1, alpha=0.5)
    ax.set_title("Ground Truth (CompDam Impact Damage)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect('equal')

    # 2. GPN prediction (P(defect))
    ax = axes[1]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=p_defect, cmap='RdYlGn_r',
                    s=1, alpha=0.5, vmin=0, vmax=1)
    ax.set_title(f"GPN Zero-Shot P(defect) — AUROC={results['auroc']:.3f}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='P(defect)')

    # 3. Epistemic uncertainty
    ax = axes[2]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=epistemic, cmap='viridis',
                    s=1, alpha=0.5)
    ax.set_title("Epistemic Uncertainty (Higher = Less Confident)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='Epistemic Uncertainty')

    plt.tight_layout()
    fig_path = out_dir / "cross_structure_zero_shot.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Structure Generalization Evaluation"
    )
    parser.add_argument("--mode", choices=["zero-shot", "few-shot", "both"],
                        default="both")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/gpn/best_gpn.pt"))
    parser.add_argument("--node_csv", type=Path,
                        default=Path("abaqus_work/compdam_flatplate/compdam_node_data.csv"))
    parser.add_argument("--elem_csv", type=Path,
                        default=Path("abaqus_work/compdam_flatplate/compdam_elements.csv"))
    parser.add_argument("--out_dir", type=Path,
                        default=Path("results/cross_structure"))
    parser.add_argument("--n_shot", type=int, default=20)
    parser.add_argument("--k_neighbors", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load CompDam data
    data = load_compdam_as_pyg(args.node_csv, args.elem_csv, args.k_neighbors)

    # Load GPN
    model, config = load_gpn_model(args.checkpoint, args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Test both raw and normalized features
    print("\n" + "="*60)
    print("=== Experiment 1: Raw features (no normalization) ===")
    print("="*60)

    # Zero-shot with raw features
    if args.mode in ("zero-shot", "both"):
        results = zero_shot_eval(model, data, args.device)
        visualize_results(results, args.out_dir)

    # Also test with normalized features for comparison
    print("\n" + "="*60)
    print("=== Experiment 2: Per-feature standardized ===")
    print("="*60)
    data_norm = data.clone()
    data_norm = normalize_features(data_norm)
    if args.mode in ("zero-shot", "both"):
        model2, _ = load_gpn_model(args.checkpoint, args.device)
        zero_shot_eval(model2, data_norm, args.device)

    # Few-shot (on raw features, more aggressive)
    if args.mode in ("few-shot", "both"):
        for n_shot in [5, 20, 50, 100]:
            model_copy, _ = load_gpn_model(args.checkpoint, args.device)
            few_shot_eval(model_copy, data, args.device,
                         n_shot=n_shot, epochs=50, lr=5e-4)

    print("\nDone!")


if __name__ == "__main__":
    main()
