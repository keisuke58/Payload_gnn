# -*- coding: utf-8 -*-
"""augment_fno_to_gnn.py — FNO surrogate → GNN healthy data augmentation.

Uses trained FNO to predict healthy stress fields, then injects them into
existing PyG graphs to create realistic healthy samples.

Strategy:
  1. Load trained FNO model
  2. For each existing defective graph, create input grid with defect_mask=0
  3. FNO predicts healthy stress field on 64x64 grid
  4. Map predicted stress back to unstructured mesh nodes
  5. Replace stress features in PyG graph, set labels to 0
  6. Add sensor noise for diversity

Usage:
  python src/augment_fno_to_gnn.py \
    --fno_model runs/fno_production/best_model.pt \
    --input data/processed_s12_thermal_600_5class \
    --output data/processed_s12_thermal_700_5class \
    --n_healthy 100 --seed 42
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from models_fno import FNO2d


def load_fno_model(model_path, norm_stats_path=None, device='cpu'):
    """Load trained FNO model and normalization stats."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Detect model config from checkpoint
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        config = ckpt.get('config', {})
    else:
        state = ckpt
        config = {}

    # Infer modes and width from state_dict shapes
    if 'conv0.weights1' in state:
        width = state['conv0.weights1'].shape[0]
        modes = state['conv0.weights1'].shape[2]
    else:
        width = config.get('width', 32)
        modes = config.get('modes', 12)
    model = FNO2d(modes1=modes, modes2=modes, width=width,
                  in_channels=4, out_channels=1)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # Load norm stats
    norm_stats = None
    if norm_stats_path and os.path.exists(norm_stats_path):
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
    else:
        # Try adjacent norm_stats.json
        parent = os.path.dirname(model_path)
        ns_path = os.path.join(parent, 'norm_stats.json')
        if os.path.exists(ns_path):
            with open(ns_path) as f:
                norm_stats = json.load(f)

    return model, norm_stats


def graph_to_fno_input(graph, grid_size=64):
    """Convert PyG graph node positions to FNO input grid (defect_mask=0).

    Returns (1, 4, H, W) tensor and mapping info for inverse transform.
    """
    pos = graph.pos  # (N, 3) — x, y, z in Cartesian
    x_coords = pos[:, 0].numpy()
    y_coords = pos[:, 1].numpy()  # axial
    z_coords = pos[:, 2].numpy()

    # Cylindrical coordinates
    theta = np.degrees(np.arctan2(z_coords, x_coords))
    y = y_coords

    # Grid parameters (match csv_to_fno_grid.py)
    theta_min, theta_max = 0.0, 30.0
    y_min, y_max = 0.0, y.max() + 1.0

    theta_edges = np.linspace(theta_min, theta_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)

    # Per-node grid indices (for inverse mapping)
    ti = np.clip(np.digitize(theta, theta_edges) - 1, 0, grid_size - 1)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, grid_size - 1)

    # Build input grid
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    theta_grid, y_grid = np.meshgrid(theta_centers, y_centers)

    theta_norm = (theta_grid - theta_min) / (theta_max - theta_min)
    y_norm = (y_grid - y_min) / (y_max - y_min)

    # Temperature from node features (dim 8 = temp in 34-dim features)
    temp_vals = graph.x[:, 8].numpy()
    temp_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    temp_counts = np.zeros((grid_size, grid_size), dtype=np.float32)
    np.add.at(temp_grid, (yi, ti), temp_vals)
    np.add.at(temp_counts, (yi, ti), 1.0)
    temp_grid = temp_grid / np.maximum(temp_counts, 1.0)
    temp_norm = np.clip((temp_grid - 100.0) / (221.0 - 100.0 + 1e-8), 0.0, 1.0)

    # Input: healthy (defect_mask=0)
    inp = np.stack([
        y_norm.astype(np.float32),
        theta_norm.astype(np.float32),
        np.zeros((grid_size, grid_size), dtype=np.float32),  # defect_mask=0
        temp_norm,
    ], axis=0)

    inp_tensor = torch.from_numpy(inp).unsqueeze(0)  # (1, 4, H, W)

    return inp_tensor, (yi, ti, y_min, y_max, theta_min, theta_max)


def fno_stress_to_nodes(stress_grid, node_mapping):
    """Map FNO output grid back to unstructured mesh nodes.

    stress_grid: (H, W) numpy array
    node_mapping: (yi, ti, ...) from graph_to_fno_input
    Returns: (N,) array of per-node stress values
    """
    yi, ti = node_mapping[0], node_mapping[1]
    return stress_grid[yi, ti]


def generate_fno_healthy(fno_model, defect_graphs, n_healthy, norm_stats,
                         seed=42, noise_mult=0.02, noise_add=0.02,
                         device='cpu'):
    """Generate healthy graphs using FNO-predicted stress fields.

    For each healthy sample:
      1. Pick a defective source graph (for mesh topology)
      2. Use FNO to predict healthy stress field (defect_mask=0)
      3. Replace stress features with FNO prediction
      4. Set all labels to healthy (0)
      5. Add noise for diversity
    """
    rng = np.random.RandomState(seed)
    n_source = len(defect_graphs)
    healthy_graphs = []

    # Stress feature indices in 34-dim feature vector
    # smises=12 in raw CSV, but in normalized 34-dim features the layout is:
    # pos(3) + normals(3) + curvatures(4) + disp(4) + temp(1) + stress(5) +
    # thermal_stress(1) + strain(3) + fiber(3) + layup(5) + boundary(2)
    # Stress dims: 14-18 (s11, s22, s12, smises, thermal_smises)
    STRESS_DIMS = [14, 15, 16, 17, 18]  # s11, s22, s12, smises, thermal_smises

    smises_mean = 0.0
    smises_std = 1.0
    if norm_stats:
        smises_mean = norm_stats.get('smises_mean', 0.0)
        smises_std = norm_stats.get('smises_std', 1.0)

    for i in range(n_healthy):
        src_idx = rng.randint(0, n_source)
        g = defect_graphs[src_idx].clone()

        # FNO prediction
        inp_tensor, node_map = graph_to_fno_input(g)
        with torch.no_grad():
            pred = fno_model(inp_tensor.to(device))  # (1, 1, H, W)
        pred_grid = pred[0, 0].cpu().numpy()

        # Denormalize FNO output
        pred_grid = pred_grid * smises_std + smises_mean
        pred_grid = np.maximum(pred_grid, 0.0)  # stress >= 0

        # Map to nodes
        node_stress = fno_stress_to_nodes(pred_grid, node_map)

        # Replace stress features with FNO-predicted healthy values
        # smises (dim 17) gets the FNO prediction
        # Other stress components scaled proportionally
        old_smises = g.x[:, 17].numpy()
        scale = np.where(old_smises > 1e-6,
                         node_stress / (old_smises + 1e-8), 1.0)
        scale = np.clip(scale, 0.1, 10.0)

        for dim in STRESS_DIMS:
            g.x[:, dim] = g.x[:, dim] * torch.from_numpy(scale.astype(np.float32))

        # Add noise for diversity
        x = g.x.clone()
        feat_std = x.std(dim=0)
        feat_std[feat_std < 1e-8] = 1.0

        eta = torch.from_numpy(
            rng.normal(0, noise_mult, size=x.shape).astype(np.float32))
        x = x * (1.0 + eta)

        eps = torch.from_numpy(
            rng.normal(0, 1.0, size=x.shape).astype(np.float32))
        x = x + eps * (feat_std.unsqueeze(0) * noise_add)

        # Preserve boundary flags (dims 32, 33)
        x[:, 32] = g.x[:, 32]
        x[:, 33] = g.x[:, 33]
        g.x = x

        # All labels = healthy
        g.y = torch.zeros_like(g.y)

        healthy_graphs.append(g)

        if (i + 1) % 25 == 0:
            print("  Generated %d/%d FNO-healthy graphs" % (i + 1, n_healthy))

    return healthy_graphs


def main():
    parser = argparse.ArgumentParser(
        description='FNO surrogate → GNN healthy augmentation')
    parser.add_argument('--fno_model', type=str,
                        default='runs/fno_production/best_model.pt')
    parser.add_argument('--input', type=str,
                        default='data/processed_s12_thermal_600_5class')
    parser.add_argument('--output', type=str,
                        default='data/processed_s12_thermal_700_5class')
    parser.add_argument('--n_healthy', type=int, default=100)
    parser.add_argument('--noise_mult', type=float, default=0.02)
    parser.add_argument('--noise_add', type=float, default=0.02)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve paths
    fno_path = os.path.join(PROJECT_ROOT, args.fno_model)
    input_dir = os.path.join(PROJECT_ROOT, args.input)
    output_dir = os.path.join(PROJECT_ROOT, args.output)

    # Load FNO
    print("Loading FNO model from %s..." % fno_path)
    fno_model, norm_stats = load_fno_model(fno_path, device=device)
    print("  norm_stats: %s" % norm_stats)

    # Load existing dataset
    print("Loading dataset from %s..." % input_dir)
    train = torch.load(os.path.join(input_dir, 'train.pt'), weights_only=False)
    val = torch.load(os.path.join(input_dir, 'val.pt'), weights_only=False)
    orig_norm = torch.load(os.path.join(input_dir, 'norm_stats.pt'),
                           weights_only=False)

    # Only use defective graphs as source
    train_defect = [g for g in train if (g.y > 0).any()]
    val_defect = [g for g in val if (g.y > 0).any()]

    n_val_new = int(args.n_healthy * args.val_ratio)
    n_train_new = args.n_healthy - n_val_new

    print("\nGenerating %d FNO-healthy train graphs..." % n_train_new)
    train_healthy = generate_fno_healthy(
        fno_model, train_defect, n_train_new, norm_stats,
        seed=args.seed, noise_mult=args.noise_mult, noise_add=args.noise_add,
        device=device)

    print("Generating %d FNO-healthy val graphs..." % n_val_new)
    val_healthy = generate_fno_healthy(
        fno_model, val_defect, n_val_new, norm_stats,
        seed=args.seed + 1000, noise_mult=args.noise_mult,
        noise_add=args.noise_add, device=device)

    # Combine
    train_all = train + train_healthy
    val_all = val + val_healthy

    # Shuffle
    rng = np.random.RandomState(args.seed)
    train_all = [train_all[i] for i in rng.permutation(len(train_all))]
    val_all = [val_all[i] for i in rng.permutation(len(val_all))]

    # Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_all, os.path.join(output_dir, 'train.pt'))
    torch.save(val_all, os.path.join(output_dir, 'val.pt'))
    torch.save(orig_norm, os.path.join(output_dir, 'norm_stats.pt'))

    # Summary
    n_healthy_train = sum(1 for g in train_all if not (g.y > 0).any())
    n_healthy_val = sum(1 for g in val_all if not (g.y > 0).any())
    print("\n" + "=" * 60)
    print("FNO-Augmented Dataset")
    print("=" * 60)
    print("Train: %d (defect=%d, healthy=%d)" % (
        len(train_all), len(train_all) - n_healthy_train, n_healthy_train))
    print("Val:   %d (defect=%d, healthy=%d)" % (
        len(val_all), len(val_all) - n_healthy_val, n_healthy_val))
    print("Saved to %s" % output_dir)


if __name__ == '__main__':
    main()
