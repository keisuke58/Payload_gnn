# -*- coding: utf-8 -*-
"""augment_healthy.py — Generate healthy graphs from defective ones via noise augmentation.

Takes existing defective PyG graphs, removes defect labels, and adds
calibration noise to create realistic healthy variants.

Strategy:
  1. Pick a defective graph
  2. Replace node features in defect region with interpolated healthy values
  3. Add multiplicative + additive noise to all features
  4. Set all labels to 0 (healthy)
  5. Keep graph topology (edge_index, edge_attr) unchanged

Usage:
  python src/augment_healthy.py \
    --input data/processed_s12_czm_thermal_200_binary \
    --output data/processed_s12_czm_thermal_mixed \
    --n_healthy 200 --seed 42
"""

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def heal_defect_nodes(graph, rng):
    """Replace defect node features with plausible healthy values.

    For each defect node, interpolate from its healthy neighbors.
    Falls back to graph-wide healthy statistics if no healthy neighbors exist.
    """
    g = graph.clone()
    defect_mask = g.y > 0
    healthy_mask = ~defect_mask

    if defect_mask.sum() == 0:
        return g

    # Precompute healthy stats as fallback
    healthy_mean = g.x[healthy_mask].mean(dim=0)
    healthy_std = g.x[healthy_mask].std(dim=0)
    healthy_std[healthy_std < 1e-8] = 1.0

    # Build adjacency for neighbor lookup
    src, dst = g.edge_index[0], g.edge_index[1]

    defect_indices = defect_mask.nonzero(as_tuple=True)[0]
    for idx in defect_indices:
        # Find neighbors of this defect node
        neighbors = dst[src == idx]
        healthy_neighbors = neighbors[~defect_mask[neighbors]]

        if len(healthy_neighbors) >= 2:
            # Interpolate from healthy neighbors (mean + small noise)
            neighbor_feats = g.x[healthy_neighbors]
            g.x[idx] = neighbor_feats.mean(dim=0)
        else:
            # Fallback: sample from healthy distribution
            noise = torch.from_numpy(
                rng.normal(0, 0.1, size=g.x.shape[1]).astype(np.float32))
            g.x[idx] = healthy_mean + noise * healthy_std

    return g


def add_sensor_noise(graph, rng, mult_std=0.02, add_std_ratio=0.02):
    """Add realistic sensor/calibration noise to all node features.

    Args:
        mult_std: Multiplicative noise std (simulates calibration drift, 2%)
        add_std_ratio: Additive noise relative to feature std (sensor noise, 2%)
    """
    g = graph.clone()
    x = g.x

    # Per-feature statistics
    feat_std = x.std(dim=0)
    feat_std[feat_std < 1e-8] = 1.0

    # Multiplicative noise: x * (1 + η), η ~ N(0, mult_std²)
    eta = torch.from_numpy(
        rng.normal(0, mult_std, size=x.shape).astype(np.float32))
    x = x * (1.0 + eta)

    # Additive noise: x + ε, ε ~ N(0, (add_std_ratio * std)²)
    eps = torch.from_numpy(
        rng.normal(0, 1.0, size=x.shape).astype(np.float32))
    eps = eps * (feat_std.unsqueeze(0) * add_std_ratio)
    x = x + eps

    # Don't corrupt position features (dims 0-2) too much
    # and keep integer-like features (boundary flags etc.) intact
    # Dims 32,33 are boundary flags (0/1), don't add noise
    x[:, 32] = g.x[:, 32]
    x[:, 33] = g.x[:, 33]

    g.x = x
    return g


def generate_healthy_graphs(defect_graphs, n_healthy, seed=42,
                            mult_std=0.02, add_std_ratio=0.02):
    """Generate n_healthy healthy graphs from defective source graphs.

    Each healthy graph:
      1. Based on a random defective graph (cycled if n_healthy > n_defective)
      2. Defect region healed (interpolated from neighbors)
      3. Sensor noise added
      4. Labels set to all-zero (healthy)
    """
    rng = np.random.RandomState(seed)
    n_source = len(defect_graphs)
    healthy_graphs = []

    for i in range(n_healthy):
        # Cycle through source graphs
        src_idx = rng.randint(0, n_source)
        g = defect_graphs[src_idx]

        # Step 1: Heal defect region
        g_healed = heal_defect_nodes(g, rng)

        # Step 2: Add sensor noise (different per variant)
        g_noisy = add_sensor_noise(g_healed, rng,
                                   mult_std=mult_std,
                                   add_std_ratio=add_std_ratio)

        # Step 3: Set all labels to healthy
        g_noisy.y = torch.zeros_like(g_noisy.y)

        healthy_graphs.append(g_noisy)

        if (i + 1) % 50 == 0:
            print("  Generated %d/%d healthy graphs" % (i + 1, n_healthy))

    return healthy_graphs


def main():
    parser = argparse.ArgumentParser(
        description="Generate healthy graphs via noise augmentation")
    parser.add_argument("--input", type=str,
                        default="data/processed_s12_czm_thermal_200_binary",
                        help="Input PyG dataset directory")
    parser.add_argument("--output", type=str,
                        default="data/processed_s12_mixed_400",
                        help="Output directory for mixed dataset")
    parser.add_argument("--n_healthy", type=int, default=200,
                        help="Number of healthy graphs to generate")
    parser.add_argument("--mult_std", type=float, default=0.02,
                        help="Multiplicative noise std (default: 2%%)")
    parser.add_argument("--add_std_ratio", type=float, default=0.02,
                        help="Additive noise std ratio (default: 2%%)")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = (os.path.join(PROJECT_ROOT, args.input)
                 if not os.path.isabs(args.input) else args.input)
    output_dir = (os.path.join(PROJECT_ROOT, args.output)
                  if not os.path.isabs(args.output) else args.output)

    # Load existing defective data
    print("Loading defective data from %s..." % input_dir)
    train_defect = torch.load(os.path.join(input_dir, "train.pt"),
                              weights_only=False)
    val_defect = torch.load(os.path.join(input_dir, "val.pt"),
                            weights_only=False)

    n_train_d = len(train_defect)
    n_val_d = len(val_defect)
    print("  Defective: train=%d, val=%d" % (n_train_d, n_val_d))

    # Generate healthy graphs proportionally
    n_val_healthy = int(args.n_healthy * args.val_ratio)
    n_train_healthy = args.n_healthy - n_val_healthy

    print("\nGenerating %d healthy train graphs..." % n_train_healthy)
    train_healthy = generate_healthy_graphs(
        train_defect, n_train_healthy, seed=args.seed,
        mult_std=args.mult_std, add_std_ratio=args.add_std_ratio)

    print("Generating %d healthy val graphs..." % n_val_healthy)
    val_healthy = generate_healthy_graphs(
        val_defect, n_val_healthy, seed=args.seed + 1000,
        mult_std=args.mult_std, add_std_ratio=args.add_std_ratio)

    # Combine: defective + healthy
    train_mixed = train_defect + train_healthy
    val_mixed = val_defect + val_healthy

    # Shuffle
    rng = np.random.RandomState(args.seed)
    train_order = rng.permutation(len(train_mixed)).tolist()
    val_order = rng.permutation(len(val_mixed)).tolist()
    train_mixed = [train_mixed[i] for i in train_order]
    val_mixed = [val_mixed[i] for i in val_order]

    # Recompute normalization stats from train set
    x_cat = torch.cat([g.x for g in train_mixed], dim=0)
    mean = x_cat.mean(dim=0)
    std = x_cat.std(dim=0)
    std[std < 1e-8] = 1.0
    norm_stats = {"mean": mean, "std": std}

    # Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_mixed, os.path.join(output_dir, "train.pt"))
    torch.save(val_mixed, os.path.join(output_dir, "val.pt"))
    torch.save(norm_stats, os.path.join(output_dir, "norm_stats.pt"))

    # Stats
    train_labels = torch.cat([g.y for g in train_mixed])
    val_labels = torch.cat([g.y for g in val_mixed])

    n_train_defect_graphs = sum(1 for g in train_mixed if (g.y > 0).any())
    n_val_defect_graphs = sum(1 for g in val_mixed if (g.y > 0).any())

    print("\n" + "=" * 60)
    print("Mixed Dataset Summary")
    print("=" * 60)
    print("Train: %d graphs (defect=%d, healthy=%d)" % (
        len(train_mixed), n_train_defect_graphs,
        len(train_mixed) - n_train_defect_graphs))
    print("Val:   %d graphs (defect=%d, healthy=%d)" % (
        len(val_mixed), n_val_defect_graphs,
        len(val_mixed) - n_val_defect_graphs))
    print("")
    print("Train nodes: defect=%d (%.3f%%), healthy=%d" % (
        (train_labels > 0).sum().item(),
        100.0 * (train_labels > 0).sum().item() / train_labels.numel(),
        (train_labels == 0).sum().item()))
    print("Val nodes:   defect=%d (%.3f%%), healthy=%d" % (
        (val_labels > 0).sum().item(),
        100.0 * (val_labels > 0).sum().item() / val_labels.numel(),
        (val_labels == 0).sum().item()))
    print("\nSaved to: %s" % output_dir)


if __name__ == "__main__":
    main()
