#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare ML Data — FEM CSV → PyG train/val split

Scans dataset dir (dataset_output_100, dataset_output_50mm_100, etc.),
builds curvature-aware graphs for each sample, splits into train/val,
saves train.pt, val.pt, norm_stats.pt.

Usage:
  python src/prepare_ml_data.py --input dataset_output_100 --output data/processed_50mm_100
  python src/prepare_ml_data.py --input dataset_output_50mm_100 --output data/processed_50mm_100 --val_ratio 0.2
"""

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from build_graph import build_curvature_graph
import pandas as pd


def load_sample(sample_dir):
    """Load nodes.csv and elements.csv from sample dir."""
    nodes_path = os.path.join(sample_dir, 'nodes.csv')
    elems_path = os.path.join(sample_dir, 'elements.csv')
    if not os.path.exists(nodes_path) or not os.path.exists(elems_path):
        return None, None
    df_nodes = pd.read_csv(nodes_path)
    df_elems = pd.read_csv(elems_path)
    return df_nodes, df_elems


def prepare_dataset(input_dir, output_dir, val_ratio=0.2, seed=42, no_geodesic=True):
    """
    Process all samples in input_dir, build graphs, split train/val, save.
    """
    input_dir = os.path.join(PROJECT_ROOT, input_dir)
    output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Collect sample dirs (sample_* + healthy_baseline)
    sample_dirs = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isdir(path):
            continue
        if name.startswith('sample_') or name == 'healthy_baseline':
            sample_dirs.append(path)

    if not sample_dirs:
        print("No sample_* or healthy_baseline directories found in %s" % input_dir)
        return

    print("Found %d samples in %s" % (len(sample_dirs), input_dir))

    # Build graphs (suppress per-sample print)
    all_data = []
    for i, sample_dir in enumerate(sample_dirs):
        df_nodes, df_elems = load_sample(sample_dir)
        if df_nodes is None:
            continue
        try:
            data = build_curvature_graph(
                df_nodes, df_elems,
                compute_geodesic=not no_geodesic,
                verbose=(i == 0),  # verbose for first sample only
            )
            all_data.append(data)
        except Exception as e:
            print("  Skip %s: %s" % (os.path.basename(sample_dir), str(e)[:80]))
            continue
        if (i + 1) % 20 == 0:
            print("  Processed %d/%d samples" % (i + 1, len(sample_dirs)))

    n = len(all_data)
    print("Built %d graphs" % n)

    if n == 0:
        print("No valid samples. Aborting.")
        return

    # Train/val split
    np.random.seed(seed)
    indices = np.random.permutation(n)
    n_val = int(n * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_data = [all_data[i] for i in train_idx]
    val_data = [all_data[i] for i in val_idx]

    # Compute normalization stats (optional — from train only)
    x_list = [d.x for d in train_data]
    x_cat = torch.cat(x_list, dim=0)
    mean = x_cat.mean(dim=0)
    std = x_cat.std(dim=0)
    std[std < 1e-8] = 1.0
    norm_stats = {'mean': mean, 'std': std}

    # Save
    torch.save(train_data, os.path.join(output_dir, 'train.pt'))
    torch.save(val_data, os.path.join(output_dir, 'val.pt'))
    torch.save(norm_stats, os.path.join(output_dir, 'norm_stats.pt'))

    # Summary
    all_labels = torch.cat([d.y for d in train_data + val_data])
    n_defect = (all_labels > 0).sum().item()
    num_classes = int(all_labels.max().item()) + 1
    print("Saved to %s" % output_dir)
    print("  Train: %d | Val: %d" % (len(train_data), len(val_data)))
    print("  Classes: %d | Total defect nodes: %d" % (num_classes, n_defect))
    for c in range(num_classes):
        n_c = (all_labels == c).sum().item()
        print("    class %d: %d nodes" % (c, n_c))


def main():
    parser = argparse.ArgumentParser(
        description='Prepare FEM dataset for ML training')
    parser.add_argument('--input', type=str, default='dataset_output_100',
                        help='Input dir with sample_* subdirs')
    parser.add_argument('--output', type=str, default='data/processed_50mm_100',
                        help='Output dir for train.pt, val.pt, norm_stats.pt')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_geodesic', action='store_true', default=True,
                        help='Skip geodesic (faster, default)')
    args = parser.parse_args()

    prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
        no_geodesic=args.no_geodesic,
    )


if __name__ == '__main__':
    main()
