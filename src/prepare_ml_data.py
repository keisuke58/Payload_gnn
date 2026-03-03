#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare ML Data — FEM CSV → PyG train/val split

Scans dataset dir (dataset_output_100, batch_s12_100, etc.),
builds curvature-aware graphs for each sample, splits into train/val,
saves train.pt, val.pt, norm_stats.pt.

Supports two directory layouts:
  1. sample_*/  — legacy (nodes.csv, elements.csv directly inside)
  2. Job-S12-D*/ — CZM S12 batch (nodes.csv, elements.csv inside results/)

Usage:
  python src/prepare_ml_data.py --input dataset_output_100 --output data/processed_50mm_100
  python src/prepare_ml_data.py --input abaqus_work/batch_s12_100 --output data/processed_s12_czm_96
"""

import argparse
import json
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


def _collect_sample_dirs(input_dir):
    """Collect data directories, supporting both legacy and S12 batch layouts."""
    sample_dirs = []

    # Check for S12 batch layout (Job-S12-D* with results/ subdir)
    batch_status_path = os.path.join(input_dir, 'batch_status.json')
    has_batch_status = os.path.exists(batch_status_path)

    # Load batch_status.json to skip failed jobs
    failed_jobs = set()
    if has_batch_status:
        with open(batch_status_path) as f:
            status = json.load(f)
        for job_name, result in status.get('results', {}).items():
            if result != 'completed':
                failed_jobs.add(job_name)
        if failed_jobs:
            print("Skipping %d failed jobs: %s" % (len(failed_jobs), sorted(failed_jobs)))

    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isdir(path):
            continue

        # S12 batch layout: Job-S12-D*/results/
        if name.startswith('Job-'):
            if name in failed_jobs:
                continue
            results_dir = os.path.join(path, 'results')
            if os.path.isdir(results_dir):
                sample_dirs.append(results_dir)
        # Legacy layout: sample_* or healthy_baseline
        elif name.startswith('sample_') or name == 'healthy_baseline':
            sample_dirs.append(path)

    return sample_dirs


def prepare_dataset(input_dir, output_dir, val_ratio=0.2, seed=42, no_geodesic=True,
                    extra_inputs=None):
    """
    Process all samples in input_dir (+ extra_inputs), build graphs, split train/val, save.
    """
    input_dir = os.path.join(PROJECT_ROOT, input_dir)
    output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sample_dirs = _collect_sample_dirs(input_dir)

    # Merge extra input directories
    for extra in (extra_inputs or []):
        extra_abs = os.path.join(PROJECT_ROOT, extra)
        extra_dirs = _collect_sample_dirs(extra_abs)
        print("Found %d samples in %s" % (len(extra_dirs), extra))
        sample_dirs.extend(extra_dirs)

    if not sample_dirs:
        print("No sample directories found in %s" % input_dir)
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
    parser.add_argument('--extra_inputs', nargs='+', default=None,
                        help='Additional input dirs to merge (e.g. batch_s12_ext200)')
    args = parser.parse_args()

    prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
        no_geodesic=args.no_geodesic,
        extra_inputs=args.extra_inputs,
    )


if __name__ == '__main__':
    main()
