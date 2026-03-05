#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare GW ML Data — Sensor CSV → PyG train/val split

Scans gw_fairing_dataset (or CSV dir), builds sensor graphs,
splits into train/val, saves train.pt, val.pt.

Usage:
  python src/prepare_gw_ml_data.py --input abaqus_work/gw_fairing_dataset --doe doe_gw_fairing.json --output data/processed_gw_100
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from build_gw_graph import build_gw_graph


def collect_gw_samples(csv_dir, doe_path=None, include_augmented=True):
    """Collect (csv_path, label) pairs.

    Healthy: Job-GW-Fair-Healthy_sensors.csv, Job-GW-Fair-Healthy-A*.csv -> 0
    Defect: Job-GW-Fair-0000_sensors.csv etc -> 1

    doe_path: optional, for defect sample list
    include_augmented: include Job-GW-Fair-Healthy-A*.csv (augmented healthy)
    """
    samples = []
    csv_dir = os.path.abspath(csv_dir)

    if not os.path.isdir(csv_dir):
        return samples

    # Healthy: Job-GW-Fair-Healthy, Job-GW-Fair-Healthy-A*, or Job-GW-Fair-Test-H*
    healthy_path = os.path.join(csv_dir, 'Job-GW-Fair-Healthy_sensors.csv')
    if os.path.exists(healthy_path):
        samples.append((healthy_path, 0))

    # Augmented healthy (Job-GW-Fair-Healthy-A000, A001, ...)
    if include_augmented:
        for f in sorted(os.listdir(csv_dir)):
            if f.endswith('_sensors.csv') and 'Healthy-A' in f:
                samples.append((os.path.join(csv_dir, f), 0))

    # Fallback: Test-H* if no Healthy/Healthy-A
    if not any(l == 0 for _, l in samples):
        for f in sorted(os.listdir(csv_dir)):
            if f.endswith('_sensors.csv') and 'Test-H' in f:
                samples.append((os.path.join(csv_dir, f), 0))
                break

    # Defect samples from DOE or by pattern
    if doe_path and os.path.exists(doe_path):
        with open(doe_path) as f:
            doe = json.load(f)
        n = doe.get('n_samples', 0)
        for i in range(n):
            name = 'Job-GW-Fair-%04d_sensors.csv' % i
            path = os.path.join(csv_dir, name)
            if os.path.exists(path):
                samples.append((path, 1))
    else:
        # Fallback: scan for Job-GW-Fair-*_sensors.csv (exclude Healthy, Test-H)
        for f in sorted(os.listdir(csv_dir)):
            if f.endswith('_sensors.csv') and f.startswith('Job-GW-Fair-'):
                if 'Healthy' in f or 'Test-H' in f:
                    continue
                path = os.path.join(csv_dir, f)
                samples.append((path, 1))

    return samples


def prepare_gw_dataset(input_dir, output_dir, doe_path=None, val_ratio=0.2, seed=42,
                       include_augmented=True):
    """Process all samples, build graphs, split train/val, save."""
    input_dir = os.path.join(PROJECT_ROOT, input_dir)
    output_dir = os.path.join(PROJECT_ROOT, output_dir)
    if doe_path:
        doe_path = os.path.join(PROJECT_ROOT, doe_path)
    os.makedirs(output_dir, exist_ok=True)

    samples = collect_gw_samples(input_dir, doe_path, include_augmented=include_augmented)
    if not samples:
        print("No samples found in %s" % input_dir)
        return

    print("Found %d samples (%d healthy, %d defect)" % (
        len(samples), sum(1 for _, l in samples if l == 0), sum(1 for _, l in samples if l == 1)))

    all_data = []
    for csv_path, label in samples:
        try:
            data = build_gw_graph(csv_path, label)
            if data is not None:
                # graph-level label for train_gw.py
                data.graph_y = data.y.squeeze(0) if data.y.dim() > 0 else data.y
                all_data.append(data)
        except Exception as e:
            print("  Skip %s: %s" % (os.path.basename(csv_path), str(e)[:60]))

    n = len(all_data)
    print("Built %d graphs" % n)
    if n == 0:
        return

    # Train/val split
    np.random.seed(seed)
    indices = np.random.permutation(n)
    n_val = int(n * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_data = [all_data[i] for i in train_idx]
    val_data = [all_data[i] for i in val_idx]

    torch.save(train_data, os.path.join(output_dir, 'train.pt'))
    torch.save(val_data, os.path.join(output_dir, 'val.pt'))
    print("Saved train.pt (%d) val.pt (%d) to %s" % (
        len(train_data), len(val_data), output_dir))


def main():
    parser = argparse.ArgumentParser(description='Prepare GW ML dataset')
    parser.add_argument('--input', type=str, default='abaqus_work/gw_fairing_dataset',
                        help='Directory with *_sensors.csv')
    parser.add_argument('--doe', type=str, default='doe_gw_fairing.json',
                        help='DOE JSON for defect sample list')
    parser.add_argument('--output', type=str, default='data/processed_gw',
                        help='Output directory')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_augmented', action='store_true',
                        help='Exclude Healthy-A* augmented samples')
    args = parser.parse_args()

    prepare_gw_dataset(
        args.input, args.output,
        doe_path=args.doe,
        val_ratio=args.val_ratio,
        seed=args.seed,
        include_augmented=not args.no_augmented,
    )


if __name__ == '__main__':
    main()
