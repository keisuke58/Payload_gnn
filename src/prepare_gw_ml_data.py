#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare GW ML Data — Sensor CSV → PyG train/val(/test) split

Scans gw_fairing_dataset (or CSV dir), builds sensor graphs,
splits into train/val(/test), saves .pt files.

Usage:
  # DOE-based (standard)
  python src/prepare_gw_ml_data.py --input abaqus_work/gw_fairing_dataset --doe doe_gw_fairing.json --output data/processed_gw_100

  # Manifest-based (arbitrary CSV + label)
  python src/prepare_gw_ml_data.py --manifest manifest_gw.json --output data/processed_gw_new

  # With test split
  python src/prepare_gw_ml_data.py --input abaqus_work/gw_fairing_dataset --no_doe --test_ratio 0.1 --output data/processed_gw_split3

Manifest JSON format:
  {
    "samples": [
      {"csv": "path/to/sensors.csv", "label": 0},
      {"csv": "path/to/sensors.csv", "label": 1, "split": "test"},
      ...
    ]
  }
  - "csv": path (absolute, or relative to project root)
  - "label": 0=healthy, 1=defect
  - "split": optional "train"/"val"/"test" to force assignment
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from build_gw_graph import build_gw_graph

# Patterns for auto-detecting healthy samples
_HEALTHY_PATTERNS = re.compile(
    r'(Healthy|Valid-H|Valid-Healthy|Test-H)', re.IGNORECASE)
# Patterns for auto-detecting defect samples (exclude healthy)
_DEFECT_PATTERNS = re.compile(
    r'(Debond|Defect|Fair-\d{4}|5mm|10mm|15mm|20mm|30mm|Micro)', re.IGNORECASE)


def collect_gw_samples(csv_dir, doe_path=None, include_augmented=True):
    """Collect (csv_path, label) pairs from a CSV directory.

    Supports standard DOE naming (Job-GW-Fair-XXXX) and non-standard names
    (Job-GW-Valid-*, Job-GW-Fair-*mm-Test, etc.) via pattern matching.

    Args:
        csv_dir: directory with *_sensors.csv files
        doe_path: optional DOE JSON for defect sample list
        include_augmented: include Healthy-A* augmented samples

    Returns:
        list of (csv_path, label) tuples
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

    # Non-standard healthy: Valid-Healthy, Test-H*, etc.
    seen = set(os.path.basename(p) for p, _ in samples)
    for f in sorted(os.listdir(csv_dir)):
        if f in seen or not f.endswith('_sensors.csv'):
            continue
        if _HEALTHY_PATTERNS.search(f) and 'Healthy-A' not in f:
            samples.append((os.path.join(csv_dir, f), 0))
            seen.add(f)

    # Defect samples from DOE or by pattern
    if doe_path and os.path.exists(doe_path):
        with open(doe_path) as f:
            doe = json.load(f)
        n = doe.get('n_samples', 0)
        for i in range(n):
            name = 'Job-GW-Fair-%04d_sensors.csv' % i
            if name in seen:
                continue
            path = os.path.join(csv_dir, name)
            if os.path.exists(path):
                samples.append((path, 1))
                seen.add(name)

    # Scan for remaining CSVs not yet collected
    for f in sorted(os.listdir(csv_dir)):
        if f in seen or not f.endswith('_sensors.csv'):
            continue
        if _HEALTHY_PATTERNS.search(f):
            continue  # already handled above
        # Anything else that looks like a sensor CSV → defect
        if _DEFECT_PATTERNS.search(f) or f.startswith('Job-GW-'):
            samples.append((os.path.join(csv_dir, f), 1))
            seen.add(f)

    return samples


def load_manifest(manifest_path):
    """Load manifest JSON → list of (csv_path, label, forced_split).

    Returns:
        list of (csv_path, label, split_or_None) tuples
    """
    manifest_path = os.path.abspath(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = manifest.get('samples', manifest if isinstance(manifest, list) else [])
    samples = []
    for entry in entries:
        csv_path = entry['csv']
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(PROJECT_ROOT, csv_path)
        label = int(entry.get('label', 1))
        split = entry.get('split', None)  # "train", "val", "test", or None
        samples.append((csv_path, label, split))

    return samples


def prepare_gw_dataset(input_dir=None, output_dir='data/processed_gw',
                       doe_path=None, manifest_path=None,
                       val_ratio=0.2, test_ratio=0.0, seed=42,
                       include_augmented=True, feature_set='baseline'):
    """Process all samples, build graphs, split train/val(/test), save.

    Data can come from either:
      1. input_dir + doe_path (standard directory scan)
      2. manifest_path (explicit CSV+label list)

    Manifest entries with "split" field are force-assigned to that split.
    Remaining entries are randomly split according to val_ratio/test_ratio.
    """
    output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Collect samples
    forced_splits = {}  # index -> split name
    if manifest_path:
        manifest_entries = load_manifest(manifest_path)
        raw_samples = [(p, l) for p, l, _ in manifest_entries]
        for i, (_, _, split) in enumerate(manifest_entries):
            if split:
                forced_splits[i] = split
        print("Manifest: %s (%d entries, %d with forced split)" % (
            manifest_path, len(raw_samples), len(forced_splits)))
    else:
        if input_dir:
            abs_input = os.path.join(PROJECT_ROOT, input_dir)
        else:
            abs_input = os.path.join(PROJECT_ROOT, 'abaqus_work/gw_fairing_dataset')
        if doe_path:
            doe_path = os.path.join(PROJECT_ROOT, doe_path)
        raw_samples = collect_gw_samples(abs_input, doe_path,
                                         include_augmented=include_augmented)

    if not raw_samples:
        print("No samples found")
        return

    n_healthy = sum(1 for _, l in raw_samples if l == 0)
    n_defect = sum(1 for _, l in raw_samples if l == 1)
    print("Found %d samples (%d healthy, %d defect)" % (
        len(raw_samples), n_healthy, n_defect))

    # Build graphs
    all_data = []
    valid_indices = []  # maps all_data index → raw_samples index
    for i, (csv_path, label) in enumerate(raw_samples):
        try:
            data = build_gw_graph(csv_path, label, feature_set=feature_set)
            if data is not None:
                data.graph_y = data.y.squeeze(0) if data.y.dim() > 0 else data.y
                data.name = os.path.basename(csv_path)
                all_data.append(data)
                valid_indices.append(i)
        except Exception as e:
            print("  Skip %s: %s" % (os.path.basename(csv_path), str(e)[:60]))

    n = len(all_data)
    print("Built %d graphs" % n)
    if n == 0:
        return

    # Split: forced assignments first, then random for the rest
    train_data, val_data, test_data = [], [], []
    auto_indices = []  # indices into all_data for random split

    for j in range(n):
        raw_idx = valid_indices[j]
        if raw_idx in forced_splits:
            split_name = forced_splits[raw_idx]
            if split_name == 'test':
                test_data.append(all_data[j])
            elif split_name == 'val':
                val_data.append(all_data[j])
            else:
                train_data.append(all_data[j])
        else:
            auto_indices.append(j)

    # Random split for non-forced entries
    np.random.seed(seed)
    perm = np.random.permutation(len(auto_indices))
    n_auto = len(auto_indices)
    n_test_auto = int(n_auto * test_ratio)
    n_val_auto = int(n_auto * val_ratio)

    for k in perm[:n_test_auto]:
        test_data.append(all_data[auto_indices[k]])
    for k in perm[n_test_auto:n_test_auto + n_val_auto]:
        val_data.append(all_data[auto_indices[k]])
    for k in perm[n_test_auto + n_val_auto:]:
        train_data.append(all_data[auto_indices[k]])

    # Save
    torch.save(train_data, os.path.join(output_dir, 'train.pt'))
    torch.save(val_data, os.path.join(output_dir, 'val.pt'))
    msg = "Saved train.pt (%d) val.pt (%d)" % (len(train_data), len(val_data))

    if test_data:
        torch.save(test_data, os.path.join(output_dir, 'test.pt'))
        msg += " test.pt (%d)" % len(test_data)

    # Save split info for reproducibility
    split_info = {
        'n_train': len(train_data),
        'n_val': len(val_data),
        'n_test': len(test_data),
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'seed': seed,
        'feature_set': feature_set,
        'manifest': manifest_path,
        'train_names': [d.name for d in train_data],
        'val_names': [d.name for d in val_data],
        'test_names': [d.name for d in test_data] if test_data else [],
    }
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    print(msg + " → %s" % output_dir)


def main():
    parser = argparse.ArgumentParser(description='Prepare GW ML dataset')
    parser.add_argument('--input', type=str, default='abaqus_work/gw_fairing_dataset',
                        help='Directory with *_sensors.csv')
    parser.add_argument('--doe', type=str, default='doe_gw_fairing.json',
                        help='DOE JSON for defect sample list')
    parser.add_argument('--no_doe', action='store_true',
                        help='Use fallback scan for defects (ignore doe file)')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Manifest JSON with explicit CSV+label list (overrides --input/--doe)')
    parser.add_argument('--output', type=str, default='data/processed_gw',
                        help='Output directory')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.0,
                        help='Test split ratio (0=no test split)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_augmented', action='store_true',
                        help='Exclude Healthy-A* augmented samples')
    parser.add_argument('--feature_set', type=str, default='baseline',
                        choices=['baseline', 'extended', 'full', 'comprehensive'],
                        help='baseline=3, extended=10, full=15, comprehensive=24')
    args = parser.parse_args()

    prepare_gw_dataset(
        input_dir=args.input,
        output_dir=args.output,
        doe_path=None if args.no_doe else args.doe,
        manifest_path=args.manifest,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        include_augmented=not args.no_augmented,
        feature_set=args.feature_set,
    )


if __name__ == '__main__':
    main()
