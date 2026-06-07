#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataset_metadata.py — Extract per-sample defect metadata from PyG graphs.

For each Data object in train.pt / val.pt, computes:
  - sample_id       : sequential index (0-based over train+val)
  - split           : 'train' or 'val'
  - defect_class    : majority non-zero label class (0=healthy)
  - n_defect_nodes  : number of nodes with y > 0
  - n_total_nodes   : total node count
  - defect_frac     : n_defect_nodes / n_total_nodes
  - cx, cy, cz      : centroid of defect nodes (mm)
  - defect_radius   : mean distance of defect nodes from centroid (mm)
  - defect_spread   : std of defect node distances (mm)
  - defect_z_min/max: z-range of defect nodes

Outputs:
  data/<tag>/metadata.csv

Usage:
  python scripts/build_dataset_metadata.py
  python scripts/build_dataset_metadata.py --data_dir data/processed_s12_thermal_700_5class --tag main
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def extract_metadata(data, sample_id, split):
    y   = data.y.numpy()        # [N]
    pos = data.pos.numpy()      # [N, 3]  (x, y, z) in mm

    n_total = len(y)
    mask    = y > 0
    n_def   = int(mask.sum())

    row = {
        'sample_id':      sample_id,
        'split':          split,
        'n_total_nodes':  n_total,
        'n_defect_nodes': n_def,
        'defect_frac':    n_def / max(n_total, 1),
    }

    if n_def == 0:
        row.update({
            'defect_class':   0,
            'cx': 0.0, 'cy': 0.0, 'cz': 0.0,
            'defect_radius':  0.0,
            'defect_spread':  0.0,
            'defect_z_min':   0.0,
            'defect_z_max':   0.0,
        })
    else:
        import collections
        cls_counts = collections.Counter(y[mask].tolist())
        majority_cls = max(cls_counts, key=cls_counts.get)

        def_pos = pos[mask]         # [n_def, 3]
        centroid = def_pos.mean(0)  # [3]
        dists    = np.linalg.norm(def_pos - centroid, axis=1)  # [n_def]

        row.update({
            'defect_class':   int(majority_cls),
            'cx':             float(centroid[0]),
            'cy':             float(centroid[1]),
            'cz':             float(centroid[2]),
            'defect_radius':  float(dists.mean()),
            'defect_spread':  float(dists.std()),
            'defect_z_min':   float(def_pos[:, 2].min()),
            'defect_z_max':   float(def_pos[:, 2].max()),
        })

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed_s12_thermal_700_5class')
    parser.add_argument('--tag', default='main',
                        help='Output written to data/<tag>/metadata.csv')
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    out_dir  = os.path.join(PROJECT_ROOT, 'data', args.tag)
    os.makedirs(out_dir, exist_ok=True)
    out_csv  = os.path.join(out_dir, 'metadata.csv')

    print('Loading', data_dir)
    train_list = torch.load(os.path.join(data_dir, 'train.pt'), map_location='cpu', weights_only=False)
    val_list   = torch.load(os.path.join(data_dir, 'val.pt'),   map_location='cpu', weights_only=False)

    print(f'  train={len(train_list)}  val={len(val_list)}')

    rows = []
    idx  = 0
    for split, lst in [('train', train_list), ('val', val_list)]:
        for data in lst:
            rows.append(extract_metadata(data, idx, split))
            idx += 1

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} rows → {out_csv}')

    # Summary statistics
    import collections
    radii     = np.array([r['defect_radius'] for r in rows])
    classes   = [r['defect_class'] for r in rows]
    healthy_n = sum(1 for r in rows if r['n_defect_nodes'] == 0)

    r_def = radii[radii > 0]
    print(f'\nSummary:')
    print(f'  Healthy: {healthy_n}  Defect: {len(rows)-healthy_n}')
    print(f'  Defect classes: {collections.Counter(c for c in classes if c > 0)}')
    print(f'  Defect radius (mm): min={r_def.min():.1f}  P25={np.percentile(r_def,25):.1f}'
          f'  P50={np.percentile(r_def,50):.1f}  P75={np.percentile(r_def,75):.1f}'
          f'  max={r_def.max():.1f}')


if __name__ == '__main__':
    main()
