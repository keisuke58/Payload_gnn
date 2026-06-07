#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_ood_split.py — OOD generalisation split for Composites B paper.

Strategy: compute per-sample "defect effective radius" from node positions
(no external metadata needed). Split:
  - Train : samples where defect_radius <  ood_threshold  (ID)
  - OOD   : samples where defect_radius >= ood_threshold  (OOD test only)
  - Healthy graphs: always included in train (radius=0 by definition)

Produces:
  data/ood_<tag>/train.pt   -- ID training set
  data/ood_<tag>/val.pt     -- ID validation set (20% of train)
  data/ood_<tag>/ood.pt     -- OOD test set
  data/ood_<tag>/meta.json  -- split statistics
  data/ood_<tag>/norm_stats.pt

Usage:
  python scripts/prepare_ood_split.py
  python scripts/prepare_ood_split.py --threshold_pct 75 --tag ood_p75
  python scripts/prepare_ood_split.py --threshold_pct 50 --tag ood_p50
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def defect_radius(data):
    """
    Effective radius of defect region = std of defect node positions (mm).
    Healthy graphs (all y==0) return 0.0.
    """
    mask = data.y > 0
    if mask.sum() == 0:
        return 0.0
    pos = data.pos[mask]          # [n_def, 3]
    centroid = pos.mean(dim=0)    # [3]
    dists = (pos - centroid).norm(dim=1)
    return float(dists.mean())    # mean distance from centroid → effective radius


def defect_n_nodes(data):
    return int((data.y > 0).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed_s12_thermal_700_5class')
    parser.add_argument('--threshold_pct', type=float, default=75.0,
                        help='Percentile of defect_radius used as OOD threshold. '
                             'Samples >= threshold go to OOD set.')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tag', default='ood_p75')
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    out_dir  = os.path.join(PROJECT_ROOT, 'data', args.tag)
    os.makedirs(out_dir, exist_ok=True)

    # ── Load all samples ──────────────────────────────────────────────────────
    print('Loading data from', data_dir)
    train_pt = torch.load(os.path.join(data_dir, 'train.pt'), map_location='cpu', weights_only=False)
    val_pt   = torch.load(os.path.join(data_dir, 'val.pt'),   map_location='cpu', weights_only=False)
    all_data = train_pt + val_pt
    print(f'  Total samples: {len(all_data)}  (orig train={len(train_pt)}, val={len(val_pt)})')

    # ── Compute defect radius per sample ──────────────────────────────────────
    radii  = np.array([defect_radius(d)   for d in all_data])
    n_defs = np.array([defect_n_nodes(d)  for d in all_data])

    healthy_mask = radii == 0
    defect_mask  = ~healthy_mask

    r_def = radii[defect_mask]
    threshold = float(np.percentile(r_def, args.threshold_pct))
    print(f'\nDefect radius distribution (n={defect_mask.sum()}):')
    print(f'  min={r_def.min():.1f}  P25={np.percentile(r_def,25):.1f}  '
          f'P50={np.percentile(r_def,50):.1f}  P75={np.percentile(r_def,75):.1f}  '
          f'max={r_def.max():.1f}  (mm)')
    print(f'  OOD threshold (P{args.threshold_pct:.0f}) = {threshold:.1f} mm')

    # ── OOD split ─────────────────────────────────────────────────────────────
    #  ID   = healthy OR defect_radius < threshold
    #  OOD  = defect_radius >= threshold
    id_mask  = healthy_mask | (radii < threshold)
    ood_mask = ~id_mask

    id_data  = [d for d, m in zip(all_data, id_mask)  if m]
    ood_data = [d for d, m in zip(all_data, ood_mask) if m]
    print(f'\nID  samples: {len(id_data):4d}  (train+val pool)')
    print(f'OOD samples: {len(ood_data):4d}  (held out test)')

    # ── Train / val split within ID ───────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(id_data))
    n_val = int(len(id_data) * args.val_ratio)
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    train_data = [id_data[i] for i in train_idx]
    val_data   = [id_data[i] for i in val_idx]

    # ── Normalisation stats (from train only) ─────────────────────────────────
    x_cat = torch.cat([d.x for d in train_data], dim=0)
    mean  = x_cat.mean(dim=0)
    std   = x_cat.std(dim=0).clamp(min=1e-8)
    norm_stats = {'mean': mean, 'std': std}

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(train_data, os.path.join(out_dir, 'train.pt'))
    torch.save(val_data,   os.path.join(out_dir, 'val.pt'))
    torch.save(ood_data,   os.path.join(out_dir, 'ood.pt'))
    torch.save(norm_stats, os.path.join(out_dir, 'norm_stats.pt'))

    # Label distributions
    def label_dist(lst):
        ys = torch.cat([d.y for d in lst])
        import collections
        return dict(sorted(collections.Counter(ys.numpy().tolist()).items()))

    meta = {
        'tag': args.tag,
        'threshold_pct': args.threshold_pct,
        'threshold_mm':  threshold,
        'n_total':  len(all_data),
        'n_id':     len(id_data),
        'n_ood':    len(ood_data),
        'n_train':  len(train_data),
        'n_val':    len(val_data),
        'radius_stats': {
            'min':  float(r_def.min()),
            'p25':  float(np.percentile(r_def, 25)),
            'p50':  float(np.percentile(r_def, 50)),
            'p75':  float(np.percentile(r_def, 75)),
            'max':  float(r_def.max()),
        },
        'label_dist': {
            'train': label_dist(train_data),
            'val':   label_dist(val_data),
            'ood':   label_dist(ood_data),
        },
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nSaved to {out_dir}')
    print(f'  train: {len(train_data)} samples')
    print(f'  val:   {len(val_data)} samples')
    print(f'  ood:   {len(ood_data)} samples  (radius >= {threshold:.1f} mm)')
    print(f'  Train label dist: {label_dist(train_data)}')
    print(f'  OOD   label dist: {label_dist(ood_data)}')


if __name__ == '__main__':
    main()
