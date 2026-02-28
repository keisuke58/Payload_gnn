#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add defect_label to nodes.csv from defect_params (geometry-based).

Use when nodes.csv was extracted without --defect_json.
Run from project root.

Usage:
  python scripts/add_defect_labels.py --doe doe_100.json --data_dir dataset_output
  python scripts/add_defect_labels.py --doe doe_100.json --data_dir dataset_output --dry_run
"""

import argparse
import json
import math
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def is_node_in_defect(x, y, z, defect_params):
    """
    Check if node (x,y,z) lies inside the circular defect zone on the cylindrical surface.
    Abaqus Revolve: Y=axial, X-Z=radial. r=sqrt(x^2+z^2), theta=atan2(z,x), z_axial=y.
    """
    theta_deg = defect_params['theta_deg']
    z_center = defect_params['z_center']
    radius = defect_params['radius']

    r_node = math.sqrt(x * x + z * z)
    theta_node_rad = math.atan2(z, x)
    theta_center_rad = math.radians(theta_deg)
    z_axial = y

    arc_mm = r_node * abs(theta_node_rad - theta_center_rad)
    dz = z_axial - z_center
    dist = math.sqrt(arc_mm * arc_mm + dz * dz)
    return dist <= radius


def add_labels_to_sample(sample_dir, defect_params, dry_run=False):
    """Add defect_label column to nodes.csv and write metadata.csv."""
    nodes_path = os.path.join(sample_dir, 'nodes.csv')
    if not os.path.exists(nodes_path):
        return None

    df = pd.read_csv(nodes_path)
    if 'defect_label' in df.columns and df['defect_label'].sum() > 0:
        return {'updated': False, 'reason': 'already has defect labels'}

    labels = []
    for _, row in df.iterrows():
        in_defect = is_node_in_defect(row['x'], row['y'], row['z'], defect_params)
        labels.append(1 if in_defect else 0)

    n_defect = sum(labels)
    df['defect_label'] = labels

    if not dry_run:
        df.to_csv(nodes_path, index=False)

        meta_path = os.path.join(sample_dir, 'metadata.csv')
        with open(meta_path, 'w') as f:
            f.write('key,value\n')
            f.write('defect_type,debonding\n')
            f.write('n_defect_nodes,%d\n' % n_defect)
            f.write('theta_deg,%s\n' % defect_params['theta_deg'])
            f.write('z_center,%s\n' % defect_params['z_center'])
            f.write('radius,%s\n' % defect_params['radius'])

    return {'updated': True, 'n_defect': n_defect, 'n_total': len(df)}


def main():
    parser = argparse.ArgumentParser(description='Add defect_label to nodes.csv from DOE')
    parser.add_argument('--doe', type=str, required=True, help='DOE JSON file')
    parser.add_argument('--data_dir', type=str, default='dataset_output')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    with open(args.doe, 'r') as f:
        doe = json.load(f)

    samples = {s['id']: s.get('defect_params') for s in doe['samples'] if s.get('defect_params')}
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)

    updated = 0
    skipped = 0
    for sample_id, defect_params in samples.items():
        sample_dir = os.path.join(data_dir, 'sample_%04d' % sample_id)
        if not os.path.isdir(sample_dir):
            continue

        result = add_labels_to_sample(sample_dir, defect_params, dry_run=args.dry_run)
        if result:
            if result.get('updated'):
                updated += 1
                print("  sample_%04d: n_defect=%d" % (sample_id, result.get('n_defect', 0)))
            else:
                skipped += 1

    print("\nDone: %d updated, %d skipped (dry_run=%s)" % (updated, skipped, args.dry_run))


if __name__ == '__main__':
    main()
