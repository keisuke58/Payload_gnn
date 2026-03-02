#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sector12_doe.py — 1/12セクター (30°) 用 DOE パラメータ生成

generate_doe.py の generate_doe() を再利用し、θ範囲を 30° セクター用に
調整したパラメータセットを生成する。

Usage:
    python src/generate_sector12_doe.py --n_samples 100 --output doe_sector12.json
    python src/generate_sector12_doe.py --n_samples 3 --output doe_test3.json --seed 123
"""

import argparse
import json
import os
import sys

# Add src/ to path for importing generate_doe
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_doe as doe_mod


def main():
    parser = argparse.ArgumentParser(
        description='Generate DOE for 1/12 sector (30 deg) CZM model')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of defective samples (default: 100)')
    parser.add_argument('--n_healthy', type=int, default=0,
                        help='Number of healthy baselines (default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--defect_types', nargs='+', default=None,
                        help='Defect types (default: all 7)')
    parser.add_argument('--output', type=str, default='doe_sector12.json',
                        help='Output JSON path (default: doe_sector12.json)')
    args = parser.parse_args()

    # Override θ range for 30° sector (2° margin from boundaries)
    original_theta = doe_mod.THETA_RANGE
    doe_mod.THETA_RANGE = (2.0, 28.0)

    try:
        doe = doe_mod.generate_doe(
            n_samples=args.n_samples,
            n_healthy=args.n_healthy,
            seed=args.seed,
            defect_types=args.defect_types,
            opening_params=None,  # No openings in sector model
        )
    finally:
        doe_mod.THETA_RANGE = original_theta

    # Rename jobs to sector12 convention
    defect_idx = 0
    healthy_idx = 0
    for sample in doe['samples']:
        dp = sample.get('defect_params')
        if dp is None or dp.get('defect_type') == 'healthy':
            healthy_idx += 1
            sample['job_name'] = 'Job-S12-H%03d' % healthy_idx
        else:
            defect_idx += 1
            sample['job_name'] = 'Job-S12-D%03d' % defect_idx

    # Add sector metadata
    doe['sector_angle'] = 30.0
    doe['theta_range'] = [2.0, 28.0]

    with open(args.output, 'w') as f:
        json.dump(doe, f, indent=2)

    print("Generated %d samples -> %s" % (len(doe['samples']), args.output))
    print("  Sector: 30 deg, theta=[2, 28] deg")
    print("  Types: %s" % doe['defect_types'])
    for dtype, count in doe.get('type_counts', {}).items():
        print("    %s: %d" % (dtype, count))


if __name__ == '__main__':
    main()
