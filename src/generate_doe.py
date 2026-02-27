#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOE Parameter Generation — Stratified Latin Hypercube Sampling

Generates debonding defect parameters for H3 fairing FEM.
Size-stratified to cover: Small (10-30mm), Medium (30-80), Large (80-150), Critical (150-250).
Based on docs/DEFECT_PLAN.md (JAXA H3 researcher acceptance).

Usage:
  python src/generate_doe.py --n_samples 50 --output doe_phase1.json
  python src/generate_doe.py --n_samples 500 --output doe_params.json
"""

import argparse
import json
import numpy as np
from scipy.stats.qmc import LatinHypercube

# JAXA-relevant bounds (see docs/DEFECT_PLAN.md)
THETA_RANGE = (5.0, 55.0)   # deg, margin from symmetry edges
Z_RANGE = (800.0, 4200.0)   # mm, Barrel section, avoid clamp/top

# Size tiers: (name, r_min, r_max, fraction)
# Minimum r=50mm ensures ~35 defect nodes at GLOBAL_SEED=50mm + DEFECT_SEED=15mm
SIZE_TIERS = [
    ('Small', 50.0, 100.0, 0.25),     # ~35-300 defect nodes
    ('Medium', 100.0, 150.0, 0.40),   # ~300-700 defect nodes
    ('Large', 150.0, 250.0, 0.30),    # ~700-2000 defect nodes
    ('Critical', 250.0, 400.0, 0.05), # Pre-failure scale
]


def generate_doe(n_samples, seed=42, n_healthy=0):
    """
    Generate DOE with stratified radius sampling.

    Returns:
        dict with 'samples' list and metadata
    """
    np.random.seed(seed)

    samples = []

    # Healthy baseline(s)
    for i in range(n_healthy):
        samples.append({
            'id': i,
            'job_name': 'H3_Healthy_%04d' % i,
            'defect_params': None,
        })

    # Assign each defective sample to a size tier
    n_defect = n_samples
    tier_counts = []
    remaining = n_defect
    for name, lo, hi, frac in SIZE_TIERS[:-1]:
        n = max(0, int(n_defect * frac))
        tier_counts.append((name, lo, hi, n))
        remaining -= n
    tier_counts.append((SIZE_TIERS[-1][0], SIZE_TIERS[-1][1], SIZE_TIERS[-1][2], remaining))

    # LHS for theta and z (2D)
    sampler = LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n=n_defect)
    theta = THETA_RANGE[0] + unit[:, 0] * (THETA_RANGE[1] - THETA_RANGE[0])
    z_center = Z_RANGE[0] + unit[:, 1] * (Z_RANGE[1] - Z_RANGE[0])

    idx = 0
    for tier_name, r_lo, r_hi, count in tier_counts:
        for _ in range(count):
            sample_id = n_healthy + idx
            r = r_lo + np.random.rand() * (r_hi - r_lo)
            params = {
                'theta_deg': round(float(theta[idx]), 2),
                'z_center': round(float(z_center[idx]), 2),
                'radius': round(float(r), 2),
            }
            samples.append({
                'id': sample_id,
                'job_name': 'H3_Debond_%04d' % sample_id,
                'defect_params': params,
                'size_tier': tier_name,
            })
            idx += 1

    doe = {
        'n_total': len(samples),
        'n_healthy': n_healthy,
        'n_defective': n_defect,
        'bounds': {
            'theta_deg': list(THETA_RANGE),
            'z_center': list(Z_RANGE),
            'radius_tiers': [{'name': t[0], 'min': t[1], 'max': t[2], 'frac': t[3]} for t in SIZE_TIERS],
        },
        'seed': seed,
        'plan_reference': 'docs/DEFECT_PLAN.md',
        'samples': samples,
    }

    return doe


def main():
    parser = argparse.ArgumentParser(
        description='Generate DOE for H3 fairing debonding (JAXA-relevant plan)')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of defective samples (default: 50 for Phase 1)')
    parser.add_argument('--n_healthy', type=int, default=0,
                        help='Number of healthy baselines to include (default: 0, use existing)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='doe_params.json')
    args = parser.parse_args()

    doe = generate_doe(
        n_samples=args.n_samples,
        n_healthy=args.n_healthy,
        seed=args.seed,
    )

    with open(args.output, 'w') as f:
        json.dump(doe, f, indent=2)

    print("Generated DOE: %d defective samples" % doe['n_defective'])
    print("  theta_deg: [%.1f, %.1f]" % tuple(doe['bounds']['theta_deg']))
    print("  z_center:  [%.1f, %.1f] mm" % tuple(doe['bounds']['z_center']))
    print("  Size tiers: Small 50-100, Medium 100-150, Large 150-250, Critical 250-400 mm")
    print("Saved to: %s" % args.output)


if __name__ == '__main__':
    main()
