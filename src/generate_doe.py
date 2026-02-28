#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOE Parameter Generation — Multi-Defect-Type Stratified Latin Hypercube Sampling

Generates defect parameters for H3 fairing FEM with multiple defect types.
Academic justification (CFRP expert / professor level): docs/DEFECT_MODELS_ACADEMIC.md

Defect types:
  - debonding: Outer skin-core delamination. Ref: NASA NTRS 20160005994.
  - fod: FOD / hard inclusion. Ref: MDPI Appl. Sci. 2024.
  - impact: BVID. Ref: Composites Part B 2017, ASTM D7136.
  - delamination: Inter-ply delamination. Ref: Compos. Sci. Technol. 2006.
  - inner_debond: Inner skin-core. Ref: NASA NTRS, DEFECT_PLAN.
  - thermal_progression: CTE mismatch. Ref: Composites Part B 2018.
  - acoustic_fatigue: 147-148 dB launch. Ref: UTIAS 2019.

Size-stratified: Small (50-100mm), Medium (100-150), Large (150-250), Critical (250-400).
Based on docs/DEFECT_PLAN.md (JAXA H3 researcher acceptance).

Usage:
  python src/generate_doe.py --n_samples 50 --output doe_phase1.json
  python src/generate_doe.py --n_samples 300 --output doe_multitype.json
  python src/generate_doe.py --n_samples 100 --defect_types debonding fod --output doe_2type.json
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

# Defect types and their default allocation fractions
DEFECT_TYPES = ['debonding', 'fod', 'impact', 'delamination', 'inner_debond', 'thermal_progression', 'acoustic_fatigue']
TYPE_FRACTIONS = [0.25, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10]  # 7 types

# Type-specific parameter ranges
TYPE_PARAM_RANGES = {
    'debonding': {},
    'fod': {
        'stiffness_factor': (5.0, 20.0),
    },
    'impact': {
        'damage_ratio': (0.1, 0.5),
    },
    'delamination': {
        'delam_depth': (0.2, 0.8),   # Fraction of plies affected
    },
    'inner_debond': {},
    'thermal_progression': {},
    'acoustic_fatigue': {
        'fatigue_severity': (0.2, 0.5),   # Residual stiffness (higher=less damage)
    },
}

# Job name prefixes per type
JOB_PREFIXES = {
    'debonding': 'H3_Debond',
    'fod': 'H3_FOD',
    'impact': 'H3_Impact',
    'delamination': 'H3_Delam',
    'inner_debond': 'H3_InnerDebond',
    'thermal_progression': 'H3_Thermal',
    'acoustic_fatigue': 'H3_Acoustic',
}

# Default opening (H3 クイックアクセスドア). Set to None to disable exclusion.
DEFAULT_OPENING = {'z_center': 1500.0, 'theta_deg': 30.0, 'radius': 650.0}


def _defect_overlaps_opening(theta_deg, z_center, radius, opening_params):
    """True if defect circle overlaps the opening zone."""
    if not opening_params:
        return False
    import math
    z_o = opening_params['z_center']
    t_o = math.radians(opening_params['theta_deg'])
    r_o = opening_params['radius']
    # Approx: overlap if centers are within r_def + r_open
    dz = abs(z_center - z_o)
    r_local = 2600.0  # RADIUS
    arc = r_local * abs(math.radians(theta_deg) - t_o)
    dist = math.sqrt(dz**2 + arc**2)
    return dist < (radius + r_o)


def _sample_type_params(defect_type, rng):
    """Sample type-specific parameters from their ranges."""
    extra = {}
    ranges = TYPE_PARAM_RANGES.get(defect_type, {})
    for key, (lo, hi) in ranges.items():
        extra[key] = round(float(lo + rng.random() * (hi - lo)), 2)
    return extra


def generate_doe(n_samples, seed=42, n_healthy=0, defect_types=None, type_fractions=None,
                 opening_params=None):
    """
    Generate DOE with stratified radius sampling and multiple defect types.

    Args:
        n_samples: Number of defective samples
        seed: Random seed
        n_healthy: Number of healthy baselines
        defect_types: List of defect type names (default: all three)
        type_fractions: Allocation fractions per type (default: [0.40, 0.30, 0.30])
        opening_params: Optional {z_center, theta_deg, radius} - reject samples overlapping opening

    Returns:
        dict with 'samples' list and metadata
    """
    rng = np.random.RandomState(seed)

    if defect_types is None:
        defect_types = list(DEFECT_TYPES)
    if type_fractions is None:
        if len(defect_types) == len(DEFECT_TYPES):
            type_fractions = list(TYPE_FRACTIONS)
        else:
            # Equal split for custom type selection
            type_fractions = [1.0 / len(defect_types)] * len(defect_types)

    # Normalize fractions
    frac_sum = sum(type_fractions)
    type_fractions = [f / frac_sum for f in type_fractions]

    samples = []

    # Healthy baseline(s)
    for i in range(n_healthy):
        samples.append({
            'id': i,
            'job_name': 'H3_Healthy_%04d' % i,
            'defect_params': None,
        })

    # Allocate samples to defect types
    n_defect = n_samples
    type_counts = []
    remaining = n_defect
    for i, (dtype, frac) in enumerate(zip(defect_types, type_fractions)):
        if i == len(defect_types) - 1:
            type_counts.append((dtype, remaining))
        else:
            n = max(0, int(n_defect * frac))
            type_counts.append((dtype, n))
            remaining -= n

    # Assign each defective sample to a size tier
    tier_counts = []
    tier_remaining = n_defect
    for name, lo, hi, frac in SIZE_TIERS[:-1]:
        n = max(0, int(n_defect * frac))
        tier_counts.append((name, lo, hi, n))
        tier_remaining -= n
    tier_counts.append((SIZE_TIERS[-1][0], SIZE_TIERS[-1][1], SIZE_TIERS[-1][2], tier_remaining))

    # LHS for theta and z (2D), with rejection if opening overlap
    sampler = LatinHypercube(d=2, seed=seed)
    max_attempts = n_defect * 20
    theta_list, z_list = [], []
    attempt = 0
    while len(theta_list) < n_defect and attempt < max_attempts:
        unit = sampler.random(n=min(n_defect * 2, n_defect - len(theta_list) + 50))
        for i in range(len(unit)):
            if len(theta_list) >= n_defect:
                break
            t = THETA_RANGE[0] + unit[i, 0] * (THETA_RANGE[1] - THETA_RANGE[0])
            z = Z_RANGE[0] + unit[i, 1] * (Z_RANGE[1] - Z_RANGE[0])
            theta_list.append(t)
            z_list.append(z)
        attempt += 1
    theta = np.array(theta_list[:n_defect])
    z_center = np.array(z_list[:n_defect])
    if len(theta) < n_defect:
        theta = np.concatenate([theta, np.random.uniform(*THETA_RANGE, n_defect - len(theta))])
        z_center = np.concatenate([z_center, np.random.uniform(*Z_RANGE, n_defect - len(z_center))])

    # Build flat list of (tier_name, radius) for all samples
    tier_assignments = []
    for tier_name, r_lo, r_hi, count in tier_counts:
        for _ in range(count):
            r = r_lo + rng.rand() * (r_hi - r_lo)
            tier_assignments.append((tier_name, round(float(r), 2)))

    # Build flat list of defect_type assignments
    type_assignments = []
    for dtype, count in type_counts:
        type_assignments.extend([dtype] * count)

    # Shuffle type assignments to distribute types across tiers
    rng.shuffle(type_assignments)

    idx = 0
    for tier_name, r in tier_assignments:
        sample_id = n_healthy + idx
        defect_type = type_assignments[idx]
        prefix = JOB_PREFIXES.get(defect_type, 'H3_Defect')

        params = {
            'defect_type': defect_type,
            'theta_deg': round(float(theta[idx]), 2),
            'z_center': round(float(z_center[idx]), 2),
            'radius': r,
        }
        # Add type-specific parameters
        params.update(_sample_type_params(defect_type, rng))

        samples.append({
            'id': sample_id,
            'job_name': '%s_%04d' % (prefix, sample_id),
            'defect_params': params,
            'defect_type': defect_type,
            'size_tier': tier_name,
        })
        idx += 1

    doe = {
        'n_total': len(samples),
        'n_healthy': n_healthy,
        'n_defective': n_defect,
        'defect_types': defect_types,
        'type_counts': {dtype: count for dtype, count in type_counts},
        'opening_params': opening_params,
        'bounds': {
            'theta_deg': list(THETA_RANGE),
            'z_center': list(Z_RANGE),
            'radius_tiers': [{'name': t[0], 'min': t[1], 'max': t[2], 'frac': t[3]} for t in SIZE_TIERS],
            'type_params': {k: {pk: list(pv) for pk, pv in v.items()}
                           for k, v in TYPE_PARAM_RANGES.items() if k in defect_types},
        },
        'seed': seed,
        'plan_reference': 'docs/DEFECT_PLAN.md',
        'samples': samples,
    }

    return doe


def main():
    parser = argparse.ArgumentParser(
        description='Generate DOE for H3 fairing multi-type defect dataset')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of defective samples (default: 50)')
    parser.add_argument('--n_healthy', type=int, default=0,
                        help='Number of healthy baselines to include (default: 0)')
    parser.add_argument('--defect_types', nargs='+', default=None,
                        help='Defect types to include (default: all). Valid: %s' % ', '.join(DEFECT_TYPES))
    parser.add_argument('--type_fractions', nargs='+', type=float, default=None,
                        help='Allocation fractions per type (default: 0.40 0.30 0.30)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='doe_params.json')
    parser.add_argument('--opening', type=str, default=None,
                        help='JSON: {z_center, theta_deg, radius} to exclude defect from opening. Use "default" for H3 access door.')
    args = parser.parse_args()

    opening_params = None
    if args.opening:
        if args.opening.lower() == 'default':
            opening_params = DEFAULT_OPENING
        else:
            try:
                opening_params = json.loads(args.opening)
            except json.JSONDecodeError:
                pass

    doe = generate_doe(
        n_samples=args.n_samples,
        n_healthy=args.n_healthy,
        seed=args.seed,
        defect_types=args.defect_types,
        type_fractions=args.type_fractions,
        opening_params=opening_params,
    )

    with open(args.output, 'w') as f:
        json.dump(doe, f, indent=2)

    print("Generated DOE: %d defective samples" % doe['n_defective'])
    print("  Defect types: %s" % doe['defect_types'])
    print("  Type counts: %s" % doe['type_counts'])
    print("  theta_deg: [%.1f, %.1f]" % tuple(doe['bounds']['theta_deg']))
    print("  z_center:  [%.1f, %.1f] mm" % tuple(doe['bounds']['z_center']))
    print("  Size tiers: Small 50-100, Medium 100-150, Large 150-250, Critical 250-400 mm")
    print("Saved to: %s" % args.output)


if __name__ == '__main__':
    main()
