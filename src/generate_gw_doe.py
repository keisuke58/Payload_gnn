#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOE for Guided Wave GNN Dataset — LHS Sampling on 30-deg Fairing Sector

Generates defect parameter sets for guided wave dynamic analysis
on a 30-degree (1/12) barrel sector of the H3 fairing.
Each sample specifies a circular debonding defect
(z_center, theta_deg, radius) for generate_gw_fairing.py.

Size stratification:
  Small:  r = 20-40 mm  (30%)  — near detection limit
  Medium: r = 40-60 mm  (45%)  — typical BVID scale
  Large:  r = 60-80 mm  (25%)  — clearly detectable

Usage:
  python src/generate_gw_doe.py --n_samples 100 --output doe_gw_fairing.json
  python src/generate_gw_doe.py --n_samples 50 --seed 123 --output doe_gw_test.json
"""

import argparse
import json
import math
import numpy as np
from scipy.stats.qmc import LatinHypercube

# Sector geometry (30-deg barrel, z=500-2500mm)
DEFAULT_Z_MIN = 500.0
DEFAULT_Z_MAX = 2500.0
DEFAULT_SECTOR_ANGLE = 30.0
RADIUS_INNER = 2600.0  # mm

# Defect position bounds (margin from sector edges and excitation)
Z_MARGIN = 100.0        # mm, avoid top/bottom edges
THETA_MARGIN = 3.0      # deg, avoid symmetry BC edges

# Size tiers for defect radius (mm)
# Default: micro-defect regime (10-30mm) for challenging benchmark
SIZE_TIERS_MICRO = [
    ('Micro',  10.0, 15.0, 0.30),   # Near/below detection limit (~λ/10–λ/7)
    ('Small',  15.0, 22.0, 0.40),   # Marginal detection (~λ/7–λ/5)
    ('Medium', 22.0, 30.0, 0.30),   # Moderate (~λ/5–λ/3)
]

# Original tiers (kept for reference / --tier_preset original)
SIZE_TIERS_ORIGINAL = [
    ('Small',  20.0, 40.0, 0.30),   # Near detection limit
    ('Medium', 40.0, 60.0, 0.45),   # Typical BVID scale
    ('Large',  60.0, 80.0, 0.25),   # Clearly detectable
]

SIZE_TIERS = SIZE_TIERS_MICRO  # default

# Defect types with weights (matching static model's 7 types)
DEFECT_TYPES = [
    ('debonding', 0.25),
    ('fod', 0.10),
    ('impact', 0.15),
    ('delamination', 0.15),
    ('inner_debond', 0.10),
    ('thermal_progression', 0.10),
    ('acoustic_fatigue', 0.15),
]

# Openings to exclude (same as generate_gw_fairing.py OPENINGS_PHASE1)
OPENINGS = [
    {'z_center': 2500.0, 'theta_deg': 20.0, 'diameter': 400.0, 'name': 'HVAC_Door'},
]

# Fixed excitation frequency
DEFAULT_FREQ_KHZ = 50


def _defect_overlaps_opening(z_center, theta_deg, radius, openings):
    """Check if defect overlaps any opening."""
    for op in openings:
        z_o = op['z_center']
        t_o = op['theta_deg']
        r_o = op.get('diameter', 0) / 2.0
        dz = abs(z_center - z_o)
        arc = RADIUS_INNER * abs(math.radians(theta_deg) - math.radians(t_o))
        dist = math.sqrt(dz ** 2 + arc ** 2)
        if dist < (radius + r_o + 20.0):  # 20mm margin
            return True
    return False


def generate_gw_doe(n_samples, seed=42, freq_khz=None,
                    z_min=None, z_max=None, sector_angle=None,
                    size_tiers=None, multi_defect=False):
    """Generate DOE for guided wave fairing dataset.

    Args:
        n_samples: Number of defective samples
        seed: Random seed for reproducibility
        freq_khz: Excitation frequency in kHz (default: 50)
        z_min: Barrel z_min in mm
        z_max: Barrel z_max in mm
        sector_angle: Sector angle in degrees

    Returns:
        dict with samples list and metadata
    """
    if freq_khz is None:
        freq_khz = DEFAULT_FREQ_KHZ
    if z_min is None:
        z_min = DEFAULT_Z_MIN
    if z_max is None:
        z_max = DEFAULT_Z_MAX
    if sector_angle is None:
        sector_angle = DEFAULT_SECTOR_ANGLE
    if size_tiers is None:
        size_tiers = SIZE_TIERS

    rng = np.random.RandomState(seed)

    # Position bounds
    z_lo = z_min + Z_MARGIN
    z_hi = z_max - Z_MARGIN
    theta_lo = THETA_MARGIN
    theta_hi = sector_angle - THETA_MARGIN

    # LHS for (z_center, theta_deg) — 2D
    sampler = LatinHypercube(d=2, seed=seed)

    # Size-stratified radius sampling
    tier_assignments = []
    remaining = n_samples
    for i, (name, r_min, r_max, frac) in enumerate(size_tiers):
        if i == len(size_tiers) - 1:
            n_tier = remaining
        else:
            n_tier = max(0, int(n_samples * frac))
            remaining -= n_tier
        for _ in range(n_tier):
            r = r_min + rng.rand() * (r_max - r_min)
            tier_assignments.append((name, round(float(r), 1)))

    # Shuffle tier assignments
    rng.shuffle(tier_assignments)

    # Generate positions with opening rejection
    max_attempts = n_samples * 20
    samples = []
    attempt = 0
    while len(samples) < n_samples and attempt < max_attempts:
        batch_n = min(n_samples * 2, n_samples - len(samples) + 50)
        unit = sampler.random(n=batch_n)
        for j in range(len(unit)):
            if len(samples) >= n_samples:
                break
            z_c = z_lo + unit[j, 0] * (z_hi - z_lo)
            t_c = theta_lo + unit[j, 1] * (theta_hi - theta_lo)

            idx = len(samples)
            tier_name, radius = tier_assignments[idx]

            # Clamp to keep defect inside sector (convert radius to angular extent)
            arc_margin = math.degrees(radius / RADIUS_INNER)
            z_c = max(z_lo + radius, min(z_hi - radius, z_c))
            t_c = max(theta_lo + arc_margin, min(theta_hi - arc_margin, t_c))

            # Reject if overlaps opening
            if _defect_overlaps_opening(z_c, t_c, radius, OPENINGS):
                continue

            defect_p = {
                'z_center': round(float(z_c), 1),
                'theta_deg': round(float(t_c), 2),
                'radius': radius,
            }
            if multi_defect:
                dt_names = [d[0] for d in DEFECT_TYPES]
                dt_weights = np.array([d[1] for d in DEFECT_TYPES])
                dt_weights /= dt_weights.sum()
                defect_p['defect_type'] = rng.choice(dt_names, p=dt_weights)
            samples.append({
                'id': idx,
                'job_name': 'Job-GW-Fair-%04d' % idx,
                'size_tier': tier_name,
                'defect_params': defect_p,
            })
        attempt += 1

    # If not enough samples (unlikely), fill with random
    while len(samples) < n_samples:
        idx = len(samples)
        tier_name, radius = tier_assignments[idx]
        z_c = rng.uniform(z_lo + radius, z_hi - radius)
        t_c = rng.uniform(theta_lo, theta_hi)
        defect_p = {
            'z_center': round(float(z_c), 1),
            'theta_deg': round(float(t_c), 2),
            'radius': radius,
        }
        if multi_defect:
            dt_names = [d[0] for d in DEFECT_TYPES]
            dt_weights = np.array([d[1] for d in DEFECT_TYPES])
            dt_weights /= dt_weights.sum()
            defect_p['defect_type'] = rng.choice(dt_names, p=dt_weights)
        samples.append({
            'id': idx,
            'job_name': 'Job-GW-Fair-%04d' % idx,
            'size_tier': tier_name,
            'defect_params': defect_p,
        })

    # Tier counts
    tier_counts = {}
    for s in samples:
        t = s['size_tier']
        tier_counts[t] = tier_counts.get(t, 0) + 1

    doe = {
        'n_samples': n_samples,
        'freq_khz': freq_khz,
        'sector_angle': sector_angle,
        'z_range': [z_min, z_max],
        'healthy_job': 'Job-GW-Fair-Healthy',
        'bounds': {
            'z_center': [z_lo, z_hi],
            'theta_deg': [theta_lo, theta_hi],
            'radius_tiers': [
                {'name': t[0], 'min': t[1], 'max': t[2], 'frac': t[3]}
                for t in size_tiers
            ],
        },
        'tier_counts': tier_counts,
        'seed': seed,
        'samples': samples,
    }
    return doe


def main():
    parser = argparse.ArgumentParser(
        description='Generate DOE for guided wave fairing GNN dataset')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of defective samples (default: 100)')
    parser.add_argument('--freq', type=int, default=50,
                        help='Excitation frequency in kHz (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--z_min', type=float, default=DEFAULT_Z_MIN,
                        help='Barrel z_min in mm (default: 500)')
    parser.add_argument('--z_max', type=float, default=DEFAULT_Z_MAX,
                        help='Barrel z_max in mm (default: 2500)')
    parser.add_argument('--sector_angle', type=float, default=DEFAULT_SECTOR_ANGLE,
                        help='Sector angle in degrees (default: 30)')
    parser.add_argument('--tier_preset', type=str, default='micro',
                        choices=['micro', 'original'],
                        help='Size tier preset: micro (10-30mm) or original (20-80mm)')
    parser.add_argument('--multi_defect', action='store_true',
                        help='Assign random defect types (7 types)')
    parser.add_argument('--output', type=str, default='doe_gw_fairing.json',
                        help='Output JSON path (default: doe_gw_fairing.json)')
    args = parser.parse_args()

    tiers = SIZE_TIERS_ORIGINAL if args.tier_preset == 'original' else SIZE_TIERS_MICRO
    doe = generate_gw_doe(
        n_samples=args.n_samples,
        seed=args.seed,
        freq_khz=args.freq,
        z_min=args.z_min,
        z_max=args.z_max,
        sector_angle=args.sector_angle,
        size_tiers=tiers,
        multi_defect=args.multi_defect,
    )

    with open(args.output, 'w') as f:
        json.dump(doe, f, indent=2)

    print("Generated GW Fairing DOE: %d defective samples + 1 healthy" % doe['n_samples'])
    print("  Frequency: %d kHz" % doe['freq_khz'])
    print("  Sector: %.0f deg, z=[%.0f, %.0f] mm" % (
        doe['sector_angle'], doe['z_range'][0], doe['z_range'][1]))
    print("  z_center: [%.1f, %.1f] mm" % tuple(doe['bounds']['z_center']))
    print("  theta_deg: [%.1f, %.1f] deg" % tuple(doe['bounds']['theta_deg']))
    print("  Size tiers: %s" % doe['tier_counts'])
    print("  Healthy ref: %s" % doe['healthy_job'])
    print("Saved to: %s" % args.output)


if __name__ == '__main__':
    main()
