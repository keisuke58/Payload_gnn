# -*- coding: utf-8 -*-
"""
Generate DOE for Multi-Defect GW FEM models.

Strategy: Pack N defects into a single 30° fairing sector,
ensuring minimum spacing between defects (no scattering interference).

Physics constraints:
  - A0 wavelength at 50 kHz ≈ 31 mm
  - Scattering decays with distance (~1/√r for cylindrical spreading)
  - At 5λ ≈ 155 mm separation, interference < 5% of local signal
  - Conservative: 300 mm minimum center-to-center spacing

Sector dimensions:
  - z: 600–2400 mm (usable, excluding near ring frames)
  - θ: 3–27° (usable, excluding ABL zones at 0° and 30°)
  - Arc span at R=2638 mm: ~1105 mm (for 24° usable range)
  - Axial span: 1800 mm

With 300 mm spacing: ~6 axial × ~4 circumferential = ~24 defects per FEM.
With 200 mm spacing: ~9 × ~5 = ~45 defects per FEM.

Usage:
  python src/generate_doe_multi_defect.py --n_defects 24 --spacing 300
  python src/generate_doe_multi_defect.py --n_defects 45 --spacing 200 --n_models 3
"""

import argparse
import json
import numpy as np
import os


# Fairing geometry
R_OUTER = 2638.0  # mm
SECTOR_ANGLE = 30.0  # degrees
ABL_WIDTH = 2.0  # degrees (absorbing boundary layer)

# Usable region (excluding ABL and ring frame neighborhoods)
Z_MIN_USABLE = 600.0   # mm (50mm away from ring frame at 500)
Z_MAX_USABLE = 2400.0  # mm (50mm away from ring frame at 2500)
THETA_MIN_USABLE = ABL_WIDTH + 1.0  # degrees
THETA_MAX_USABLE = SECTOR_ANGLE - ABL_WIDTH - 1.0  # degrees

# Defect size range
RADIUS_TIERS = [
    {'name': 'small', 'min': 20, 'max': 40, 'weight': 0.30},
    {'name': 'medium', 'min': 40, 'max': 60, 'weight': 0.45},
    {'name': 'large', 'min': 60, 'max': 80, 'weight': 0.25},
]

DEFECT_TYPES = [
    'debonding', 'fod', 'impact', 'delamination',
    'inner_debond', 'thermal_progression', 'acoustic_fatigue',
]


def arc_mm(theta_deg):
    """Convert angle to arc length at outer skin radius."""
    return R_OUTER * np.radians(theta_deg)


def place_defects_grid(n_defects, min_spacing_mm, rng):
    """Place defects on a jittered grid within the usable sector.

    Returns list of (z_center, theta_deg).
    """
    z_span = Z_MAX_USABLE - Z_MIN_USABLE
    arc_span = arc_mm(THETA_MAX_USABLE - THETA_MIN_USABLE)

    # Compute grid dimensions
    aspect = arc_span / z_span
    n_theta = max(2, int(np.sqrt(n_defects * aspect) + 0.5))
    n_z = max(2, int(n_defects / n_theta + 0.5))

    # Adjust to not exceed n_defects
    while n_z * n_theta > n_defects * 1.2:
        if n_z > n_theta:
            n_z -= 1
        else:
            n_theta -= 1

    actual_n = n_z * n_theta

    # Grid spacing
    dz = z_span / (n_z + 1)
    dtheta = (THETA_MAX_USABLE - THETA_MIN_USABLE) / (n_theta + 1)

    # Check minimum spacing
    actual_spacing_z = dz
    actual_spacing_arc = arc_mm(dtheta)
    actual_min = min(actual_spacing_z, actual_spacing_arc)

    if actual_min < min_spacing_mm * 0.8:
        print(f"  WARNING: Grid spacing ({actual_min:.0f} mm) < "
              f"requested min ({min_spacing_mm} mm). Reducing n_defects.")
        # Reduce grid to meet spacing
        n_z = max(2, int(z_span / min_spacing_mm))
        n_theta = max(2, int(arc_span / min_spacing_mm))
        actual_n = n_z * n_theta
        dz = z_span / (n_z + 1)
        dtheta = (THETA_MAX_USABLE - THETA_MIN_USABLE) / (n_theta + 1)

    positions = []
    jitter_z = dz * 0.3  # 30% jitter
    jitter_theta = dtheta * 0.3

    for iz in range(n_z):
        for it in range(n_theta):
            z = Z_MIN_USABLE + dz * (iz + 1) + rng.uniform(-jitter_z, jitter_z)
            theta = THETA_MIN_USABLE + dtheta * (it + 1) + rng.uniform(-jitter_theta, jitter_theta)
            # Clamp to usable region
            z = np.clip(z, Z_MIN_USABLE, Z_MAX_USABLE)
            theta = np.clip(theta, THETA_MIN_USABLE, THETA_MAX_USABLE)
            positions.append((z, theta))

    return positions[:n_defects], n_z, n_theta


def generate_multi_defect_doe(n_defects=24, min_spacing=300,
                               n_models=1, seed=42):
    """Generate DOE with multiple defects per model.

    Args:
        n_defects: target number of defects per model
        min_spacing: minimum center-to-center spacing in mm
        n_models: number of FEM models to generate
        seed: random seed

    Returns:
        dict: DOE specification
    """
    rng = np.random.RandomState(seed)

    models = []
    for model_idx in range(n_models):
        # Place defects on jittered grid
        positions, n_z, n_theta = place_defects_grid(n_defects, min_spacing, rng)

        defects = []
        for z, theta in positions:
            # Random radius from tier distribution
            tier_probs = [t['weight'] for t in RADIUS_TIERS]
            tier_idx = rng.choice(len(RADIUS_TIERS), p=tier_probs)
            tier = RADIUS_TIERS[tier_idx]
            radius = rng.uniform(tier['min'], tier['max'])

            # Random defect type
            defect_type = rng.choice(DEFECT_TYPES)

            defects.append({
                'z_center': float(round(z, 1)),
                'theta_deg': float(round(theta, 2)),
                'radius': float(round(radius, 1)),
                'defect_type': defect_type,
                'size_tier': tier['name'],
            })

        job_name = f"Job-GW-Multi-{model_idx:04d}"
        models.append({
            'job_name': job_name,
            'model_idx': model_idx,
            'n_defects': len(defects),
            'defects': defects,
        })

    # Sensor grid specification (same as v3 with n_sensors=100)
    z_span = Z_MAX_USABLE - Z_MIN_USABLE
    arc_span = arc_mm(THETA_MAX_USABLE - THETA_MIN_USABLE)
    aspect = arc_span / z_span
    n_s_theta = max(2, int(np.sqrt(100 * aspect) + 0.5))
    n_s_z = max(2, int(100 / n_s_theta + 0.5))

    doe = {
        'type': 'multi_defect_gw',
        'n_models': n_models,
        'n_defects_per_model': n_defects,
        'min_spacing_mm': min_spacing,
        'seed': seed,
        'bounds': {
            'z_center': [Z_MIN_USABLE, Z_MAX_USABLE],
            'theta_deg': [THETA_MIN_USABLE, THETA_MAX_USABLE],
            'radius_tiers': RADIUS_TIERS,
        },
        'sensor_grid': {
            'n_theta': n_s_theta,
            'n_z': n_s_z,
            'n_total': n_s_theta * n_s_z,
            'z_range': [Z_MIN_USABLE - 50, Z_MAX_USABLE + 50],
            'theta_range': [THETA_MIN_USABLE, THETA_MAX_USABLE],
        },
        'models': models,
    }

    # Summary
    total_defects = sum(m['n_defects'] for m in models)
    print(f"\n=== Multi-Defect DOE Summary ===")
    print(f"  Models: {n_models}")
    print(f"  Defects per model: {n_defects} (actual: {models[0]['n_defects']})")
    print(f"  Total defect samples: {total_defects}")
    print(f"  Min spacing: {min_spacing} mm")
    print(f"  Sensor grid: {n_s_theta} × {n_s_z} = {n_s_theta * n_s_z}")
    print(f"  FEM cost: {n_models} runs × ~10h = ~{n_models * 10}h")
    print(f"  Equivalent single-defect FEM: {total_defects} × ~10h = ~{total_defects * 10}h")
    print(f"  Speedup: {total_defects}× (from {total_defects * 10}h → {n_models * 10}h)")

    return doe


def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-defect GW DOE')
    parser.add_argument('--n_defects', type=int, default=24,
                        help='Defects per FEM model')
    parser.add_argument('--spacing', type=int, default=300,
                        help='Min spacing between defects (mm)')
    parser.add_argument('--n_models', type=int, default=3,
                        help='Number of FEM models')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='doe_gw_multi_defect.json')
    args = parser.parse_args()

    doe = generate_multi_defect_doe(
        n_defects=args.n_defects,
        min_spacing=args.spacing,
        n_models=args.n_models,
        seed=args.seed,
    )

    with open(args.output, 'w') as f:
        json.dump(doe, f, indent=2, ensure_ascii=False)
    print(f"\n  DOE saved: {args.output}")


if __name__ == '__main__':
    main()
