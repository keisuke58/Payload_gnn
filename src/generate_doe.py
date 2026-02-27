#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOE Parameter Generation — Latin Hypercube Sampling

Generates debonding defect parameters using Latin Hypercube Sampling (LHS)
for efficient coverage of the parameter space.

Usage:
  python src/generate_doe.py --n_samples 500 --output doe_params.json
  python src/generate_doe.py --n_samples 50 --seed 123 --output doe_test.json

Output: JSON file with sampled parameters for batch FEM generation.
"""

import argparse
import json
import sys

from scipy.stats.qmc import LatinHypercube
import numpy as np

# Parameter bounds for H3 fairing (1/6 sector, 60° arc)
DEFAULT_BOUNDS = {
    'theta_deg': (10.0, 50.0),     # degrees (within 60° arc, margin from edges)
    'z_center':  (500.0, 4500.0),  # mm (margin from clamped bottom & top)
    'radius':    (50.0, 300.0),    # mm (debonding patch size)
}


def generate_doe(n_samples, bounds=None, seed=42, n_healthy=1):
    """
    Generate DOE parameters using Latin Hypercube Sampling.

    Args:
        n_samples: number of defective samples to generate
        bounds: dict of parameter bounds {name: (low, high)}
        seed: random seed for reproducibility
        n_healthy: number of healthy baseline samples (prepended)

    Returns:
        dict with 'samples' list and metadata
    """
    if bounds is None:
        bounds = DEFAULT_BOUNDS

    param_names = list(bounds.keys())
    d = len(param_names)
    lows = np.array([bounds[k][0] for k in param_names])
    highs = np.array([bounds[k][1] for k in param_names])

    # Latin Hypercube Sampling
    sampler = LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    # Scale to physical bounds
    scaled = lows + unit_samples * (highs - lows)

    # Build sample list
    samples = []

    # Healthy baseline(s) first
    for i in range(n_healthy):
        samples.append({
            'id': i,
            'job_name': 'H3_Healthy_%04d' % i,
            'defect_params': None,
        })

    # Defective samples
    for i in range(n_samples):
        sample_id = n_healthy + i
        params = {param_names[j]: round(float(scaled[i, j]), 2)
                  for j in range(d)}
        samples.append({
            'id': sample_id,
            'job_name': 'H3_Debond_%04d' % sample_id,
            'defect_params': params,
        })

    doe = {
        'n_total': len(samples),
        'n_healthy': n_healthy,
        'n_defective': n_samples,
        'bounds': {k: list(v) for k, v in bounds.items()},
        'seed': seed,
        'samples': samples,
    }

    return doe


def main():
    parser = argparse.ArgumentParser(
        description='Generate DOE parameters for debonding FEM batch')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of defective samples (default: 500)')
    parser.add_argument('--n_healthy', type=int, default=1,
                        help='Number of healthy baselines (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='doe_params.json',
                        help='Output JSON file path')
    args = parser.parse_args()

    doe = generate_doe(
        n_samples=args.n_samples,
        n_healthy=args.n_healthy,
        seed=args.seed,
    )

    with open(args.output, 'w') as f:
        json.dump(doe, f, indent=2)

    print("Generated DOE: %d total samples (%d healthy + %d defective)" %
          (doe['n_total'], doe['n_healthy'], doe['n_defective']))
    print("  theta_deg: [%.1f, %.1f]" % tuple(doe['bounds']['theta_deg']))
    print("  z_center:  [%.1f, %.1f]" % tuple(doe['bounds']['z_center']))
    print("  radius:    [%.1f, %.1f]" % tuple(doe['bounds']['radius']))
    print("Saved to: %s" % args.output)


if __name__ == '__main__':
    main()
