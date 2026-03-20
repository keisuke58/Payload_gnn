#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active DOE — Bayesian Optimization for FEM parameter selection.

Uses trained GNN model uncertainty to guide next FEM simulation parameters.
Goal: maximize information gain with minimum FEM runs.

Strategy:
  1. Define parameter space (n_stuck_bolts, spring_stiffness, bolt_positions)
  2. For each candidate, predict anomaly detection performance via GNN
  3. Use acquisition function (Expected Improvement) to select next case
  4. Generate and submit FEM job automatically

Usage:
  python src/active_doe.py --model results/separation_gnn/best_model.pt \
      --data_dir data/processed_separation \
      --n_suggest 5
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from itertools import product


def define_parameter_space():
    """Define the DOE parameter space for fairing separation."""
    return {
        'n_stuck_bolts': list(range(0, 13)),  # 0-12
        'spring_stiffness': [300, 500, 1000, 2000, 3000, 5000, 8000, 12000],
        'stuck_pattern': ['uniform', 'clustered', 'random'],
    }


def load_completed_cases(results_dir='results/separation'):
    """Load already-completed DOE cases and their metrics."""
    completed = []
    import glob
    for csv_path in glob.glob(os.path.join(results_dir, '*_time_history.csv')):
        case_name = os.path.basename(csv_path).replace('_time_history.csv', '')
        # Parse parameters from case name
        params = parse_case_params(case_name)
        if params is not None:
            # Extract key metric: max stress asymmetry
            metric = extract_case_metric(csv_path)
            params['metric'] = metric
            params['case_name'] = case_name
            completed.append(params)
    return completed


def parse_case_params(case_name):
    """Parse n_stuck and spring_k from case name."""
    params = {'n_stuck_bolts': 0, 'spring_stiffness': 5000}

    if 'Normal' in case_name:
        params['n_stuck_bolts'] = 0
    elif 'Stuck3' in case_name:
        params['n_stuck_bolts'] = 3
    elif 'Stuck6' in case_name:
        params['n_stuck_bolts'] = 6
    elif 'DOE-S' in case_name:
        # Sep-DOE-S04, Sep-DOE-S06K300
        import re
        m = re.search(r'S(\d+)', case_name)
        if m:
            params['n_stuck_bolts'] = int(m.group(1))
        m = re.search(r'K(\d+)', case_name)
        if m:
            params['spring_stiffness'] = int(m.group(1))
    elif 'DOE-K' in case_name:
        import re
        params['n_stuck_bolts'] = 3  # default for K-sweep
        m = re.search(r'K(\d+)', case_name)
        if m:
            params['spring_stiffness'] = int(m.group(1))
    else:
        return None

    return params


def extract_case_metric(csv_path):
    """Extract anomaly severity metric from time history CSV."""
    import csv
    max_stress_q1 = 0
    max_stress_q2 = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['step'] != 'Step-Separation':
                continue
            s = float(row['s_mises_max_MPa'])
            if 'Q1' in row['instance'] and 'SKIN' in row['instance'].upper():
                max_stress_q1 = max(max_stress_q1, s)
            if 'Q2' in row['instance'] and 'SKIN' in row['instance'].upper():
                max_stress_q2 = max(max_stress_q2, s)
    # Asymmetry metric: how different Q1 and Q2 are
    if max_stress_q2 > 0:
        return abs(max_stress_q1 - max_stress_q2) / max_stress_q2
    return 0.0


def gaussian_process_surrogate(completed, candidates):
    """Simple GP surrogate using RBF kernel for acquisition function.

    Returns predicted mean and uncertainty for each candidate.
    """
    if len(completed) < 2:
        # Not enough data for GP, return uniform uncertainty
        return np.zeros(len(candidates)), np.ones(len(candidates))

    # Feature matrix from completed cases
    X_train = np.array([[c['n_stuck_bolts'], np.log10(c['spring_stiffness'])]
                         for c in completed])
    y_train = np.array([c['metric'] for c in completed])

    # Normalize
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-6
    X_norm = (X_train - X_mean) / X_std
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-6
    y_norm = (y_train - y_mean) / y_std

    # RBF kernel
    length_scale = 1.0
    def rbf_kernel(X1, X2):
        dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * dists / length_scale ** 2)

    K = rbf_kernel(X_norm, X_norm) + 1e-4 * np.eye(len(X_norm))
    K_inv = np.linalg.inv(K)

    # Candidate features
    X_cand = np.array([[c['n_stuck_bolts'], np.log10(c['spring_stiffness'])]
                        for c in candidates])
    X_cand_norm = (X_cand - X_mean) / X_std

    K_star = rbf_kernel(X_cand_norm, X_norm)
    K_ss = rbf_kernel(X_cand_norm, X_cand_norm)

    # Posterior mean and variance
    mu = K_star @ K_inv @ y_norm
    sigma2 = np.diag(K_ss - K_star @ K_inv @ K_star.T)
    sigma = np.sqrt(np.maximum(sigma2, 1e-6))

    # Denormalize
    mu = mu * y_std + y_mean
    sigma = sigma * y_std

    return mu, sigma


def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Expected Improvement acquisition function."""
    from scipy.stats import norm
    imp = mu - y_best - xi
    Z = imp / (sigma + 1e-8)
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei


def suggest_next_cases(completed, n_suggest=5):
    """Suggest next DOE cases using Bayesian optimization."""
    space = define_parameter_space()

    # Generate all candidate parameter combinations
    completed_set = set()
    for c in completed:
        completed_set.add((c['n_stuck_bolts'], c['spring_stiffness']))

    candidates = []
    for n_stuck in space['n_stuck_bolts']:
        for k_spring in space['spring_stiffness']:
            if (n_stuck, k_spring) not in completed_set:
                candidates.append({
                    'n_stuck_bolts': n_stuck,
                    'spring_stiffness': k_spring,
                })

    if not candidates:
        print("All parameter combinations already explored!")
        return []

    # GP surrogate + acquisition
    mu, sigma = gaussian_process_surrogate(completed, candidates)

    if len(completed) > 0:
        y_best = max(c['metric'] for c in completed)
        ei = expected_improvement(mu, sigma, y_best)
    else:
        ei = sigma  # Pure exploration

    # Sort by EI descending
    ranked = sorted(zip(candidates, ei, mu, sigma),
                     key=lambda x: -x[1])

    suggestions = []
    for cand, ei_val, mu_val, sig_val in ranked[:n_suggest]:
        suggestions.append({
            'n_stuck_bolts': cand['n_stuck_bolts'],
            'spring_stiffness': cand['spring_stiffness'],
            'expected_improvement': float(ei_val),
            'predicted_metric': float(mu_val),
            'uncertainty': float(sig_val),
        })

    return suggestions


def generate_qsub_commands(suggestions):
    """Generate qsub commands for suggested cases."""
    commands = []
    for s in suggestions:
        n = s['n_stuck_bolts']
        k = s['spring_stiffness']
        job_name = f"Sep-BO-S{n:02d}K{k}"
        args = f"--n_stuck_bolts {n} --spring_stiffness {k}"
        cmd = (f"qsub -v JOB_NAME={job_name},"
               f"EXTRA_ARGS=\"{args}\" "
               f"scripts/qsub_fairing_sep_run.sh")
        commands.append({
            'job_name': job_name,
            'command': cmd,
            'params': s,
        })
    return commands


def main():
    parser = argparse.ArgumentParser(
        description='Active DOE — Bayesian optimization for separation FEM')
    parser.add_argument('--results_dir', default='results/separation')
    parser.add_argument('--n_suggest', type=int, default=5)
    parser.add_argument('--submit', action='store_true',
                        help='Actually submit qsub jobs')
    args = parser.parse_args()

    print("=" * 60)
    print("Active DOE — Bayesian Optimization")
    print("=" * 60)

    # Load completed cases
    completed = load_completed_cases(args.results_dir)
    print(f"\nCompleted cases: {len(completed)}")
    for c in completed:
        print(f"  {c['case_name']}: stuck={c['n_stuck_bolts']}, "
              f"k={c['spring_stiffness']}, metric={c['metric']:.4f}")

    # Suggest next cases
    suggestions = suggest_next_cases(completed, args.n_suggest)
    print(f"\nSuggested next {len(suggestions)} cases:")
    for i, s in enumerate(suggestions):
        print(f"  {i+1}. stuck={s['n_stuck_bolts']}, k={s['spring_stiffness']}")
        print(f"     EI={s['expected_improvement']:.4f}, "
              f"pred={s['predicted_metric']:.4f}, "
              f"unc={s['uncertainty']:.4f}")

    # Generate commands
    commands = generate_qsub_commands(suggestions)
    print(f"\nqsub commands:")
    for c in commands:
        print(f"  {c['command']}")

    if args.submit:
        import subprocess
        for c in commands:
            result = subprocess.run(c['command'], shell=True,
                                     capture_output=True, text=True)
            print(f"  Submitted {c['job_name']}: {result.stdout.strip()}")

    # Save suggestions
    out_path = os.path.join(args.results_dir, 'bo_suggestions.json')
    with open(out_path, 'w') as f:
        json.dump({
            'completed': completed,
            'suggestions': suggestions,
            'commands': [c['command'] for c in commands],
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
