# -*- coding: utf-8 -*-
"""
reliability_analysis.py  -- Post-processing for PCE UQ results

Loads pce_results.json produced by pce_driver.py, then:
  1. Rebuilds PCE surrogate
  2. Runs MC on surrogate (N=100,000)
  3. Computes reliability metrics (Pf, beta index)
  4. Compares PCE surrogate vs. simple NN surrogate
  5. Generates publication-quality figures

Usage:
    python3 src/reliability_analysis.py \\
        --results abaqus_work/pce_uq/pce_results.json \\
        --outdir  figures/pce_uq
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import chaospy as cp
except ImportError:
    sys.exit("ERROR: pip install chaospy")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("ERROR: pip install matplotlib")

# Reuse distribution definition from pce_driver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pce_driver import build_joint_distribution, PARAM_NAMES, NOMINAL, COV


# ---------------------------------------------------------------------------
# Helper: FORM approximation (first-order reliability method)
# ---------------------------------------------------------------------------
def form_beta(pce_model, joint, qoi_threshold, n_mc=100_000, seed=42):
    """
    Estimate reliability index beta via inverse normal CDF of Pf.
    Uses MC on PCE surrogate as reference.

    Returns: Pf (float), beta (float)
    """
    rng     = np.random.default_rng(seed)
    samples = joint.sample(n_mc, seed=seed)
    y_hat   = cp.call(pce_model, samples)
    pf      = float((y_hat > qoi_threshold).mean())
    if pf <= 0.0:
        return pf, float('inf')
    if pf >= 1.0:
        return pf, float('-inf')
    from scipy.stats import norm
    beta = -norm.ppf(pf)
    return pf, beta


# ---------------------------------------------------------------------------
# Helper: Simple NN surrogate (MLP)
# ---------------------------------------------------------------------------
def train_nn_surrogate(X_train, Y_train, hidden=(64, 64, 32),
                       epochs=500, lr=1e-3, seed=42):
    """
    Train a small MLP surrogate on (X_train, Y_train).
    X_train: (n_samples, n_params)
    Y_train: (n_samples,)
    Returns predict_fn(X) -> ndarray
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  [WARN] PyTorch not available. Skipping NN comparison.")
        return None

    torch.manual_seed(seed)
    n_in = X_train.shape[1]

    layers = []
    prev   = n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    model = nn.Sequential(*layers)

    # Standardise inputs
    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0) + 1e-8
    Y_mean = Y_train.mean()
    Y_std  = Y_train.std() + 1e-8

    Xt = torch.tensor((X_train - X_mean) / X_std, dtype=torch.float32)
    Yt = torch.tensor((Y_train - Y_mean) / Y_std, dtype=torch.float32).unsqueeze(1)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, Yt)
        loss.backward()
        opt.step()

    def predict_fn(X_new):
        model.eval()
        with torch.no_grad():
            Xn = torch.tensor((X_new - X_mean) / X_std, dtype=torch.float32)
            return (model(Xn).squeeze().numpy() * Y_std + Y_mean)

    return predict_fn


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_sobol(stats, qoi_name, outdir):
    """Bar chart of first-order and total Sobol indices."""
    s1 = [stats['sobol_first'][n] for n in PARAM_NAMES]
    sT = [stats['sobol_total'][n] for n in PARAM_NAMES]

    x   = np.arange(len(PARAM_NAMES))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, s1, w, label='First-order $S_i$',  color='steelblue')
    ax.bar(x + w/2, sT, w, label='Total-order $S_T^i$', color='tomato', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES)
    ax.set_ylabel('Sobol Sensitivity Index')
    ax.set_title('Sobol Indices — %s' % qoi_name)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'sobol_%s.pdf' % qoi_name)
    plt.savefig(path)
    plt.close()
    print("  Saved: %s" % path)


def plot_pdf_comparison(y_pce, y_nn, y_mc_ref, qoi_name, threshold, outdir):
    """Overlay PDFs: MC reference vs PCE surrogate vs NN surrogate."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 60

    if y_mc_ref is not None:
        ax.hist(y_mc_ref, bins=bins, density=True, alpha=0.4,
                color='gray',      label='MC reference (N=10k)', histtype='stepfilled')
    ax.hist(y_pce, bins=bins, density=True, alpha=0.6,
            color='steelblue',  label='PCE surrogate (N=100k)',  histtype='step', linewidth=1.8)
    if y_nn is not None:
        ax.hist(y_nn, bins=bins, density=True, alpha=0.6,
                color='tomato',  label='NN surrogate  (N=100k)',  histtype='step',
                linewidth=1.8, linestyle='--')

    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle=':', linewidth=1.5,
                   label='Threshold = %.3g' % threshold)

    ax.set_xlabel(qoi_name)
    ax.set_ylabel('Probability density')
    ax.set_title('PDF Comparison — %s' % qoi_name)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'pdf_%s.pdf' % qoi_name)
    plt.savefig(path)
    plt.close()
    print("  Saved: %s" % path)


def plot_convergence(joint, pce_model, qoi_name, outdir, seed=42):
    """
    Plot convergence of PCE mean/std estimate as a function of MC sample size.
    Shows how quickly PCE surrogate-based MC converges compared to raw MC.
    """
    ns_list = [100, 300, 500, 1000, 3000, 5000, 10000, 30000, 100000]
    means, stds = [], []
    for ns in ns_list:
        s = joint.sample(ns, seed=seed)
        y = cp.call(pce_model, s)
        means.append(y.mean())
        stds.append(y.std())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.semilogx(ns_list, means, 'o-', color='steelblue')
    ax1.axhline(means[-1], color='gray', linestyle='--', linewidth=0.8)
    ax1.set_xlabel('N MC samples on PCE surrogate')
    ax1.set_ylabel('Mean of %s' % qoi_name)
    ax1.set_title('Convergence of Mean')
    ax1.grid(alpha=0.3)

    ax2.semilogx(ns_list, stds, 'o-', color='tomato')
    ax2.axhline(stds[-1], color='gray', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('N MC samples on PCE surrogate')
    ax2.set_ylabel('Std of %s' % qoi_name)
    ax2.set_title('Convergence of Std')
    ax2.grid(alpha=0.3)

    plt.suptitle('PCE Surrogate MC Convergence — %s' % qoi_name)
    plt.tight_layout()
    path = os.path.join(outdir, 'convergence_%s.pdf' % qoi_name)
    plt.savefig(path)
    plt.close()
    print("  Saved: %s" % path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Post-process PCE results: reliability + comparison plots')
    parser.add_argument('--results', required=True,
                        help='Path to pce_results.json')
    parser.add_argument('--outdir', default='figures/pce_uq',
                        help='Output directory for figures and stats')
    parser.add_argument('--degree', type=int, default=3,
                        help='PCE degree used in pce_driver.py (default: 3)')
    parser.add_argument('--rule', default='gaussian',
                        choices=['gaussian', 'halton', 'sobol'])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Load results ----
    with open(args.results) as f:
        data = json.load(f)
    results = data['results']
    print("Loaded %d samples from %s" % (len(results), args.results))

    # ---- Rebuild distribution & nodes ----
    joint = build_joint_distribution()

    completed_ids = [r['sample_id'] for r in results]
    n_params   = len(PARAM_NAMES)

    # Rebuild node matrix from stored params
    nodes_all = np.array(
        [[r['params'][n] for n in PARAM_NAMES] for r in results]
    ).T   # (n_params, n_samples)

    if args.rule == 'gaussian':
        nodes_q, weights_q = cp.generate_quadrature(
            args.degree, joint, rule='gaussian', sparse=True)
        W_all = weights_q
        # Use regression if sparse weights are negative
        if (W_all < 0).any():
            print("  [INFO] Sparse grid negative weights -> using regression")
            X = nodes_all
            W = None
        else:
            X = nodes_q[:, completed_ids]
            W = W_all[completed_ids]
    else:
        X = nodes_all
        W = None

    expansion = cp.generate_expansion(args.degree, joint)

    qoi_names   = ['max_smises', 'max_sdeg', 'max_disp']
    thresholds  = {'max_smises': 1200.0,   # MPa
                   'max_sdeg':   0.5,       # partial debond
                   'max_disp':   None}      # no reliability threshold
    pce_models  = {}
    all_stats   = {}

    # Training data for NN
    X_train = nodes_all.T   # (n_samples, n_params)

    for qoi_name in qoi_names:
        Y = np.array([r['qoi'].get(qoi_name, 0.0) for r in results])

        # --- PCE fit ---
        if W is not None:
            approx = cp.fit_quadrature(expansion, X, W, Y)
        else:
            approx = cp.fit_regression(expansion, X, Y)
        pce_models[qoi_name] = approx

        mean   = float(cp.E(approx, joint))
        std    = float(cp.Std(approx, joint))
        sobol1 = cp.Sens_m(approx, joint)
        sobolT = cp.Sens_t(approx, joint)

        stats = {
            'mean': mean,
            'std':  std,
            'cv_pct': 100 * std / abs(mean) if mean != 0 else 0,
            'sobol_first': {n: float(sobol1[j])
                            for j, n in enumerate(PARAM_NAMES)},
            'sobol_total': {n: float(sobolT[j])
                            for j, n in enumerate(PARAM_NAMES)},
        }

        # --- Reliability ---
        thresh = thresholds.get(qoi_name)
        if thresh is not None:
            pf, beta = form_beta(approx, joint, thresh)
            stats['pf']   = pf
            stats['beta'] = beta
            print("[%s]  Mean=%.3g  Std=%.3g  Pf=%.2e  beta=%.2f" % (
                qoi_name, mean, std, pf, beta))
        else:
            print("[%s]  Mean=%.3g  Std=%.3g" % (qoi_name, mean, std))

        all_stats[qoi_name] = stats

        # --- MC samples on PCE ---
        mc_samples = joint.sample(100_000)
        y_pce_mc   = cp.call(approx, mc_samples)

        # --- NN surrogate ---
        nn_fn = train_nn_surrogate(X_train, Y)
        y_nn_mc = None
        if nn_fn is not None:
            nn_mc_X = joint.sample(100_000).T
            y_nn_mc = nn_fn(nn_mc_X)

            # NN leave-one-out error
            y_nn_loo = nn_fn(X_train)
            rmse_nn  = float(np.sqrt(np.mean((y_nn_loo - Y) ** 2)))
            # PCE leave-one-out
            y_pce_loo = np.array([float(np.atleast_1d(approx(*X_train[i]))[0])
                                   for i in range(len(Y))])
            rmse_pce  = float(np.sqrt(np.mean((y_pce_loo - Y) ** 2)))
            stats['rmse_pce_loo'] = rmse_pce
            stats['rmse_nn_loo']  = rmse_nn
            print("  RMSE (LOO): PCE=%.4g  NN=%.4g" % (rmse_pce, rmse_nn))

        # --- Plots ---
        plot_sobol(stats, qoi_name, args.outdir)
        plot_pdf_comparison(y_pce_mc, y_nn_mc, Y, qoi_name, thresh, args.outdir)
        plot_convergence(joint, approx, qoi_name, args.outdir)

    # Save summary
    stats_path = os.path.join(args.outdir, 'reliability_summary.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print("\nSummary -> %s" % stats_path)
    print("Figures -> %s/" % args.outdir)


if __name__ == '__main__':
    main()
