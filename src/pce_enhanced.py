# -*- coding: utf-8 -*-
"""
pce_enhanced.py  -- Enhanced post-processing for PCE UQ results
                    (no new Abaqus runs required — works on existing pce_results.json)

New analyses vs. reliability_analysis.py:
  1. Proper LOO cross-validation with R² metric
  2. Second-order Sobol indices heatmap (S_ij interactions)
  3. Bootstrap confidence intervals on Sobol indices
  4. Exceedance probability curve (P(Y > threshold) vs threshold)
  5. Scatter plots: QoI vs each input parameter
  6. Tornado diagram: individual CoV contribution to output variance
  7. Publication-quality figure styling

Usage:
    python3 src/pce_enhanced.py \\
        --results abaqus_work/pce_uq_deg3/pce_results.json \\
        --outdir  figures/pce_enhanced \\
        --degree 3
"""

import os, sys, json, argparse
import numpy as np

try:
    import chaospy as cp
except ImportError:
    sys.exit("pip install chaospy")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
except ImportError:
    sys.exit("pip install matplotlib")

# Reuse core definitions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pce_driver import build_joint_distribution, PARAM_NAMES, NOMINAL, COV

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.labelsize':   12,
    'axes.titlesize':   13,
    'legend.fontsize':  9,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'lines.linewidth':  1.8,
})
COLORS = ['#2c7bb6', '#d7191c', '#1a9641', '#fdae61', '#756bb1']


# ── 1. Proper LOO cross-validation ────────────────────────────────────────────
def loo_cv(expansion, X_nodes, Y_obs, joint, use_regression=True):
    """
    Leave-one-out cross-validation.
    For each i: fit PCE on all except i, predict at i.
    Returns y_pred (n,), r2, rmse.
    """
    n = X_nodes.shape[1]
    y_pred = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr = X_nodes[:, mask]
        Y_tr = Y_obs[mask]
        approx_i = cp.fit_regression(expansion, X_tr, Y_tr)
        val = np.atleast_1d(approx_i(*X_nodes[:, i]))
        y_pred[i] = float(val[0])

    ss_res = np.sum((Y_obs - y_pred) ** 2)
    ss_tot = np.sum((Y_obs - Y_obs.mean()) ** 2)
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(ss_res / n))
    return y_pred, r2, rmse


def plot_loo(Y_obs, y_pred, qoi_name, r2, rmse, outdir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Y_obs, y_pred, s=18, alpha=0.7, color=COLORS[0], edgecolors='none')
    lo = min(Y_obs.min(), y_pred.min())
    hi = max(Y_obs.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.2, label='Ideal')
    ax.set_xlabel('Abaqus (true)')
    ax.set_ylabel('PCE prediction (LOO)')
    ax.set_title('LOO Cross-Validation — %s\n$R^2$=%.5f, RMSE=%.4g' % (qoi_name, r2, rmse))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'loo_%s.pdf' % qoi_name)
    plt.savefig(path)
    plt.close()
    print("  Saved: %s" % path)
    return r2, rmse


# ── 2. Second-order Sobol indices ─────────────────────────────────────────────
def second_order_sobol(approx, joint):
    """
    Returns matrix S2[i,j] (i<j) using chaospy Sens_m2.
    """
    try:
        s2_flat = cp.Sens_m2(approx, joint)  # shape (n_params, n_params)
        return np.array(s2_flat)
    except Exception as e:
        print("  [WARN] Sens_m2 failed: %s" % e)
        return None


def plot_sobol2_heatmap(s2_matrix, s1, qoi_name, outdir):
    n = len(PARAM_NAMES)
    # Build full symmetric matrix for display
    mat = np.zeros((n, n))
    for i, name in enumerate(PARAM_NAMES):
        mat[i, i] = s1[name]     # diagonal = first-order (s1 is a dict)
    if s2_matrix is not None:
        for i in range(n):
            for j in range(n):
                if i != j:
                    mat[i, j] = abs(float(s2_matrix[i, j]))  # symmetrize

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto',
                   norm=Normalize(vmin=0, vmax=mat.max()))
    ax.set_xticks(range(n)); ax.set_xticklabels(PARAM_NAMES)
    ax.set_yticks(range(n)); ax.set_yticklabels(PARAM_NAMES)
    ax.set_title('Sobol Sensitivity Heatmap — %s\n(diagonal=S$_i$, off-diag=S$_{ij}$)' % qoi_name)

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            color = 'white' if val > 0.5 * mat.max() else 'black'
            ax.text(j, i, '%.4f' % val, ha='center', va='center',
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='Sobol index')
    plt.tight_layout()
    path = os.path.join(outdir, 'sobol2_heatmap_%s.pdf' % qoi_name)
    plt.savefig(path)
    plt.close()
    print("  Saved: %s" % path)


# ── 3. Bootstrap CIs on Sobol indices ─────────────────────────────────────────
def _bootstrap_sobol_mc(approx, joint, n_boot=200, ci=95, seed=42):
    """
    Bootstrap CI for Sobol indices using repeated MC Saltelli estimation.
    Returns dict: param -> (S1_mean, S1_lo, S1_hi)
    """
    alpha = (100 - ci) / 2
    s1_boot = {name: [] for name in PARAM_NAMES}

    for b in range(n_boot):
        s1 = _sobol_mc(approx, joint, n=30_000, seed=seed + b)
        for name in PARAM_NAMES:
            s1_boot[name].append(s1[name])

    result = {}
    for name in PARAM_NAMES:
        arr = np.array(s1_boot[name])
        result[name] = (arr.mean(),
                        np.percentile(arr, alpha),
                        np.percentile(arr, 100 - alpha))
    return result


def plot_sobol_ci(s1_ci, s1_point, qoi_name, outdir):
    names  = PARAM_NAMES
    means  = [s1_point[n] for n in names]
    lo_err = [s1_point[n] - s1_ci[n][1] for n in names]
    hi_err = [s1_ci[n][2] - s1_point[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(names))
    ax.bar(x, means, color=COLORS[0], alpha=0.8, label='$S_i$ (point)')
    ax.errorbar(x, means, yerr=[lo_err, hi_err],
                fmt='none', color='black', capsize=4, linewidth=1.5, label='95% Bootstrap CI')
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('First-order Sobol Index $S_i$')
    ax.set_ylim(bottom=0)
    ax.set_title('Sobol Indices with Bootstrap CI — %s' % qoi_name)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'sobol_ci_%s.pdf' % qoi_name)
    plt.savefig(path)
    plt.close()
    print("  Saved: %s" % path)


# ── 4. Exceedance probability curve ───────────────────────────────────────────
def plot_exceedance(approx, joint, qoi_name, outdir,
                    extra_thresholds=None, n_mc=200_000, seed=42):
    """
    Plot P(Y > threshold) vs threshold (complementary CDF).
    """
    samples = joint.sample(n_mc, seed=seed)
    y_mc    = cp.call(approx, samples)

    y_sorted = np.sort(y_mc)
    pexceed  = 1.0 - np.arange(1, n_mc + 1) / n_mc

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(y_sorted, pexceed, color=COLORS[0], linewidth=1.8)
    ax.set_xlabel('%s' % qoi_name)
    ax.set_ylabel('Exceedance Probability $P(Y > y)$')
    ax.set_title('Exceedance Probability — %s' % qoi_name)
    ax.grid(which='both', alpha=0.3)

    # Mark key thresholds
    if extra_thresholds:
        for thresh, label, color in extra_thresholds:
            pf = float((y_mc > thresh).mean())
            if pf > 0:
                ax.axvline(thresh, color=color, linestyle='--',
                           linewidth=1.3,
                           label='%s=%.3g\n$P_f$=%.2e' % (label, thresh, pf))
            else:
                ax.axvline(thresh, color=color, linestyle='--',
                           linewidth=1.3,
                           label='%s=%.3g ($P_f$=0)' % (label, thresh))
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(outdir, 'exceedance_%s.pdf' % qoi_name)
    plt.savefig(path)
    plt.close()
    print("  Saved: %s" % path)


# ── 5. Scatter plots: QoI vs each input ───────────────────────────────────────
def plot_scatter_matrix(X_nodes, Y_obs, qoi_name, outdir):
    """
    Grid of scatter plots: each input parameter vs QoI.
    Shows linearity / nonlinearity of response surface.
    """
    n_params = len(PARAM_NAMES)
    fig, axes = plt.subplots(1, n_params, figsize=(3.5 * n_params, 3.5),
                              sharey=True)

    for j, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
        x_vals = X_nodes[j, :]
        ax.scatter(x_vals, Y_obs, s=10, alpha=0.5, color=COLORS[j % len(COLORS)],
                   edgecolors='none')
        # Pearson r
        r = float(np.corrcoef(x_vals, Y_obs)[0, 1])
        ax.set_xlabel(name + '\n($r$=%.3f)' % r)
        if j == 0:
            ax.set_ylabel(qoi_name)
        ax.grid(alpha=0.25)
        ax.set_title(name)

    fig.suptitle('Scatter: Input Parameters vs %s' % qoi_name, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, 'scatter_%s.pdf' % qoi_name)
    plt.savefig(path)
    plt.close()
    print("  Saved: %s" % path)


# ── 6. Tornado diagram ────────────────────────────────────────────────────────
def plot_tornado(approx, joint, qoi_name, outdir, n_mc=100_000, seed=42):
    """
    For each param: fix all others at nominal, vary this param by ±2σ.
    Shows marginal contribution to output range.
    """
    base_sample = np.array([[NOMINAL[n]] for n in PARAM_NAMES], dtype=float)
    base_val    = float(np.atleast_1d(approx(*base_sample))[0])

    ranges = []
    for j, name in enumerate(PARAM_NAMES):
        sigma = NOMINAL[name] * COV[name]
        lo_pt = base_sample.copy(); lo_pt[j, 0] = NOMINAL[name] - 2 * sigma
        hi_pt = base_sample.copy(); hi_pt[j, 0] = NOMINAL[name] + 2 * sigma
        val_lo = float(np.atleast_1d(approx(*lo_pt))[0])
        val_hi = float(np.atleast_1d(approx(*hi_pt))[0])
        ranges.append((name, val_lo, val_hi))

    # Sort by range width descending
    ranges.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors_lr = [COLORS[1], COLORS[0]]  # red=decrease, blue=increase

    for i, (name, val_lo, val_hi) in enumerate(ranges):
        lo_delta = val_lo - base_val
        hi_delta = val_hi - base_val
        # Left bar (negative effect if val_lo < base_val)
        ax.barh(i, lo_delta, left=base_val, color=colors_lr[0], alpha=0.7)
        ax.barh(i, hi_delta, left=base_val, color=colors_lr[1], alpha=0.7)

    ax.set_yticks(range(len(ranges)))
    ax.set_yticklabels([r[0] for r in ranges])
    ax.axvline(base_val, color='black', linewidth=1.5, linestyle='--',
               label='Nominal = %.3g' % base_val)
    ax.set_xlabel(qoi_name)
    ax.set_title('Tornado Diagram — %s\n(±2σ variation of each parameter)' % qoi_name)
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'tornado_%s.pdf' % qoi_name)
    plt.savefig(path)
    plt.close()
    print("  Saved: %s" % path)


# ── Sobol via MC variance decomposition (Saltelli 2002, avoids cp.Sens_m bug) ─
def _sobol_mc(approx, joint, n=100_000, seed=42):
    """First-order Sobol indices via Saltelli estimator."""
    rng = np.random.default_rng(seed)
    d   = len(PARAM_NAMES)

    # Draw two independent sample matrices A, B
    A = joint.sample(n, seed=seed)              # (d, n)
    B = joint.sample(n, seed=seed + 99999)      # (d, n)

    f_A = cp.call(approx, A)
    f_B = cp.call(approx, B)

    var_total = np.var(np.concatenate([f_A, f_B]))
    s1 = {}
    for j, name in enumerate(PARAM_NAMES):
        # C_j: B with j-th column replaced by A's j-th column
        C_j = B.copy()
        C_j[j, :] = A[j, :]
        f_C = cp.call(approx, C_j)
        # Saltelli 2002 estimator: S_i = Var(E[Y|X_i]) / Var(Y)
        si = float(np.mean(f_A * (f_C - f_B)) / var_total) if var_total > 0 else 0.0
        s1[name] = max(0.0, si)   # clip negative due to sampling noise
    return s1


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Enhanced PCE post-processing')
    parser.add_argument('--results', required=True)
    parser.add_argument('--outdir',  default='figures/pce_enhanced')
    parser.add_argument('--degree',  type=int, default=3)
    parser.add_argument('--n_boot',  type=int, default=200,
                        help='Bootstrap samples for Sobol CI (default 200)')
    parser.add_argument('--skip_loo', action='store_true',
                        help='Skip LOO CV (slow for large n)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    with open(args.results) as f:
        data = json.load(f)
    results = data['results']
    print("Loaded %d samples" % len(results))

    joint     = build_joint_distribution()
    expansion = cp.generate_expansion(args.degree, joint)

    # Rebuild node matrix
    X_nodes = np.array(
        [[r['params'][n] for n in PARAM_NAMES] for r in results]
    ).T  # (n_params, n_samples)

    qoi_names = ['max_smises', 'max_disp']   # skip max_sdeg (all zero)
    thresholds_smises = [
        (1200.0, r'$\sigma_{ult}$=1200 MPa', COLORS[1]),
        (600.0,  r'$\sigma_{FPF}$=600 MPa',  COLORS[2]),
        (300.0,  r'$\sigma_{allow}$=300 MPa', COLORS[3]),
    ]
    thresholds_disp = [
        (12.0,  r'$u_{allow}$=12 mm',  COLORS[1]),
        (15.0,  r'$u_{limit}$=15 mm',  COLORS[2]),
    ]

    summary = {}

    for qoi_name in qoi_names:
        print("\n======= %s =======" % qoi_name)
        Y = np.array([r['qoi'].get(qoi_name, 0.0) for r in results])

        if Y.std() < 1e-10:
            print("  [SKIP] All values identical — skipping %s" % qoi_name)
            continue

        # Fit PCE surrogate
        approx = cp.fit_regression(expansion, X_nodes, Y)

        # Compute mean/std via MC on surrogate (avoids chaospy/numpy2 moment bug)
        mc_large = joint.sample(500_000, seed=0)
        y_mc_large = cp.call(approx, mc_large)
        mean = float(y_mc_large.mean())
        std  = float(y_mc_large.std())

        # Sobol indices via MC variance decomposition (Saltelli estimator)
        s1_dict = _sobol_mc(approx, joint, seed=1)

        print("  Mean=%.4g  Std=%.4g  CV=%.2f%%" % (mean, std, 100*std/abs(mean)))

        # 1. LOO cross-validation
        if not args.skip_loo:
            print("  Running LOO-CV (%d-fold)..." % len(Y))
            y_loo, r2_loo, rmse_loo = loo_cv(expansion, X_nodes, Y, joint)
            print("  LOO R²=%.6f  RMSE=%.4g" % (r2_loo, rmse_loo))
            plot_loo(Y, y_loo, qoi_name, r2_loo, rmse_loo, args.outdir)
        else:
            r2_loo, rmse_loo = None, None

        # 2. Second-order Sobol (heatmap using existing s1; s2 from cp.Sens_m2 if available)
        s2_mat = second_order_sobol(approx, joint)
        plot_sobol2_heatmap(s2_mat, s1_dict, qoi_name, args.outdir)

        # 3. Bootstrap Sobol CIs (MC-based bootstrap)
        print("  Running bootstrap CI (n=%d)..." % args.n_boot)
        s1_ci = _bootstrap_sobol_mc(approx, joint, n_boot=args.n_boot)
        plot_sobol_ci(s1_ci, s1_dict, qoi_name, args.outdir)

        # 4. Exceedance curve
        thresh_list = thresholds_smises if 'smises' in qoi_name else thresholds_disp
        plot_exceedance(approx, joint, qoi_name, args.outdir,
                        extra_thresholds=thresh_list)

        # 5. Scatter plots
        plot_scatter_matrix(X_nodes, Y, qoi_name, args.outdir)

        # 6. Tornado diagram
        plot_tornado(approx, joint, qoi_name, args.outdir)

        # Save individual summary
        summary[qoi_name] = {
            'mean': mean, 'std': std, 'cv_pct': 100*std/abs(mean),
            'sobol_first': s1_dict,
            'r2_loo': r2_loo, 'rmse_loo': rmse_loo,
        }

        # Print failure probabilities for smises
        if 'smises' in qoi_name:
            mc_samples = joint.sample(200_000)
            y_mc = cp.call(approx, mc_samples)
            print("\n  Failure probabilities:")
            for thresh, label, _ in thresholds_smises:
                pf = float((y_mc > thresh).mean())
                print("    P(σ > %6.0f MPa) = %.3e  [%s]" % (thresh, pf, label))

    # Save summary JSON
    out_json = os.path.join(args.outdir, 'enhanced_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nAll done. Results -> %s" % args.outdir)


if __name__ == '__main__':
    main()
