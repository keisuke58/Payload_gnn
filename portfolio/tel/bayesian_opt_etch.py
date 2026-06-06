#!/usr/bin/env python3
"""
Bayesian Optimization for Plasma Etch Recipe
==============================================
Portfolio: Tokyo Electron (TEL) — ML/Optimization

Optimizes plasma etch parameters (RF power, pressure, gas flow, bias)
to minimize etch non-uniformity and maximize etch rate using
Bayesian Optimization with Gaussian Process Regression.

ML/Optimization Concepts:
  - Gaussian Process Regression (GPR) from scratch (RBF kernel)
  - Expected Improvement (EI) acquisition function
  - Multi-objective scalarization (weighted sum)
  - Sequential model-based optimization (SMBO)
  - Latin Hypercube Sampling (LHS) for initial DOE

Physics Model (surrogate for real etch process):
  - Etch rate: Arrhenius-like dependence on RF power, Paschen curve for pressure
  - Uniformity: radial profile depends on gas flow distribution and chamber geometry
  - Ion bombardment energy: proportional to bias voltage

Author: Nishioka
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import os
import time

np.random.seed(42)

# ============================================================
# Etch Process Simulator (Ground Truth)
# ============================================================
# Parameters: [RF_power(W), Pressure(mTorr), CF4_flow(sccm), Bias(V)]
PARAM_BOUNDS = np.array([
    [200, 1500],   # RF power [W]
    [5, 100],      # Pressure [mTorr]
    [20, 200],     # CF4 flow [sccm]
    [50, 500],     # Bias voltage [V]
])
PARAM_NAMES = ['RF Power [W]', 'Pressure [mTorr]', 'CF4 Flow [sccm]', 'Bias [V]']

def etch_simulator(params):
    """Simulate etch process: returns (etch_rate [nm/min], non_uniformity [%]).

    Realistic multi-physics inspired model:
    - Etch rate: plasma density (RF power) x ion energy (bias) / pressure
    - Non-uniformity: gas flow distribution + edge effects
    """
    rf, press, flow, bias = params
    # Normalize to [0,1]
    rf_n = (rf - 200) / 1300
    pr_n = (press - 5) / 95
    fl_n = (flow - 20) / 180
    bi_n = (bias - 50) / 450

    # Etch rate model: peak around mid-pressure (Paschen-like)
    plasma_density = rf_n**0.7 * np.exp(-0.5 * ((pr_n - 0.3)/0.3)**2)
    ion_energy = bi_n**0.5
    chem_etch = fl_n**0.4 * (1 - 0.3 * fl_n)  # saturation at high flow
    etch_rate = 500 * plasma_density * ion_energy * chem_etch + 50

    # Non-uniformity model: sweet spot at moderate conditions
    flow_uni = np.exp(-3 * (fl_n - 0.5)**2)
    press_uni = np.exp(-4 * (pr_n - 0.4)**2)
    bias_uni = 1 - 0.5 * bi_n**2  # high bias = more non-uniform
    non_uniformity = 15 * (1 - 0.7 * flow_uni * press_uni * bias_uni) + \
                     2 * np.random.randn() * 0.3  # measurement noise

    return etch_rate, max(0.5, non_uniformity)

def objective(params):
    """Combined objective: maximize rate, minimize non-uniformity.
    Returns negative (we minimize).
    """
    rate, nu = etch_simulator(params)
    # Weighted scalarization: high rate + low non-uniformity
    return -(rate / 500 - 2.0 * nu / 15)

# ============================================================
# Gaussian Process Regression (from scratch)
# ============================================================
class GaussianProcessRegressor:
    """GPR with RBF kernel and noise, implemented from scratch."""

    def __init__(self, length_scale=1.0, signal_var=1.0, noise_var=0.01):
        self.l = length_scale
        self.sf2 = signal_var
        self.sn2 = noise_var
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None

    def rbf_kernel(self, X1, X2):
        """Squared exponential (RBF) kernel."""
        sq_dist = np.sum(X1**2, axis=1, keepdims=True) + \
                  np.sum(X2**2, axis=1, keepdims=True).T - \
                  2 * X1 @ X2.T
        return self.sf2 * np.exp(-0.5 * sq_dist / self.l**2)

    def fit(self, X, y):
        """Fit GP to training data."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.y_mean = np.mean(y)
        y_centered = y - self.y_mean

        K = self.rbf_kernel(X, X) + self.sn2 * np.eye(len(X))
        # Cholesky for numerical stability
        L = np.linalg.cholesky(K + 1e-8 * np.eye(len(K)))
        self.L = L
        self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))

    def predict(self, X_test):
        """Predict mean and variance at test points."""
        K_star = self.rbf_kernel(X_test, self.X_train)
        mu = K_star @ self.alpha + self.y_mean
        v = np.linalg.solve(self.L, K_star.T)
        K_ss = self.rbf_kernel(X_test, X_test)
        var = np.diag(K_ss) - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-10)
        return mu, var

# ============================================================
# Acquisition Functions
# ============================================================
def expected_improvement(X, gp, y_best, xi=0.01):
    """Expected Improvement acquisition function."""
    mu, var = gp.predict(X)
    sigma = np.sqrt(var)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (y_best - mu - xi) / sigma
        ei = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.where(sigma < 1e-10, 0.0, ei)
    return ei

def ucb(X, gp, beta=2.0):
    """Upper Confidence Bound (for minimization: Lower Confidence Bound)."""
    mu, var = gp.predict(X)
    return -(mu - beta * np.sqrt(var))  # negative for minimization

# ============================================================
# Latin Hypercube Sampling
# ============================================================
def latin_hypercube_sampling(n_samples, bounds):
    """LHS for space-filling initial design."""
    n_dim = len(bounds)
    samples = np.zeros((n_samples, n_dim))
    for d in range(n_dim):
        perm = np.random.permutation(n_samples)
        samples[:, d] = (perm + np.random.rand(n_samples)) / n_samples
        samples[:, d] = bounds[d, 0] + samples[:, d] * (bounds[d, 1] - bounds[d, 0])
    return samples

# ============================================================
# Bayesian Optimization Loop
# ============================================================
def bayesian_optimization(n_init=10, n_iter=40, acq_type='EI'):
    """Run Bayesian Optimization."""
    # Normalize parameters to [0, 1] for GP
    def normalize(X):
        return (X - PARAM_BOUNDS[:, 0]) / (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0])

    def denormalize(X_norm):
        return X_norm * (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]) + PARAM_BOUNDS[:, 0]

    # Initial sampling (LHS)
    X_init = latin_hypercube_sampling(n_init, PARAM_BOUNDS)
    y_init = np.array([objective(x) for x in X_init])
    rates_init = np.array([etch_simulator(x)[0] for x in X_init])
    nu_init = np.array([etch_simulator(x)[1] for x in X_init])

    X_all = X_init.copy()
    y_all = y_init.copy()
    rates_all = list(rates_init)
    nu_all = list(nu_init)

    gp = GaussianProcessRegressor(length_scale=0.3, signal_var=1.0, noise_var=0.05)
    best_history = [np.min(y_all)]

    print(f"  Initial best: obj={np.min(y_all):.4f}")

    for i in range(n_iter):
        # Fit GP
        X_norm = normalize(X_all)
        gp.fit(X_norm, y_all)

        # Optimize acquisition function
        best_val = np.min(y_all)

        # Random restarts for acquisition optimization
        best_acq = -np.inf
        best_x_next = None

        for _ in range(100):
            x0 = np.random.rand(4)
            if acq_type == 'EI':
                acq_val = expected_improvement(x0.reshape(1, -1), gp, best_val)
            else:
                acq_val = ucb(x0.reshape(1, -1), gp)
            if acq_val[0] > best_acq:
                best_acq = acq_val[0]
                best_x_next = x0

        # Evaluate new point
        x_new = denormalize(best_x_next)
        x_new = np.clip(x_new, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
        y_new = objective(x_new)
        rate_new, nu_new = etch_simulator(x_new)

        X_all = np.vstack([X_all, x_new])
        y_all = np.append(y_all, y_new)
        rates_all.append(rate_new)
        nu_all.append(nu_new)
        best_history.append(min(best_history[-1], y_new))

    # Find best result
    best_idx = np.argmin(y_all)
    best_params = X_all[best_idx]
    best_rate, best_nu = etch_simulator(best_params)

    return {
        'X_all': X_all, 'y_all': y_all,
        'rates': np.array(rates_all), 'nus': np.array(nu_all),
        'best_params': best_params, 'best_rate': best_rate,
        'best_nu': best_nu, 'best_obj': y_all[best_idx],
        'best_history': best_history, 'gp': gp,
        'n_init': n_init,
    }

# ============================================================
# Random Search Baseline
# ============================================================
def random_search(n_total=50):
    """Random search baseline for comparison."""
    X = latin_hypercube_sampling(n_total, PARAM_BOUNDS)
    y = np.array([objective(x) for x in X])
    best_history = [y[0]]
    for i in range(1, len(y)):
        best_history.append(min(best_history[-1], y[i]))
    return best_history

# ============================================================
# Visualization
# ============================================================
def plot_results(results, random_baseline, save_path):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Bayesian Optimization for Plasma Etch Recipe (TEL)',
                 fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    n_init = results['n_init']

    # (a) Convergence: BO vs Random
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results['best_history'], 'r-', linewidth=2, label='Bayesian Opt (EI)')
    ax1.plot(random_baseline, 'b--', linewidth=1.5, alpha=0.7, label='Random Search')
    ax1.axvline(n_init, color='gray', linestyle=':', label=f'Init phase ({n_init} LHS)')
    ax1.set_xlabel('Evaluation')
    ax1.set_ylabel('Best Objective')
    ax1.set_title('(a) Convergence: BO vs Random Search')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (b) Etch Rate vs Non-Uniformity (Pareto front)
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(results['rates'], results['nus'],
                     c=np.arange(len(results['rates'])), cmap='viridis',
                     s=50, alpha=0.7, edgecolors='k', linewidth=0.3)
    ax2.scatter(results['best_rate'], results['best_nu'],
               color='red', s=200, marker='*', zorder=5, label='Best')
    plt.colorbar(sc, ax=ax2, label='Evaluation #')
    ax2.set_xlabel('Etch Rate [nm/min]')
    ax2.set_ylabel('Non-Uniformity [%]')
    ax2.set_title('(b) Rate vs Uniformity Trade-off')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (c) Parameter exploration over iterations
    ax3 = fig.add_subplot(gs[0, 2])
    n_eval = len(results['X_all'])
    for i, name in enumerate(PARAM_NAMES):
        vals = (results['X_all'][:, i] - PARAM_BOUNDS[i, 0]) / \
               (PARAM_BOUNDS[i, 1] - PARAM_BOUNDS[i, 0])
        ax3.scatter(np.arange(n_eval), vals, s=15, alpha=0.6, label=name)
    ax3.axvline(n_init, color='gray', linestyle=':')
    ax3.set_xlabel('Evaluation')
    ax3.set_ylabel('Normalized Parameter Value')
    ax3.set_title('(c) Parameter Exploration')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # (d) GP Surrogate (1D slice: RF power)
    ax4 = fig.add_subplot(gs[1, 0])
    best = results['best_params']
    rf_range = np.linspace(0, 1, 200)
    X_test = np.tile(
        (best - PARAM_BOUNDS[:, 0]) / (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]),
        (200, 1))
    X_test[:, 0] = rf_range
    mu, var = results['gp'].predict(X_test)
    sigma = np.sqrt(var)
    rf_phys = rf_range * 1300 + 200
    ax4.plot(rf_phys, mu, 'r-', linewidth=2, label='GP Mean')
    ax4.fill_between(rf_phys, mu - 2*sigma, mu + 2*sigma,
                     alpha=0.2, color='red', label='95% CI')
    # Plot training points projected onto this slice
    X_norm = (results['X_all'] - PARAM_BOUNDS[:, 0]) / \
             (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0])
    ax4.scatter(results['X_all'][:, 0], results['y_all'],
               s=20, color='black', alpha=0.4, zorder=3, label='Observations')
    ax4.set_xlabel('RF Power [W]')
    ax4.set_ylabel('Objective')
    ax4.set_title('(d) GP Surrogate (RF Power slice)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # (e) Best recipe summary
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    bp = results['best_params']
    summary = [
        ['Parameter', 'Optimal Value', 'Range'],
        ['RF Power', f'{bp[0]:.0f} W', '200-1500 W'],
        ['Pressure', f'{bp[1]:.1f} mTorr', '5-100 mTorr'],
        ['CF4 Flow', f'{bp[2]:.1f} sccm', '20-200 sccm'],
        ['Bias Voltage', f'{bp[3]:.0f} V', '50-500 V'],
        ['---', '---', '---'],
        ['Etch Rate', f'{results["best_rate"]:.1f} nm/min', ''],
        ['Non-Uniformity', f'{results["best_nu"]:.2f} %', ''],
        ['Evaluations', f'{len(results["y_all"])}', f'(init: {n_init})'],
    ]
    table = ax5.table(cellText=summary[1:], colLabels=summary[0],
                       cellLoc='center', loc='center',
                       colColours=['#e6f3ff']*3)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
    ax5.set_title('(e) Optimized Etch Recipe', fontsize=12, pad=20)

    # (f) Acquisition function landscape (2D: RF power x Pressure)
    ax6 = fig.add_subplot(gs[1, 2])
    n_grid = 50
    rf_grid = np.linspace(0, 1, n_grid)
    pr_grid = np.linspace(0, 1, n_grid)
    RF, PR = np.meshgrid(rf_grid, pr_grid)
    X_grid = np.zeros((n_grid*n_grid, 4))
    best_norm = (best - PARAM_BOUNDS[:, 0]) / (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0])
    X_grid[:, 0] = RF.ravel()
    X_grid[:, 1] = PR.ravel()
    X_grid[:, 2] = best_norm[2]
    X_grid[:, 3] = best_norm[3]
    ei_vals = expected_improvement(X_grid, results['gp'], np.min(results['y_all']))
    EI_map = ei_vals.reshape(n_grid, n_grid)
    c = ax6.contourf(rf_grid*1300+200, pr_grid*95+5, EI_map, levels=20, cmap='YlOrRd')
    plt.colorbar(c, ax=ax6, label='Expected Improvement')
    ax6.scatter(results['X_all'][:, 0], results['X_all'][:, 1],
               s=20, c='blue', alpha=0.5, edgecolors='white', linewidth=0.5)
    ax6.scatter(bp[0], bp[1], s=200, marker='*', color='lime', edgecolors='black',
               linewidth=1.5, zorder=5, label='Best')
    ax6.set_xlabel('RF Power [W]')
    ax6.set_ylabel('Pressure [mTorr]')
    ax6.set_title('(f) EI Acquisition (RF x Pressure)')
    ax6.legend()

    fig.text(0.5, 0.01,
             'GPR from scratch (RBF kernel) | EI Acquisition | LHS Initial DOE | '
             'Multi-objective Scalarization',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("Bayesian Optimization for Plasma Etch Recipe")
    print("Portfolio: Tokyo Electron (TEL) — ML/Optimization")
    print("=" * 60)

    print("\nRunning Bayesian Optimization (EI)...")
    results = bayesian_optimization(n_init=10, n_iter=40, acq_type='EI')
    print(f"  Best etch rate: {results['best_rate']:.1f} nm/min")
    print(f"  Best non-uniformity: {results['best_nu']:.2f} %")
    print(f"  Best params: RF={results['best_params'][0]:.0f}W, "
          f"P={results['best_params'][1]:.1f}mTorr, "
          f"CF4={results['best_params'][2]:.1f}sccm, "
          f"Bias={results['best_params'][3]:.0f}V")

    print("\nRunning Random Search baseline...")
    random_bl = random_search(n_total=50)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "bayesian_opt_results.png")
    print("\nPlotting results...")
    plot_results(results, random_bl, save_path)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
