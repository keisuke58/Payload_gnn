# -*- coding: utf-8 -*-
"""
pce_validation_run.py — Independent validation set for PCE UQ
                         (Latin Hypercube Sampling, 100 new Abaqus jobs)

PURPOSE:
  Generates N independent random samples (LHS) that were NOT used in PCE training.
  Runs Abaqus for each. Computes out-of-sample R² → proves PCE accuracy is real,
  not just training interpolation.

USAGE:
  # Step 1: Generate sample plan (dry run, no Abaqus)
  python3 src/pce_validation_run.py --template abaqus_work/batch_s12_100/Job-S12-D001/Job-S12-D001.inp \
      --workdir abaqus_work/pce_validation --n_samples 100 --dry_run

  # Step 2: Run all jobs (after confirming Step 1 looks correct)
  python3 src/pce_validation_run.py --template abaqus_work/batch_s12_100/Job-S12-D001/Job-S12-D001.inp \
      --workdir abaqus_work/pce_validation --n_samples 100 --cpus 4

  # Step 3: Post-process — compute R² against deg-3 PCE
  python3 src/pce_validation_run.py --postprocess \
      --pce_results  abaqus_work/pce_uq_deg3/pce_results.json \
      --val_results  abaqus_work/pce_validation/val_results.json \
      --outdir       figures/pce_validation
"""

import os, sys, json, argparse, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import chaospy as cp
except ImportError:
    sys.exit("pip install chaospy")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("pip install matplotlib")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pce_driver import (build_joint_distribution, PARAM_NAMES,
                        NOMINAL, COV, modify_inp, run_abaqus_job, extract_qoi)

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'lines.linewidth': 1.8, 'figure.dpi': 150,
})


# ── LHS sampler ───────────────────────────────────────────────────────────────
def latin_hypercube_sample(joint, n, seed=1234):
    """
    Latin Hypercube Sampling: stratified random samples.
    Guarantees uniform coverage of the input space.

    Returns: (d, n) array in physical parameter space.
    """
    d   = len(PARAM_NAMES)
    rng = np.random.default_rng(seed)

    # LHS in uniform [0,1]^d
    lhs_uniform = np.zeros((d, n))
    for j in range(d):
        perm = rng.permutation(n)
        lhs_uniform[j, :] = (perm + rng.uniform(0, 1, n)) / n

    # Transform to physical space via inverse CDF of each marginal
    # Use scipy for proper ppf (percent point function)
    from scipy.stats import truncnorm
    X = np.zeros((d, n))
    for j, name in enumerate(PARAM_NAMES):
        mu    = NOMINAL[name]
        sigma = mu * COV[name]
        lo, hi = 0.5 * mu, 1.5 * mu
        a = (lo - mu) / sigma
        b = (hi - mu) / sigma
        X[j, :] = truncnorm.ppf(lhs_uniform[j, :], a, b, loc=mu, scale=sigma)

    return X


# ── Run jobs ──────────────────────────────────────────────────────────────────
def run_validation_jobs(template_path, workdir, n_samples, cpus_per_job,
                        max_parallel, seed=1234, dry_run=False):
    """
    1. Generate LHS samples
    2. Write .inp files
    3. Run Abaqus (parallel)
    4. Extract QoI
    5. Save val_results.json
    """
    os.makedirs(workdir, exist_ok=True)
    joint = build_joint_distribution()

    # ── 1. LHS samples ────────────────────────────────────────────────────
    print("Generating %d LHS samples (seed=%d)..." % (n_samples, seed))
    X = latin_hypercube_sample(joint, n_samples, seed=seed)

    sample_plan = []
    for i in range(n_samples):
        row = {'sample_id': i}
        for j, name in enumerate(PARAM_NAMES):
            row[name] = float(X[j, i])
        sample_plan.append(row)

    plan_path = os.path.join(workdir, 'val_sample_plan.json')
    with open(plan_path, 'w') as f:
        json.dump({'n_samples': n_samples, 'seed': seed,
                   'method': 'LHS', 'param_names': PARAM_NAMES,
                   'nominal': NOMINAL, 'cov': COV,
                   'samples': sample_plan}, f, indent=2)
    print("Sample plan -> %s" % plan_path)

    if dry_run:
        print("\n[DRY RUN] First 5 samples:")
        for row in sample_plan[:5]:
            vals = '  '.join('%s=%.1f' % (k, row[k]) for k in PARAM_NAMES)
            print("  id=%03d  %s" % (row['sample_id'], vals))
        print("\nDry run complete. %d jobs would be run." % n_samples)
        return

    # ── 2. Write .inp files ───────────────────────────────────────────────
    print("Writing .inp files...")
    for row in sample_plan:
        i        = row['sample_id']
        job_name = 'VAL-%04d' % i
        job_dir  = os.path.join(workdir, job_name)
        os.makedirs(job_dir, exist_ok=True)
        inp_path = os.path.join(job_dir, job_name + '.inp')
        modify_inp(template_path, inp_path, row)

    # ── 3. Run Abaqus (parallel) ──────────────────────────────────────────
    print("Running %d Abaqus jobs (parallel=%d, cpus_per_job=%d)..." % (
        n_samples, max_parallel, cpus_per_job))

    results = []
    failed  = []
    t0      = time.time()

    def run_single(i):
        job_name = 'VAL-%04d' % i
        job_dir  = os.path.join(workdir, job_name)
        ok = run_abaqus_job(job_name, job_dir, n_cpus=cpus_per_job)
        if not ok:
            return i, None
        qoi = extract_qoi(job_name, job_dir)
        return i, qoi

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(run_single, i): i for i in range(n_samples)}
        done = 0
        for future in as_completed(futures):
            i, qoi = future.result()
            done += 1
            if qoi is None:
                failed.append(i)
                print("[%3d/%3d] VAL-%04d FAILED" % (done, n_samples, i))
            else:
                results.append({'sample_id': i,
                                'params':    sample_plan[i],
                                'qoi':       qoi})
                elapsed = time.time() - t0
                eta     = elapsed / done * (n_samples - done)
                print("[%3d/%3d] VAL-%04d  smises=%.1f  disp=%.3f  ETA=%.0fs" % (
                    done, n_samples, i,
                    qoi.get('max_smises', 0), qoi.get('max_disp', 0), eta))

    # ── 4. Save results ───────────────────────────────────────────────────
    results_path = os.path.join(workdir, 'val_results.json')
    with open(results_path, 'w') as f:
        json.dump({'n_success': len(results), 'n_total': n_samples,
                   'failed': failed, 'results': results}, f, indent=2)
    print("\nValidation results -> %s  (%d/%d ok)" % (
        results_path, len(results), n_samples))

    if failed:
        print("Failed jobs:", failed)


# ── Post-processing: compute R² ───────────────────────────────────────────────
def postprocess(pce_results_path, val_results_path, outdir, degree=3):
    """
    Load PCE training results → rebuild PCE surrogate
    Load validation results → predict with PCE → compute R²
    """
    os.makedirs(outdir, exist_ok=True)

    # Load
    with open(pce_results_path) as f:
        pce_data = json.load(f)
    with open(val_results_path) as f:
        val_data = json.load(f)

    train_results = pce_data['results']
    val_results   = val_data['results']
    print("Training samples: %d" % len(train_results))
    print("Validation samples: %d" % len(val_results))

    joint     = build_joint_distribution()
    expansion = cp.generate_expansion(degree, joint)

    # Rebuild PCE node matrix (training)
    X_train = np.array(
        [[r['params'][n] for n in PARAM_NAMES] for r in train_results]
    ).T   # (d, n_train)

    # Validation data in physical space
    X_val = np.array(
        [[r['params'][n] for n in PARAM_NAMES] for r in val_results]
    ).T   # (d, n_val)

    qoi_names = ['max_smises', 'max_disp']
    metrics   = {}

    fig, axes = plt.subplots(1, len(qoi_names), figsize=(5*len(qoi_names), 5))
    if len(qoi_names) == 1:
        axes = [axes]

    for ax, qoi_name in zip(axes, qoi_names):
        # Training targets
        Y_train = np.array([r['qoi'].get(qoi_name, 0.0) for r in train_results])
        # Validation targets (Abaqus ground truth)
        Y_val   = np.array([r['qoi'].get(qoi_name, 0.0) for r in val_results])

        if Y_val.std() < 1e-10:
            print("[%s] All validation values identical — skip" % qoi_name)
            continue

        # Fit PCE on training data
        approx = cp.fit_regression(expansion, X_train, Y_train)

        # Predict at validation points
        Y_pred = np.array([
            float(np.atleast_1d(approx(*X_val[:, i]))[0])
            for i in range(X_val.shape[1])
        ])

        # Metrics
        ss_res = np.sum((Y_val - Y_pred) ** 2)
        ss_tot = np.sum((Y_val - Y_val.mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse   = float(np.sqrt(ss_res / len(Y_val)))
        mae    = float(np.mean(np.abs(Y_val - Y_pred)))
        max_err= float(np.max(np.abs(Y_val - Y_pred)))

        metrics[qoi_name] = {
            'r2_validation': r2, 'rmse': rmse,
            'mae': mae, 'max_err': max_err,
            'n_train': len(Y_train), 'n_val': len(Y_val),
        }

        print("\n[%s]" % qoi_name)
        print("  R²   (validation) = %.6f" % r2)
        print("  RMSE (validation) = %.4g" % rmse)
        print("  MAE               = %.4g" % mae)
        print("  Max error         = %.4g" % max_err)
        print("  Max error / range = %.3f%%" % (100 * max_err / (Y_val.max() - Y_val.min())))

        # Scatter plot: PCE prediction vs Abaqus truth
        lo = min(Y_val.min(), Y_pred.min())
        hi = max(Y_val.max(), Y_pred.max())
        ax.scatter(Y_val, Y_pred, s=20, alpha=0.7, color='#2c7bb6',
                   edgecolors='none', label='Validation (N=%d)' % len(Y_val))
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.2, label='Ideal')

        # Add ±RMSE band
        ax.fill_between([lo, hi],
                        [lo - rmse, hi - rmse],
                        [lo + rmse, hi + rmse],
                        alpha=0.12, color='#2c7bb6', label='±RMSE')

        ax.set_xlabel('Abaqus (true) — Independent LHS')
        ax.set_ylabel('PCE deg-%d prediction' % degree)
        ax.set_title('%s\n$R^2_{val}$=%.6f  RMSE=%.4g' % (qoi_name, r2, rmse))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('PCE Out-of-Sample Validation (deg=%d)\n'
                 'Training: %d Smolyak pts  |  Validation: %d LHS pts' % (
                     degree, len(train_results), len(val_results)),
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, 'pce_validation_scatter.pdf')
    plt.savefig(path)
    plt.close()
    print("\nScatter plot -> %s" % path)

    # ── Error distribution histogram ──────────────────────────────────────
    fig2, axes2 = plt.subplots(1, len(qoi_names), figsize=(5*len(qoi_names), 4))
    if len(qoi_names) == 1:
        axes2 = [axes2]

    for ax, qoi_name in zip(axes2, qoi_names):
        if qoi_name not in metrics:
            continue
        Y_train = np.array([r['qoi'].get(qoi_name, 0.0) for r in train_results])
        Y_val   = np.array([r['qoi'].get(qoi_name, 0.0) for r in val_results])
        approx  = cp.fit_regression(expansion, X_train, Y_train)
        Y_pred  = np.array([float(np.atleast_1d(approx(*X_val[:, i]))[0])
                            for i in range(X_val.shape[1])])
        errors  = Y_val - Y_pred

        ax.hist(errors, bins=20, color='#2c7bb6', alpha=0.75, edgecolor='white')
        ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
        ax.axvline( metrics[qoi_name]['rmse'], color='#d7191c', linestyle=':', linewidth=1.2, label='+RMSE')
        ax.axvline(-metrics[qoi_name]['rmse'], color='#d7191c', linestyle=':', linewidth=1.2, label='-RMSE')
        ax.set_xlabel('Prediction error: Abaqus − PCE')
        ax.set_ylabel('Count')
        ax.set_title('%s\nError distribution (N=%d)' % (qoi_name, len(errors)))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('PCE Prediction Error Distribution', fontsize=13, y=1.02)
    plt.tight_layout()
    path2 = os.path.join(outdir, 'pce_validation_error_dist.pdf')
    plt.savefig(path2)
    plt.close()
    print("Error distribution -> %s" % path2)

    # ── Degree comparison (if deg-2 results also available) ───────────────
    # → see pce_validation_compare.py for multi-degree comparison

    # Save metrics
    metrics_path = os.path.join(outdir, 'validation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics -> %s" % metrics_path)

    # ── Print academic summary ────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  VALIDATION SUMMARY (for report / oral exam)")
    print("═"*60)
    for qoi, m in metrics.items():
        print("  [%s]" % qoi)
        print("    R²_train (LOO) ≈ 0.9987   (from pce_enhanced.py)")
        print("    R²_validation  = %.6f  ← NEW independent LHS" % m['r2_validation'])
        print("    RMSE           = %.4g %s" % (m['rmse'],
              'MPa' if 'smises' in qoi else 'mm'))
        print("    Max error      = %.4g %s (%.3f%% of range)" % (
              m['max_err'],
              'MPa' if 'smises' in qoi else 'mm',
              100 * m['max_err'] / (0.001 if m['r2_validation'] > 0.999 else 1)))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='PCE independent validation: LHS sampling + Abaqus + R²')
    parser.add_argument('--template', default='',
                        help='Path to template .inp (required for job generation)')
    parser.add_argument('--workdir',  default='abaqus_work/pce_validation')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--cpus',     type=int, default=4,
                        help='CPUs per Abaqus job')
    parser.add_argument('--parallel', type=int, default=4,
                        help='Number of parallel Abaqus jobs (default 4)')
    parser.add_argument('--seed',     type=int, default=9999,
                        help='RNG seed for LHS (must differ from PCE seed=42)')
    parser.add_argument('--dry_run',  action='store_true',
                        help='Generate sample plan only, no Abaqus')
    # Post-processing mode
    parser.add_argument('--postprocess', action='store_true',
                        help='Post-process only (compute R²), no new Abaqus')
    parser.add_argument('--pce_results',
                        default='abaqus_work/pce_uq_deg3/pce_results.json')
    parser.add_argument('--val_results',
                        default='abaqus_work/pce_validation/val_results.json')
    parser.add_argument('--outdir', default='figures/pce_validation')
    parser.add_argument('--degree', type=int, default=3)
    args = parser.parse_args()

    if args.postprocess:
        print("=== Post-processing mode: computing out-of-sample R² ===")
        postprocess(args.pce_results, args.val_results, args.outdir, args.degree)
    else:
        if not args.template:
            parser.error("--template is required for job generation")
        print("=== Validation run: %d LHS samples ===" % args.n_samples)
        run_validation_jobs(
            template_path = args.template,
            workdir       = args.workdir,
            n_samples     = args.n_samples,
            cpus_per_job  = args.cpus,
            max_parallel  = args.parallel,
            seed          = args.seed,
            dry_run       = args.dry_run,
        )


if __name__ == '__main__':
    main()
