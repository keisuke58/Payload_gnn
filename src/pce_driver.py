# -*- coding: utf-8 -*-
"""
pce_driver.py  -- PCE-based UQ for H3 Fairing Static Analysis

Uncertain inputs (5 variables):
  xi1: E1_CFRP     [MPa]  N(160000, CoV=5%)   -> CFRP_T1000G *Elastic LAMINA col1
  xi2: G12_CFRP    [MPa]  N(5000,   CoV=10%)  -> CFRP_T1000G *Elastic LAMINA col4-6
  xi3: Kn_adhesive [MPa]  N(100000, CoV=15%)  -> Mat-Adhesive *Elastic TRACTION col1
  xi4: GIc         [N/mm] N(0.3,    CoV=20%)  -> Mat-Adhesive *Damage Evolution col1
  xi5: tn          [MPa]  N(50,     CoV=15%)  -> Mat-Adhesive *Damage Initiation col1

QoI (extracted from .odb via extract_pce_qoi.py):
  - max_smises : max von Mises stress in CFRP face sheet [MPa]
  - max_sdeg   : max damage variable in adhesive (0=intact, 1=fully debonded)
  - max_disp   : max displacement magnitude [mm]

Usage (on server):
    cd /home/nishioka/Payload2026
    python3 src/pce_driver.py \\
        --template abaqus_work/batch_s12_100/Job-S12-D001/Job-S12-D001.inp \\
        --workdir  abaqus_work/pce_uq \\
        --degree 3 --rule gaussian

    # Dry run (generate sample plan only, no Abaqus):
    python3 src/pce_driver.py --template ... --dry_run
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import re
import numpy as np

try:
    import chaospy as cp
except ImportError:
    sys.exit("ERROR: chaospy not found. Run: pip install chaospy")


# ---------------------------------------------------------------------------
# Nominal values and CoV
# (source: material_properties.py + manufacturing_variability.py)
# ---------------------------------------------------------------------------
NOMINAL = {
    'E1':   160000.0,   # MPa  CFRP fiber-direction modulus
    'G12':    5000.0,   # MPa  CFRP in-plane shear modulus
    'Kn':   100000.0,   # MPa/mm  adhesive normal traction stiffness
    'GIc':       0.3,   # N/mm  mode-I fracture energy
    'tn':       50.0,   # MPa  mode-I traction strength
}
COV = {
    'E1':  0.05,   # 5%  (fiber-dominated, low scatter)
    'G12': 0.10,   # 10% (matrix/shear)
    'Kn':  0.15,   # 15% (CZM stiffness)
    'GIc': 0.20,   # 20% (fracture energy, high scatter)
    'tn':  0.15,   # 15% (CZM strength)
}
PARAM_NAMES = ['E1', 'G12', 'Kn', 'GIc', 'tn']


# ---------------------------------------------------------------------------
# 1. Build chaospy joint distribution
# ---------------------------------------------------------------------------
def build_joint_distribution():
    """Truncated Normal for each parameter (clipped to [0.5*mu, 1.5*mu])."""
    dists = []
    for name in PARAM_NAMES:
        mu    = NOMINAL[name]
        sigma = mu * COV[name]
        lo, hi = 0.5 * mu, 1.5 * mu
        dists.append(cp.TruncNormal(lower=lo, upper=hi, mu=mu, sigma=sigma))
    return cp.J(*dists)


# ---------------------------------------------------------------------------
# 2. Modify .inp file with new material parameters
# ---------------------------------------------------------------------------
def modify_inp(template_path, out_path, params):
    """
    Read template .inp, substitute material cards, write to out_path.

    Parameters
    ----------
    params : dict  keys = E1, G12, Kn, GIc, tn
    """
    E1  = params['E1']
    G12 = params['G12']
    Kn  = params['Kn']
    GIc = params['GIc']
    tn  = params['tn']

    # Derived values (maintain nominal ratios)
    E2   = 10000.0 * (E1  / NOMINAL['E1'])       # transverse modulus scales with E1
    G13  = G12                                    # in-plane == out-of-plane approx
    G23  = 3000.0  * (G12 / NOMINAL['G12'])       # scale with G12
    Ks   = Kn / 2.0                               # shear stiffness = Kn/2 (nominal)
    ts   = tn * 0.8                               # ts/tn = 40/50 nominal
    GIIc = GIc * (1.0 / 0.3)                     # GIIc/GIc = 1.0/0.3 nominal

    with open(template_path, 'r') as f:
        content = f.read()

    # -- CFRP_T1000G LAMINA: E1, E2, nu12=0.3, G12, G13, G23 --
    lamina_new = '*Elastic, type=LAMINA\n%.1f,%.1f,   0.3, %.1f, %.1f, %.1f' % (
        E1, E2, G12, G13, G23)
    content = re.sub(
        r'\*Elastic,\s*type=LAMINA\r?\n[\d.,\s]+',
        lamina_new + '\n',
        content
    )

    # -- Mat-Adhesive TRACTION stiffness (first occurrence = undamaged) --
    traction_new = '*Elastic, type=TRACTION\n%.1f,%.1f,%.1f' % (Kn, Ks, Ks)
    content = re.sub(
        r'\*Elastic,\s*type=TRACTION\r?\n100000\.,50000\.,50000\.',
        traction_new,
        content,
        count=1
    )

    # -- Mat-Adhesive MAXS damage initiation: tn, ts, ts --
    maxs_new = '*Damage Initiation, criterion=MAXS\n%.1f,%.1f,%.1f' % (tn, ts, ts)
    content = re.sub(
        r'\*Damage Initiation,\s*criterion=MAXS\r?\n50\.,40\.,40\.',
        maxs_new,
        content,
        count=1
    )

    # -- Mat-Adhesive BK damage evolution: GIc, GIIc, GIIc --
    bk_new = ('*Damage Evolution, type=ENERGY, mixed mode behavior=BK, power=2.284\n'
              ' %.4f,%.4f,%.4f') % (GIc, GIIc, GIIc)
    content = re.sub(
        r'\*Damage Evolution,\s*type=ENERGY,\s*mixed mode behavior=BK,\s*power=2\.284\r?\n\s*0\.3,1\.,1\.',
        bk_new,
        content,
        count=1
    )

    with open(out_path, 'w') as f:
        f.write(content)


# ---------------------------------------------------------------------------
# 3. Run Abaqus batch job
# ---------------------------------------------------------------------------
def run_abaqus_job(job_name, work_dir, n_cpus=4):
    """Run Abaqus interactively from work_dir. Returns True if succeeded."""
    cmd = ['abaqus', 'job=%s' % job_name, 'interactive',
           'cpus=%d' % n_cpus]
    result = subprocess.run(cmd, cwd=work_dir,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("  [WARN] Abaqus non-zero exit for %s" % job_name)
    # Check .sta for COMPLETED
    sta_path = os.path.join(work_dir, job_name + '.sta')
    if os.path.exists(sta_path):
        with open(sta_path) as f:
            return 'COMPLETED SUCCESSFULLY' in f.read()
    return result.returncode == 0


# ---------------------------------------------------------------------------
# 4. Extract QoI from .odb
# ---------------------------------------------------------------------------
def extract_qoi(job_name, work_dir):
    """
    Call  abaqus python extract_pce_qoi.py  to write <job>_qoi.json.
    Returns dict {max_smises, max_sdeg, max_disp} or None on failure.
    """
    odb_path = os.path.abspath(os.path.join(work_dir, job_name + '.odb'))
    out_json = os.path.abspath(os.path.join(work_dir, job_name + '_qoi.json'))
    script   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'extract_pce_qoi.py')

    cmd = ['abaqus', 'python', script,
           '--odb', odb_path, '--output', out_json]
    subprocess.run(cmd, cwd=work_dir,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not os.path.exists(out_json):
        print("  [WARN] QoI JSON not found for %s" % job_name)
        return None
    with open(out_json) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 5. Main driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='PCE UQ driver for H3 fairing (Abaqus static)'
    )
    parser.add_argument('--template', required=True,
                        help='Path to template .inp file')
    parser.add_argument('--workdir', default='abaqus_work/pce_uq',
                        help='Output directory for PCE jobs')
    parser.add_argument('--degree', type=int, default=2,
                        help='PCE polynomial degree (default: 2)')
    parser.add_argument('--rule', default='gaussian',
                        choices=['gaussian', 'halton', 'sobol'],
                        help='Quadrature/sampling rule (default: gaussian)')
    parser.add_argument('--sparse', action='store_true', default=True,
                        help='Use sparse (Smolyak) quadrature — default True')
    parser.add_argument('--cpus', type=int, default=1,
                        help='CPUs per Abaqus job (default: 1)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Generate sample plan only, skip Abaqus')
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)

    # ---- 1. Distributions & sample points --------------------------------
    joint = build_joint_distribution()
    print("=== PCE UQ: H3 Fairing Static Analysis ===")
    print("Parameters (%d): %s" % (len(PARAM_NAMES), ', '.join(PARAM_NAMES)))

    if args.rule == 'gaussian':
        nodes, weights = cp.generate_quadrature(
            args.degree, joint, rule='gaussian', sparse=args.sparse)
        n_samples = nodes.shape[1]
        label = 'sparse Gauss' if args.sparse else 'full Gauss'
        print("%s quadrature  degree=%d -> %d points" % (label, args.degree, n_samples))
    else:
        expansion_size = cp.generate_expansion(args.degree, joint).size
        n_samples = max(2 * expansion_size, 50)
        nodes   = joint.sample(n_samples, rule=args.rule)
        weights = None
        print("%s sampling  n=%d" % (args.rule, n_samples))

    # Save sample plan
    sample_plan = [
        dict({'sample_id': i},
             **{name: float(nodes[j, i]) for j, name in enumerate(PARAM_NAMES)})
        for i in range(n_samples)
    ]
    plan_path = os.path.join(args.workdir, 'pce_sample_plan.json')
    with open(plan_path, 'w') as f:
        json.dump({
            'n_samples': n_samples,
            'degree': args.degree,
            'rule': args.rule,
            'param_names': PARAM_NAMES,
            'nominal': NOMINAL,
            'cov': COV,
            'samples': sample_plan,
        }, f, indent=2)
    print("Sample plan -> %s" % plan_path)

    if args.dry_run:
        print("Dry run done.  %d Abaqus jobs would be run." % n_samples)
        # Preview first 3 points
        for row in sample_plan[:3]:
            print("  id=%d  " % row['sample_id'] +
                  '  '.join('%s=%.2f' % (k, row[k]) for k in PARAM_NAMES))
        return

    # ---- 2. Run Abaqus for each sample point ----------------------------
    results = []
    failed  = []

    for i, row in enumerate(sample_plan):
        job_name = 'PCE-S%04d' % i
        job_dir  = os.path.join(args.workdir, job_name)
        os.makedirs(job_dir, exist_ok=True)

        inp_path = os.path.join(job_dir, job_name + '.inp')
        modify_inp(args.template, inp_path, row)

        print("[%d/%d] %s" % (i + 1, n_samples, job_name))
        ok = run_abaqus_job(job_name, job_dir, n_cpus=args.cpus)
        if not ok:
            failed.append(i)
            continue

        qoi = extract_qoi(job_name, job_dir)
        if qoi is None:
            failed.append(i)
            continue

        results.append({'sample_id': i, 'params': row, 'qoi': qoi})
        print("  -> smises=%.1f  sdeg=%.3f  disp=%.3f" % (
            qoi.get('max_smises', 0),
            qoi.get('max_sdeg', 0),
            qoi.get('max_disp', 0)))

    # Save raw results
    results_path = os.path.join(args.workdir, 'pce_results.json')
    with open(results_path, 'w') as f:
        json.dump({'results': results, 'failed': failed,
                   'n_success': len(results), 'n_total': n_samples}, f, indent=2)
    print("Raw results -> %s  (%d/%d ok)" % (
        results_path, len(results), n_samples))

    if len(results) < 0.8 * n_samples:
        print("[ERROR] >20%% jobs failed. Check Abaqus logs in %s" % args.workdir)
        return

    # ---- 3. Fit PCE surrogate -------------------------------------------
    expansion = cp.generate_expansion(args.degree, joint)
    print("\nPCE expansion: %d terms (degree %d, %d vars)" % (
        expansion.size, args.degree, len(PARAM_NAMES)))

    completed_ids = [r['sample_id'] for r in results]
    X = nodes[:, completed_ids]
    # sparse quadrature weights can be negative — use regression for robustness
    # if any weight is negative or quadrature is sparse, fall back to regression
    if weights is not None and (weights < 0).any():
        print("  [INFO] Negative weights detected (sparse grid) -> using regression")
        W = None
    elif weights is not None:
        W = weights[completed_ids]
    else:
        W = None

    qoi_names  = ['max_smises', 'max_sdeg', 'max_disp']
    pce_models = {}
    stats      = {}

    for qoi_name in qoi_names:
        Y = np.array([r['qoi'].get(qoi_name, 0.0) for r in results])
        if W is not None:
            approx = cp.fit_quadrature(expansion, X, W, Y)
        else:
            approx = cp.fit_regression(expansion, X, Y)

        pce_models[qoi_name] = approx
        mean   = float(cp.E(approx, joint))
        std    = float(cp.Std(approx, joint))
        sobol1 = cp.Sens_m(approx, joint)   # first-order Sobol indices
        sobolT = cp.Sens_t(approx, joint)   # total-order Sobol indices

        stats[qoi_name] = {
            'mean': mean,
            'std':  std,
            'cv_pct': 100 * std / abs(mean) if mean != 0 else 0,
            'sobol_first': {n: float(sobol1[j])
                            for j, n in enumerate(PARAM_NAMES)},
            'sobol_total': {n: float(sobolT[j])
                            for j, n in enumerate(PARAM_NAMES)},
        }

        print("\n--- QoI: %s ---" % qoi_name)
        print("  Mean = %.4g   Std = %.4g   CV = %.1f%%" % (
            mean, std, stats[qoi_name]['cv_pct']))
        print("  Sobol 1st order:")
        for n in PARAM_NAMES:
            print("    %-6s  S1=%.3f  ST=%.3f" % (
                n,
                stats[qoi_name]['sobol_first'][n],
                stats[qoi_name]['sobol_total'][n]))

    # ---- 4. Reliability analysis (MC on PCE surrogate) ------------------
    print("\n=== Reliability Analysis (MC on PCE, N=100,000) ===")
    mc_samples = joint.sample(100_000)

    SIGMA_YIELD  = 1200.0   # MPa  (CFRP T1000G tensile / safety factor 2)
    SDEG_THRESH  = 0.5      # 50% damage initiation = partial debond

    pf_stress  = None
    pf_debond  = None

    if 'max_smises' in pce_models:
        s_mc      = cp.call(pce_models['max_smises'], mc_samples)
        pf_stress = float((s_mc > SIGMA_YIELD).mean())
        print("  P(sigma_max > %.0f MPa)  = %.3e" % (SIGMA_YIELD, pf_stress))

    if 'max_sdeg' in pce_models:
        d_mc     = cp.call(pce_models['max_sdeg'], mc_samples)
        pf_debond = float((d_mc > SDEG_THRESH).mean())
        print("  P(SDEG > %.1f)           = %.3e  [debonding]" % (
            SDEG_THRESH, pf_debond))

    stats['reliability'] = {
        'sigma_yield_MPa':   SIGMA_YIELD,
        'sdeg_threshold':    SDEG_THRESH,
        'pf_stress_overload': pf_stress,
        'pf_debonding':      pf_debond,
        'n_mc_samples':      100_000,
    }

    stats_path = os.path.join(args.workdir, 'pce_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print("\nAll statistics -> %s" % stats_path)
    print("Done.")


if __name__ == '__main__':
    main()
