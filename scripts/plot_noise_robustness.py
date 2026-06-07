#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_noise_robustness.py — Plot OptF1 vs. noise level for Composites B Figure.

Usage:
  python scripts/plot_noise_robustness.py results/noise_robustness.json
  python scripts/plot_noise_robustness.py results/noise_robustness.json --out figures/noise_robustness.pdf
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LUH colours
BLUE  = '#00509B'
RED   = '#CC0000'
GRAY  = '#808080'
LGREEN= '#3CB371'
ORANGE= '#E07B39'

# Style map: label substring → (color, linestyle, marker)
STYLE_MAP = {
    'lgsta':    (BLUE,   '-',  'o'),
    'sage':     (LGREEN, '--', 's'),
    'gcn':      (ORANGE, '-.',  '^'),
    'gin':      (GRAY,   ':',  'D'),
    'gatv2':    (RED,    '--', 'v'),
    'meshgnn':  ('#9B59B6', '-.', 'P'),
    'gat':      ('#F39C12', ':',  'X'),
    'transolver': (GRAY, ':', 'x'),
}

PINN_LS = '-'
NOPINN_LS = '--'


def get_style(label):
    label_lo = label.lower()
    for key, (c, ls, mk) in STYLE_MAP.items():
        if key in label_lo:
            ls2 = PINN_LS if 'pinn' in label_lo.replace('no-pinn', '') else NOPINN_LS
            return c, ls2, mk
    return GRAY, '-', 'o'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', help='Path to noise_robustness.json')
    parser.add_argument('--out', default=None, help='Output figure path (PDF/PNG)')
    parser.add_argument('--highlight', nargs='+', default=['lgsta'],
                        help='Model names to highlight (bold)')
    args = parser.parse_args()

    with open(args.json_path) as f:
        results = json.load(f)

    if not results:
        print('Empty results file.')
        return

    # Filter out Transolver (collapsed baseline — not interesting for noise)
    # and sort by σ=0 OptF1 (descending)
    def clean_zero_f1(r):
        c = r['noise_curve']
        return c[0]['opt_f1_mean'] if c else 0.0

    results = [r for r in results if 'transolver' not in r['label'].lower()
               or clean_zero_f1(r) > 0.05]
    results.sort(key=clean_zero_f1, reverse=True)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    x_pct = [e['sigma_pct'] for e in results[0]['noise_curve']]
    x_log = np.log10([max(p, 0.001) for p in x_pct])   # for log-x axis

    # Panel A: OptF1 vs σ%
    ax = axes[0]
    for r in results:
        curve = r['noise_curve']
        y     = [e['opt_f1_mean'] for e in curve]
        yerr  = [e['opt_f1_std']  for e in curve]
        c, ls, mk = get_style(r['label'])
        lw = 2.0 if any(h in r['label'].lower() for h in args.highlight) else 1.2
        ax.errorbar(x_pct, y, yerr=yerr, label=r['label'],
                    color=c, linestyle=ls, marker=mk, markersize=5,
                    linewidth=lw, capsize=3, alpha=0.85)

    ax.set_xlabel('Gaussian feature noise σ (% of feature std)', fontsize=11)
    ax.set_ylabel('OptF1 ↑', fontsize=11)
    ax.set_title('(a) Noise robustness — OptF1 vs. σ', fontsize=12)
    ax.set_xscale('symlog', linthresh=0.5)
    ax.set_xticks([0, 1, 3, 10, 30])
    ax.set_xticklabels(['0', '1%', '3%', '10%', '30%'])
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    ax.legend(fontsize=8, ncol=2, loc='lower left')

    # Panel B: relative F1 drop = (F1(σ) - F1(0)) / F1(0)
    ax = axes[1]
    for r in results:
        curve = r['noise_curve']
        f0 = curve[0]['opt_f1_mean']
        if f0 < 0.05:
            continue
        y_rel = [(e['opt_f1_mean'] - f0) / f0 * 100 for e in curve]
        c, ls, mk = get_style(r['label'])
        lw = 2.0 if any(h in r['label'].lower() for h in args.highlight) else 1.2
        ax.plot(x_pct, y_rel, label=r['label'],
                color=c, linestyle=ls, marker=mk, markersize=5,
                linewidth=lw, alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Gaussian feature noise σ (% of feature std)', fontsize=11)
    ax.set_ylabel('Relative F1 change (%)', fontsize=11)
    ax.set_title('(b) Relative degradation vs. σ', fontsize=12)
    ax.set_xscale('symlog', linthresh=0.5)
    ax.set_xticks([0, 1, 3, 10, 30])
    ax.set_xticklabels(['0', '1%', '3%', '10%', '30%'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='lower left')

    plt.tight_layout()

    out = args.out or os.path.join(PROJECT_ROOT, 'figures', 'slides', 'noise_robustness.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved: {out}')

    # ── Print summary table ────────────────────────────────────────────────────
    print('\n' + '='*80)
    print(f'{"Model":<35} {"σ=0%":>8} {"σ=1%":>8} {"σ=3%":>8} {"σ=10%":>9} {"σ=30%":>9}')
    print('-'*80)
    for r in results:
        curve = r['noise_curve']
        vals  = [f'{e["opt_f1_mean"]:.4f}' for e in curve]
        print(f'{r["label"]:<35} ' + '  '.join(f'{v:>7}' for v in vals))


if __name__ == '__main__':
    main()
