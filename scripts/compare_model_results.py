#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare best validation metrics across model runs.

Usage:
  python scripts/compare_model_results.py runs/
  python scripts/compare_model_results.py runs/ --sort opt_f1 --csv out.csv
"""

import argparse
import csv
import glob
import os

CLASS_NAMES = ['healthy', 'debonding', 'fod', 'impact', 'delamination']
SHORT = ['hea', 'deb', 'fod', 'imp', 'del']


def load_run(path):
    import torch
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    val = ckpt.get('val_metrics', {})
    run_name = os.path.basename(os.path.dirname(path))
    a = ckpt.get('args', {})
    task = a.get('task', 'node_cls')
    arch = a.get('arch', run_name.split('_')[0])
    lambda_stress = a.get('physics_lambda_stress', 0.0)
    lambda_eq = a.get('physics_lambda_eq', 0.0)

    r = {
        'run': run_name,
        'arch': arch,
        'task': task,
        'epoch': ckpt.get('epoch', -1),
        'val_f1': val.get('f1', 0.0),
        'val_opt_f1': val.get('opt_f1', val.get('f1', 0.0)),
        'val_auc': val.get('auc', 0.0),
        'val_loss': val.get('loss', 1e9),
        'lambda_stress': lambda_stress,
        'lambda_eq': lambda_eq,
    }
    # Per-class F1
    for cls in CLASS_NAMES:
        r['f1_%s' % cls] = val.get('f1_%s' % cls, 0.0)
    # Multitask regression RMSE (mm)
    r['reg_rmse_mm'] = val.get('reg_rmse_mm', float('nan'))
    return r


def fmt(v):
    if v != v:  # nan
        return '   —  '
    return '%6.3f' % v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs_dir', type=str, default='runs', nargs='?')
    parser.add_argument('--sort', type=str, default='opt_f1',
                        choices=['opt_f1', 'f1', 'auc', 'loss'])
    parser.add_argument('--csv', type=str, default='',
                        help='Also write CSV to this path')
    parser.add_argument('--pattern', type=str, default='*',
                        help='Glob pattern for run subdirs (e.g. pinn_*)')
    args = parser.parse_args()

    runs_dir = os.path.abspath(args.runs_dir)
    results = []
    for path in sorted(glob.glob(os.path.join(runs_dir, args.pattern, 'best_model.pt'))):
        try:
            results.append(load_run(path))
        except Exception as e:
            print('  Skip %s: %s' % (os.path.dirname(path), e))

    if not results:
        print('No best_model.pt found in %s/%s' % (runs_dir, args.pattern))
        return

    rev = (args.sort != 'loss')
    sort_key = 'val_opt_f1' if args.sort == 'opt_f1' else ('val_' + args.sort)
    results.sort(key=lambda r: r.get(sort_key, 0.0), reverse=rev)

    # Header
    W = 100
    print('\n' + '=' * W)
    print('  Model Comparison (sorted by %s)' % args.sort)
    print('=' * W)
    hdr = '%-32s %-10s %-10s %5s %7s %7s %7s | %s'
    print(hdr % ('Run', 'Arch', 'Task', 'Ep',
                 'OptF1', 'F1', 'AUC',
                 ' '.join('%6s' % s for s in SHORT) + '  RegRMSE'))
    print('-' * W)
    for r in results:
        pinn = ''
        if r['lambda_stress'] > 0 or r['lambda_eq'] > 0:
            parts = []
            if r['lambda_stress'] > 0:
                parts.append('Ls%.2f' % r['lambda_stress'])
            if r['lambda_eq'] > 0:
                parts.append('Leq%.2f' % r['lambda_eq'])
            pinn = '+' + '+'.join(parts)
        task_str = ('MT' if r['task'] == 'multitask' else 'NC') + pinn
        per_cls = ' '.join(fmt(r.get('f1_%s' % c, 0.0)) for c in CLASS_NAMES)
        reg = ('  %6.1f' % r['reg_rmse_mm']) if r['reg_rmse_mm'] == r['reg_rmse_mm'] else '     —'
        print('%-32s %-10s %-10s %5d %7.4f %7.4f %7.4f | %s%s' % (
            r['run'][:32], r['arch'][:10], task_str[:10],
            r['epoch'], r['val_opt_f1'], r['val_f1'], r['val_auc'],
            per_cls, reg))
    print('=' * W)
    best = results[0]
    print('Best: %s  OptF1=%.4f  AUC=%.4f' % (
        best['run'], best['val_opt_f1'], best['val_auc']))

    if args.csv:
        fields = (['run', 'arch', 'task', 'epoch', 'val_opt_f1', 'val_f1',
                   'val_auc', 'val_loss', 'lambda_stress', 'lambda_eq'] +
                  ['f1_%s' % c for c in CLASS_NAMES] + ['reg_rmse_mm'])
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(results)
        print('Saved CSV: %s' % args.csv)


if __name__ == '__main__':
    main()
