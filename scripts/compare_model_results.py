#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare best validation metrics across model runs.

Usage:
  python scripts/compare_model_results.py runs/
  python scripts/compare_model_results.py runs/ --sort f1
"""

import argparse
import json
import os
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs_dir', type=str, default='runs', nargs='?',
                        help='Directory containing run subdirs (e.g. runs/gat_*, runs/gcn_*)')
    parser.add_argument('--sort', type=str, default='f1',
                        choices=['f1', 'auc', 'loss', 'accuracy'])
    args = parser.parse_args()

    runs_dir = os.path.abspath(args.runs_dir)
    if not os.path.isdir(runs_dir):
        print("Directory not found: %s" % runs_dir)
        return

    results = []
    for path in glob.glob(os.path.join(runs_dir, '*', 'best_model.pt')):
        run_dir = os.path.dirname(path)
        try:
            import torch
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            val = ckpt.get('val_metrics', {})
            arch = os.path.basename(run_dir).split('_')[0]
            results.append({
                'run': os.path.basename(run_dir),
                'arch': arch,
                'epoch': ckpt.get('epoch', -1),
                'val_f1': val.get('f1', 0),
                'val_auc': val.get('auc', 0),
                'val_loss': val.get('loss', 1e9),
                'val_acc': val.get('accuracy', 0),
            })
        except Exception as e:
            print("  Skip %s: %s" % (run_dir, e))

    if not results:
        print("No best_model.pt found in %s" % runs_dir)
        return

    key = 'val_' + args.sort
    results.sort(key=lambda r: r.get(key, 0), reverse=(args.sort != 'loss'))

    print("\n" + "=" * 70)
    print(" Model Comparison (sorted by %s)" % args.sort)
    print("=" * 70)
    print("%-35s %6s %8s %8s %8s" % ("Run", "Epoch", "F1", "AUC", "Loss"))
    print("-" * 70)
    for r in results:
        print("%-35s %6d %8.4f %8.4f %8.4f" %
              (r['run'], r['epoch'], r['val_f1'], r['val_auc'], r['val_loss']))
    print("=" * 70)
    best = results[0]
    print("Best: %s (F1=%.4f, AUC=%.4f)" % (best['run'], best['val_f1'], best['val_auc']))


if __name__ == '__main__':
    main()
