#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GW 特徴量アブレーション — baseline (3 feat) vs extended (10 feat) の精度比較

データセットが存在する場合:
  1. baseline で prepare → train → 記録
  2. extended で prepare → train → 記録
  3. 結果を比較して報告

必要: Healthy + 欠陥サンプルが両方あること（例: batch_generate_gw_dataset.sh all 後）

Usage:
  python scripts/ablation_gw_features.py --input abaqus_work/gw_fairing_dataset
  python scripts/ablation_gw_features.py --input abaqus_work/gw_fairing_dataset --epochs 100
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, cwd=None, capture=False):
    """Run command, return success. If capture=True, return (success, stdout_lines)."""
    cwd = cwd or PROJECT_ROOT
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=capture, text=True)
    if capture:
        return r.returncode == 0, (r.stdout or '').splitlines()
    return r.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='GW feature ablation: baseline vs extended')
    parser.add_argument('--input', type=str, default='abaqus_work/gw_fairing_dataset',
                        help='Directory with *_sensors.csv')
    parser.add_argument('--data_1k', action='store_true',
                        help='Use data/processed_gw_1000_* (run prepare_gw_1000.sh first)')
    parser.add_argument('--doe', type=str, default='doe_gw_fairing.json',
                        help='DOE JSON for defect list; use --no_doe for fallback scan')
    parser.add_argument('--no_doe', action='store_true',
                        help='Use fallback scan (for abaqus_work with Test-H/D only)')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--arch', type=str, default='gat')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.data_1k:
        # Use pre-prepared 1000-sample datasets
        input_dir = None
        out_base = 'data'
        for feat_set in ('baseline', 'extended', 'full'):
            out_dir = '%s/processed_gw_1000_%s' % (out_base, feat_set)
            if not os.path.exists(os.path.join(PROJECT_ROOT, out_dir, 'train.pt')):
                print("ERROR: %s not found. Run: bash scripts/prepare_gw_1000.sh" % out_dir)
                sys.exit(1)
    else:
        input_dir = os.path.join(PROJECT_ROOT, args.input)
        if not os.path.isdir(input_dir):
            print("ERROR: Input dir not found: %s" % input_dir)
            print("  Run batch_generate_gw_dataset.sh all first, or specify --input")
            sys.exit(1)
        csvs = [f for f in os.listdir(input_dir) if f.endswith('_sensors.csv')]
        if len(csvs) < 2:
            print("ERROR: Need at least 2 CSV files in %s (found %d)" % (input_dir, len(csvs)))
            sys.exit(1)

    print("=== GW Feature Ablation ===")
    if args.data_1k:
        print("Mode: 1000-sample datasets (data/processed_gw_1000_*)")
    else:
        print("Input: %s (%d CSVs)" % (args.input, len(csvs)))
    print("Epochs: %d | Arch: %s" % (args.epochs, args.arch))
    print()

    out_base = 'data'
    run_base = os.path.join(PROJECT_ROOT, 'runs', 'ablation_gw')
    results = {}

    for feat_set in ('baseline', 'extended', 'full'):
        out_dir = '%s/processed_gw_1000_%s' % (out_base, feat_set) if args.data_1k else '%s/processed_gw_ablation_%s' % (out_base, feat_set)
        run_dir = os.path.join(run_base, feat_set)
        os.makedirs(run_dir, exist_ok=True)

        print("--- %s ---" % feat_set)
        if not args.data_1k:
            prep_cmd = "python src/prepare_gw_ml_data.py --input %s --output %s --feature_set %s" % (
                args.input, out_dir, feat_set)
            if args.no_doe:
                prep_cmd += " --no_doe"
            else:
                prep_cmd += " --doe %s" % args.doe
            ok = run(prep_cmd)
        else:
            ok = True  # data already prepared
        if not ok:
            print("  FAILED: prepare")
            results[feat_set] = {'run_dir': None, 'best_f1': None}
            continue

        # Train and capture Best val F1
        ok, lines = run(
            "python src/train_gw.py --data_dir %s --arch %s --epochs %d --run_dir %s"
            % (out_dir, args.arch, args.epochs, run_dir),
            capture=True
        )
        if not ok:
            print("  FAILED: train")
            results[feat_set] = {'run_dir': run_dir, 'best_f1': None}
            continue

        best_f1 = None
        for line in lines:
            if 'Best val F1' in line:
                try:
                    best_f1 = float(line.split('F1:')[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        results[feat_set] = {'run_dir': run_dir, 'best_f1': best_f1}
        print("  Done. Best val F1: %s" % (best_f1 if best_f1 is not None else 'N/A'))
        print()

    print("=== Summary ===")
    for feat_set, res in results.items():
        if res and res.get('run_dir'):
            f1 = res.get('best_f1')
            f1_str = '%.3f' % f1 if f1 is not None else 'N/A'
            print("  %s: Best val F1 = %s | %s" % (feat_set, f1_str, res['run_dir']))
        else:
            print("  %s: FAILED" % feat_set)

    if all(r and r.get('best_f1') is not None for r in results.values()):
        b_f1 = results.get('baseline', {}).get('best_f1') or 0
        e_f1 = results.get('extended', {}).get('best_f1') or 0
        f_f1 = results.get('full', {}).get('best_f1') or 0
        print()
        print("  extended - baseline: %+.3f" % (e_f1 - b_f1))
        print("  full - extended: %+.3f" % (f_f1 - e_f1))

    print()
    print("Compare: tensorboard --logdir %s" % run_base)


if __name__ == '__main__':
    main()
