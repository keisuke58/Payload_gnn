#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
欠陥データの物理量検証 — Numerical/Physical Validation of Defect Data

応力・変位・温度の物理的妥当性、欠陥ゾーン vs 健全ゾーンの差異を検証。
CFRP サンドイッチの熱応力・変位の期待範囲（文献値）と照合。

Usage:
  python scripts/validate_defect_physics.py --data dataset_multitype_100
  python scripts/validate_defect_physics.py --data dataset_multitype_100 --output report.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 物理的期待範囲（CFRP サンドイッチ、熱荷重 20→120°C）
# Ref: Thermal stress ~E*alpha*DT = 160e3 * 2e-6 * 100 ≈ 32 MPa (order)
# Displacement: thermal expansion ~alpha*L*DT, L~5m → ~1 mm (order)
EXPECTED = {
    'temp_min': 20.0,
    'temp_max': 125.0,
    'temp_mean': (20 + 120) / 2,
    'u_mag_min': 0.0,      # mm (固定点で0の場合あり)
    'u_mag_max': 500.0,    # mm (クランプ端・オジーブ先端で大きい場合あり)
    'smises_min': 1e-12,   # MPa (数値ゼロ回避、熱のみ荷重で小さい場合あり)
    'smises_max': 1000.0,  # MPa
    'n_defect_min': 1,     # 欠陥サンプルは少なくとも1ノード
}


def load_meta(path):
    """Load metadata.csv as dict."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df['key'], df['value']))


def compute_u_mag(df):
    """Displacement magnitude (mm)."""
    return np.sqrt(df['ux']**2 + df['uy']**2 + df['uz']**2)


def validate_sample(sample_dir):
    """Validate a single sample. Returns (results, stats)."""
    nodes_path = os.path.join(sample_dir, 'nodes.csv')
    meta_path = os.path.join(sample_dir, 'metadata.csv')
    results = []
    stats = {}

    if not os.path.exists(nodes_path):
        results.append(('Files.nodes_csv', False, 'nodes.csv not found'))
        return results, stats

    df = pd.read_csv(nodes_path)
    meta = load_meta(meta_path)

    n_nodes = len(df)
    n_defect = (df['defect_label'] != 0).sum()
    defect_type = meta.get('defect_type', 'unknown')

    stats['n_nodes'] = n_nodes
    stats['n_defect'] = int(n_defect)
    stats['defect_type'] = defect_type

    # Temperature
    if 'temp' in df.columns:
        t_min, t_max, t_mean = df['temp'].min(), df['temp'].max(), df['temp'].mean()
        stats['temp_min'] = float(t_min)
        stats['temp_max'] = float(t_max)
        stats['temp_mean'] = float(t_mean)
        ok_t = EXPECTED['temp_min'] <= t_min and t_max <= EXPECTED['temp_max']
        results.append(('Physics.Temp_range', ok_t, f'temp=[{t_min:.1f}, {t_max:.1f}] °C'))
    else:
        results.append(('Physics.Temp_range', False, 'temp column missing'))

    # Displacement
    if all(c in df.columns for c in ['ux', 'uy', 'uz']):
        u_mag = compute_u_mag(df)
        u_min, u_max, u_mean = u_mag.min(), u_mag.max(), u_mag.mean()
        stats['u_mag_min'] = float(u_min)
        stats['u_mag_max'] = float(u_max)
        stats['u_mag_mean'] = float(u_mean)
        ok_u = EXPECTED['u_mag_min'] <= u_min and u_max <= EXPECTED['u_mag_max']
        results.append(('Physics.Displacement', ok_u, f'|u|=[{u_min:.4f}, {u_max:.4f}] mm'))
    else:
        results.append(('Physics.Displacement', False, 'ux/uy/uz missing'))

    # Stress (von Mises)
    if 'smises' in df.columns:
        s_min, s_max = df['smises'].min(), df['smises'].max()
        s_mean = df['smises'].mean()
        stats['smises_min'] = float(s_min)
        stats['smises_max'] = float(s_max)
        stats['smises_mean'] = float(s_mean)
        ok_s = s_max >= EXPECTED['smises_min'] and s_max <= EXPECTED['smises_max']
        results.append(('Physics.Stress', ok_s, f'smises=[{s_min:.2e}, {s_max:.2e}] MPa'))
    else:
        results.append(('Physics.Stress', False, 'smises missing'))

    # Defect vs healthy contrast (for defect samples)
    if n_defect > 0 and 'smises' in df.columns:
        df_healthy = df[df['defect_label'] == 0]
        df_defect = df[df['defect_label'] != 0]
        s_healthy_mean = df_healthy['smises'].mean()
        s_defect_mean = df_defect['smises'].mean()
        u_healthy = compute_u_mag(df_healthy) if all(c in df.columns for c in ['ux', 'uy', 'uz']) else pd.Series([0])
        u_defect = compute_u_mag(df_defect)
        u_healthy_mean = u_healthy.mean()
        u_defect_mean = u_defect.mean()

        stats['smises_healthy_mean'] = float(s_healthy_mean)
        stats['smises_defect_mean'] = float(s_defect_mean)
        stats['u_defect_mean'] = float(u_defect_mean)
        stats['u_healthy_mean'] = float(u_healthy_mean)

        # Defect zone vs healthy (physical signature; thermal stress can be small)
        s_ratio = s_defect_mean / (s_healthy_mean + 1e-12)
        u_ratio = u_defect_mean / (u_healthy_mean + 1e-12)
        has_contrast = (abs(s_ratio - 1.0) > 0.05) or (abs(u_ratio - 1.0) > 0.05)
        results.append(('Physics.Defect_contrast', has_contrast,
                       f'Defect/Healthy: smises={s_ratio:.3f}, u={u_ratio:.3f}'))
    else:
        if n_defect == 0 and defect_type != 'healthy':
            results.append(('Physics.Defect_contrast', False, 'No defect nodes but defect_type=%s' % defect_type))
        else:
            results.append(('Physics.Defect_contrast', True, 'N/A (healthy or no defect)'))

    # Defect label consistency with metadata
    meta_n_defect = int(meta.get('n_defect_nodes', -1))
    ok_meta = meta_n_defect == n_defect
    results.append(('Consistency.Meta_defect_count', ok_meta,
                    f'meta n_defect={meta_n_defect} vs actual={n_defect}'))

    return results, stats


def run_validation(data_dir, max_samples=100):
    """Run validation on all samples."""
    data_dir = os.path.join(PROJECT_ROOT, data_dir) if not os.path.isabs(data_dir) else data_dir
    if not os.path.exists(data_dir):
        return [], {}, {'error': 'Directory not found: %s' % data_dir}

    samples = sorted([d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)) and
                     (d.startswith('sample_') or d == 'healthy_baseline')])[:max_samples]

    all_results = []
    all_stats = []
    for name in samples:
        sample_dir = os.path.join(data_dir, name)
        results, stats = validate_sample(sample_dir)
        if stats:
            stats['sample'] = name
            all_stats.append(stats)
        for r in results:
            all_results.append((name, r[0], r[1], r[2]))

    # Aggregate
    passed = sum(1 for r in all_results if r[2])
    total = len(all_results)
    by_check = {}
    for _, check, ok, msg in all_results:
        if check not in by_check:
            by_check[check] = {'pass': 0, 'fail': 0}
        if ok:
            by_check[check]['pass'] += 1
        else:
            by_check[check]['fail'] += 1

    summary = {
        'data_dir': data_dir,
        'n_samples': len(samples),
        'passed': passed,
        'total': total,
        'all_pass': passed == total,
        'by_check': by_check,
    }
    return all_results, all_stats, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset_multitype_100')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--max_samples', type=int, default=100)
    args = parser.parse_args()

    all_results, all_stats, summary = run_validation(args.data, args.max_samples)

    print("=" * 70)
    print("Defect Physics Validation Report")
    print("=" * 70)
    print(f"Data: {summary.get('data_dir', args.data)}")
    print(f"Samples: {summary.get('n_samples', 0)}")
    print()

    # Group by check
    checks = {}
    for name, check, ok, msg in all_results:
        if check not in checks:
            checks[check] = []
        checks[check].append((name, ok, msg))

    for check, items in sorted(checks.items()):
        passed = sum(1 for _, ok, _ in items if ok)
        total = len(items)
        status = "PASS" if passed == total else "FAIL"
        print(f"  [{status}] {check}: {passed}/{total}")
        if passed < total:
            for name, ok, msg in items[:3]:
                if not ok:
                    print(f"       - {name}: {msg}")
    print()
    print(f"Result: {summary.get('passed', 0)}/{summary.get('total', 0)} passed" +
          (" - ALL PASS" if summary.get('all_pass') else " - FAILED"))
    print("=" * 70)

    if args.output:
        out = {
            'summary': summary,
            'stats': all_stats[:50] if len(all_stats) > 50 else all_stats,
        }
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Report saved: {args.output}")

    sys.exit(0 if summary.get('all_pass') else 1)


if __name__ == '__main__':
    main()
