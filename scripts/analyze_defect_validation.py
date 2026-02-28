#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
欠陥検証の数値分析 — Numerical Analysis for Defect Validation

データセット全体の統計量、欠陥タイプ別の数値サマリ、欠陥 vs 健全ゾーンの差異を算出。
出力: JSON レポート + Markdown テーブル（Wiki 掲載用）

Usage:
  python scripts/analyze_defect_validation.py --data dataset_multitype_100
  python scripts/analyze_defect_validation.py --data dataset_multitype_100 --output docs/defect_analysis_report.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_meta(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df['key'], df['value']))


def compute_u_mag(df):
    return np.sqrt(df['ux']**2 + df['uy']**2 + df['uz']**2)


def run_analysis(data_dir, max_samples=200):
    """Run full numerical analysis on defect dataset."""
    data_dir = os.path.join(PROJECT_ROOT, data_dir) if not os.path.isabs(data_dir) else data_dir
    if not os.path.exists(data_dir):
        return {'error': f'Directory not found: {data_dir}'}

    samples = sorted([d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)) and
                     (d.startswith('sample_') or d == 'healthy_baseline')])[:max_samples]

    rows = []
    all_smises_healthy, all_smises_defect = [], []
    all_u_healthy, all_u_defect = [], []

    for name in samples:
        sample_dir = os.path.join(data_dir, name)
        nodes_path = os.path.join(sample_dir, 'nodes.csv')
        if not os.path.exists(nodes_path):
            continue

        df = pd.read_csv(nodes_path)
        meta = load_meta(os.path.join(sample_dir, 'metadata.csv'))
        dtype = meta.get('defect_type', 'unknown')

        df['u_mag'] = compute_u_mag(df)
        n_def = (df['defect_label'] != 0).sum()
        df_h = df[df['defect_label'] == 0]
        df_d = df[df['defect_label'] != 0]

        row = {
            'sample': name,
            'defect_type': dtype,
            'n_nodes': len(df),
            'n_defect': int(n_def),
            'defect_ratio': n_def / len(df) if len(df) > 0 else 0,
        }
        if 'smises' in df.columns:
            row['smises_mean'] = float(df['smises'].mean())
            row['smises_std'] = float(df['smises'].std()) if len(df) > 1 else 0
            row['smises_healthy_mean'] = float(df_h['smises'].mean()) if len(df_h) > 0 else 0
            row['smises_defect_mean'] = float(df_d['smises'].mean()) if len(df_d) > 0 else 0
            row['smises_ratio_defect_healthy'] = row['smises_defect_mean'] / (row['smises_healthy_mean'] + 1e-12)
            all_smises_healthy.extend(df_h['smises'].tolist())
            all_smises_defect.extend(df_d['smises'].tolist())
        if 'temp' in df.columns:
            row['temp_mean'] = float(df['temp'].mean())
            row['temp_min'] = float(df['temp'].min())
            row['temp_max'] = float(df['temp'].max())
        row['u_mag_mean'] = float(df['u_mag'].mean())
        row['u_mag_healthy_mean'] = float(df_h['u_mag'].mean()) if len(df_h) > 0 else 0
        row['u_mag_defect_mean'] = float(df_d['u_mag'].mean()) if len(df_d) > 0 else 0
        row['u_ratio_defect_healthy'] = row['u_mag_defect_mean'] / (row['u_mag_healthy_mean'] + 1e-12)
        all_u_healthy.extend(df_h['u_mag'].tolist())
        all_u_defect.extend(df_d['u_mag'].tolist())

        rows.append(row)

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        return {'error': 'No valid samples found', 'data_dir': data_dir}

    # Aggregate by defect type
    by_type = df_all.groupby('defect_type').agg({
        'sample': 'count',
        'n_defect': ['mean', 'std', 'min', 'max'],
        'smises_mean': 'mean',
        'smises_ratio_defect_healthy': 'mean',
        'u_mag_mean': 'mean',
        'u_ratio_defect_healthy': 'mean',
        'temp_mean': 'mean',
    }).round(6)

    by_type.columns = ['n_samples', 'n_defect_mean', 'n_defect_std', 'n_defect_min', 'n_defect_max',
                       'smises_mean', 'smises_ratio', 'u_mag_mean', 'u_ratio', 'temp_mean']
    by_type = by_type.reset_index()

    # Global statistics
    global_stats = {
        'n_samples': len(df_all),
        'defect_types': df_all['defect_type'].nunique(),
        'total_nodes': int(df_all['n_nodes'].sum()),
        'total_defect_nodes': int(df_all['n_defect'].sum()),
        'overall_defect_ratio': float(df_all['n_defect'].sum() / df_all['n_nodes'].sum()) if df_all['n_nodes'].sum() > 0 else 0,
    }
    if all_smises_healthy and all_smises_defect:
        global_stats['smises_healthy_global_mean'] = float(np.mean(all_smises_healthy))
        global_stats['smises_defect_global_mean'] = float(np.mean(all_smises_defect))
        global_stats['smises_global_ratio'] = float(np.mean(all_smises_defect) / (np.mean(all_smises_healthy) + 1e-12))
    if all_u_healthy and all_u_defect:
        global_stats['u_healthy_global_mean'] = float(np.mean(all_u_healthy))
        global_stats['u_defect_global_mean'] = float(np.mean(all_u_defect))
        global_stats['u_global_ratio'] = float(np.mean(all_u_defect) / (np.mean(all_u_healthy) + 1e-12))

    # Sample count per type (for dataset ratio reference)
    type_counts = df_all['defect_type'].value_counts().to_dict()

    return {
        'data_dir': data_dir,
        'global': global_stats,
        'by_defect_type': by_type.to_dict(orient='records'),
        'type_counts': type_counts,
        'sample_details': rows[:20],  # First 20 for brevity
    }


def to_markdown_table(report):
    """Generate Markdown table for wiki."""
    lines = []
    lines.append("## 数値分析サマリ")
    lines.append("")
    lines.append("### グローバル統計")
    g = report.get('global', {})
    lines.append("| 指標 | 値 |")
    lines.append("|------|-----|")
    lines.append(f"| サンプル数 | {g.get('n_samples', '-')} |")
    lines.append(f"| 欠陥タイプ数 | {g.get('defect_types', '-')} |")
    lines.append(f"| 総ノード数 | {g.get('total_nodes', '-'):,} |")
    lines.append(f"| 総欠陥ノード数 | {g.get('total_defect_nodes', '-'):,} |")
    lines.append(f"| ノードレベル欠陥比率 | {g.get('overall_defect_ratio', 0)*100:.2f}% |")
    if 'smises_global_ratio' in g:
        lines.append(f"| 応力比 (欠陥/健全) | {g['smises_global_ratio']:.3f} |")
    if 'u_global_ratio' in g:
        lines.append(f"| 変位比 (欠陥/健全) | {g['u_global_ratio']:.3f} |")
    lines.append("")
    lines.append("### 欠陥タイプ別統計")
    lines.append("")
    by_type = report.get('by_defect_type', [])
    if by_type:
        lines.append("| 欠陥タイプ | サンプル数 | 平均欠陥ノード数 | 応力比 | 変位比 |")
        lines.append("|------------|------------|-------------------|--------|--------|")
        for r in by_type:
            s_ratio = r.get('smises_ratio', 0) or 0
            u_ratio = r.get('u_ratio', 0) or 0
            lines.append(f"| {r.get('defect_type', '-')} | {r.get('n_samples', '-')} | "
                        f"{r.get('n_defect_mean', 0):.0f} | {s_ratio:.3f} | {u_ratio:.3f} |")
    lines.append("")
    lines.append("### サンプル数（タイプ別）")
    lines.append("")
    tc = report.get('type_counts', {})
    total = sum(tc.values())
    lines.append("| 欠陥タイプ | 個数 | 割合 |")
    lines.append("|------------|------|------|")
    for t, c in sorted(tc.items(), key=lambda x: -x[1]):
        pct = 100 * c / total if total > 0 else 0
        lines.append(f"| {t} | {c} | {pct:.1f}% |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset_multitype_100')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--markdown', type=str, default='', help='Output path for Markdown table')
    parser.add_argument('--max_samples', type=int, default=200)
    args = parser.parse_args()

    report = run_analysis(args.data, args.max_samples)

    if 'error' in report and report.get('error'):
        print("Error:", report['error'])
        sys.exit(1)

    print("=" * 60)
    print("Defect Validation Numerical Analysis")
    print("=" * 60)
    g = report.get('global', {})
    print(f"Samples: {g.get('n_samples')}, Defect types: {g.get('defect_types')}")
    print(f"Total nodes: {g.get('total_nodes'):,}, Defect nodes: {g.get('total_defect_nodes'):,}")
    print(f"Defect ratio (node-level): {g.get('overall_defect_ratio', 0)*100:.2f}%")
    print()
    print("By defect type:")
    for r in report.get('by_defect_type', []):
        print(f"  {r.get('defect_type')}: n={r.get('n_samples')}, "
              f"mean_defect_nodes={r.get('n_defect_mean', 0):.0f}, "
              f"smises_ratio={r.get('smises_ratio', 0):.3f}, u_ratio={r.get('u_ratio', 0):.3f}")
    print("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved: {args.output}")

    if args.markdown:
        os.makedirs(os.path.dirname(args.markdown) or '.', exist_ok=True)
        md = to_markdown_table(report)
        with open(args.markdown, 'w') as f:
            f.write(md)
        print(f"Markdown saved: {args.markdown}")


if __name__ == '__main__':
    main()
