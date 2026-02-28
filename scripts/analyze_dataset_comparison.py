#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Comparison: 欠陥あり vs 欠陥なし の可視化・統計比較

Usage:
  python scripts/analyze_dataset_comparison.py --output_dir figures/dataset_comparison
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def load_nodes_csv(path):
    """Load nodes.csv, return DataFrame and format type."""
    df = pd.read_csv(path)
    if 's11' in df.columns and 'defect_label' in df.columns:
        fmt = 'stress'
    elif 'ux' in df.columns:
        fmt = 'displacement'
    else:
        fmt = 'unknown'
    return df, fmt


def load_doe_samples(doe_path):
    """Load DOE samples for defect params."""
    with open(doe_path) as f:
        doe = json.load(f)
    return {s['id']: s.get('defect_params') for s in doe['samples']}


def compute_stats(df, fmt):
    """Compute summary statistics for a nodes DataFrame."""
    stats = {'n_nodes': len(df)}
    if fmt == 'stress':
        for col in ['s11', 's22', 's12', 'dspss']:
            if col in df.columns:
                stats[f'{col}_mean'] = float(df[col].mean())
                stats[f'{col}_std'] = float(df[col].std())
                stats[f'{col}_min'] = float(df[col].min())
                stats[f'{col}_max'] = float(df[col].max())
        if 'defect_label' in df.columns:
            stats['n_defect'] = int((df['defect_label'] == 1).sum())
    elif fmt == 'displacement':
        for col in ['ux', 'uy', 'uz']:
            if col in df.columns:
                stats[f'{col}_mean'] = float(df[col].mean())
                stats[f'{col}_std'] = float(df[col].std())
        if 'temp' in df.columns:
            stats['temp_mean'] = float(df['temp'].mean())
            stats['temp_std'] = float(df['temp'].std())
    # Spatial
    for col in ['x', 'y', 'z']:
        if col in df.columns:
            stats[f'{col}_min'] = float(df[col].min())
            stats[f'{col}_max'] = float(df[col].max())
    return stats


def plot_spatial_comparison(healthy_df, defect_dfs, fmt, out_dir):
    """Spatial distribution comparison (x,y,z)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, coord in zip(axes, ['x', 'y', 'z']):
        if coord not in healthy_df.columns:
            continue
        ax.hist(healthy_df[coord], bins=50, alpha=0.6, label='Healthy', density=True, color='green')
        for i, (name, df) in enumerate(defect_dfs[:5]):  # max 5 defect samples
            ax.hist(df[coord], bins=50, alpha=0.3, label=name, density=True)
        ax.set_xlabel(coord + ' (mm)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.set_title(f'{coord} distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'spatial_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_stress_comparison(healthy_df, defect_dfs, out_dir):
    """Stress (dspss) distribution comparison."""
    if 'dspss' not in healthy_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(healthy_df['dspss'], bins=80, alpha=0.6, label='Healthy', density=True, color='green')
    for name, df in defect_dfs[:5]:
        if 'dspss' in df.columns:
            ax.hist(df['dspss'], bins=80, alpha=0.3, label=name)
    ax.set_xlabel('DSPSS (MPa)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title('Stress (DSPSS) distribution: Healthy vs Defect')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'stress_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_defect_labels(defect_df, out_dir, name='sample'):
    """Defect label spatial distribution (scatter)."""
    if 'defect_label' not in defect_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    healthy = defect_df[defect_df['defect_label'] == 0]
    defect = defect_df[defect_df['defect_label'] == 1]
    if 'x' in defect_df.columns and 'z' in defect_df.columns:
        ax.scatter(healthy['x'], healthy['z'], s=1, alpha=0.5, c='green', label='Healthy')
        ax.scatter(defect['x'], defect['z'], s=5, alpha=0.8, c='red', label='Defect')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('z (mm)')
        ax.legend()
        ax.set_title(f'Defect spatial distribution ({name})')
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'defect_spatial_{name}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_r_z_scatter(df, out_dir, name, color_col=None):
    """r vs z scatter (cylindrical view)."""
    if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
        return
    r = np.sqrt(df['x']**2 + df['y']**2)
    z = df['z']
    fig, ax = plt.subplots(figsize=(7, 6))
    if color_col and color_col in df.columns:
        sc = ax.scatter(r, z, c=df[color_col], s=2, cmap='viridis')
        plt.colorbar(sc, ax=ax, label=color_col)
    else:
        ax.scatter(r, z, s=2, alpha=0.6)
    ax.set_xlabel('r (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_title(f'Geometry: {name}')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'r_z_{name}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='dataset_output')
    parser.add_argument('--output_dir', default='figures/dataset_comparison')
    parser.add_argument('--doe', default='doe_phase1.json')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load DOE for defect params
    doe_samples = {}
    if os.path.exists(args.doe):
        doe_samples = load_doe_samples(args.doe)

    # Load healthy
    healthy_path = data_dir / 'healthy_baseline' / 'nodes.csv'
    if not healthy_path.exists():
        print("healthy_baseline/nodes.csv not found")
        return
    healthy_df, healthy_fmt = load_nodes_csv(healthy_path)
    healthy_stats = compute_stats(healthy_df, healthy_fmt)
    healthy_stats['name'] = 'healthy_baseline'
    healthy_stats['defect_type'] = 'healthy'

    # Load defect samples
    defect_dfs = []
    defect_stats_list = []
    for i in range(20):
        sample_dir = data_dir / f'sample_{i:04d}'
        nodes_path = sample_dir / 'nodes.csv'
        if not nodes_path.exists():
            continue
        df, fmt = load_nodes_csv(nodes_path)
        stats = compute_stats(df, fmt)
        stats['name'] = f'sample_{i:04d}'
        stats['defect_type'] = 'defect'
        stats['defect_params'] = doe_samples.get(i, {})
        defect_dfs.append((f'sample_{i:04d}', df))
        defect_stats_list.append(stats)

        # Defect spatial plot (for stress-format samples with defect_label)
        if fmt == 'stress' and 'defect_label' in df.columns and (df['defect_label'] == 1).any():
            plot_defect_labels(df, out_dir, f'sample_{i:04d}')

    # Comparison stats table
    all_stats = [healthy_stats] + defect_stats_list
    rows = []
    for s in all_stats:
        row = {
            'name': s['name'],
            'n_nodes': s['n_nodes'],
            'n_defect': s.get('n_defect', '-'),
        }
        if 'dspss_mean' in s:
            row['dspss_mean'] = f"{s['dspss_mean']:.6f}"
            row['dspss_std'] = f"{s['dspss_std']:.6f}"
        if 'defect_params' in s and s['defect_params']:
            dp = s['defect_params']
            row['theta_deg'] = dp.get('theta_deg', '-')
            row['z_center'] = dp.get('z_center', '-')
            row['radius'] = dp.get('radius', '-')
        rows.append(row)
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out_dir / 'comparison_stats.csv', index=False)

    # Plots
    if defect_dfs:
        plot_spatial_comparison(healthy_df, defect_dfs, healthy_fmt, out_dir)
        if healthy_fmt == 'stress':
            plot_stress_comparison(healthy_df, defect_dfs, out_dir)
        elif healthy_fmt == 'displacement':
            plot_disp_temp_comparison(healthy_df, defect_dfs, out_dir)
        color_col = None
        if 'dspss' in healthy_df.columns:
            color_col = 'dspss'
        elif 'temp' in healthy_df.columns:
            color_col = 'temp'
        plot_r_z_scatter(healthy_df, out_dir, 'healthy_baseline', color_col)
        if defect_dfs:
            d0 = defect_dfs[0][1]
            c0 = 'dspss' if 'dspss' in d0.columns else 'temp' if 'temp' in d0.columns else None
            plot_r_z_scatter(d0, out_dir, 'sample_0000', c0)

    # Aggregate comparison (n_nodes, n_defect distribution)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(['Healthy'], [healthy_stats['n_nodes']], color='green', alpha=0.7)
    if defect_stats_list:
        n_nodes_defect = [s['n_nodes'] for s in defect_stats_list]
        axes[0].bar(['Defect (avg)'], [np.mean(n_nodes_defect)], color='orange', alpha=0.7)
        axes[0].set_ylabel('Number of nodes')
        axes[1].hist(n_nodes_defect, bins=10, color='orange', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Number of nodes (defect samples)')
        axes[1].set_ylabel('Count')
    axes[0].set_title('Node count: Healthy vs Defect')
    axes[1].set_title('Defect samples node count distribution')
    plt.tight_layout()
    plt.savefig(out_dir / 'node_count_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Summary for wiki
    summary = {
        'n_healthy': 1,
        'n_defect': len(defect_stats_list),
        'healthy_n_nodes': healthy_stats['n_nodes'],
        'defect_n_nodes_mean': float(np.mean([s['n_nodes'] for s in defect_stats_list])) if defect_stats_list else 0,
        'defect_n_nodes_std': float(np.std([s['n_nodes'] for s in defect_stats_list])) if defect_stats_list else 0,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("Analysis complete. Output:", out_dir)
    print("  - comparison_stats.csv")
    print("  - summary.json")
    print("  - *.png figures")


if __name__ == '__main__':
    main()
