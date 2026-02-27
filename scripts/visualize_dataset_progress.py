#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセット進捗可視化 — 教授との進捗確認用

既存 dataset_output の 3D 可視化・特徴量分布・サマリを
見やすく出力。figures/progress/ に保存。

Usage:
  python scripts/visualize_dataset_progress.py
  python scripts/visualize_dataset_progress.py --data dataset_output --output figures/progress
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 日本語フォント（利用可能なら）
plt.rcParams['font.family'] = ['DejaVu Sans', 'IPAexGothic', 'Noto Sans CJK JP', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'figures', 'progress')


def load_metadata(sample_dir):
    """metadata.csv を dict で返す"""
    path = os.path.join(sample_dir, 'metadata.csv')
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df['key'], df['value']))


def load_nodes(sample_dir):
    """nodes.csv を DataFrame で返す"""
    path = os.path.join(sample_dir, 'nodes.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def collect_dataset_stats(data_dir):
    """全サンプルの統計を収集"""
    samples = []
    for name in sorted(os.listdir(data_dir)):
        if not name.startswith('sample_'):
            continue
        sample_dir = os.path.join(data_dir, name)
        if not os.path.isdir(sample_dir):
            continue
        meta = load_metadata(sample_dir)
        nodes = load_nodes(sample_dir)
        if nodes is None:
            continue
        # 品質チェック
        ux_mean = nodes['ux'].abs().mean()
        temp_mean = nodes['temp'].mean()
        good = (ux_mean > 0.001) and (temp_mean > 10)
        samples.append({
            'id': int(name.split('_')[1]),
            'theta': float(meta.get('theta_deg', 0)),
            'z_center': float(meta.get('z_center', 0)),
            'radius': float(meta.get('radius', 0)),
            'n_defect': int(meta.get('n_defect_nodes', 0)),
            'n_total': int(meta.get('n_total_nodes', 0)),
            'ux_mean': ux_mean,
            'temp_mean': temp_mean,
            'good': good,
        })
    return samples


def plot_3d_sample(nodes_path, output_path, color_by='displacement', sample_id=None):
    """1サンプルの 3D メッシュ可視化"""
    df = pd.read_csv(nodes_path)
    x, y, z = df['x'].values / 1000, df['y'].values / 1000, df['z'].values / 1000

    if color_by == 'displacement':
        ux, uy, uz = df['ux'].values, df['uy'].values, df['uz'].values
        val = np.sqrt(ux**2 + uy**2 + uz**2)
        label = 'Displacement [mm]'
        cmap = 'viridis'
    elif color_by == 'temp':
        val = df['temp'].values
        label = 'Temperature [°C]'
        cmap = 'hot'
    elif color_by == 'defect':
        val = df['defect_label'].values
        label = 'Defect (0=Healthy, 1=Defect)'
        cmap = 'RdYlGn_r'
    elif color_by == 'smises':
        val = np.maximum(df['smises'].values, 1e-10)
        val = np.log10(val + 1e-12)
        label = 'log10(Mises Stress) [MPa]'
        cmap = 'inferno'
    else:
        val = np.zeros(len(df))
        label = ''
        cmap = 'viridis'

    # サブサンプリング（大量点で重い場合）
    n = len(df)
    step = max(1, n // 8000)
    idx = np.arange(0, n, step)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x[idx], y[idx], z[idx], c=val[idx], cmap=cmap, s=2, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=15)
    cbar.set_label(label, fontsize=11)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y (Axis) [m]')
    ax.set_zlabel('Z [m]')
    title = f'H3 Fairing FEM Mesh — {color_by}'
    if sample_id is not None:
        title += f' (Sample {sample_id:04d})'
    ax.set_title(title, fontsize=14)
    ax.set_box_aspect([1, 2, 1])
    ax.view_init(elev=20, azim=45)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_unfolded_2d(nodes_path, output_path, color_by='displacement', sample_id=None):
    """展開図 (θ  vs 軸方向 y) — 円筒形状を直感的に表示"""
    df = pd.read_csv(nodes_path)
    x, y, z = df['x'].values, df['y'].values, df['z'].values
    r = np.sqrt(x**2 + z**2)
    theta_rad = np.arctan2(z, x)
    theta_deg = np.degrees(theta_rad)
    arc_mm = r * np.radians(theta_deg)  # 弧長近似

    if color_by == 'displacement':
        ux, uy, uz = df['ux'].values, df['uy'].values, df['uz'].values
        val = np.sqrt(ux**2 + uy**2 + uz**2)
        label = 'Displacement [mm]'
        cmap = 'viridis'
    elif color_by == 'temp':
        val = df['temp'].values
        label = 'Temperature [°C]'
        cmap = 'hot'
    elif color_by == 'defect':
        val = df['defect_label'].values
        label = 'Defect'
        cmap = 'RdYlGn_r'
    else:
        val = np.zeros(len(df))
        label = ''
        cmap = 'viridis'

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(arc_mm, y, c=val, cmap=cmap, s=1, alpha=0.7)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label(label, fontsize=11)

    ax.set_xlabel('Arc Length (R×θ) [mm]')
    ax.set_ylabel('Axis Y [mm]')
    title = f'Unfolded View — {color_by}'
    if sample_id is not None:
        title += f' (Sample {sample_id:04d})'
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_feature_distributions(data_dir, output_path):
    """全サンプルの特徴量分布（ヒストグラム）"""
    samples = collect_dataset_stats(data_dir)
    if not samples:
        print("No samples found")
        return

    stats = pd.DataFrame(samples)

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))

    # 欠陥半径
    ax = axes[0, 0]
    ax.hist(stats['radius'], bins=25, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Defect Radius [mm]')
    ax.set_ylabel('Count')
    ax.set_title('Defect Size Distribution')
    ax.axvline(stats['radius'].median(), color='red', linestyle='--', label=f'Median: {stats["radius"].median():.0f} mm')
    ax.legend()

    # 欠陥位置 θ
    ax = axes[0, 1]
    ax.hist(stats['theta'], bins=25, color='coral', edgecolor='white', alpha=0.8)
    ax.set_xlabel('θ [deg]')
    ax.set_ylabel('Count')
    ax.set_title('Defect Location (θ)')

    # 欠陥位置 z
    ax = axes[0, 2]
    ax.hist(stats['z_center'], bins=25, color='seagreen', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Z center [mm]')
    ax.set_ylabel('Count')
    ax.set_title('Defect Location (Z)')

    # 欠陥ノード数
    ax = axes[1, 0]
    ax.hist(stats['n_defect'], bins=30, color='purple', edgecolor='white', alpha=0.7)
    ax.set_xlabel('Defect Nodes per Sample')
    ax.set_ylabel('Count')
    ax.set_title('Defect Node Count')

    # 品質 (Good vs Bad)
    ax = axes[1, 1]
    good = stats['good'].sum()
    bad = len(stats) - good
    ax.bar(['Good\n(disp+temp)', 'Need\nRe-run'], [good, bad], color=['green', 'orange'], alpha=0.8)
    ax.set_ylabel('Samples')
    ax.set_title(f'Dataset Quality ({good}/{len(stats)} verified)')

    # 欠陥サイズ階層
    ax = axes[1, 2]
    tiers = [
        (0, 50, 'Small', 'lightblue'),
        (50, 100, 'Medium', 'skyblue'),
        (100, 200, 'Large', 'steelblue'),
        (200, 400, 'Critical', 'darkblue'),
    ]
    for lo, hi, name, color in tiers:
        count = ((stats['radius'] >= lo) & (stats['radius'] < hi)).sum()
        ax.bar(name, count, color=color, alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_title('Size Tier Distribution')

    plt.suptitle('Dataset Overview — Feature Distributions', fontsize=16, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_summary_dashboard(data_dir, output_path):
    """進捗サマリダッシュボード（1枚）"""
    samples = collect_dataset_stats(data_dir)
    if not samples:
        print("No samples found")
        return

    stats = pd.DataFrame(samples)
    n_total = len(stats)
    n_good = stats['good'].sum()
    n_defect_total = stats['n_defect'].sum()

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # タイトル
    fig.suptitle('GNN-SHM Dataset Progress — H3 Fairing Debonding', fontsize=18, y=1.02)

    # サマリテキスト
    ax = fig.add_subplot(gs[0, :])
    ax.axis('off')
    text = f"""
    [Dataset Overview]
    - Total samples: {n_total}
    - Verified (disp + temp OK): {n_good} ({100*n_good/n_total:.1f}%)
    - Total defect nodes: {n_defect_total:,}
    - Mesh: ~10,900 nodes/sample (GLOBAL_SEED=50mm)
    - Thermal: 20C init -> 120C outer skin (Step-1)

    [Defect Parameter Range]
    - theta: {stats['theta'].min():.1f} - {stats['theta'].max():.1f} deg
    - Z: {stats['z_center'].min():.0f} - {stats['z_center'].max():.0f} mm
    - Radius: {stats['radius'].min():.1f} - {stats['radius'].max():.1f} mm
    """
    ax.text(0.05, 0.9, text.strip(), transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 品質円グラフ
    ax = fig.add_subplot(gs[1, 0])
    ax.pie([n_good, n_total - n_good], labels=['Verified', 'Pending'], autopct='%1.0f%%',
           colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0.05, 0))

    # 欠陥半径分布
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(stats['radius'], bins=20, color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Radius [mm]')
    ax.set_ylabel('Count')
    ax.set_title('Defect Radius')

    # 欠陥ノード数分布
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(stats['n_defect'], bins=25, color='purple', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Defect Nodes')
    ax.set_ylabel('Count')
    ax.set_title('Defect Node Count')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Dataset progress visualization for meetings')
    parser.add_argument('--data', type=str, default=os.path.join(PROJECT_ROOT, 'dataset_output'))
    parser.add_argument('--output', type=str, default=OUTPUT_DIR)
    parser.add_argument('--samples', type=str, default='1,10,15',
                        help='Comma-separated sample IDs for 3D/2D viz (e.g. 1,10,15)')
    args = parser.parse_args()

    data_dir = args.data
    out_dir = args.output
    sample_ids = [int(s.strip()) for s in args.samples.split(',')]

    print("=" * 60)
    print(" Dataset Progress Visualization")
    print("=" * 60)
    print(f" Data: {data_dir}")
    print(f" Output: {out_dir}")
    print()

    # 1. サマリダッシュボード
    print("[1] Summary dashboard...")
    plot_summary_dashboard(data_dir, os.path.join(out_dir, '01_summary_dashboard.png'))

    # 2. 特徴量分布
    print("[2] Feature distributions...")
    plot_feature_distributions(data_dir, os.path.join(out_dir, '02_feature_distributions.png'))

    # 3. 代表サンプルの 3D / 2D
    print("[3] Sample 3D & unfolded views...")
    for sid in sample_ids:
        sample_dir = os.path.join(data_dir, f'sample_{sid:04d}')
        nodes_path = os.path.join(sample_dir, 'nodes.csv')
        if not os.path.exists(nodes_path):
            print(f"  Skip sample_{sid:04d}: not found")
            continue
        for color_by in ['displacement', 'temp', 'defect']:
            plot_3d_sample(nodes_path,
                           os.path.join(out_dir, f'03_3d_sample{sid:04d}_{color_by}.png'),
                           color_by=color_by, sample_id=sid)
        for color_by in ['displacement', 'defect']:
            plot_unfolded_2d(nodes_path,
                             os.path.join(out_dir, f'04_unfolded_sample{sid:04d}_{color_by}.png'),
                             color_by=color_by, sample_id=sid)

    print()
    print("=" * 60)
    print(f" All figures saved to: {out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
