#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
欠陥データの物理量可視化 — Defect Physics Validation Visualizations

応力・変位・温度の分布、欠陥タイプ別統計、健全 vs 欠陥の比較を可視化。
出力: figures/defect_validation/ および wiki_repo/images/defect_validation/

Usage:
  python scripts/visualize_defect_validation.py --data dataset_multitype_100
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_meta(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df['key'], df['value']))


def load_sample(sample_dir):
    nodes_path = os.path.join(sample_dir, 'nodes.csv')
    if not os.path.exists(nodes_path):
        return None, None
    df = pd.read_csv(nodes_path)
    df['u_mag'] = np.sqrt(df['ux']**2 + df['uy']**2 + df['uz']**2)
    meta = load_meta(os.path.join(sample_dir, 'metadata.csv'))
    return df, meta


def plot_physics_distributions(data_dir, out_dir):
    """応力・変位・温度の分布（全サンプル集約）"""
    data_dir = os.path.join(PROJECT_ROOT, data_dir)
    out_dir = os.path.join(PROJECT_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    samples = sorted([d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('sample_')])[:100]

    all_smises, all_u, all_temp = [], [], []
    defect_smises, defect_u = [], []
    type_stats = {}

    for name in samples:
        df, meta = load_sample(os.path.join(data_dir, name))
        if df is None:
            continue
        dtype = meta.get('defect_type', 'unknown')
        if dtype not in type_stats:
            type_stats[dtype] = {'smises': [], 'u_mag': [], 'n_defect': []}
        type_stats[dtype]['smises'].extend(df['smises'].tolist())
        type_stats[dtype]['u_mag'].extend(df['u_mag'].tolist())
        type_stats[dtype]['n_defect'].append((df['defect_label'] != 0).sum())

        all_smises.extend(df['smises'].tolist())
        all_u.extend(df['u_mag'].tolist())
        all_temp.extend(df['temp'].tolist())
        df_def = df[df['defect_label'] != 0]
        if len(df_def) > 0:
            defect_smises.extend(df_def['smises'].tolist())
            defect_u.extend(df_def['u_mag'].tolist())

    all_smises = np.array(all_smises)
    all_u = np.array(all_u)
    all_temp = np.array(all_temp)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. von Mises stress distribution
    ax = axes[0, 0]
    ax.hist(np.log10(all_smises[all_smises > 1e-12] + 1e-12), bins=60, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('log10(smises + 1e-12) [MPa]')
    ax.set_ylabel('Count')
    ax.set_title('Von Mises Stress Distribution (All Nodes)')
    ax.axvline(np.log10(np.median(all_smises) + 1e-12), color='red', linestyle='--', label='median')
    ax.legend()

    # 2. Displacement magnitude distribution
    ax = axes[0, 1]
    ax.hist(all_u, bins=60, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('|u| [mm]')
    ax.set_ylabel('Count')
    ax.set_title('Displacement Magnitude Distribution')
    ax.axvline(np.median(all_u), color='red', linestyle='--', label='median')
    ax.legend()

    # 3. Temperature distribution
    ax = axes[1, 0]
    ax.hist(all_temp, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax.set_xlabel('Temperature [°C]')
    ax.set_ylabel('Count')
    ax.set_title('Temperature Distribution (Thermal Load 120°C)')
    ax.axvline(120, color='red', linestyle='--', label='expected outer')

    # 4. Defect type vs mean stress
    ax = axes[1, 1]
    types = list(type_stats.keys())
    means = [np.mean(type_stats[t]['smises']) for t in types]
    colors = plt.cm.viridis(np.linspace(0, 1, len(types)))
    bars = ax.bar(types, [m * 1e9 for m in means], color=colors)  # scale for visibility
    ax.set_xlabel('Defect Type')
    ax.set_ylabel('Mean smises × 1e9 [MPa]')
    ax.set_title('Mean Stress by Defect Type')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '01_physics_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_physics_distributions.png")


def plot_defect_contrast(data_dir, out_dir):
    """欠陥ゾーン vs 健全ゾーンの応力・変位比較"""
    data_dir = os.path.join(PROJECT_ROOT, data_dir)
    out_dir = os.path.join(PROJECT_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    samples = sorted([d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('sample_')])[:50]

    healthy_s, defect_s, healthy_u, defect_u = [], [], [], []
    for name in samples:
        df, meta = load_sample(os.path.join(data_dir, name))
        if df is None:
            continue
        n_def = (df['defect_label'] != 0).sum()
        if n_def == 0:
            continue
        df_h = df[df['defect_label'] == 0]
        df_d = df[df['defect_label'] != 0]
        healthy_s.append(df_h['smises'].mean())
        defect_s.append(df_d['smises'].mean())
        healthy_u.append(df_h['u_mag'].mean())
        defect_u.append(df_d['u_mag'].mean())

    if not healthy_s:
        print("No defect samples for contrast plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(healthy_s, defect_s, alpha=0.6, s=30)
    lim = max(max(healthy_s), max(defect_s)) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', label='y=x')
    ax.set_xlabel('Healthy zone mean smises [MPa]')
    ax.set_ylabel('Defect zone mean smises [MPa]')
    ax.set_title('Stress: Healthy vs Defect Zone')
    ax.legend()
    ax.set_aspect('equal')

    ax = axes[1]
    ax.scatter(healthy_u, defect_u, alpha=0.6, s=30)
    lim = max(max(healthy_u), max(defect_u)) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', label='y=x')
    ax.set_xlabel('Healthy zone mean |u| [mm]')
    ax.set_ylabel('Defect zone mean |u| [mm]')
    ax.set_title('Displacement: Healthy vs Defect Zone')
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '02_defect_contrast.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_defect_contrast.png")


def plot_spatial_field(sample_dir, out_dir, sample_id='sample_0000'):
    """1サンプルの展開図（θ-z）で応力・変位・欠陥ラベルを表示"""
    sample_dir = os.path.join(PROJECT_ROOT, sample_dir)
    out_dir = os.path.join(PROJECT_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(sample_dir, sample_id)
    df, meta = load_sample(path)
    if df is None:
        print(f"Sample {sample_id} not found")
        return

    r = np.sqrt(df['x']**2 + df['z']**2)
    theta = np.degrees(np.arctan2(df['z'], df['x']))
    theta = np.where(theta < 0, theta + 360, theta)
    z = df['y'].values  # Abaqus Y = axial
    arc = r * np.radians(theta)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, field, label, cmap in [
        (axes[0, 0], df['smises'].values, 'Von Mises Stress [MPa]', 'hot'),
        (axes[0, 1], df['u_mag'].values, 'Displacement |u| [mm]', 'viridis'),
        (axes[1, 0], df['temp'].values, 'Temperature [°C]', 'coolwarm'),
        (axes[1, 1], df['defect_label'].values, 'Defect Label', 'RdYlGn_r'),
    ]:
        sc = ax.scatter(arc, z, c=field, s=1, cmap=cmap)
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel('Arc length [mm]')
        ax.set_ylabel('z (axial) [mm]')
        ax.set_title(label)
        ax.set_aspect('equal')

    dtype = meta.get('defect_type', 'unknown')
    fig.suptitle(f'{sample_id} | defect_type={dtype} | n_defect={(df["defect_label"]!=0).sum()}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '03_spatial_field_%s.png' % sample_id), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_spatial_field_%s.png" % sample_id)


def plot_correlation_heatmap(data_dir, out_dir):
    """物理量間の相関行列（欠陥サンプルの欠陥ゾーン）"""
    data_dir = os.path.join(PROJECT_ROOT, data_dir)
    out_dir = os.path.join(PROJECT_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    samples = sorted([d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('sample_')])[:80]

    rows = []
    for name in samples:
        df, meta = load_sample(os.path.join(data_dir, name))
        if df is None or (df['defect_label'] != 0).sum() == 0:
            continue
        df_d = df[df['defect_label'] != 0][['smises', 'u_mag', 'temp']].copy()
        df_d['defect_type'] = meta.get('defect_type', 'unknown')
        rows.append(df_d)

    if not rows:
        print("No defect samples for correlation plot")
        return
    df_all = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    cols = ['smises', 'u_mag', 'temp']
    corr = df_all[cols].corr()
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(['Von Mises\n[MPa]', '|u| [mm]', 'Temp [°C]'])
    ax.set_yticklabels(['Von Mises', '|u|', 'Temp'])
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=12)
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('Correlation Matrix (Defect Zone Nodes)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '05_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 05_correlation_heatmap.png")


def plot_defect_ratio_by_type(data_dir, out_dir):
    """欠陥タイプ別の欠陥ノード比率（ノードレベル）"""
    data_dir = os.path.join(PROJECT_ROOT, data_dir)
    out_dir = os.path.join(PROJECT_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    samples = sorted([d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('sample_')])

    rows = []
    for name in samples:
        df, meta = load_sample(os.path.join(data_dir, name))
        if df is None:
            continue
        n_def = (df['defect_label'] != 0).sum()
        ratio = n_def / len(df) * 100 if len(df) > 0 else 0
        rows.append({'defect_type': meta.get('defect_type', 'unknown'), 'defect_ratio_pct': ratio})

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    df_all.boxplot(column='defect_ratio_pct', by='defect_type', ax=ax)
    ax.set_ylabel('Defect Node Ratio [%]')
    ax.set_xlabel('Defect Type')
    ax.set_title('Defect Node Ratio by Type (Node-level Imbalance)')
    plt.suptitle('')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '06_defect_ratio_by_type.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 06_defect_ratio_by_type.png")


def plot_defect_type_stats(data_dir, out_dir):
    """欠陥タイプ別の統計（ノード数、応力、変位）"""
    data_dir = os.path.join(PROJECT_ROOT, data_dir)
    out_dir = os.path.join(PROJECT_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    samples = sorted([d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('sample_')])

    rows = []
    for name in samples:
        df, meta = load_sample(os.path.join(data_dir, name))
        if df is None:
            continue
        dtype = meta.get('defect_type', 'unknown')
        n_def = (df['defect_label'] != 0).sum()
        rows.append({
            'sample': name,
            'defect_type': dtype,
            'n_defect': n_def,
            'smises_mean': df['smises'].mean(),
            'u_mag_mean': df['u_mag'].mean(),
            'temp_mean': df['temp'].mean(),
        })

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        print("No data for type stats")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # n_defect by type
    ax = axes[0, 0]
    by_type = df_all.groupby('defect_type')['n_defect'].agg(['mean', 'std', 'count'])
    by_type.plot(kind='bar', y='mean', yerr='std', ax=ax, capsize=3)
    ax.set_ylabel('Mean n_defect_nodes')
    ax.set_title('Defect Nodes per Type')
    ax.tick_params(axis='x', rotation=45)

    # smises by type
    ax = axes[0, 1]
    df_all.boxplot(column='smises_mean', by='defect_type', ax=ax)
    ax.set_ylabel('Mean smises [MPa]')
    ax.set_title('Stress by Defect Type')
    plt.suptitle('')

    # u_mag by type
    ax = axes[1, 0]
    df_all.boxplot(column='u_mag_mean', by='defect_type', ax=ax)
    ax.set_ylabel('Mean |u| [mm]')
    ax.set_title('Displacement by Defect Type')
    plt.suptitle('')

    # Sample count by type
    ax = axes[1, 1]
    counts = df_all['defect_type'].value_counts()
    counts.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_ylabel('Count')
    ax.set_title('Samples per Defect Type')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '04_defect_type_stats.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 04_defect_type_stats.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset_multitype_100')
    parser.add_argument('--output', type=str, default='figures/defect_validation')
    parser.add_argument('--wiki', action='store_true', help='Also copy to wiki_repo/images/')
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(out_dir, exist_ok=True)

    print("Generating defect validation visualizations...")
    plot_physics_distributions(args.data, out_dir)
    plot_defect_contrast(args.data, out_dir)
    plot_spatial_field(args.data, out_dir, 'sample_0000')
    plot_spatial_field(args.data, out_dir, 'sample_0010')
    plot_correlation_heatmap(args.data, out_dir)
    plot_defect_ratio_by_type(args.data, out_dir)
    plot_defect_type_stats(args.data, out_dir)

    if args.wiki:
        wiki_dir = os.path.join(PROJECT_ROOT, 'wiki_repo', 'images', 'defect_validation')
        os.makedirs(wiki_dir, exist_ok=True)
        import shutil
        for f in os.listdir(out_dir):
            if f.endswith('.png'):
                shutil.copy(os.path.join(out_dir, f), os.path.join(wiki_dir, f))
        print("Copied to wiki_repo/images/defect_validation/")


if __name__ == '__main__':
    main()
