#!/usr/bin/env python3
"""Ground Truth FEM 3D可視化 — 30°セクターフェアリングの物理場分布 + GT vs Baseline ヒストグラム比較."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_CSV = os.path.join(BASE_DIR, 'dataset_gt', 'sample_healthy', 'nodes.csv')
BASELINE_CSV = os.path.join(BASE_DIR, 'dataset_realistic_25mm_100', 'sample_0000', 'nodes.csv')
OUT_DIR = os.path.join(BASE_DIR, 'wiki_repo', 'images')

os.makedirs(OUT_DIR, exist_ok=True)


def load_csv(path):
    print("  Loading %s ..." % path)
    df = pd.read_csv(path)
    print("    -> %d nodes" % len(df))
    return df


def plot_3d_field(df, field, cmap, unit, out_path,
                  elev=20, azim=-60, point_size=0.8):
    """3D scatter: Abaqus Y=axial -> matplotlib Z(vertical)."""
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    x = df['x'].values
    y_axial = df['y'].values
    z = df['z'].values
    vals = df[field].values

    vmin, vmax = np.nanmin(vals), np.nanmax(vals)

    sc = ax.scatter(x, z, y_axial,
                    c=vals, cmap=cmap, s=point_size,
                    vmin=vmin, vmax=vmax,
                    edgecolors='none', alpha=0.9)

    ax.set_xlabel('X [mm]', fontsize=9)
    ax.set_ylabel('Z [mm]', fontsize=9)
    ax.set_zlabel('Y (axial) [mm]', fontsize=9)
    ax.view_init(elev=elev, azim=azim)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('%s [%s]' % (field, unit), fontsize=10)

    ax.set_title('%s  (range: %.4g ~ %.4g %s)' % (field, vmin, vmax, unit),
                 fontsize=12, fontweight='bold')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: %s" % out_path)


def plot_comparison(gt_df, bl_df, out_path):
    """GT vs Baseline histogram comparison (2x2)."""
    fields = [
        ('ux', 'mm', 'Displacement ux'),
        ('temp', '\u00b0C', 'Temperature'),
        ('smises', 'MPa', 'von Mises Stress'),
        ('thermal_smises', 'MPa', 'Thermal von Mises'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)

    for ax, (col, unit, label) in zip(axes.flat, fields):
        gt_vals = gt_df[col].values
        bl_vals = bl_df[col].values

        lo = min(np.nanmin(gt_vals), np.nanmin(bl_vals))
        hi = max(np.nanmax(gt_vals), np.nanmax(bl_vals))
        margin = (hi - lo) * 0.05 if hi != lo else 1.0
        bins = np.linspace(lo - margin, hi + margin, 80)

        ax.hist(bl_vals, bins=bins, alpha=0.6, color='tab:orange',
                label='Baseline (N=%d)' % len(bl_vals), density=True)
        ax.hist(gt_vals, bins=bins, alpha=0.6, color='tab:blue',
                label='GT (N=%d)' % len(gt_vals), density=True)

        ax.set_xlabel('%s [%s]' % (label, unit), fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    fig.suptitle('Ground Truth vs Baseline Comparison', fontsize=13,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: %s" % out_path)


def main():
    print("=" * 60)
    print("Ground Truth 3D Visualization")
    print("=" * 60)

    if not os.path.exists(GT_CSV):
        print("ERROR: GT CSV not found: %s" % GT_CSV)
        sys.exit(1)

    gt_df = load_csv(GT_CSV)

    plots = [
        ('u_mag',          'jet',      'mm',  'gt_3d_displacement.png'),
        ('temp',           'coolwarm', '\u00b0C',  'gt_3d_temperature.png'),
        ('smises',         'jet',      'MPa', 'gt_3d_smises.png'),
        ('thermal_smises', 'coolwarm', 'MPa', 'gt_3d_thermal_smises.png'),
    ]

    print("\n--- 3D scatter plots ---")
    for field, cmap, unit, fname in plots:
        out_path = os.path.join(OUT_DIR, fname)
        plot_3d_field(gt_df, field, cmap, unit, out_path)

    print("\n--- GT vs Baseline comparison ---")
    if os.path.exists(BASELINE_CSV):
        bl_df = load_csv(BASELINE_CSV)
        out_path = os.path.join(OUT_DIR, 'gt_vs_baseline_comparison.png')
        plot_comparison(gt_df, bl_df, out_path)
    else:
        print("WARNING: Baseline CSV not found, skipping comparison.")

    print("\n" + "=" * 60)
    print("Done! Generated images:")
    for _, _, _, fname in plots:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print("  %s  (%.0f KB)" % (fpath, size_kb))
    comp_path = os.path.join(OUT_DIR, 'gt_vs_baseline_comparison.png')
    if os.path.exists(comp_path):
        size_kb = os.path.getsize(comp_path) / 1024
        print("  %s  (%.0f KB)" % (comp_path, size_kb))


if __name__ == '__main__':
    main()
