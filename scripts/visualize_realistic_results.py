#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize realistic fairing FEM results: Phase 1 vs Phase 2.
Generates comparison plots of stress, displacement, and temperature fields.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE1_DIR = os.path.join(PROJECT_ROOT, 'dataset_realistic', 'phase1')
PHASE2_DIR = os.path.join(PROJECT_ROOT, 'dataset_realistic', 'phase2')
FIG_DIR = os.path.join(PROJECT_ROOT, 'figures', 'realistic_fairing')
os.makedirs(FIG_DIR, exist_ok=True)


def load_nodes(path):
    df = pd.read_csv(os.path.join(path, 'nodes.csv'))
    df['r'] = np.sqrt(df['x']**2 + df['z']**2)
    df['theta_deg'] = np.degrees(np.arctan2(df['z'], df['x']))
    df['u_mag'] = np.sqrt(df['ux']**2 + df['uy']**2 + df['uz']**2)
    return df


def plot_field_comparison(df1, df2, field, label, cmap='viridis', vmin=None, vmax=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, df, title in [(axes[0], df1, 'Phase 1 (AccessDoor + 6 Frames)'),
                           (axes[1], df2, 'Phase 2 (5 Openings + 4 Frames)')]:
        sc = ax.scatter(df['theta_deg'], df['y'], c=df[field], s=0.3,
                        cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_xlabel('Theta (deg)')
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 10500)
        plt.colorbar(sc, ax=ax, label=label, shrink=0.8)
    axes[0].set_ylabel('Axial Position Y (mm)')
    fig.suptitle('H3 Realistic Fairing: %s' % label, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_stress_histograms(df1, df2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df, title in [(axes[0], df1, 'Phase 1'), (axes[1], df2, 'Phase 2')]:
        s = df['smises']
        ax.hist(s, bins=100, range=(0, min(s.quantile(0.99), 200)),
                color='steelblue', edgecolor='none', alpha=0.8)
        ax.axvline(s.median(), color='red', ls='--', label='Median: %.2f MPa' % s.median())
        ax.axvline(s.quantile(0.95), color='orange', ls='--',
                   label='95th pct: %.2f MPa' % s.quantile(0.95))
        ax.set_xlabel('von Mises Stress (MPa)')
        ax.set_ylabel('Node Count')
        ax.set_title(title)
        ax.legend(fontsize=9)
    fig.suptitle('von Mises Stress Distribution', fontsize=13)
    fig.tight_layout()
    return fig


def plot_displacement_profile(df1, df2):
    fig, ax = plt.subplots(figsize=(10, 6))
    for df, label, color in [(df1, 'Phase 1', 'steelblue'), (df2, 'Phase 2', 'coral')]:
        bins = np.linspace(0, 10500, 60)
        centers = 0.5 * (bins[:-1] + bins[1:])
        groups = df.groupby(pd.cut(df['y'], bins))
        means = groups['u_mag'].mean().values
        maxs = groups['u_mag'].max().values
        ax.plot(centers, means, '-', color=color, label='%s (mean)' % label)
        ax.fill_between(centers, means, maxs, alpha=0.15, color=color, label='%s (max)' % label)
    ax.set_xlabel('Axial Position Y (mm)')
    ax.set_ylabel('Displacement Magnitude (mm)')
    ax.set_title('Displacement Profile Along Fairing Axis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_opening_detail(df, opening_name, theta_c, z_c, r_half, field='smises'):
    margin = r_half * 1.5
    mask = ((df['theta_deg'] > theta_c - 15) & (df['theta_deg'] < theta_c + 15) &
            (df['y'] > z_c - margin) & (df['y'] < z_c + margin))
    sub = df[mask]
    if len(sub) < 10:
        return None
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(sub['theta_deg'], sub['y'], c=sub[field], s=2, cmap='hot', rasterized=True)
    plt.colorbar(sc, ax=ax, label='von Mises Stress (MPa)')
    ax.set_xlabel('Theta (deg)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('%s Detail: Stress Concentration' % opening_name)
    theta_extent = np.degrees(r_half / 2638.0)
    rect = plt.Rectangle((theta_c - theta_extent, z_c - r_half),
                          2 * theta_extent, 2 * r_half,
                          fill=False, edgecolor='lime', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def print_summary(df, label):
    print("\n=== %s ===" % label)
    print("  Nodes: %d" % len(df))
    print("  von Mises: min=%.3f, median=%.3f, mean=%.3f, 95pct=%.3f, max=%.3f MPa" % (
        df['smises'].min(), df['smises'].median(), df['smises'].mean(),
        df['smises'].quantile(0.95), df['smises'].max()))
    print("  Disp mag: min=%.4f, mean=%.4f, max=%.4f mm" % (
        df['u_mag'].min(), df['u_mag'].mean(), df['u_mag'].max()))
    print("  Temp: min=%.1f, max=%.1f C" % (df['temp'].min(), df['temp'].max()))


def main():
    print("Loading Phase 1: %s" % PHASE1_DIR)
    df1 = load_nodes(PHASE1_DIR)
    print("Loading Phase 2: %s" % PHASE2_DIR)
    df2 = load_nodes(PHASE2_DIR)

    print_summary(df1, 'Phase 1 (AccessDoor + 6 Frames)')
    print_summary(df2, 'Phase 2 (5 Openings + 4 Frames)')

    vmax_s = max(df1['smises'].quantile(0.98), df2['smises'].quantile(0.98))

    fig = plot_field_comparison(df1, df2, 'smises', 'von Mises Stress (MPa)', cmap='hot', vmin=0, vmax=vmax_s)
    fig.savefig(os.path.join(FIG_DIR, 'stress_comparison.png'), dpi=150, bbox_inches='tight')
    print("Saved: stress_comparison.png")

    fig = plot_field_comparison(df1, df2, 'u_mag', 'Displacement (mm)', cmap='viridis', vmin=0)
    fig.savefig(os.path.join(FIG_DIR, 'displacement_comparison.png'), dpi=150, bbox_inches='tight')
    print("Saved: displacement_comparison.png")

    fig = plot_field_comparison(df1, df2, 'temp', 'Temperature (C)', cmap='coolwarm')
    fig.savefig(os.path.join(FIG_DIR, 'temperature_comparison.png'), dpi=150, bbox_inches='tight')
    print("Saved: temperature_comparison.png")

    fig = plot_stress_histograms(df1, df2)
    fig.savefig(os.path.join(FIG_DIR, 'stress_histogram.png'), dpi=150, bbox_inches='tight')
    print("Saved: stress_histogram.png")

    fig = plot_displacement_profile(df1, df2)
    fig.savefig(os.path.join(FIG_DIR, 'displacement_profile.png'), dpi=150, bbox_inches='tight')
    print("Saved: displacement_profile.png")

    openings = [('AccessDoor', 30.0, 1500.0, 650.0), ('HVAC_Door', 20.0, 2500.0, 200.0), ('RF_Window', 40.0, 4000.0, 200.0)]
    for name, theta, z, r in openings:
        fig = plot_opening_detail(df2, name, theta, z, r)
        if fig:
            fig.savefig(os.path.join(FIG_DIR, 'detail_%s.png' % name), dpi=150, bbox_inches='tight')
            print("Saved: detail_%s.png" % name)

    fig = plot_opening_detail(df1, 'AccessDoor_P1', 30.0, 1500.0, 650.0)
    if fig:
        fig.savefig(os.path.join(FIG_DIR, 'detail_AccessDoor_P1.png'), dpi=150, bbox_inches='tight')
        print("Saved: detail_AccessDoor_P1.png")

    plt.close('all')
    print("\nAll figures saved to: %s" % FIG_DIR)


if __name__ == '__main__':
    main()
