#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare fairing separation results: Normal vs Stuck bolts.

Reads CSV time-history and energy files from results/separation/
and generates comparison plots.

Usage:
    python scripts/plot_separation_comparison.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'separation')
OUTPUT_DIR = RESULT_DIR


def load_time_histories():
    """Load all time history CSVs."""
    cases = {}
    for csv_path in sorted(glob.glob(os.path.join(RESULT_DIR, '*_time_history.csv'))):
        job_name = os.path.basename(csv_path).replace('_time_history.csv', '')
        df = pd.read_csv(csv_path)
        cases[job_name] = df
        print(f"  Loaded {job_name}: {len(df)} rows, instances: {df['instance'].unique()}")
    return cases


def load_energies():
    """Load all energy CSVs."""
    cases = {}
    for csv_path in sorted(glob.glob(os.path.join(RESULT_DIR, '*_energy.csv'))):
        job_name = os.path.basename(csv_path).replace('_energy.csv', '')
        df = pd.read_csv(csv_path)
        cases[job_name] = df
        print(f"  Loaded {job_name} energy: {len(df)} rows")
    return cases


def plot_displacement_comparison(cases):
    """Plot max displacement vs time for each case."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Overall max displacement (all instances combined)
    ax = axes[0, 0]
    for job_name, df in cases.items():
        grouped = df.groupby('time_s')['u_mag_max_mm'].max().reset_index()
        label = job_name.replace('Sep-', '')
        ax.plot(grouped['time_s'] * 1000, grouped['u_mag_max_mm'],
                label=label, linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Max Displacement (mm)')
    ax.set_title('(a) Max Displacement — All Instances')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q1-InnerSkin displacement
    ax = axes[0, 1]
    for job_name, df in cases.items():
        df_q1 = df[df['instance'].str.contains('Q1-InnerSkin', case=False)]
        if len(df_q1) == 0:
            continue
        label = job_name.replace('Sep-', '')
        ax.plot(df_q1['time_s'] * 1000, df_q1['u_mag_max_mm'],
                label=label, linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Max Displacement (mm)')
    ax.set_title('(b) Q1-InnerSkin Displacement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Max velocity
    ax = axes[1, 0]
    for job_name, df in cases.items():
        grouped = df.groupby('time_s')['v_mag_max_mm_s'].max().reset_index()
        label = job_name.replace('Sep-', '')
        ax.plot(grouped['time_s'] * 1000, grouped['v_mag_max_mm_s'] / 1000.0,
                label=label, linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Max Velocity (m/s)')
    ax.set_title('(c) Max Velocity — All Instances')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Max Mises stress
    ax = axes[1, 1]
    for job_name, df in cases.items():
        grouped = df.groupby('time_s')['s_mises_max_MPa'].max().reset_index()
        label = job_name.replace('Sep-', '')
        ax.plot(grouped['time_s'] * 1000, grouped['s_mises_max_MPa'],
                label=label, linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Max von Mises Stress (MPa)')
    ax.set_title('(d) Max Stress — All Instances')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Fairing Separation: Normal vs Stuck Bolts', fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'separation_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def plot_energy_comparison(energies):
    """Plot energy history comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (var, label, title) in enumerate([
        ('ALLKE_mJ', 'Kinetic Energy (mJ)', '(a) Kinetic Energy'),
        ('ALLSE_mJ', 'Strain Energy (mJ)', '(b) Strain Energy'),
        ('ETOTAL_mJ', 'Total Energy (mJ)', '(c) Total Energy'),
    ]):
        ax = axes[i]
        for job_name, df in energies.items():
            name = job_name.replace('Sep-', '')
            ax.plot(df['time_s'] * 1000, df[var], label=name, linewidth=1.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Energy History: Normal vs Stuck Bolts', fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'separation_energy.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def plot_instance_breakdown(cases):
    """Plot displacement per instance for each case."""
    # Get all unique instances
    all_instances = set()
    for df in cases.values():
        all_instances.update(df['instance'].unique())
    skin_instances = sorted([i for i in all_instances if 'Skin' in i or 'Core' in i])

    if not skin_instances:
        print("  No skin/core instances found, skipping breakdown")
        return

    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, 1, figsize=(12, 4 * n_cases), squeeze=False)

    for i, (job_name, df) in enumerate(cases.items()):
        ax = axes[i, 0]
        for inst in skin_instances:
            df_inst = df[df['instance'] == inst]
            if len(df_inst) == 0:
                continue
            short_name = inst.replace('Q1-', 'Q1 ').replace('Q2-', 'Q2 ')
            ax.plot(df_inst['time_s'] * 1000, df_inst['u_mag_max_mm'],
                    label=short_name, linewidth=1.0, alpha=0.8)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Max Displacement (mm)')
        ax.set_title(f'{job_name} — Per-Instance Displacement')
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'separation_per_instance.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def print_summary(cases):
    """Print summary statistics for comparison."""
    print("\n" + "=" * 70)
    print("SEPARATION COMPARISON SUMMARY")
    print("=" * 70)

    for job_name, df in cases.items():
        sep_df = df[df['step'] == 'Step-Separation']
        if len(sep_df) == 0:
            continue
        print(f"\n{job_name}:")
        print(f"  Max displacement: {sep_df['u_mag_max_mm'].max():.2f} mm")
        print(f"  Max velocity:     {sep_df['v_mag_max_mm_s'].max() / 1000:.2f} m/s")
        print(f"  Max Mises stress: {sep_df['s_mises_max_MPa'].max():.1f} MPa")

        # Final frame displacement per Q1/Q2
        t_max = sep_df['time_s'].max()
        final = sep_df[sep_df['time_s'] == t_max]
        for inst in ['Q1-InnerSkin', 'Q2-InnerSkin']:
            inst_data = final[final['instance'] == inst]
            if len(inst_data) > 0:
                u = inst_data['u_mag_max_mm'].values[0]
                print(f"  Final {inst}: {u:.2f} mm")


def main():
    print("Loading time history data...")
    cases = load_time_histories()
    if not cases:
        print("ERROR: No time history CSVs found in %s" % RESULT_DIR)
        sys.exit(1)

    print("\nLoading energy data...")
    energies = load_energies()

    print("\nPlotting displacement comparison...")
    plot_displacement_comparison(cases)

    if energies:
        print("\nPlotting energy comparison...")
        plot_energy_comparison(energies)

    print("\nPlotting per-instance breakdown...")
    plot_instance_breakdown(cases)

    print_summary(cases)
    print("\nDone!")


if __name__ == '__main__':
    main()
