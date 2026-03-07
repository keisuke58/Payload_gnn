#!/usr/bin/env python3
"""
Visualize CompDam_DGD damage results and compare with project empirical factors.

Reads: abaqus_work/compdam_flatplate/compdam_stiffness_reduction.json
       abaqus_work/compdam_flatplate/compdam_damage_results.csv
Writes: figures/compdam_validation/

Usage:
    python scripts/visualize_compdam_results.py
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project empirical stiffness reduction factors
PROJECT_FACTORS = {
    'debonding':     {'E1': 0.01, 'E2': 0.01, 'G12': 0.01},
    'impact':        {'E1': 0.70, 'E2': 0.30, 'G12': 0.30},
    'delamination':  {'E1': 0.90, 'E2': 0.50, 'G12': 0.50},
    'thermal':       {'E1': 0.05, 'E2': 0.05, 'G12': 0.05},
    'acoustic_fat':  {'E1': 0.50, 'E2': 0.35, 'G12': 0.35},
}

# Model parameters (for coordinate reconstruction)
PLATE_LX = 100.0
PLATE_LY = 100.0
ELEM_XY = 2.0
NX = int(PLATE_LX / ELEM_XY)
NY = int(PLATE_LY / ELEM_XY)
ELEMS_PER_LAYER = NX * NY


def load_results(base_dir):
    """Load CSV and JSON results."""
    csv_path = base_dir / 'compdam_damage_results.csv'
    json_path = base_dir / 'compdam_stiffness_reduction.json'

    data = []
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = {}
                for k, v in row.items():
                    try:
                        d[k] = float(v)
                    except (ValueError, TypeError):
                        d[k] = v
                data.append(d)

    summary = {}
    if json_path.exists():
        with open(json_path) as f:
            summary = json.load(f)

    return data, summary


def plot_damage_map(data, outdir):
    """Fig C1: Damage distribution map (matrix + fiber) per ply."""

    # Group by ply layer
    n_plies = 8
    n_core = 5
    n_layers = 2 * n_plies + n_core

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('CompDam Damage Distribution per Ply', fontsize=14, y=0.98)

    ply_labels = [
        'Inner 45°', 'Inner 0°', 'Inner -45°', 'Inner 90°',
        'Inner 90°', 'Inner -45°', 'Inner 0°', 'Inner 45°',
        'Outer 45°', 'Outer 0°', 'Outer -45°', 'Outer 90°',
        'Outer 90°', 'Outer -45°', 'Outer 0°', 'Outer 45°',
    ]

    ply_layer_indices = list(range(n_plies)) + list(range(n_plies + n_core, n_layers))

    for idx, (layer_idx, label) in enumerate(zip(ply_layer_indices, ply_labels)):
        ax = axes[idx // 4, idx % 4]

        # Extract d2 for this layer
        d2_map = np.zeros((NY, NX))
        for d in data:
            elem = int(d.get('elem', 0))
            local_eid = elem - layer_idx * ELEMS_PER_LAYER
            if 1 <= local_eid <= ELEMS_PER_LAYER:
                iy = (local_eid - 1) // NX
                ix = (local_eid - 1) % NX
                d2_map[iy, ix] = d.get('CDM_d2', 0.0)

        im = ax.imshow(d2_map, extent=[0, PLATE_LX, 0, PLATE_LY],
                       origin='lower', cmap='hot_r', vmin=0, vmax=1,
                       aspect='equal')
        ax.set_title(f'Ply {idx+1}: {label}', fontsize=9)
        ax.set_xlabel('X [mm]', fontsize=8)
        ax.set_ylabel('Y [mm]', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.colorbar(im, ax=axes, label='CDM_d2 (Matrix Damage)', shrink=0.6)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    path = outdir / '01_damage_map_per_ply.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fiber_damage(data, outdir):
    """Fig C2: Fiber damage (d1T, d1C) on outer skin."""

    n_plies = 8
    n_core = 5

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('CompDam Fiber Damage - Outer Skin', fontsize=14)

    outer_layers = list(range(n_plies + n_core, 2 * n_plies + n_core))
    labels = ['45°', '0°', '-45°', '90°', '90°', '-45°', '0°', '45°']

    for idx, (layer_idx, label) in enumerate(zip(outer_layers, labels)):
        row = 0 if idx < 4 else 1
        col = idx % 4
        ax = axes[row, col]

        d1T_map = np.zeros((NY, NX))
        d1C_map = np.zeros((NY, NX))

        for d in data:
            elem = int(d.get('elem', 0))
            local_eid = elem - layer_idx * ELEMS_PER_LAYER
            if 1 <= local_eid <= ELEMS_PER_LAYER:
                iy = (local_eid - 1) // NX
                ix = (local_eid - 1) % NX
                d1T_map[iy, ix] = d.get('CDM_d1T', 0.0)
                d1C_map[iy, ix] = d.get('CDM_d1C', 0.0)

        # Combined fiber damage
        combined = np.maximum(d1T_map, d1C_map)
        im = ax.imshow(combined, extent=[0, PLATE_LX, 0, PLATE_LY],
                       origin='lower', cmap='YlOrRd', vmin=0, vmax=1,
                       aspect='equal')
        ax.set_title(f'Outer Ply {idx+1}: {label}', fontsize=9)
        ax.set_xlabel('X [mm]', fontsize=8)
        ax.set_ylabel('Y [mm]', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.colorbar(im, ax=axes, label='max(CDM_d1T, CDM_d1C)', shrink=0.6)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    path = outdir / '02_fiber_damage_outer_skin.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_stiffness_comparison(summary, outdir):
    """Fig C3: CompDam vs project empirical stiffness reduction factors."""

    if 'global' not in summary:
        print("  No global summary data, skipping stiffness comparison")
        return

    g = summary['global']

    fig, ax = plt.subplots(figsize=(10, 6))

    # CompDam results
    compdam = {
        'E1': g['E1_eff_ratio'],
        'E2': g['E2_eff_ratio'],
        'G12': g['G12_eff_ratio'],
    }

    props = ['E1', 'E2', 'G12']
    x = np.arange(len(props))
    width = 0.12

    # CompDam bar
    compdam_vals = [compdam[p] for p in props]
    ax.bar(x - 2.5*width, compdam_vals, width, label='CompDam (this study)',
           color='#2196F3', edgecolor='black', linewidth=0.5)

    # Project empirical factors
    colors = ['#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#795548']
    for i, (defect, factors) in enumerate(PROJECT_FACTORS.items()):
        vals = [factors[p] for p in props]
        ax.bar(x + (i - 1.5)*width, vals, width, label=f'Project: {defect}',
               color=colors[i], edgecolor='black', linewidth=0.5, alpha=0.8)

    ax.set_xlabel('Property', fontsize=12)
    ax.set_ylabel('Effective Stiffness Ratio (damaged/undamaged)', fontsize=12)
    ax.set_title('CompDam Progressive Damage vs Project Empirical Factors', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(props, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = outdir / '03_stiffness_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ply_damage_profile(summary, outdir):
    """Fig C4: Through-thickness damage profile."""

    if 'plies' not in summary:
        print("  No ply data, skipping through-thickness profile")
        return

    plies = summary['plies']

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Through-Thickness Damage Profile', fontsize=14)

    ply_names = sorted(plies.keys(), key=lambda k: plies[k]['layer_idx'])
    ply_positions = list(range(len(ply_names)))

    for ax, (var, label, color) in zip(axes, [
        ('d2_max', 'Matrix Damage (CDM_d2)', '#F44336'),
        ('d1T_max', 'Fiber Tension (CDM_d1T)', '#2196F3'),
        ('d1C_max', 'Fiber Compression (CDM_d1C)', '#4CAF50'),
    ]):
        vals = [plies[name].get(var, 0) for name in ply_names]
        ax.barh(ply_positions, vals, color=color, edgecolor='black', linewidth=0.5)
        ax.set_yticks(ply_positions)
        ax.set_yticklabels(ply_names, fontsize=8)
        ax.set_xlabel(f'Max {label}', fontsize=10)
        ax.set_xlim(0, max(max(vals) * 1.2, 0.01))
        ax.set_title(label, fontsize=11)
        ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    path = outdir / '04_through_thickness_damage.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    base_dir = Path('abaqus_work/compdam_flatplate')
    outdir = Path('figures/compdam_validation')
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading CompDam results...")
    data, summary = load_results(base_dir)

    if not data:
        print("ERROR: No data found. Run extract_compdam_results.py first.")
        return

    print(f"  Loaded {len(data)} element records")

    print("\nGenerating figures...")
    plot_damage_map(data, outdir)
    plot_fiber_damage(data, outdir)
    plot_stiffness_comparison(summary, outdir)
    plot_ply_damage_profile(summary, outdir)

    print("\nDone!")


if __name__ == '__main__':
    main()
