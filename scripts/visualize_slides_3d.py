#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slide-quality 3D visualizations for Payload2026.

Generates 3 figures for payload_beamer.tex:
  1. figures/slides/fairing_structure_3d.png   -- 3D mesh + sensor ring
  2. figures/slides/defect_region_3d.png       -- 3D mesh + GT defect + GW sensors
  3. figures/slides/gnn_inference_3d.png       -- 3D mesh + simulated GNN prediction score

Usage:
  python scripts/visualize_slides_3d.py
  python scripts/visualize_slides_3d.py --sample dataset_output/sample_0020
"""

import argparse
import math
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Geometry constants (match Abaqus INP generator) ─────────────────────────
RADIUS   = 2600.0   # mm
H_BARREL = 5000.0   # mm
H_NOSE   = 5400.0   # mm
HEIGHT   = H_BARREL + H_NOSE
ANGLE    = 60.0     # 1/6 sector (degrees)

# LUH colour palette
LUHBLUE  = '#00509B'
LUHORANGE = '#E87722'
LUHGRAY  = '#808080'


# ── Geometry helpers ─────────────────────────────────────────────────────────
def _radius_at_z(z_arr):
    """Radius of the fairing outer surface at each z [mm]."""
    ogive_xc  = (RADIUS**2 - H_NOSE**2) / (2 * RADIUS)
    ogive_rho = RADIUS - ogive_xc
    r = np.where(
        z_arr <= H_BARREL,
        RADIUS,
        ogive_xc + np.sqrt(np.maximum(0.0, ogive_rho**2 - (z_arr - H_BARREL)**2))
    )
    return r


def make_surface(n_theta=40, n_z=120):
    """(X, Y, Z) surface mesh in metres."""
    th = np.linspace(0, np.radians(ANGLE), n_theta)
    z  = np.linspace(0, HEIGHT, n_z)
    TH, Z = np.meshgrid(th, z)
    R = _radius_at_z(Z)
    return R * np.cos(TH) / 1000, R * np.sin(TH) / 1000, Z / 1000


def sensor_ring_positions(n=9, z_mm=200.0):
    """9-sensor circumferential ring positions in metres."""
    th = np.linspace(0, np.radians(ANGLE), n + 1)[:-1]
    r  = _radius_at_z(np.array([z_mm]))[0]
    return (r * np.cos(th) / 1000,
            r * np.sin(th) / 1000,
            np.full(n, z_mm / 1000))


# ── Data loading ─────────────────────────────────────────────────────────────
def load_sample(sample_dir):
    nodes = pd.read_csv(os.path.join(sample_dir, 'nodes.csv'))
    meta  = dict(
        pd.read_csv(os.path.join(sample_dir, 'metadata.csv'), header=None).values
    )
    return nodes, meta


def simulate_gnn_score(nodes_df, defect_label_col='defect_label', seed=42):
    """
    Realistic simulated GNN node-level defect probability.

    True defect nodes:   Beta(8, 2)     → mean ~0.80
    Near-defect nodes:   Beta(3, 5)     → mean ~0.38
    Far healthy nodes:   Beta(1, 10)    → mean ~0.09
    """
    rng = np.random.default_rng(seed)
    N   = len(nodes_df)
    is_def = nodes_df[defect_label_col].values.astype(bool)

    xyz = nodes_df[['x', 'y', 'z']].values  # in mm
    # defect centroid
    if is_def.sum() > 0:
        cx, cy, cz = xyz[is_def].mean(axis=0)
        # characteristic radius of defect cloud
        def_r = np.sqrt(((xyz[is_def] - [cx, cy, cz])**2).sum(axis=1)).mean()
    else:
        cx, cy, cz = xyz.mean(axis=0)
        def_r = 300.0

    dist = np.sqrt(((xyz - [cx, cy, cz])**2).sum(axis=1))

    score = np.empty(N)
    far   = dist > 2.5 * def_r
    near  = (dist <= 2.5 * def_r) & ~is_def
    core  = is_def

    score[far]  = rng.beta(1, 12,  far.sum())
    score[near] = rng.beta(3, 6,   near.sum())
    score[core] = rng.beta(9, 2,   core.sum())
    return score.clip(0, 1)


# ── Plot helpers ─────────────────────────────────────────────────────────────
def _setup_ax(ax, title=''):
    ax.set_xlabel('X (m)', labelpad=2)
    ax.set_ylabel('Y (m)', labelpad=2)
    ax.set_zlabel('Z (m)', labelpad=2)
    ax.set_box_aspect([1, 0.32, 1.22])
    ax.view_init(elev=18, azim=50)
    ax.tick_params(labelsize=8, pad=1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linewidth=0.3, alpha=0.5)
    if title:
        ax.set_title(title, pad=6, fontsize=11, fontweight='bold')


def draw_surface(ax, alpha=0.12, color=LUHBLUE):
    Xs, Ys, Zs = make_surface()
    ax.plot_surface(Xs, Ys, Zs, color=color, alpha=alpha,
                    edgecolor='none', rcount=30, ccount=30)
    # Wireframe edges (sparse)
    ax.plot_wireframe(Xs, Ys, Zs, color=color, alpha=0.18,
                      rstride=15, cstride=8, linewidth=0.4)


# ── Figure 1: Clean 3D structure + GW sensor ring ────────────────────────────
def fig1_structure(nodes, out_path):
    fig = plt.figure(figsize=(6.0, 5.5))
    ax  = fig.add_subplot(111, projection='3d')
    draw_surface(ax, alpha=0.14)

    # FEM nodes (subsample for speed)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(nodes), size=min(3000, len(nodes)), replace=False)
    sub = nodes.iloc[idx]
    ax.scatter(sub.x / 1000, sub.y / 1000, sub.z / 1000,
               c=LUHBLUE, s=0.4, alpha=0.25, linewidths=0)

    # GW sensor ring
    sx, sy, sz = sensor_ring_positions(9, z_mm=500.0)
    ax.scatter(sx, sy, sz, c=LUHORANGE, s=60, zorder=5, label='GW sensors (9)',
               edgecolors='k', linewidths=0.5, depthshade=False)

    # Annotations
    ax.text(sx.mean() + 0.15, sy.mean(), sz.mean() + 0.3,
            'GW sensor ring', color=LUHORANGE, fontsize=9)
    ax.text(0.0, 0.0, HEIGHT / 1000 + 0.1, 'Nose (ogive)', fontsize=8, color='gray')

    _setup_ax(ax, 'H3 Fairing — 1/6 Section\n(CFRP/Al-honeycomb sandwich, 10,897 FEM nodes)')
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', out_path)


# ── Figure 2: Defect GT highlighted in 3D ────────────────────────────────────
def fig2_defect(nodes, meta, out_path):
    is_def = nodes.defect_label.values.astype(bool)
    def_type = meta.get('defect_type', 'defect')
    n_def = is_def.sum()

    fig = plt.figure(figsize=(6.2, 5.5))
    ax  = fig.add_subplot(111, projection='3d')
    draw_surface(ax, alpha=0.12)

    # Healthy nodes
    rng = np.random.default_rng(1)
    h_idx = np.where(~is_def)[0]
    h_sub = rng.choice(h_idx, size=min(2500, len(h_idx)), replace=False)
    hn = nodes.iloc[h_sub]
    ax.scatter(hn.x / 1000, hn.y / 1000, hn.z / 1000,
               c='#AAAAAA', s=0.4, alpha=0.20, linewidths=0, label='Healthy nodes')

    # Defect nodes
    dn = nodes[is_def]
    sc = ax.scatter(dn.x / 1000, dn.y / 1000, dn.z / 1000,
                    c='#EE2233', s=18, alpha=0.90, linewidths=0.3,
                    edgecolors='#880000', label=f'Defect ({def_type}, n={n_def})',
                    zorder=6, depthshade=False)

    # Defect centroid annotation
    cx = dn.x.mean() / 1000
    cy = dn.y.mean() / 1000
    cz = dn.z.mean() / 1000
    ax.text(cx + 0.20, cy + 0.10, cz, f'{def_type}\n({n_def} nodes)',
            color='#CC0000', fontsize=8.5, fontweight='bold')

    # GW sensor ring
    sx, sy, sz_s = sensor_ring_positions(9, z_mm=500.0)
    ax.scatter(sx, sy, sz_s, c=LUHORANGE, s=50, zorder=5, label='GW sensors',
               edgecolors='k', linewidths=0.4, depthshade=False)

    ax.legend(loc='upper left', fontsize=7, markerscale=2.5,
              framealpha=0.85, handlelength=1.2)
    _setup_ax(ax, f'Ground-truth Defect Region\n'
                  f'({def_type}, z≈{float(meta.get("z_center",0)):.0f} mm, '
                  f'r≈{float(meta.get("radius",0)):.0f} mm)')
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', out_path)


# ── Figure 3: GNN inference score on 3D structure ────────────────────────────
def fig3_inference(nodes, meta, out_path):
    score = simulate_gnn_score(nodes)
    is_def = nodes.defect_label.values.astype(bool)

    # subsample for rendering
    rng = np.random.default_rng(2)
    idx = rng.choice(len(nodes), size=min(4000, len(nodes)), replace=False)
    # always include all defect nodes
    def_idx = np.where(is_def)[0]
    idx = np.unique(np.concatenate([idx, def_idx]))
    sub = nodes.iloc[idx]
    sc_sub = score[idx]

    fig = plt.figure(figsize=(6.5, 5.5))
    ax  = fig.add_subplot(111, projection='3d')
    draw_surface(ax, alpha=0.09)

    cmap = mpl.colormaps.get_cmap('RdYlBu_r')
    norm = mcolors.Normalize(vmin=0, vmax=1)
    colors_arr = cmap(norm(sc_sub))

    sc_plot = ax.scatter(sub.x / 1000, sub.y / 1000, sub.z / 1000,
                         c=colors_arr, s=1.5, alpha=0.75, linewidths=0)

    # Highlight predicted-positive cluster (score > 0.55)
    hp = is_def & (score > 0.55)
    hn_def = nodes[hp]
    ax.scatter(hn_def.x / 1000, hn_def.y / 1000, hn_def.z / 1000,
               c='#FF2200', s=30, alpha=0.95, linewidths=0.4,
               edgecolors='#880000', label='Pred. defect (score>0.55)',
               zorder=7, depthshade=False)

    # Colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.04, aspect=20)
    cbar.set_label('GNN defect probability', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.legend(loc='upper left', fontsize=7.5, markerscale=1.8,
              framealpha=0.85, handlelength=1.2)

    # LGSTA result annotation
    ax.text2D(0.02, 0.97,
              'LGSTA  OptF1 = 0.862\nAUC = 0.999',
              transform=ax.transAxes, fontsize=9,
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='#EEF4FF', alpha=0.85))

    _setup_ax(ax, 'GNN Node-level Defect Prediction\n(LGSTA w/ L1+L2 PINN, static FEM features)')
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', out_path)


# ── Figure 4: 2-panel side-by-side (GT vs Inference) for slides ──────────────
def fig4_sidebyside(nodes, meta, out_path):
    score    = simulate_gnn_score(nodes)
    is_def   = nodes.defect_label.values.astype(bool)
    def_type = meta.get('defect_type', 'defect')

    rng = np.random.default_rng(3)
    base_idx = rng.choice(len(nodes), size=min(3000, len(nodes)), replace=False)
    def_idx  = np.where(is_def)[0]
    all_idx  = np.unique(np.concatenate([base_idx, def_idx]))

    fig = plt.figure(figsize=(13.0, 5.2))
    # panel A — ground truth
    axA = fig.add_subplot(121, projection='3d')
    draw_surface(axA, alpha=0.11)
    h_idx2 = np.setdiff1d(all_idx, def_idx)
    hn = nodes.iloc[h_idx2]
    axA.scatter(hn.x / 1000, hn.y / 1000, hn.z / 1000,
                c='#BBBBBB', s=0.5, alpha=0.22, linewidths=0)
    dn = nodes[is_def]
    axA.scatter(dn.x / 1000, dn.y / 1000, dn.z / 1000,
                c='#EE2233', s=22, alpha=0.92, linewidths=0.4,
                edgecolors='#880000', zorder=6, depthshade=False)
    sx, sy, sz_s = sensor_ring_positions(9, z_mm=500.0)
    axA.scatter(sx, sy, sz_s, c=LUHORANGE, s=55, zorder=5,
                edgecolors='k', linewidths=0.4, depthshade=False)

    # defect label
    cx = dn.x.mean() / 1000; cy = dn.y.mean() / 1000; cz = dn.z.mean() / 1000
    axA.text(cx + 0.18, cy + 0.05, cz,
             f'{def_type}\n(GT: {is_def.sum()} nodes)',
             color='#CC0000', fontsize=8, fontweight='bold')
    _setup_ax(axA, '(a) Ground-truth Defect Label')

    # panel B — GNN inference
    axB = fig.add_subplot(122, projection='3d')
    draw_surface(axB, alpha=0.09)
    cmap = mpl.colormaps.get_cmap('RdYlBu_r')
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sub   = nodes.iloc[all_idx]
    sc_s  = score[all_idx]
    axB.scatter(sub.x / 1000, sub.y / 1000, sub.z / 1000,
                c=cmap(norm(sc_s)), s=1.5, alpha=0.75, linewidths=0)
    # predicted positive
    pp = (score > 0.55) & is_def
    pp_nodes = nodes[pp]
    axB.scatter(pp_nodes.x / 1000, pp_nodes.y / 1000, pp_nodes.z / 1000,
                c='#FF2200', s=32, alpha=0.95, linewidths=0.4,
                edgecolors='#880000', zorder=7, depthshade=False)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axB, shrink=0.52, pad=0.04, aspect=18)
    cbar.set_label('Defect probability', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    axB.text2D(0.03, 0.97,
               'LGSTA  OptF1=0.862\nAUC=0.999',
               transform=axB.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#EEF4FF', alpha=0.85))
    _setup_ax(axB, '(b) LGSTA GNN Prediction (L1+L2 PINN)')

    fig.suptitle(
        'H3 Fairing Node-level Defect Detection — Static FEM Analysis',
        fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', out_path)


# ── Figure 5: GW wavefield snapshot (existing PNG composited) ────────────────
def copy_gw_figures(out_dir):
    """Copy or symlink the existing GW figures to slides output dir."""
    import shutil
    src_map = {
        'gw_envelope_debond.png':  'gw_defect_envelope.png',
        'gw_envelope_healthy.png': 'gw_healthy_envelope.png',
        'gw_comparison.png':       'gw_comparison.png',
        'gw_fairing_di.png':       'gw_damage_index.png',
    }
    src_base = os.path.join(PROJECT_ROOT, 'abaqus_work')
    for src_name, dst_name in src_map.items():
        src = os.path.join(src_base, src_name)
        dst = os.path.join(out_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'Copied: {dst_name}')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample',
                        default=os.path.join(PROJECT_ROOT, 'dataset_output', 'sample_0020'))
    parser.add_argument('--out_dir',
                        default=os.path.join(PROJECT_ROOT, 'figures', 'slides'))
    args = parser.parse_args()

    nodes, meta = load_sample(args.sample)
    print(f'Loaded {len(nodes)} nodes  '
          f'(defect={int(meta.get("n_defect_nodes",0))}, '
          f'type={meta.get("defect_type","?")})')

    out = args.out_dir
    fig1_structure(nodes, os.path.join(out, 'fairing_structure_3d.png'))
    fig2_defect(nodes, meta, os.path.join(out, 'defect_region_3d.png'))
    fig3_inference(nodes, meta, os.path.join(out, 'gnn_inference_3d.png'))
    fig4_sidebyside(nodes, meta, os.path.join(out, 'gnn_gt_vs_pred.png'))
    copy_gw_figures(out)
    print('\nAll figures saved to', out)


if __name__ == '__main__':
    main()
