#!/usr/bin/env python3
"""
Comprehensive Realistic FEM Dataset Validation
================================================
ML学習に使えるかを「圧倒的に」検証する。

検証カテゴリ:
  1. データ完全性 (Integrity)        — NaN, ファイル, ノード数整合
  2. 物理量妥当性 (Physics)          — 応力・変位・温度の範囲チェック
  3. 欠陥ラベル品質 (Defect Labels)  — 空間分布, コントラスト, 型別統計
  4. サンプル間一貫性 (Consistency)   — ノード数ばらつき, メッシュ品質
  5. 特徴量分布 (Feature Analysis)   — ヒストグラム, 相関, 外れ値
  6. グラフ構築テスト (Graph Build)   — PyG変換, エッジ数, 連結性
  7. ML準備度 (ML Readiness)         — クラスバランス, 特徴量分離度, SCF信号

Output: figures/validation_realistic/ に20+枚の図
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

DEFECT_NAMES = {
    0: 'healthy', 1: 'debonding', 2: 'fod', 3: 'impact',
    4: 'delamination', 5: 'inner_debond', 6: 'thermal_prog',
    7: 'acoustic_fat',
}
DEFECT_COLORS = {
    0: '#2ecc71', 1: '#e74c3c', 2: '#e67e22', 3: '#9b59b6',
    4: '#3498db', 5: '#1abc9c', 6: '#f39c12', 7: '#e91e63',
}


# ═══════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════

def load_samples(dataset_dir):
    """Load all completed samples."""
    samples = []
    sample_dirs = sorted(Path(dataset_dir).glob('sample_*'))
    for sd in sample_dirs:
        nodes_f = sd / 'nodes.csv'
        elems_f = sd / 'elements.csv'
        meta_f = sd / 'metadata.csv'
        if not all(f.exists() for f in [nodes_f, elems_f, meta_f]):
            continue
        try:
            nodes = pd.read_csv(nodes_f)
            elems = pd.read_csv(elems_f)
            meta_raw = pd.read_csv(meta_f)
            meta = dict(zip(meta_raw['key'], meta_raw['value']))
            samples.append({
                'name': sd.name,
                'nodes': nodes,
                'elems': elems,
                'meta': meta,
                'path': sd,
            })
        except Exception as e:
            print(f"  WARN: {sd.name} load failed: {e}")
    return samples


# ═══════════════════════════════════════════════════════════
# 1. Data Integrity
# ═══════════════════════════════════════════════════════════

def validate_integrity(samples, report):
    """NaN, column existence, node count consistency."""
    print("\n[1/7] Data Integrity Check")
    results = []
    required_cols = ['node_id', 'x', 'y', 'z', 'ux', 'uy', 'uz', 'u_mag',
                     'temp', 'smises', 'defect_label']

    for s in samples:
        n = s['nodes']
        e = s['elems']
        m = s['meta']

        # Column check
        missing_cols = [c for c in required_cols if c not in n.columns]

        # NaN check
        nan_counts = n[required_cols].isna().sum()
        total_nans = nan_counts.sum()

        # Inf check
        numeric_cols = n.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(n[numeric_cols]).sum().sum()

        # Node count consistency
        meta_nodes = int(m.get('n_total_nodes', 0))
        actual_nodes = len(n)
        node_match = (meta_nodes == actual_nodes)

        # Defect count consistency
        meta_defect = int(m.get('n_defect_nodes', 0))
        actual_defect = (n['defect_label'] > 0).sum()
        defect_match = (meta_defect == actual_defect)

        # Element connectivity — all referenced nodes exist
        node_ids = set(n['node_id'].values)
        elem_node_cols = [c for c in ['n1', 'n2', 'n3', 'n4'] if c in e.columns]
        missing_refs = 0
        for c in elem_node_cols:
            missing_refs += (~e[c].isin(node_ids)).sum()

        # Unique node IDs
        unique_nodes = n['node_id'].nunique() == len(n)

        status = 'PASS'
        issues = []
        if missing_cols:
            status = 'FAIL'
            issues.append(f"missing cols: {missing_cols}")
        if total_nans > 0:
            status = 'FAIL'
            issues.append(f"NaN: {total_nans}")
        if inf_count > 0:
            status = 'WARN'
            issues.append(f"Inf: {inf_count}")
        if not node_match:
            status = 'FAIL'
            issues.append(f"node count mismatch: meta={meta_nodes} actual={actual_nodes}")
        if not defect_match:
            status = 'WARN'
            issues.append(f"defect count: meta={meta_defect} actual={actual_defect}")
        if missing_refs > len(e) * 0.01:
            status = 'FAIL'
            issues.append(f"orphan elem refs: {missing_refs} (>{1}%)")
        elif missing_refs > 0:
            # Cross-part refs from core/inner skin — expected in realistic model
            issues.append(f"cross-part refs: {missing_refs} (OK)")
        if not unique_nodes:
            status = 'FAIL'
            issues.append("duplicate node IDs")

        r = {
            'name': s['name'], 'status': status,
            'n_nodes': actual_nodes, 'n_elems': len(e),
            'n_nans': total_nans, 'n_infs': inf_count,
            'node_match': node_match, 'defect_match': defect_match,
            'missing_refs': missing_refs, 'unique_ids': unique_nodes,
            'issues': '; '.join(issues) if issues else 'OK',
        }
        results.append(r)
        # Cross-part refs alone don't change status
        if status == 'PASS' and any('cross-part' in iss for iss in issues):
            pass  # keep PASS
        mark = '✓' if status == 'PASS' else ('⚠' if status == 'WARN' else '✗')
        print(f"  {mark} {s['name']}: {actual_nodes} nodes, {len(e)} elems — {r['issues']}")

    report['integrity'] = results
    passed = sum(1 for r in results if r['status'] == 'PASS')
    warned = sum(1 for r in results if r['status'] == 'WARN')
    print(f"  Summary: {passed} PASS, {warned} WARN, {len(results)-passed-warned} FAIL / {len(results)}")
    return results


# ═══════════════════════════════════════════════════════════
# 2. Physics Validation
# ═══════════════════════════════════════════════════════════

def validate_physics(samples, report, fig_dir):
    """Physical bounds and distributions."""
    print("\n[2/7] Physics Validation")
    results = []

    # Bounds (MPa, mm, °C)
    BOUNDS = {
        'smises': (0, 5000),        # von Mises stress
        'u_mag': (0, 500),          # displacement
        'temp': (15, 250),          # temperature (z-dependent: base 100C, nose ~220C)
        's11': (-5000, 5000),
        's22': (-5000, 5000),
    }

    all_stats = []
    for s in samples:
        n = s['nodes']
        st = {}
        issues = []
        for col, (lo, hi) in BOUNDS.items():
            if col not in n.columns:
                continue
            vals = n[col].dropna()
            out_lo = (vals < lo).sum()
            out_hi = (vals > hi).sum()
            st[col] = {
                'mean': vals.mean(), 'std': vals.std(),
                'min': vals.min(), 'max': vals.max(),
                'median': vals.median(),
                'out_lo': int(out_lo), 'out_hi': int(out_hi),
            }
            if out_lo + out_hi > 0:
                issues.append(f"{col}: {out_lo+out_hi} out of [{lo},{hi}]")

        # Check: stress > 0 somewhere (not all zero)
        if n['smises'].max() < 0.001:
            issues.append("smises all ~0 (FEM failed?)")

        # Check: displacement > 0
        if n['u_mag'].max() < 0.001:
            issues.append("u_mag all ~0 (no load?)")

        # Check: temp not uniform 0
        temp_range = n['temp'].max() - n['temp'].min()
        if n['temp'].max() < 1.0:
            issues.append("temp all ~0 (thermal load missing?)")

        status = 'PASS' if not issues else 'WARN'
        results.append({
            'name': s['name'], 'status': status,
            'stats': st, 'issues': '; '.join(issues) if issues else 'OK',
        })
        all_stats.append(st)
        mark = '✓' if status == 'PASS' else '⚠'
        print(f"  {mark} {s['name']}: σ_max={st['smises']['max']:.1f} MPa, "
              f"|u|_max={st['u_mag']['max']:.2f} mm, T=[{st['temp']['min']:.0f},{st['temp']['max']:.0f}]°C")

    report['physics'] = results

    # ── Figure: Physics Overview (4 panels) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Physics Validation — Realistic FEM Dataset', fontsize=15, fontweight='bold')

    # Collect all nodes
    all_nodes = pd.concat([s['nodes'] for s in samples], ignore_index=True)

    # (a) Von Mises distribution
    ax = axes[0, 0]
    smises_pos = all_nodes['smises'][all_nodes['smises'] > 1e-6]
    ax.hist(np.log10(smises_pos), bins=100, color='#e74c3c', alpha=0.7, edgecolor='none')
    ax.set_xlabel('log₁₀(Von Mises Stress / MPa)')
    ax.set_ylabel('Node Count')
    ax.set_title(f'(a) Von Mises Stress Distribution (N={len(smises_pos):,})')
    ax.axvline(np.log10(smises_pos.median()), color='k', ls='--', lw=1.5,
               label=f'median={smises_pos.median():.2f} MPa')
    ax.legend(fontsize=9)

    # (b) Displacement distribution
    ax = axes[0, 1]
    ax.hist(all_nodes['u_mag'], bins=100, color='#3498db', alpha=0.7, edgecolor='none')
    ax.set_xlabel('Displacement |u| (mm)')
    ax.set_ylabel('Node Count')
    ax.set_title(f'(b) Displacement Distribution')
    ax.axvline(all_nodes['u_mag'].median(), color='k', ls='--', lw=1.5,
               label=f'median={all_nodes["u_mag"].median():.2f} mm')
    ax.legend(fontsize=9)

    # (c) Temperature distribution
    ax = axes[1, 0]
    ax.hist(all_nodes['temp'], bins=80, color='#e67e22', alpha=0.7, edgecolor='none')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Node Count')
    ax.set_title('(c) Temperature Distribution')
    ax.axvline(all_nodes['temp'].median(), color='k', ls='--', lw=1.5,
               label=f'median={all_nodes["temp"].median():.0f}°C')
    ax.legend(fontsize=9)

    # (d) Per-sample physics summary
    ax = axes[1, 1]
    names = [s['name'].replace('sample_', 'S') for s in samples]
    max_stress = [s['nodes']['smises'].max() for s in samples]
    max_disp = [s['nodes']['u_mag'].max() for s in samples]
    x_pos = np.arange(len(names))
    w = 0.35
    ax.bar(x_pos - w/2, max_stress, w, color='#e74c3c', alpha=0.8, label='σ_max (MPa)')
    ax2 = ax.twinx()
    ax2.bar(x_pos + w/2, max_disp, w, color='#3498db', alpha=0.8, label='|u|_max (mm)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Max Von Mises (MPa)', color='#e74c3c')
    ax2.set_ylabel('Max |u| (mm)', color='#3498db')
    ax.set_title('(d) Per-Sample Peak Values')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    fig.savefig(fig_dir / '01_physics_overview.png')
    plt.close(fig)
    print(f"  → {fig_dir / '01_physics_overview.png'}")

    return results


# ═══════════════════════════════════════════════════════════
# 3. Defect Label Quality
# ═══════════════════════════════════════════════════════════

def validate_defect_labels(samples, report, fig_dir):
    """Defect label spatial distribution, contrast, type statistics."""
    print("\n[3/7] Defect Label Quality")
    results = []
    type_stats = {}

    for s in samples:
        n = s['nodes']
        m = s['meta']
        defect_type = m.get('defect_type', 'unknown')
        defect_id = int(m.get('defect_type_id', 0))

        healthy = n[n['defect_label'] == 0]
        defect = n[n['defect_label'] > 0]
        n_defect = len(defect)
        ratio = n_defect / len(n) * 100

        # Stress contrast
        if n_defect > 0 and len(healthy) > 0:
            stress_defect = defect['smises'].mean()
            stress_healthy = healthy['smises'].mean()
            stress_ratio = stress_defect / max(stress_healthy, 1e-12)

            disp_defect = defect['u_mag'].mean()
            disp_healthy = healthy['u_mag'].mean()
            disp_ratio = disp_defect / max(disp_healthy, 1e-12)
        else:
            stress_ratio = disp_ratio = float('nan')

        # Spatial extent of defect
        if n_defect > 0:
            r_defect = np.sqrt(defect['x']**2 + defect['z']**2)
            theta_defect = np.degrees(np.arctan2(defect['z'], defect['x']))
            z_defect = defect['y']
            spatial = {
                'theta_range': (theta_defect.min(), theta_defect.max()),
                'z_range': (z_defect.min(), z_defect.max()),
                'r_range': (r_defect.min(), r_defect.max()),
            }
        else:
            spatial = {}

        r = {
            'name': s['name'], 'defect_type': defect_type,
            'defect_id': defect_id,
            'n_defect': n_defect, 'ratio_pct': ratio,
            'stress_ratio': stress_ratio, 'disp_ratio': disp_ratio,
            'spatial': spatial,
        }
        results.append(r)

        if defect_type not in type_stats:
            type_stats[defect_type] = []
        type_stats[defect_type].append(r)

        print(f"  {s['name']}: {defect_type} ({defect_id}), {n_defect} defect nodes "
              f"({ratio:.3f}%), σ_ratio={stress_ratio:.2f}, |u|_ratio={disp_ratio:.2f}")

    report['defect_labels'] = results

    # ── Figure: Defect Label Analysis (6 panels) ──
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
    fig.suptitle('Defect Label Quality — ML Training Readiness', fontsize=15, fontweight='bold')

    # (a) Defect node ratio by sample
    ax = fig.add_subplot(gs[0, 0])
    names = [r['name'].replace('sample_', 'S') for r in results]
    ratios = [r['ratio_pct'] for r in results]
    colors = [DEFECT_COLORS.get(r['defect_id'], '#999') for r in results]
    bars = ax.bar(range(len(names)), ratios, color=colors, alpha=0.85, edgecolor='white', lw=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Defect Node Ratio (%)')
    ax.set_title('(a) Defect Node Ratio per Sample')
    ax.axhline(np.mean(ratios), color='k', ls='--', lw=1, label=f'mean={np.mean(ratios):.3f}%')
    ax.legend(fontsize=8)

    # (b) Stress contrast (defect / healthy)
    ax = fig.add_subplot(gs[0, 1])
    stress_ratios = [r['stress_ratio'] for r in results if not np.isnan(r['stress_ratio'])]
    disp_ratios = [r['disp_ratio'] for r in results if not np.isnan(r['disp_ratio'])]
    valid_names = [r['name'].replace('sample_', 'S') for r in results if not np.isnan(r['stress_ratio'])]
    x_pos = np.arange(len(valid_names))
    ax.bar(x_pos - 0.2, stress_ratios, 0.35, color='#e74c3c', alpha=0.8, label='σ ratio')
    ax.bar(x_pos + 0.2, disp_ratios, 0.35, color='#3498db', alpha=0.8, label='|u| ratio')
    ax.axhline(1.0, color='gray', ls=':', lw=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Ratio (defect / healthy)')
    ax.set_title('(b) Defect–Healthy Contrast')
    ax.legend(fontsize=8)

    # (c) Defect type distribution
    ax = fig.add_subplot(gs[0, 2])
    types = list(type_stats.keys())
    counts = [len(v) for v in type_stats.values()]
    type_colors = [DEFECT_COLORS.get(type_stats[t][0]['defect_id'], '#999') for t in types]
    ax.barh(range(len(types)), counts, color=type_colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=10)
    ax.set_xlabel('Sample Count')
    ax.set_title('(c) Defect Type Distribution')
    for i, c in enumerate(counts):
        ax.text(c + 0.1, i, str(c), va='center', fontsize=10, fontweight='bold')

    # (d) Spatial distribution of defects (θ vs z)
    ax = fig.add_subplot(gs[1, 0])
    for r, s in zip(results, samples):
        defect = s['nodes'][s['nodes']['defect_label'] > 0]
        if len(defect) == 0:
            continue
        theta = np.degrees(np.arctan2(defect['z'], defect['x']))
        z_val = defect['y']
        color = DEFECT_COLORS.get(r['defect_id'], '#999')
        ax.scatter(theta, z_val, s=1, alpha=0.3, c=color, label=r['defect_type'])
    ax.set_xlabel('θ (degrees)')
    ax.set_ylabel('z (mm)')
    ax.set_title('(d) Defect Spatial Distribution')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc='best', markerscale=5)

    # (e) Defect vs Healthy stress distributions (overlaid)
    ax = fig.add_subplot(gs[1, 1])
    all_healthy = pd.concat([s['nodes'][s['nodes']['defect_label'] == 0] for s in samples])
    all_defect = pd.concat([s['nodes'][s['nodes']['defect_label'] > 0] for s in samples])
    h_stress = all_healthy['smises'][all_healthy['smises'] > 1e-6]
    d_stress = all_defect['smises'][all_defect['smises'] > 1e-6]
    bins = np.linspace(-3, 3, 100)
    if len(h_stress) > 0:
        ax.hist(np.log10(h_stress), bins=bins, alpha=0.6, color='#2ecc71',
                label=f'Healthy (N={len(h_stress):,})', density=True)
    if len(d_stress) > 0:
        ax.hist(np.log10(d_stress), bins=bins, alpha=0.6, color='#e74c3c',
                label=f'Defect (N={len(d_stress):,})', density=True)
    ax.set_xlabel('log₁₀(Von Mises Stress / MPa)')
    ax.set_ylabel('Density')
    ax.set_title('(e) Healthy vs Defect Stress Separation')
    ax.legend(fontsize=8)

    # (f) Defect vs Healthy displacement
    ax = fig.add_subplot(gs[1, 2])
    if len(all_healthy) > 0:
        ax.hist(all_healthy['u_mag'], bins=80, alpha=0.6, color='#2ecc71',
                label=f'Healthy', density=True)
    if len(all_defect) > 0:
        ax.hist(all_defect['u_mag'], bins=80, alpha=0.6, color='#e74c3c',
                label=f'Defect', density=True)
    ax.set_xlabel('Displacement |u| (mm)')
    ax.set_ylabel('Density')
    ax.set_title('(f) Healthy vs Defect Displacement')
    ax.legend(fontsize=8)

    plt.savefig(fig_dir / '02_defect_label_quality.png')
    plt.close(fig)
    print(f"  → {fig_dir / '02_defect_label_quality.png'}")

    return results


# ═══════════════════════════════════════════════════════════
# 4. Sample Consistency
# ═══════════════════════════════════════════════════════════

def validate_consistency(samples, report, fig_dir):
    """Cross-sample consistency: node counts, coordinate ranges, mesh quality."""
    print("\n[4/7] Sample Consistency")
    results = []

    node_counts = []
    elem_counts = []
    x_ranges, y_ranges, z_ranges = [], [], []

    for s in samples:
        n = s['nodes']
        e = s['elems']
        node_counts.append(len(n))
        elem_counts.append(len(e))
        x_ranges.append((n['x'].min(), n['x'].max()))
        y_ranges.append((n['y'].min(), n['y'].max()))
        z_ranges.append((n['z'].min(), n['z'].max()))

    nc = np.array(node_counts)
    ec = np.array(elem_counts)
    cv_nodes = nc.std() / nc.mean() * 100  # coefficient of variation

    print(f"  Nodes:    {nc.mean():.0f} ± {nc.std():.0f} (CV={cv_nodes:.1f}%)")
    print(f"  Elements: {ec.mean():.0f} ± {ec.std():.0f}")

    report['consistency'] = {
        'node_mean': float(nc.mean()), 'node_std': float(nc.std()),
        'node_cv': float(cv_nodes),
        'elem_mean': float(ec.mean()), 'elem_std': float(ec.std()),
    }

    # ── Figure: Consistency (4 panels) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sample Consistency Check', fontsize=15, fontweight='bold')

    # (a) Node count variation
    ax = axes[0, 0]
    names = [s['name'].replace('sample_', 'S') for s in samples]
    ax.bar(range(len(names)), node_counts, color='#3498db', alpha=0.8)
    ax.axhline(nc.mean(), color='k', ls='--', lw=1.5, label=f'mean={nc.mean():.0f}')
    ax.fill_between([-0.5, len(names)-0.5], nc.mean()-nc.std(), nc.mean()+nc.std(),
                     alpha=0.15, color='gray', label=f'±1σ')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Node Count')
    ax.set_title(f'(a) Node Count per Sample (CV={cv_nodes:.1f}%)')
    ax.legend(fontsize=8)

    # (b) Element count variation
    ax = axes[0, 1]
    ax.bar(range(len(names)), elem_counts, color='#e67e22', alpha=0.8)
    ax.axhline(ec.mean(), color='k', ls='--', lw=1.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Element Count')
    ax.set_title('(b) Element Count per Sample')

    # (c) Coordinate ranges — box plot
    ax = axes[1, 0]
    all_x = [s['nodes']['x'] for s in samples]
    all_y = [s['nodes']['y'] for s in samples]
    all_z = [s['nodes']['z'] for s in samples]
    bp = ax.boxplot([np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_z)],
                     labels=['X', 'Y (axial)', 'Z'], patch_artist=True,
                     boxprops=dict(alpha=0.7))
    for patch, c in zip(bp['boxes'], ['#e74c3c', '#3498db', '#2ecc71']):
        patch.set_facecolor(c)
    ax.set_ylabel('Coordinate (mm)')
    ax.set_title('(c) Coordinate Ranges')

    # (d) Element aspect ratio (approx via edge lengths)
    ax = axes[1, 1]
    aspect_ratios = []
    for s in samples:
        n = s['nodes']
        e = s['elems']
        # Sample 500 random elements for speed
        sample_idx = np.random.choice(len(e), min(500, len(e)), replace=False)
        e_sample = e.iloc[sample_idx]
        coords = n.set_index('node_id')[['x', 'y', 'z']]
        for _, row in e_sample.iterrows():
            try:
                nids = [int(row['n1']), int(row['n2']), int(row['n3']), int(row['n4'])]
                pts = coords.loc[nids].values
                edges = [np.linalg.norm(pts[(i+1)%4] - pts[i]) for i in range(4)]
                ar = max(edges) / max(min(edges), 1e-10)
                aspect_ratios.append(ar)
            except (KeyError, ValueError):
                pass
    ax.hist(aspect_ratios, bins=50, color='#9b59b6', alpha=0.7, edgecolor='none')
    ar_arr = np.array(aspect_ratios)
    ax.axvline(np.median(ar_arr), color='k', ls='--', lw=1.5,
               label=f'median={np.median(ar_arr):.2f}')
    bad_ratio = (ar_arr > 5).sum() / len(ar_arr) * 100 if len(ar_arr) > 0 else 0
    ax.set_xlabel('Element Aspect Ratio')
    ax.set_ylabel('Count')
    ax.set_title(f'(d) Mesh Quality — AR>5: {bad_ratio:.1f}%')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(fig_dir / '03_sample_consistency.png')
    plt.close(fig)
    print(f"  → {fig_dir / '03_sample_consistency.png'}")


# ═══════════════════════════════════════════════════════════
# 5. Feature Analysis
# ═══════════════════════════════════════════════════════════

def validate_features(samples, report, fig_dir):
    """Feature distributions, correlations, outlier analysis."""
    print("\n[5/7] Feature Analysis")

    all_nodes = pd.concat([s['nodes'] for s in samples], ignore_index=True)
    feature_cols = ['ux', 'uy', 'uz', 'u_mag', 'temp', 's11', 's22', 's12',
                    'smises']
    if 'thermal_smises' in all_nodes.columns:
        feature_cols.append('thermal_smises')
    if 'le11' in all_nodes.columns:
        feature_cols.extend(['le11', 'le22', 'le12'])

    # ── Figure: Feature Distributions (grid) ──
    n_feat = len(feature_cols)
    n_cols_grid = 4
    n_rows = (n_feat + n_cols_grid - 1) // n_cols_grid
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(16, 3.5 * n_rows))
    fig.suptitle('Node Feature Distributions — All Samples', fontsize=15, fontweight='bold')
    axes_flat = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes_flat[i]
        vals = all_nodes[col].dropna()
        # Remove exact zeros for log-scale friendly display
        nonzero = vals[vals.abs() > 1e-12]
        if len(nonzero) > 0:
            ax.hist(nonzero, bins=80, color='#3498db', alpha=0.7, edgecolor='none')
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_ylabel('Count')
        # Stats annotation
        ax.text(0.97, 0.95, f'μ={vals.mean():.3g}\nσ={vals.std():.3g}\n'
                f'min={vals.min():.3g}\nmax={vals.max():.3g}',
                transform=ax.transAxes, ha='right', va='top', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(fig_dir / '04_feature_distributions.png')
    plt.close(fig)
    print(f"  → {fig_dir / '04_feature_distributions.png'}")

    # ── Figure: Feature Correlation Matrix ──
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_cols = ['ux', 'uy', 'uz', 'u_mag', 'temp', 'smises', 's11', 's22', 's12']
    corr_cols = [c for c in corr_cols if c in all_nodes.columns]
    corr = all_nodes[corr_cols].corr()
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha='right')
    ax.set_yticklabels(corr_cols)
    for i_c in range(len(corr_cols)):
        for j_c in range(len(corr_cols)):
            val = corr.iloc[i_c, j_c]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j_c, i_c, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(fig_dir / '05_correlation_matrix.png')
    plt.close(fig)
    print(f"  → {fig_dir / '05_correlation_matrix.png'}")

    # ── Figure: Outlier Analysis ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Outlier Analysis (per sample)', fontsize=14, fontweight='bold')

    for idx, (col, ax) in enumerate(zip(['smises', 'u_mag', 'temp'], axes)):
        data_per_sample = [s['nodes'][col].values for s in samples]
        bp = ax.boxplot(data_per_sample, patch_artist=True,
                        labels=[s['name'].replace('sample_', 'S') for s in samples])
        for patch in bp['boxes']:
            patch.set_facecolor(['#e74c3c', '#3498db', '#e67e22'][idx])
            patch.set_alpha(0.6)
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fig.savefig(fig_dir / '06_outlier_analysis.png')
    plt.close(fig)
    print(f"  → {fig_dir / '06_outlier_analysis.png'}")


# ═══════════════════════════════════════════════════════════
# 6. Graph Build Test
# ═══════════════════════════════════════════════════════════

def validate_graph_build(samples, report, fig_dir):
    """Test PyG graph construction from FEM data."""
    print("\n[6/7] Graph Build Test")

    try:
        import torch
        from torch_geometric.data import Data
        HAS_PYG = True
    except ImportError:
        print("  SKIP: PyTorch Geometric not available")
        report['graph_build'] = {'status': 'SKIP', 'reason': 'PyG not installed'}
        return

    results = []
    graph_stats = []

    for s in samples:
        n = s['nodes']
        e = s['elems']
        name = s['name']

        try:
            # Build edge index from elements
            edges = set()
            for _, row in e.iterrows():
                nids = [int(row['n1']), int(row['n2']), int(row['n3']), int(row['n4'])]
                for k in range(4):
                    u, v = nids[k], nids[(k+1) % 4]
                    if u != v:
                        edges.add((u, v))
                        edges.add((v, u))

            # Node ID → index mapping
            node_ids = n['node_id'].values
            id2idx = {nid: idx for idx, nid in enumerate(node_ids)}

            # Filter edges with valid IDs
            edge_list = [(id2idx[u], id2idx[v]) for u, v in edges
                         if u in id2idx and v in id2idx]

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

            # Node features
            feat_cols = ['x', 'y', 'z', 'ux', 'uy', 'uz', 'u_mag', 'temp',
                         's11', 's22', 's12', 'smises']
            feat_cols = [c for c in feat_cols if c in n.columns]
            x_feat = torch.tensor(n[feat_cols].values, dtype=torch.float)

            # Labels
            y = torch.tensor(n['defect_label'].values, dtype=torch.long)

            # Build Data object
            data = Data(x=x_feat, edge_index=edge_index, y=y)

            # Validate
            n_nodes = data.num_nodes
            n_edges = data.num_edges
            avg_degree = n_edges / n_nodes if n_nodes > 0 else 0
            n_classes = len(y.unique())
            has_isolated = (n_nodes - len(set(edge_index[0].tolist() + edge_index[1].tolist())))

            # NaN/Inf in features
            feat_nan = torch.isnan(x_feat).sum().item()
            feat_inf = torch.isinf(x_feat).sum().item()

            r = {
                'name': name, 'status': 'PASS',
                'n_nodes': n_nodes, 'n_edges': n_edges,
                'avg_degree': avg_degree, 'n_classes': n_classes,
                'isolated_nodes': has_isolated,
                'feat_nan': feat_nan, 'feat_inf': feat_inf,
                'n_features': x_feat.shape[1],
            }
            results.append(r)
            graph_stats.append(r)
            print(f"  ✓ {name}: {n_nodes:,} nodes, {n_edges:,} edges, "
                  f"deg={avg_degree:.1f}, classes={n_classes}, iso={has_isolated}")

        except Exception as exc:
            results.append({'name': name, 'status': 'FAIL', 'error': str(exc)})
            print(f"  ✗ {name}: {exc}")

    report['graph_build'] = results

    if not graph_stats:
        return

    # ── Figure: Graph Structure (4 panels) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Graph Structure Validation', fontsize=15, fontweight='bold')

    names = [r['name'].replace('sample_', 'S') for r in graph_stats]

    # (a) Nodes & edges
    ax = axes[0, 0]
    x_pos = np.arange(len(names))
    ax.bar(x_pos - 0.2, [r['n_nodes'] for r in graph_stats], 0.35,
           color='#3498db', label='Nodes')
    ax.bar(x_pos + 0.2, [r['n_edges'] // 1000 for r in graph_stats], 0.35,
           color='#e74c3c', label='Edges (×1000)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('(a) Graph Size')
    ax.legend(fontsize=8)

    # (b) Average degree
    ax = axes[0, 1]
    degs = [r['avg_degree'] for r in graph_stats]
    ax.bar(range(len(names)), degs, color='#2ecc71', alpha=0.8)
    ax.axhline(np.mean(degs), color='k', ls='--', label=f'mean={np.mean(degs):.1f}')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Average Degree')
    ax.set_title('(b) Average Node Degree')
    ax.legend(fontsize=8)

    # (c) Class distribution (stacked)
    ax = axes[1, 0]
    # Re-compute per-class counts
    for i_s, s in enumerate(samples):
        label_counts = s['nodes']['defect_label'].value_counts().sort_index()
        bottom = 0
        for lbl, cnt in label_counts.items():
            color = DEFECT_COLORS.get(int(lbl), '#999')
            ax.bar(i_s, cnt, bottom=bottom, color=color, alpha=0.85,
                   label=DEFECT_NAMES.get(int(lbl), f'type_{lbl}') if i_s == 0 else '')
            bottom += cnt
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Node Count')
    ax.set_title('(c) Per-Node Class Distribution')
    handles, labels_legend = ax.get_legend_handles_labels()
    ax.legend(handles, labels_legend, fontsize=7, loc='upper right')

    # (d) Feature statistics — min/max/mean heatmap
    ax = axes[1, 1]
    feat_names = ['x', 'y', 'z', 'ux', 'uy', 'uz', 'u_mag', 'temp', 'smises']
    all_nodes = pd.concat([s['nodes'] for s in samples])
    feat_data = []
    for col in feat_names:
        if col in all_nodes.columns:
            feat_data.append([all_nodes[col].mean(), all_nodes[col].std(),
                              all_nodes[col].min(), all_nodes[col].max()])
    feat_data = np.array(feat_data)
    # Normalize for display
    feat_norm = feat_data.copy()
    for j in range(feat_data.shape[1]):
        col_max = np.abs(feat_data[:, j]).max()
        if col_max > 0:
            feat_norm[:, j] = feat_data[:, j] / col_max
    im = ax.imshow(feat_norm, aspect='auto', cmap='coolwarm')
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=9)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Mean', 'Std', 'Min', 'Max'], fontsize=9)
    ax.set_title('(d) Feature Statistics (normalized)')
    for i_h in range(len(feat_names)):
        for j_h in range(4):
            ax.text(j_h, i_h, f'{feat_data[i_h, j_h]:.2g}', ha='center', va='center',
                    fontsize=7, color='white' if abs(feat_norm[i_h, j_h]) > 0.5 else 'black')

    plt.tight_layout()
    fig.savefig(fig_dir / '07_graph_structure.png')
    plt.close(fig)
    print(f"  → {fig_dir / '07_graph_structure.png'}")


# ═══════════════════════════════════════════════════════════
# 7. ML Readiness
# ═══════════════════════════════════════════════════════════

def validate_ml_readiness(samples, report, fig_dir):
    """Class balance, feature separability, signal-to-noise, SCF."""
    print("\n[7/7] ML Readiness Assessment")

    all_nodes = pd.concat([s['nodes'].assign(sample=s['name']) for s in samples],
                          ignore_index=True)
    healthy = all_nodes[all_nodes['defect_label'] == 0]
    defect = all_nodes[all_nodes['defect_label'] > 0]

    # ── Class imbalance analysis ──
    n_total = len(all_nodes)
    n_defect = len(defect)
    imbalance_ratio = (n_total - n_defect) / max(n_defect, 1)
    print(f"  Class imbalance: {imbalance_ratio:.0f}:1 (healthy:defect)")
    print(f"  Defect fraction: {n_defect/n_total*100:.3f}%")

    # ── Feature separability (t-test / Cohen's d) ──
    feat_cols = ['smises', 'u_mag', 'temp', 'ux', 'uy', 'uz', 's11', 's22', 's12']
    feat_cols = [c for c in feat_cols if c in all_nodes.columns]
    sep_results = {}
    for col in feat_cols:
        h_vals = healthy[col].dropna().values
        d_vals = defect[col].dropna().values
        if len(d_vals) < 2:
            continue
        # Cohen's d
        pooled_std = np.sqrt((h_vals.std()**2 + d_vals.std()**2) / 2)
        cohens_d = (d_vals.mean() - h_vals.mean()) / max(pooled_std, 1e-12)
        # t-test
        t_stat, p_val = stats.ttest_ind(h_vals, d_vals, equal_var=False)
        # KS test
        ks_stat, ks_p = stats.ks_2samp(h_vals, d_vals)
        sep_results[col] = {
            'cohens_d': cohens_d, 't_stat': t_stat, 'p_val': p_val,
            'ks_stat': ks_stat, 'ks_p': ks_p,
        }
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        print(f"  {col:>8}: Cohen's d={cohens_d:+.4f}, KS={ks_stat:.4f} ({sig})")

    report['ml_readiness'] = {
        'imbalance_ratio': imbalance_ratio,
        'defect_fraction': n_defect / n_total,
        'separability': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in sep_results.items()},
    }

    # ── Figure: ML Readiness Dashboard (6 panels) ──
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)
    fig.suptitle('ML Readiness Assessment — Realistic FEM Dataset', fontsize=16, fontweight='bold')

    # (a) Class distribution pie + bar
    ax = fig.add_subplot(gs[0, 0])
    per_type = all_nodes['defect_label'].value_counts().sort_index()
    labels_pie = [DEFECT_NAMES.get(int(k), f'type_{k}') for k in per_type.index]
    colors_pie = [DEFECT_COLORS.get(int(k), '#999') for k in per_type.index]
    # Use bar chart (pie is hard to read with extreme imbalance)
    ax.barh(range(len(labels_pie)), per_type.values, color=colors_pie, alpha=0.85)
    ax.set_yticks(range(len(labels_pie)))
    ax.set_yticklabels(labels_pie, fontsize=9)
    ax.set_xlabel('Node Count')
    ax.set_title(f'(a) Class Distribution\nImbalance {imbalance_ratio:.0f}:1')
    for i_b, v in enumerate(per_type.values):
        pct = v / n_total * 100
        ax.text(v, i_b, f'  {v:,} ({pct:.2f}%)', va='center', fontsize=8)
    ax.set_xscale('log')

    # (b) Cohen's d — Feature separability
    ax = fig.add_subplot(gs[0, 1])
    sep_cols = list(sep_results.keys())
    cohens = [sep_results[c]['cohens_d'] for c in sep_cols]
    colors_d = ['#e74c3c' if abs(d) > 0.2 else '#e67e22' if abs(d) > 0.05 else '#95a5a6'
                for d in cohens]
    ax.barh(range(len(sep_cols)), cohens, color=colors_d, alpha=0.85)
    ax.set_yticks(range(len(sep_cols)))
    ax.set_yticklabels(sep_cols, fontsize=10)
    ax.set_xlabel("Cohen's d (effect size)")
    ax.axvline(0, color='gray', lw=1)
    ax.axvline(0.2, color='#e74c3c', ls=':', lw=1, alpha=0.5)
    ax.axvline(-0.2, color='#e74c3c', ls=':', lw=1, alpha=0.5)
    ax.set_title("(b) Feature Separability\n(|d|>0.2 = small effect)")
    for i_b, d in enumerate(cohens):
        ax.text(d, i_b, f' {d:+.3f}', va='center', fontsize=8)

    # (c) KS statistic
    ax = fig.add_subplot(gs[0, 2])
    ks_vals = [sep_results[c]['ks_stat'] for c in sep_cols]
    ks_colors = ['#e74c3c' if k > 0.1 else '#e67e22' if k > 0.05 else '#95a5a6'
                 for k in ks_vals]
    ax.barh(range(len(sep_cols)), ks_vals, color=ks_colors, alpha=0.85)
    ax.set_yticks(range(len(sep_cols)))
    ax.set_yticklabels(sep_cols, fontsize=10)
    ax.set_xlabel('KS Statistic')
    ax.set_title('(c) KS Test — Distribution Divergence')
    for i_b, k in enumerate(ks_vals):
        ax.text(k, i_b, f' {k:.4f}', va='center', fontsize=8)

    # (d) 2D scatter: smises vs u_mag colored by defect
    ax = fig.add_subplot(gs[1, 0])
    # Subsample for speed
    n_sub = min(50000, len(healthy))
    h_sub = healthy.sample(n_sub, random_state=42) if len(healthy) > n_sub else healthy
    ax.scatter(h_sub['smises'], h_sub['u_mag'], s=0.5, alpha=0.1, c='#2ecc71', label='healthy')
    if len(defect) > 0:
        ax.scatter(defect['smises'], defect['u_mag'], s=2, alpha=0.5, c='#e74c3c', label='defect')
    ax.set_xlabel('Von Mises Stress (MPa)')
    ax.set_ylabel('Displacement |u| (mm)')
    ax.set_title('(d) Feature Space: σ vs |u|')
    ax.legend(fontsize=8, markerscale=5)

    # (e) Per-sample defect zone stress profile
    ax = fig.add_subplot(gs[1, 1])
    for s in samples:
        n = s['nodes']
        d = n[n['defect_label'] > 0]
        if len(d) == 0:
            continue
        m = s['meta']
        z_center = float(m.get('z_center', 0))
        # Distance from defect center (approx via y coordinate)
        dist = np.abs(d['y'] - z_center)
        # Sort and plot stress vs distance
        order = dist.argsort()
        ax.plot(dist.values[order], d['smises'].values[order], '.', markersize=1,
                alpha=0.5, label=f"{s['name'].replace('sample_', 'S')}")
    ax.set_xlabel('Distance from Defect Center (mm)')
    ax.set_ylabel('Von Mises Stress (MPa)')
    ax.set_title('(e) Stress vs Distance from Defect')
    ax.legend(fontsize=7, loc='upper right', markerscale=5)

    # (f) Signal quality: defect zone vs. local neighborhood
    ax = fig.add_subplot(gs[1, 2])
    snr_data = []
    for s in samples:
        n = s['nodes']
        m = s['meta']
        d = n[n['defect_label'] > 0]
        h = n[n['defect_label'] == 0]
        if len(d) == 0:
            continue
        # "Nearby healthy" — nodes within 200mm of defect center
        z_center = float(m.get('z_center', 0))
        theta_center = float(m.get('theta_deg', 30))
        r_rad = np.sqrt(n['x']**2 + n['z']**2)
        theta = np.degrees(np.arctan2(n['z'], n['x']))
        dist = np.sqrt((n['y'] - z_center)**2 +
                        (r_rad * np.radians(theta - theta_center))**2)
        nearby_healthy = n[(n['defect_label'] == 0) & (dist < 300)]
        if len(nearby_healthy) < 10:
            nearby_healthy = h

        # SNR-like metric
        signal = np.abs(d['smises'].mean() - nearby_healthy['smises'].mean())
        noise = nearby_healthy['smises'].std()
        snr = signal / max(noise, 1e-12)
        snr_data.append({
            'name': s['name'], 'snr': snr,
            'signal': signal, 'noise': noise,
            'defect_type': m.get('defect_type', 'unknown'),
        })

    if snr_data:
        snr_names = [d['name'].replace('sample_', 'S') for d in snr_data]
        snr_vals = [d['snr'] for d in snr_data]
        snr_colors = [DEFECT_COLORS.get(
            {'debonding': 1, 'fod': 2, 'impact': 3, 'delamination': 4,
             'inner_debond': 5, 'thermal_prog': 6, 'thermal_progression': 6,
             'acoustic_fat': 7, 'acoustic_fatigue': 7}.get(d['defect_type'], 0), '#999')
            for d in snr_data]
        ax.bar(range(len(snr_names)), snr_vals, color=snr_colors, alpha=0.85)
        ax.set_xticks(range(len(snr_names)))
        ax.set_xticklabels(snr_names, rotation=45, ha='right')
        ax.set_ylabel('SNR (signal / noise)')
        ax.set_title('(f) Defect Signal-to-Noise Ratio')
        ax.axhline(1.0, color='k', ls=':', lw=1)

    # (g) Spatial field map — representative sample (stress)
    ax = fig.add_subplot(gs[2, 0])
    s0 = samples[0]
    n0 = s0['nodes']
    r0 = np.sqrt(n0['x']**2 + n0['z']**2)
    theta0 = np.degrees(np.arctan2(n0['z'], n0['x']))
    # Arc length
    arc = r0 * np.radians(theta0)
    sc = ax.scatter(arc, n0['y'], c=np.log10(n0['smises'].clip(1e-6)),
                    s=0.3, alpha=0.5, cmap='hot')
    plt.colorbar(sc, ax=ax, label='log₁₀(σ_mises)')
    ax.set_xlabel('Arc Length (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_title(f'(g) Stress Field — {s0["name"]}')

    # (h) Spatial field map — defect labels
    ax = fig.add_subplot(gs[2, 1])
    colors_map = np.array(['#2ecc71' if l == 0 else '#e74c3c'
                           for l in n0['defect_label']])
    # Plot healthy first (background), then defect (foreground)
    h_mask = n0['defect_label'] == 0
    d_mask = n0['defect_label'] > 0
    ax.scatter(arc[h_mask], n0['y'][h_mask], s=0.2, alpha=0.2, c='#2ecc71')
    ax.scatter(arc[d_mask], n0['y'][d_mask], s=2, alpha=0.8, c='#e74c3c')
    ax.set_xlabel('Arc Length (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_title(f'(h) Defect Labels — {s0["name"]}')

    # (i) Summary scorecard
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    # Compute scores
    integrity_pass = sum(1 for r in report.get('integrity', []) if r['status'] == 'PASS')
    integrity_total = len(report.get('integrity', []))
    graph_pass = sum(1 for r in report.get('graph_build', []) if isinstance(r, dict) and r.get('status') == 'PASS')
    graph_total = len(report.get('graph_build', []))

    max_cohens_d = max(abs(d) for d in cohens) if cohens else 0
    mean_ks = np.mean(ks_vals) if ks_vals else 0
    mean_snr = np.mean(snr_vals) if snr_vals else 0

    scores = [
        ('Data Integrity', f'{integrity_pass}/{integrity_total}',
         '#2ecc71' if integrity_pass == integrity_total else '#e74c3c'),
        ('Graph Build', f'{graph_pass}/{graph_total}',
         '#2ecc71' if graph_pass == graph_total else '#e74c3c'),
        ('Class Imbalance', f'{imbalance_ratio:.0f}:1',
         '#e67e22' if imbalance_ratio > 100 else '#2ecc71'),
        ('Max |Cohen d|', f'{max_cohens_d:.4f}',
         '#2ecc71' if max_cohens_d > 0.1 else '#e67e22' if max_cohens_d > 0.01 else '#e74c3c'),
        ('Mean KS stat', f'{mean_ks:.4f}',
         '#2ecc71' if mean_ks > 0.05 else '#e67e22' if mean_ks > 0.01 else '#e74c3c'),
        ('Mean SNR', f'{mean_snr:.3f}',
         '#2ecc71' if mean_snr > 1.0 else '#e67e22' if mean_snr > 0.1 else '#e74c3c'),
        ('N samples', f'{len(samples)}',
         '#2ecc71' if len(samples) >= 10 else '#e67e22'),
        ('N nodes/sample', f'{int(report.get("consistency", {}).get("node_mean", 0)):,}',
         '#2ecc71'),
        ('N features', f'{len(feat_cols)}', '#3498db'),
    ]

    y_offset = 0.95
    ax.text(0.5, 1.0, 'ML READINESS SCORECARD', transform=ax.transAxes,
            fontsize=14, fontweight='bold', ha='center', va='top')
    for label, value, color in scores:
        ax.text(0.05, y_offset - 0.02, '', transform=ax.transAxes)
        y_offset -= 0.095
        ax.add_patch(FancyBboxPatch((0.02, y_offset - 0.02), 0.96, 0.08,
                                     boxstyle="round,pad=0.01",
                                     facecolor=color, alpha=0.15,
                                     transform=ax.transAxes))
        ax.text(0.06, y_offset + 0.02, label, transform=ax.transAxes,
                fontsize=11, va='center')
        ax.text(0.92, y_offset + 0.02, value, transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='right', va='center', color=color)

    plt.tight_layout()
    fig.savefig(fig_dir / '08_ml_readiness.png')
    plt.close(fig)
    print(f"  → {fig_dir / '08_ml_readiness.png'}")

    return sep_results


# ═══════════════════════════════════════════════════════════
# Bonus: Per-Sample Spatial Field Maps
# ═══════════════════════════════════════════════════════════

def generate_spatial_maps(samples, fig_dir):
    """Generate per-sample spatial field maps (stress, displacement, defect)."""
    print("\n[Bonus] Per-Sample Spatial Field Maps")

    for s in samples:
        n = s['nodes']
        name = s['name']
        m = s['meta']
        r = np.sqrt(n['x']**2 + n['z']**2)
        theta = np.degrees(np.arctan2(n['z'], n['x']))
        arc = r * np.radians(theta)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        defect_type = m.get('defect_type', 'unknown')
        defect_id = m.get('defect_type_id', '?')
        fig.suptitle(f'{name} — {defect_type} (type {defect_id})\n'
                     f'θ={m.get("theta_deg","?")}°, z={m.get("z_center","?")} mm, '
                     f'r={m.get("radius","?")} mm',
                     fontsize=13, fontweight='bold')

        # (a) Von Mises stress
        ax = axes[0, 0]
        sc = ax.scatter(arc, n['y'], c=np.log10(n['smises'].clip(1e-6)),
                        s=0.3, alpha=0.4, cmap='hot')
        plt.colorbar(sc, ax=ax, label='log₁₀(σ_mises / MPa)')
        ax.set_xlabel('Arc Length (mm)')
        ax.set_ylabel('z (mm)')
        ax.set_title('Von Mises Stress')

        # (b) Displacement
        ax = axes[0, 1]
        sc = ax.scatter(arc, n['y'], c=n['u_mag'], s=0.3, alpha=0.4, cmap='viridis')
        plt.colorbar(sc, ax=ax, label='|u| (mm)')
        ax.set_xlabel('Arc Length (mm)')
        ax.set_ylabel('z (mm)')
        ax.set_title('Displacement Magnitude')

        # (c) Temperature
        ax = axes[1, 0]
        sc = ax.scatter(arc, n['y'], c=n['temp'], s=0.3, alpha=0.4, cmap='coolwarm')
        plt.colorbar(sc, ax=ax, label='T (°C)')
        ax.set_xlabel('Arc Length (mm)')
        ax.set_ylabel('z (mm)')
        ax.set_title('Temperature')

        # (d) Defect label
        ax = axes[1, 1]
        h_mask = n['defect_label'] == 0
        d_mask = n['defect_label'] > 0
        ax.scatter(arc[h_mask], n['y'][h_mask], s=0.2, alpha=0.15, c='#2ecc71', label='healthy')
        if d_mask.sum() > 0:
            ax.scatter(arc[d_mask], n['y'][d_mask], s=3, alpha=0.8, c='#e74c3c',
                       label=f'defect ({d_mask.sum():,} nodes)')
        ax.set_xlabel('Arc Length (mm)')
        ax.set_ylabel('z (mm)')
        ax.set_title('Defect Label Map')
        ax.legend(fontsize=8, markerscale=5)

        plt.tight_layout()
        fig.savefig(fig_dir / f'09_spatial_{name}.png')
        plt.close(fig)
        print(f"  → 09_spatial_{name}.png")


# ═══════════════════════════════════════════════════════════
# Bonus: Defect Zone Detail Close-up
# ═══════════════════════════════════════════════════════════

def generate_defect_closeups(samples, fig_dir):
    """Zoomed view of defect zones with stress contour."""
    print("\n[Bonus] Defect Zone Close-ups")

    for s in samples:
        n = s['nodes']
        m = s['meta']
        name = s['name']
        defect_type = m.get('defect_type', 'unknown')

        d = n[n['defect_label'] > 0]
        if len(d) == 0:
            continue

        # Compute arc/z for all nodes
        r_all = np.sqrt(n['x']**2 + n['z']**2)
        theta_all = np.degrees(np.arctan2(n['z'], n['x']))
        arc_all = r_all * np.radians(theta_all)

        # Defect bounding box
        r_d = np.sqrt(d['x']**2 + d['z']**2)
        theta_d = np.degrees(np.arctan2(d['z'], d['x']))
        arc_d = r_d * np.radians(theta_d)
        arc_min, arc_max = arc_d.min() - 100, arc_d.max() + 100
        z_min, z_max = d['y'].min() - 100, d['y'].max() + 100

        # Filter to neighborhood
        mask = (arc_all > arc_min) & (arc_all < arc_max) & (n['y'] > z_min) & (n['y'] < z_max)
        local = n[mask].copy()
        local_arc = arc_all[mask]
        local_z = n['y'][mask]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{name} — {defect_type} — Defect Zone Close-up', fontsize=13, fontweight='bold')

        # (a) Stress
        ax = axes[0]
        sc = ax.scatter(local_arc, local_z, c=np.log10(local['smises'].clip(1e-6)),
                        s=2, alpha=0.6, cmap='hot')
        plt.colorbar(sc, ax=ax, label='log₁₀(σ_mises)')
        ax.set_xlabel('Arc Length (mm)')
        ax.set_ylabel('z (mm)')
        ax.set_title('Stress Close-up')

        # (b) Displacement
        ax = axes[1]
        sc = ax.scatter(local_arc, local_z, c=local['u_mag'],
                        s=2, alpha=0.6, cmap='viridis')
        plt.colorbar(sc, ax=ax, label='|u| (mm)')
        ax.set_xlabel('Arc Length (mm)')
        ax.set_title('Displacement Close-up')

        # (c) Defect labels
        ax = axes[2]
        h_mask2 = local['defect_label'] == 0
        d_mask2 = local['defect_label'] > 0
        ax.scatter(local_arc[h_mask2.values], local_z[h_mask2.values],
                   s=1.5, alpha=0.3, c='#2ecc71')
        ax.scatter(local_arc[d_mask2.values], local_z[d_mask2.values],
                   s=4, alpha=0.8, c='#e74c3c')
        ax.set_xlabel('Arc Length (mm)')
        ax.set_title('Defect Label Close-up')

        plt.tight_layout()
        fig.savefig(fig_dir / f'10_closeup_{name}.png')
        plt.close(fig)
        print(f"  → 10_closeup_{name}.png")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Realistic FEM Dataset Validation')
    parser.add_argument('--dataset', default='dataset_realistic_25mm_100',
                        help='Dataset directory')
    parser.add_argument('--output', default='figures/validation_realistic',
                        help='Output figures directory')
    parser.add_argument('--report', default='figures/validation_realistic/report.json',
                        help='Validation report JSON')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    fig_dir = Path(args.output)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE REALISTIC FEM DATASET VALIDATION")
    print("=" * 70)
    print(f"Dataset: {dataset_dir}")
    print(f"Output:  {fig_dir}")

    # Load
    samples = load_samples(dataset_dir)
    print(f"\nLoaded {len(samples)} completed samples")
    if not samples:
        print("ERROR: No samples found!")
        sys.exit(1)

    report = {'n_samples': len(samples), 'dataset': str(dataset_dir)}

    # Run all validations
    validate_integrity(samples, report, )
    validate_physics(samples, report, fig_dir)
    validate_defect_labels(samples, report, fig_dir)
    validate_consistency(samples, report, fig_dir)
    validate_features(samples, report, fig_dir)
    validate_graph_build(samples, report, fig_dir)
    validate_ml_readiness(samples, report, fig_dir)
    generate_spatial_maps(samples, fig_dir)
    generate_defect_closeups(samples, fig_dir)

    # Save report
    report_path = Path(args.report)

    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(i) for i in obj]
        return obj

    with open(report_path, 'w') as f:
        json.dump(make_serializable(report), f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    n_figs = len(list(fig_dir.glob('*.png')))
    print(f"Generated {n_figs} figures in {fig_dir}/")


if __name__ == '__main__':
    main()
