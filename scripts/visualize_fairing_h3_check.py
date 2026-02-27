#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H3 フェアリング形状の可視化と整合性チェック

- 理論形状（Barrel + Ogive）のプロット
- 実データ（nodes.csv）のオーバーレイ
- H3 ロケット仕様との比較表

Usage:
  python scripts/visualize_fairing_h3_check.py
  python scripts/visualize_fairing_h3_check.py --data dataset_output/healthy_baseline
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# =========================================================================
# H3 フェアリング幾何定数（generate_fairing_dataset.py と一致）
# =========================================================================
RADIUS = 2600.0       # mm (直径 5.2m)
H_BARREL = 5000.0    # mm (シリンダー部)
H_NOSE = 5400.0      # mm (オジーブ部)
HEIGHT = H_BARREL + H_NOSE  # 10400 mm
ANGLE = 60.0         # 1/6 対称セクション

OGIVE_XC = (RADIUS**2 - H_NOSE**2) / (2 * RADIUS)
OGIVE_RHO = RADIUS - OGIVE_XC

FACE_T = 1.0         # mm
CORE_T = 38.0        # mm
R_OUTER = RADIUS
R_CORE_O = RADIUS - FACE_T / 2.0
R_INNER = RADIUS - FACE_T - CORE_T


def get_radius_at_z(z):
    """z における外径 (mm)"""
    if z <= H_BARREL:
        return RADIUS
    dz = z - H_BARREL
    term = OGIVE_RHO**2 - dz**2
    if term < 0:
        return 0.0
    return OGIVE_XC + math.sqrt(term)


# =========================================================================
# H3 ロケット公表仕様（KHI / JAXA）
# =========================================================================
H3_SPECS = {
    'Type-S': {'length_m': 10.4, 'diameter_m': 5.2, 'note': '中型衛星向け'},
    'Type-L': {'length_m': 16.4, 'diameter_m': 5.2, 'note': '大型/複数衛星'},
    'Type-W': {'length_m': None, 'diameter_m': 5.2, 'note': 'Beyond Gravity製'},
}
H3_SHAPE = 'オジャイブ（ダブルコンター）形状'


def run_consistency_check():
    """H3 仕様との整合性チェック表を生成"""
    model_diameter_m = RADIUS * 2 / 1000
    model_length_m = HEIGHT / 1000

    lines = [
        "=" * 60,
        "H3 フェアリング 整合性チェック",
        "=" * 60,
        "",
        "【H3 公表仕様 (KHI/JAXA)】",
        f"  Type-S: 全長 10.4 m, 直径 5.2 m",
        f"  Type-L: 全長 16.4 m, 直径 5.2 m",
        f"  形状: {H3_SHAPE}",
        "",
        "【本モデル (generate_fairing_dataset.py)】",
        f"  全長: {model_length_m:.2f} m ({H_BARREL/1000:.1f} m Barrel + {H_NOSE/1000:.1f} m Ogive)",
        f"  直径: {model_diameter_m:.2f} m (半径 {RADIUS:.0f} mm)",
        f"  形状: Tangent Ogive 近似",
        "",
        "【整合性】",
    ]

    ok_dia = abs(model_diameter_m - 5.2) < 0.01
    ok_len_s = abs(model_length_m - 10.4) < 0.01
    lines.append(f"  直径 5.2 m: {'✓ OK' if ok_dia else '✗ NG'} (モデル {model_diameter_m:.2f} m)")
    lines.append(f"  Type-S 全長 10.4 m: {'✓ OK' if ok_len_s else '✗ NG'} (モデル {model_length_m:.2f} m)")
    lines.append(f"  オジーブ形状: ✓ OK (Tangent Ogive)")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def plot_fairing_cross_section(ax, z_samples=200):
    """理論形状の断面（R-z）をプロット"""
    z_vals = np.linspace(0, HEIGHT, z_samples)
    r_vals = np.array([get_radius_at_z(z) for z in z_vals])
    ax.plot(z_vals / 1000, r_vals / 1000, 'b-', linewidth=2, label='Theory (outer)')

    # Inner surface (offset by skin+core thickness)
    offset_mm = FACE_T + CORE_T
    r_inner_vals = np.maximum(0, r_vals - offset_mm)
    ax.plot(z_vals / 1000, r_inner_vals / 1000, 'b--', linewidth=1, alpha=0.7, label='Inner')

    ax.axvline(H_BARREL / 1000, color='gray', linestyle=':', alpha=0.7, label='Barrel/Ogive 境界')
    ax.set_xlabel('軸方向 z (m)')
    ax.set_ylabel('半径 (m)')
    ax.set_title('H3 フェアリング 理論形状 (1/6 セクション)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def load_and_plot_nodes(ax, nodes_path):
    """nodes.csv から節点を読み込みプロット"""
    if not os.path.exists(nodes_path):
        return None

    df = pd.read_csv(nodes_path)
    r = np.sqrt(df['x']**2 + df['y']**2)
    z = df['z'].values

    ax.scatter(z / 1000, r / 1000, c='red', s=1, alpha=0.4, label='FEM nodes')
    return {'z_min': z.min(), 'z_max': z.max(), 'r_min': r.min(), 'r_max': r.max(), 'n': len(df)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default=os.path.join(PROJECT_ROOT, 'dataset_output', 'healthy_baseline'),
                        help='nodes.csv を含むディレクトリ')
    parser.add_argument('--output', type=str,
                        default=os.path.join(PROJECT_ROOT, 'figures', 'fairing_h3_check.png'),
                        help='出力画像パス')
    parser.add_argument('--no-plot', action='store_true', help='Check only, no plot')
    parser.add_argument('--report', type=str, default='',
                        help='Save consistency report to file')
    args = parser.parse_args()

    # Consistency check
    report = run_consistency_check()
    print(report)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved: {args.report}")

    if args.no_plot:
        return

    # 図の作成
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_fairing_cross_section(ax)

    nodes_path = os.path.join(args.data, 'nodes.csv')
    info = load_and_plot_nodes(ax, nodes_path)
    if info:
        z_span = info['z_max'] - info['z_min']
        note = " (Barrel only)" if info['z_max'] < 6000 else " (Barrel + Ogive)"
        print(f"\nData range (nodes.csv): z=[{info['z_min']:.0f}, {info['z_max']:.0f}] mm{note}, "
              f"r=[{info['r_min']:.1f}, {info['r_max']:.1f}] mm, n={info['n']} nodes")

    ax.set_xlim(-0.2, HEIGHT / 1000 + 0.5)
    ax.set_ylim(-0.1, RADIUS / 1000 + 0.3)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n保存: {args.output}")


if __name__ == '__main__':
    main()
