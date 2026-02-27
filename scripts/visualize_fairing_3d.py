#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H3 フェアリング 3次元可視化

Barrel + Ogive 形状の 1/6 セクションを 3D プロット。
FEM 節点のオーバーレイ対応。

Usage:
  python scripts/visualize_fairing_3d.py
  python scripts/visualize_fairing_3d.py --data dataset_output/healthy_baseline --output figures/fairing_3d.png
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Geometry (match generate_fairing_dataset.py)
RADIUS = 2600.0
H_BARREL = 5000.0
H_NOSE = 5400.0
HEIGHT = H_BARREL + H_NOSE
ANGLE = 60.0  # deg, 1/6 sector

OGIVE_XC = (RADIUS**2 - H_NOSE**2) / (2 * RADIUS)
OGIVE_RHO = RADIUS - OGIVE_XC


def get_radius_at_z(z):
    if z <= H_BARREL:
        return RADIUS
    dz = z - H_BARREL
    term = OGIVE_RHO**2 - dz**2
    if term < 0:
        return 0.0
    return OGIVE_XC + math.sqrt(term)


def create_fairing_surface(n_theta=30, n_z=80):
    """Create (X, Y, Z) mesh for 1/6 sector outer surface."""
    theta_deg = np.linspace(0, ANGLE, n_theta)
    z_vals = np.linspace(0, HEIGHT, n_z)
    theta_rad = np.radians(theta_deg)

    Z, Theta = np.meshgrid(z_vals, theta_rad)
    R = np.array([[get_radius_at_z(z) for z in row] for row in Z])
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    return X / 1000, Y / 1000, Z / 1000  # convert to m


def load_nodes_3d(nodes_path):
    """Load nodes.csv and return (x, y, z) in meters."""
    if not os.path.exists(nodes_path):
        return None
    df = pd.read_csv(nodes_path)
    return df['x'].values / 1000, df['y'].values / 1000, df['z'].values / 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default=os.path.join(PROJECT_ROOT, 'dataset_output', 'healthy_baseline'))
    parser.add_argument('--output', type=str,
                        default=os.path.join(PROJECT_ROOT, 'figures', 'fairing_3d.png'))
    parser.add_argument('--no-nodes', action='store_true', help='Skip FEM nodes overlay')
    args = parser.parse_args()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Fairing surface (1/6 sector)
    X, Y, Z = create_fairing_surface()
    ax.plot_surface(X, Y, Z, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=0.2)

    # Bottom cap (z=0 circle)
    theta_cap = np.linspace(0, np.radians(ANGLE), 20)
    x_cap = (RADIUS / 1000) * np.cos(theta_cap)
    y_cap = (RADIUS / 1000) * np.sin(theta_cap)
    ax.plot(x_cap, y_cap, np.zeros_like(x_cap), 'k-', linewidth=1)

    # FEM nodes overlay
    if not args.no_nodes:
        xyz = load_nodes_3d(os.path.join(args.data, 'nodes.csv'))
        if xyz is not None:
            ax.scatter(xyz[0], xyz[1], xyz[2], c='red', s=0.5, alpha=0.3, label='FEM nodes')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('H3 Fairing 3D View (1/6 sector, Barrel + Ogive)')
    ax.set_box_aspect([1, 0.3, 1.2])
    ax.view_init(elev=20, azim=45)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
