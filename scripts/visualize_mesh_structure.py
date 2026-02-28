#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
メッシュ構造の可視化 — 要素タイプ・接続・パーティション

外板の S4R/S3 シェル要素をワイヤーフレームで表示。
四面体ではなく四辺形シェルであることを可視化。

Usage:
  python scripts/visualize_mesh_structure.py
  python scripts/visualize_mesh_structure.py --data dataset_output/sample_0001
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


def load_mesh(sample_dir):
    """nodes.csv と elements.csv を読み込み"""
    nodes_path = os.path.join(sample_dir, 'nodes.csv')
    elems_path = os.path.join(sample_dir, 'elements.csv')
    if not os.path.exists(nodes_path) or not os.path.exists(elems_path):
        return None, None
    df_n = pd.read_csv(nodes_path)
    df_e = pd.read_csv(elems_path)
    return df_n, df_e


def plot_mesh_wireframe(df_nodes, df_elems, output_path, sample_id=None):
    """メッシュをワイヤーフレームで表示（要素エッジを描画）"""
    node_id_to_idx = {int(r['node_id']): i for i, r in df_nodes.iterrows()}
    coords = df_nodes[['x', 'y', 'z']].values / 1000  # mm -> m

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    n_s4r = n_s3 = 0
    for _, row in df_elems.iterrows():
        etype = row.get('elem_type', 'S4R')
        n1, n2, n3 = int(row['n1']), int(row['n2']), int(row['n3'])
        n4 = row.get('n4', -1)
        if pd.isna(n4) or n4 < 0:
            n4 = None

        pts = []
        for nid in [n1, n2, n3] + ([n4] if n4 is not None else []):
            if nid in node_id_to_idx:
                pts.append(coords[node_id_to_idx[nid]])
        if len(pts) < 3:
            continue

        if etype == 'S4R' and len(pts) == 4:
            n_s4r += 1
            # 四辺形: 4 エッジを描画
            for i in range(4):
                j = (i + 1) % 4
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                        [pts[i][2], pts[j][2]], 'b-', linewidth=0.3, alpha=0.6)
        else:
            n_s3 += 1
            # 三角形
            for i in range(3):
                j = (i + 1) % 3
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                        [pts[i][2], pts[j][2]], 'r-', linewidth=0.4, alpha=0.8)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y (Axis) [m]')
    ax.set_zlabel('Z [m]')
    title = 'Mesh Structure: S4R (blue) + S3 (red) — Shell Elements'
    if sample_id is not None:
        title += f' (Sample {sample_id:04d})'
    ax.set_title(title, fontsize=12)
    ax.set_box_aspect([1, 2, 1])
    ax.view_init(elev=20, azim=45)

    # 凡例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', linewidth=2, label=f'S4R (4-node quad): {n_s4r}'),
        Line2D([0], [0], color='r', linewidth=2, label=f'S3 (3-node tri): {n_s3}'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path} (S4R={n_s4r}, S3={n_s3})")


def plot_element_type_distribution(df_elems, output_path):
    """要素タイプの分布"""
    counts = df_elems['elem_type'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=['steelblue', 'coral'], edgecolor='white')
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 50,
                str(int(b.get_height())), ha='center', fontsize=11)
    ax.set_ylabel('Count')
    ax.set_title('Element Type Distribution (Outer Skin)\nS4R=4-node quad, S3=3-node tri — NOT tetrahedral')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default=os.path.join(PROJECT_ROOT, 'dataset_output', 'sample_0001'))
    parser.add_argument('--output', type=str,
                        default=os.path.join(PROJECT_ROOT, 'figures', 'mesh_structure'))
    args = parser.parse_args()

    df_n, df_e = load_mesh(args.data)
    if df_n is None:
        print("Error: nodes.csv or elements.csv not found in %s" % args.data)
        sys.exit(1)

    sample_id = int(os.path.basename(args.data).replace('sample_', '')) if 'sample_' in args.data else None

    print("Mesh structure visualization")
    print("  Nodes: %d | Elements: %d" % (len(df_n), len(df_e)))
    print("  Element types:", df_e['elem_type'].value_counts().to_dict())

    plot_element_type_distribution(df_e, os.path.join(args.output, '01_element_types.png'))

    # サブサンプルでワイヤーフレーム（全要素は重い）
    step = max(1, len(df_e) // 1500)
    df_e_sub = df_e.iloc[::step]
    plot_mesh_wireframe(df_n, df_e_sub, os.path.join(args.output, '02_mesh_wireframe.png'), sample_id)

    print("\nDone. See figures/mesh_structure/")


if __name__ == '__main__':
    main()
