#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wiki 用可視化: メッシュズーム / 欠陥差分 / グラフ構造
出力先: wiki_repo/images/mesh_graph/
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, 'wiki_repo', 'images', 'mesh_graph')
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset_realistic_25mm_100')
PYG_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_realistic_25mm')


def load_sample_csv(sample_id):
    """CSV からノード・要素データを読む"""
    sample_dir = os.path.join(DATASET_DIR, 'sample_%04d' % sample_id)
    nodes_file = os.path.join(sample_dir, 'nodes.csv')
    elems_file = os.path.join(sample_dir, 'elements.csv')

    # nodes.csv
    nodes = np.genfromtxt(nodes_file, delimiter=',', names=True)
    # elements.csv
    elems_raw = []
    with open(elems_file, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            elems_raw.append(parts)

    return nodes, elems_raw


def to_cylindrical(x, y, z):
    """Abaqus座標 → 円筒展開 (theta, axial)"""
    r = np.sqrt(x**2 + z**2)
    theta = np.arctan2(z, x)
    return theta, y, r


# =========================================================================
# Fig 1: メッシュズーム — 欠陥部分の要素密度を見せる
# =========================================================================
def plot_mesh_zoom(sample_id=1):
    """欠陥周辺メッシュの拡大図 (展開座標)"""
    print("[1/3] Mesh zoom: sample_%04d" % sample_id)
    nodes, elems_raw = load_sample_csv(sample_id)

    x = nodes['x']
    y = nodes['y']
    z = nodes['z']
    label = nodes['defect_label']
    smises = nodes['smises']
    node_ids = nodes['node_id'].astype(int)

    theta, axial, _ = to_cylindrical(x, y, z)

    # ノードIDから配列インデックスへのマップ
    id_to_idx = {}
    for i, nid in enumerate(node_ids):
        id_to_idx[nid] = i

    # 欠陥領域の中心を特定
    defect_mask = label > 0.5
    if defect_mask.sum() == 0:
        print("  No defect nodes, skipping")
        return
    theta_c = np.median(theta[defect_mask])
    axial_c = np.median(axial[defect_mask])

    # ズーム範囲
    d_theta = 0.15  # rad
    d_axial = 400   # mm
    mask_zoom = (
        (np.abs(theta - theta_c) < d_theta) &
        (np.abs(axial - axial_c) < d_axial)
    )
    zoom_idx = set(np.where(mask_zoom)[0])

    # エッジ収集 (要素の辺)
    edges = []
    for parts in elems_raw:
        elem_type = parts[1]
        if elem_type in ('S4R', 'S4'):
            nids = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]
            pairs = [(0,1),(1,2),(2,3),(3,0)]
        elif elem_type in ('S3', 'S3R'):
            nids = [int(parts[2]), int(parts[3]), int(parts[4])]
            pairs = [(0,1),(1,2),(2,0)]
        else:
            continue
        idxs = [id_to_idx.get(n) for n in nids]
        if any(i is None for i in idxs):
            continue
        if not any(i in zoom_idx for i in idxs):
            continue
        for a, b in pairs:
            edges.append((idxs[a], idxs[b]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_i, (ax, color_data, cmap, title, clabel) in enumerate([
        (axes[0], smises, 'hot', 'von Mises Stress', 'MPa'),
        (axes[1], label, 'RdYlBu_r', 'Defect Label', ''),
    ]):
        # エッジ描画
        segments = []
        for i1, i2 in edges:
            segments.append([(theta[i1], axial[i1]), (theta[i2], axial[i2])])

        if ax_i == 0:
            lc = LineCollection(segments, colors='gray', linewidths=0.3, alpha=0.5)
        else:
            lc = LineCollection(segments, colors='gray', linewidths=0.3, alpha=0.3)
        ax.add_collection(lc)

        # ノード散布
        sc = ax.scatter(theta[mask_zoom], axial[mask_zoom],
                       c=color_data[mask_zoom], cmap=cmap,
                       s=4, edgecolors='none', zorder=2)
        plt.colorbar(sc, ax=ax, label=clabel, shrink=0.8)

        # 欠陥境界
        if defect_mask[mask_zoom].any():
            defect_in_zoom = mask_zoom & defect_mask
            t_def = theta[defect_in_zoom]
            a_def = axial[defect_in_zoom]
            ax.scatter(t_def, a_def, facecolors='none', edgecolors='lime',
                      s=8, linewidths=0.5, alpha=0.6, zorder=3)

        ax.set_xlim(theta_c - d_theta, theta_c + d_theta)
        ax.set_ylim(axial_c - d_axial, axial_c + d_axial)
        ax.set_xlabel('Theta (rad)')
        ax.set_ylabel('Axial Position Y (mm)')
        ax.set_title(title)
        ax.set_aspect('auto')

    fig.suptitle('Mesh Zoom: sample_%04d (defect region)' % sample_id,
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, '01_mesh_zoom_sample%04d.png' % sample_id)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: %s" % out)


# =========================================================================
# Fig 2: 欠陥あり vs なし の差分
# =========================================================================
def plot_defect_diff(sample_id=1):
    """欠陥サンプルの応力・変位を healthy と比較"""
    print("[2/3] Defect diff: sample_%04d" % sample_id)
    nodes, _ = load_sample_csv(sample_id)

    x = nodes['x']
    y_pos = nodes['y']
    z = nodes['z']
    label = nodes['defect_label']
    smises = nodes['smises']
    u_mag = nodes['u_mag']

    theta, axial, _ = to_cylindrical(x, y_pos, z)

    defect_mask = label > 0.5
    healthy_mask = ~defect_mask

    if defect_mask.sum() == 0:
        print("  No defect nodes, skipping")
        return

    # 欠陥中心
    theta_c = np.median(theta[defect_mask])
    axial_c = np.median(axial[defect_mask])

    # 広域表示
    d_theta = 0.3
    d_axial = 800
    mask_view = (
        (np.abs(theta - theta_c) < d_theta) &
        (np.abs(axial - axial_c) < d_axial)
    )

    # 健全領域の基準値
    healthy_in_view = mask_view & healthy_mask
    stress_baseline = np.median(smises[healthy_in_view]) if healthy_in_view.any() else 0
    disp_baseline = np.median(u_mag[healthy_in_view]) if healthy_in_view.any() else 0

    # 差分
    stress_diff = smises - stress_baseline
    disp_diff = u_mag - disp_baseline

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    plots = [
        (axes[0,0], smises, 'hot', 'von Mises Stress (MPa)', 'MPa'),
        (axes[0,1], stress_diff, 'seismic', 'Stress Anomaly (vs healthy median)', 'MPa'),
        (axes[1,0], u_mag, 'viridis', 'Displacement Magnitude (mm)', 'mm'),
        (axes[1,1], disp_diff, 'seismic', 'Displacement Anomaly (vs healthy median)', 'mm'),
    ]

    for ax, data, cmap, title, clabel in plots:
        if 'Anomaly' in title:
            vmax = np.percentile(np.abs(data[mask_view]), 98)
            vmin = -vmax
            sc = ax.scatter(theta[mask_view], axial[mask_view],
                           c=data[mask_view], cmap=cmap, s=3,
                           vmin=vmin, vmax=vmax, edgecolors='none')
        else:
            vmax = np.percentile(data[mask_view], 99)
            sc = ax.scatter(theta[mask_view], axial[mask_view],
                           c=data[mask_view], cmap=cmap, s=3,
                           vmin=0, vmax=vmax, edgecolors='none')
        plt.colorbar(sc, ax=ax, label=clabel, shrink=0.8)

        # 欠陥境界 (circle)
        defect_in_view = mask_view & defect_mask
        if defect_in_view.any():
            ax.scatter(theta[defect_in_view], axial[defect_in_view],
                      facecolors='none', edgecolors='lime', s=5,
                      linewidths=0.4, alpha=0.5, zorder=3)

        ax.set_xlim(theta_c - d_theta, theta_c + d_theta)
        ax.set_ylim(axial_c - d_axial, axial_c + d_axial)
        ax.set_xlabel('Theta (rad)')
        ax.set_ylabel('Axial Y (mm)')
        ax.set_title(title)

    fig.suptitle('Defect vs Healthy: sample_%04d' % sample_id,
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, '02_defect_diff_sample%04d.png' % sample_id)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: %s" % out)


# =========================================================================
# Fig 3: GNN グラフ構造の可視化
# =========================================================================
def plot_graph_structure(pyg_idx=0):
    """PyG グラフ: ノード + エッジ (展開座標)、欠陥ラベル色分け"""
    print("[3/3] Graph structure: PyG sample %d" % pyg_idx)

    train_path = os.path.join(PYG_DIR, 'train.pt')
    if not os.path.exists(train_path):
        print("  PyG data not found, skipping")
        return

    data_list = torch.load(train_path, weights_only=False)
    if pyg_idx >= len(data_list):
        print("  Index %d out of range (N=%d)" % (pyg_idx, len(data_list)))
        return

    g = data_list[pyg_idx]
    pos = g.pos.numpy()        # (N, 3)
    edge_index = g.edge_index.numpy()  # (2, E)
    y = g.y.numpy()            # (N,)

    x_coord, y_coord, z_coord = pos[:, 0], pos[:, 1], pos[:, 2]
    theta, axial, _ = to_cylindrical(x_coord, y_coord, z_coord)

    defect_mask = y > 0.5
    theta_c = np.median(theta[defect_mask]) if defect_mask.any() else np.median(theta)
    axial_c = np.median(axial[defect_mask]) if defect_mask.any() else np.median(axial)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # --- Panel 1: 全体グラフ (サブサンプル) ---
    ax = axes[0]
    # サブサンプル (全エッジは多すぎるので 1% 描画)
    n_edges = edge_index.shape[1]
    rng = np.random.RandomState(42)
    edge_sample = rng.choice(n_edges, min(n_edges // 100, 10000), replace=False)
    segments = []
    for ei in edge_sample:
        i, j = edge_index[0, ei], edge_index[1, ei]
        segments.append([(theta[i], axial[i]), (theta[j], axial[j])])
    lc = LineCollection(segments, colors='steelblue', linewidths=0.15, alpha=0.3)
    ax.add_collection(lc)

    # 全ノード
    colors = np.where(y > 0.5, 1.0, 0.0)
    sc = ax.scatter(theta, axial, c=colors, cmap='RdYlBu_r', s=0.5,
                   edgecolors='none', alpha=0.7)
    ax.set_xlabel('Theta (rad)')
    ax.set_ylabel('Axial Y (mm)')
    ax.set_title('Full Graph (N=%d, E=%d)\nedges: 1%% sampled' % (len(y), n_edges))
    ax.set_aspect('auto')

    # --- Panel 2: 欠陥周辺ズーム (エッジ付き) ---
    ax = axes[1]
    d_theta = 0.15
    d_axial = 400
    mask_zoom = (
        (np.abs(theta - theta_c) < d_theta) &
        (np.abs(axial - axial_c) < d_axial)
    )
    zoom_set = set(np.where(mask_zoom)[0])

    segments_zoom = []
    for ei in range(n_edges):
        i, j = edge_index[0, ei], edge_index[1, ei]
        if i in zoom_set and j in zoom_set:
            segments_zoom.append([(theta[i], axial[i]), (theta[j], axial[j])])

    # サブサンプル
    if len(segments_zoom) > 20000:
        idx_s = rng.choice(len(segments_zoom), 20000, replace=False)
        segments_zoom = [segments_zoom[i] for i in idx_s]

    lc = LineCollection(segments_zoom, colors='steelblue', linewidths=0.3, alpha=0.4)
    ax.add_collection(lc)

    sc = ax.scatter(theta[mask_zoom], axial[mask_zoom],
                   c=y[mask_zoom], cmap='RdYlBu_r', s=6,
                   edgecolors='none', zorder=2)
    ax.set_xlim(theta_c - d_theta, theta_c + d_theta)
    ax.set_ylim(axial_c - d_axial, axial_c + d_axial)
    ax.set_xlabel('Theta (rad)')
    ax.set_ylabel('Axial Y (mm)')
    ax.set_title('Graph Zoom: Defect Region\n(edges show k-NN connectivity)')

    # --- Panel 3: 超ズーム — 個別エッジが見えるスケール ---
    ax = axes[2]
    d_theta2 = 0.04
    d_axial2 = 100
    mask_ultra = (
        (np.abs(theta - theta_c) < d_theta2) &
        (np.abs(axial - axial_c) < d_axial2)
    )
    ultra_set = set(np.where(mask_ultra)[0])

    segments_ultra = []
    for ei in range(n_edges):
        i, j = edge_index[0, ei], edge_index[1, ei]
        if i in ultra_set and j in ultra_set:
            segments_ultra.append([(theta[i], axial[i]), (theta[j], axial[j])])

    lc = LineCollection(segments_ultra, colors='steelblue', linewidths=0.6, alpha=0.5)
    ax.add_collection(lc)

    sc = ax.scatter(theta[mask_ultra], axial[mask_ultra],
                   c=y[mask_ultra], cmap='RdYlBu_r', s=25,
                   edgecolors='k', linewidths=0.3, zorder=2)
    ax.set_xlim(theta_c - d_theta2, theta_c + d_theta2)
    ax.set_ylim(axial_c - d_axial2, axial_c + d_axial2)
    ax.set_xlabel('Theta (rad)')
    ax.set_ylabel('Axial Y (mm)')
    ax.set_title('Graph Ultra-Zoom\n(individual nodes & edges)')
    plt.colorbar(sc, ax=ax, label='Defect Label', shrink=0.8)

    fig.suptitle('GNN Graph Structure (PyG)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, '03_graph_structure.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: %s" % out)


if __name__ == '__main__':
    # 良いサンプルを選ぶ (データがあるもの)
    sample_id = 1
    if len(sys.argv) > 1:
        sample_id = int(sys.argv[1])

    plot_mesh_zoom(sample_id)
    plot_defect_diff(sample_id)
    plot_graph_structure(0)

    print("\nAll visualizations saved to: %s" % OUT_DIR)
