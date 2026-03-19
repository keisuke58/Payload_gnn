#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build PyG graph datasets from fairing separation FEM results.

Converts extracted CSV (nodes + elements) from separation ODB into
PyTorch Geometric Data objects for GNN-based anomaly detection.

Anomaly labeling strategy:
  - Normal case (Sep-v2-Normal): all nodes label=0 (healthy)
  - Stuck bolt cases: nodes near stuck bolt positions label=1 (anomaly)
    Proximity criterion: nodes within stuck_radius of bolt position
    exhibit abnormal stress/displacement patterns

Usage:
  python src/build_separation_graph.py \
      --results_dir results/separation \
      --output_dir data/processed_separation \
      --cases Sep-v2-Normal Sep-v2-Stuck3 Sep-v2-Stuck6
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def build_graph_from_csv(nodes_csv, elems_csv, label_mode='normal',
                          anomaly_nodes=None, verbose=True):
    """Build PyG Data from separation CSV files.

    Args:
        nodes_csv: path to nodes CSV (from extract_separation_results.py --graph)
        elems_csv: path to elements CSV
        label_mode: 'normal' (all 0), 'anomaly' (use anomaly_nodes set)
        anomaly_nodes: set of node_ids to label as anomaly (label=1)

    Returns:
        PyG Data object
    """
    df_nodes = pd.read_csv(nodes_csv)
    df_elems = pd.read_csv(elems_csv)

    n_nodes = len(df_nodes)
    if verbose:
        print("  Nodes: %d, Elements: %d" % (n_nodes, len(df_elems)))

    # Node coordinates
    coords = df_nodes[['x', 'y', 'z']].values.astype(np.float32)

    # Node features: position(3) + displacement(4) + stress(4) + velocity(4) = 15 dim
    feat_cols = []

    # Position (3)
    feat_cols.extend(['x', 'y', 'z'])

    # Displacement (4)
    for col in ['ux', 'uy', 'uz', 'u_mag']:
        if col in df_nodes.columns:
            feat_cols.append(col)

    # Stress (4)
    for col in ['s11', 's22', 's12', 'smises']:
        if col in df_nodes.columns:
            feat_cols.append(col)

    # Velocity (4)
    for col in ['vx', 'vy', 'vz', 'v_mag']:
        if col in df_nodes.columns:
            feat_cols.append(col)

    x = torch.tensor(df_nodes[feat_cols].values, dtype=torch.float32)
    if verbose:
        print("  Node features: %d dims (%s)" % (x.shape[1], ', '.join(feat_cols)))

    # Build edge_index from element connectivity
    edges = set()
    for _, row in df_elems.iterrows():
        nodes = [int(row['n1']), int(row['n2']), int(row['n3'])]
        n4 = int(row['n4'])
        if n4 > 0:
            nodes.append(n4)

        # All edges within element
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                ni, nj = nodes[i], nodes[j]
                if ni > 0 and nj > 0:
                    # Convert to 0-indexed
                    edges.add((ni - 1, nj - 1))
                    edges.add((nj - 1, ni - 1))

    if not edges:
        print("  WARNING: No edges found!")
        return None

    edge_list = sorted(edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    if verbose:
        print("  Edges: %d" % edge_index.shape[1])

    # Edge features: relative position (3) + distance (1) = 4 dim
    src, dst = edge_index[0], edge_index[1]
    pos_tensor = torch.tensor(coords, dtype=torch.float32)
    diff = pos_tensor[dst] - pos_tensor[src]
    dist = torch.norm(diff, dim=1, keepdim=True)
    edge_attr = torch.cat([diff, dist], dim=1)

    # Labels
    if label_mode == 'anomaly' and anomaly_nodes is not None:
        y = torch.zeros(n_nodes, dtype=torch.long)
        node_ids = df_nodes['node_id'].values
        for nid in anomaly_nodes:
            idx = np.where(node_ids == nid)[0]
            if len(idx) > 0:
                y[idx[0]] = 1
        n_anomaly = int(y.sum())
        if verbose:
            print("  Labels: %d anomaly / %d total (%.2f%%)" % (
                n_anomaly, n_nodes, 100.0 * n_anomaly / n_nodes))
    else:
        y = torch.zeros(n_nodes, dtype=torch.long)
        if verbose:
            print("  Labels: all healthy (normal case)")

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        pos=pos_tensor,
    )
    return data


def find_anomaly_nodes_by_stress(normal_csv, stuck_csv, threshold_factor=2.0):
    """Identify anomaly nodes by comparing stress between normal and stuck cases.

    Nodes where |stress_stuck - stress_normal| / stress_normal > threshold
    are labeled as anomaly.
    """
    df_normal = pd.read_csv(normal_csv)
    df_stuck = pd.read_csv(stuck_csv)

    if 'smises' not in df_normal.columns or 'smises' not in df_stuck.columns:
        print("  WARNING: smises not available, falling back to displacement")
        col = 'u_mag'
    else:
        col = 'smises'

    # Align by node_id
    normal_vals = df_normal.set_index('node_id')[col]
    stuck_vals = df_stuck.set_index('node_id')[col]

    common_ids = normal_vals.index.intersection(stuck_vals.index)
    n_vals = normal_vals.loc[common_ids]
    s_vals = stuck_vals.loc[common_ids]

    # Relative difference
    safe_denom = n_vals.abs().clip(lower=1e-6)
    rel_diff = ((s_vals - n_vals).abs() / safe_denom)

    # Threshold
    anomaly_mask = rel_diff > threshold_factor
    anomaly_ids = set(common_ids[anomaly_mask].tolist())

    print("  Anomaly detection (%s, threshold=%.1fx):" % (col, threshold_factor))
    print("    %d / %d nodes (%.2f%%)" % (
        len(anomaly_ids), len(common_ids),
        100.0 * len(anomaly_ids) / len(common_ids)))

    return anomaly_ids


def main():
    parser = argparse.ArgumentParser(
        description='Build PyG graph data from fairing separation results')
    parser.add_argument('--results_dir', default='results/separation',
                        help='Directory with extracted CSVs')
    parser.add_argument('--output_dir', default='data/processed_separation',
                        help='Output directory for PyG data')
    parser.add_argument('--normal_case', default='Sep-v2-Normal',
                        help='Normal (reference) case name')
    parser.add_argument('--stuck_cases', nargs='+',
                        default=['Sep-v2-Stuck3', 'Sep-v2-Stuck6'],
                        help='Stuck bolt case names')
    parser.add_argument('--threshold', type=float, default=2.0,
                        help='Anomaly threshold (stress ratio)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check if graph CSVs exist
    normal_nodes = os.path.join(args.results_dir,
                                 '%s_nodes.csv' % args.normal_case)
    normal_elems = os.path.join(args.results_dir,
                                 '%s_elements.csv' % args.normal_case)

    if not os.path.exists(normal_nodes):
        print("ERROR: Normal case nodes CSV not found: %s" % normal_nodes)
        print("Run: qsub -v JOB_NAME=%s scripts/qsub_extract_sep_graph.sh" %
              args.normal_case)
        return

    print("=" * 60)
    print("Building separation graph dataset")
    print("=" * 60)

    all_graphs = []

    # 1. Normal case (label=0 for all nodes)
    print("\n[Normal] %s" % args.normal_case)
    data_normal = build_graph_from_csv(normal_nodes, normal_elems,
                                        label_mode='normal')
    if data_normal is not None:
        data_normal.case_name = args.normal_case
        data_normal.is_anomaly = False
        all_graphs.append(data_normal)

    # 2. Stuck cases (anomaly labels based on stress comparison)
    for case_name in args.stuck_cases:
        stuck_nodes = os.path.join(args.results_dir,
                                    '%s_nodes.csv' % case_name)
        stuck_elems = os.path.join(args.results_dir,
                                    '%s_elements.csv' % case_name)

        if not os.path.exists(stuck_nodes):
            print("\n[SKIP] %s: nodes CSV not found" % case_name)
            continue

        print("\n[Anomaly] %s" % case_name)

        # Find anomaly nodes by comparing with normal case
        anomaly_ids = find_anomaly_nodes_by_stress(
            normal_nodes, stuck_nodes,
            threshold_factor=args.threshold)

        data_stuck = build_graph_from_csv(stuck_nodes, stuck_elems,
                                           label_mode='anomaly',
                                           anomaly_nodes=anomaly_ids)
        if data_stuck is not None:
            data_stuck.case_name = case_name
            data_stuck.is_anomaly = True
            all_graphs.append(data_stuck)

    if not all_graphs:
        print("\nERROR: No graphs built")
        return

    # Save dataset
    print("\n" + "=" * 60)
    print("Dataset summary:")
    for g in all_graphs:
        n_anom = int(g.y.sum()) if g.y is not None else 0
        print("  %s: %d nodes, %d edges, %d anomaly nodes" % (
            g.case_name, g.x.shape[0], g.edge_index.shape[1], n_anom))

    # Save as list for train.py compatibility
    dataset_path = os.path.join(args.output_dir, 'separation_dataset.pt')
    torch.save(all_graphs, dataset_path)
    print("\nSaved: %s (%d graphs)" % (dataset_path, len(all_graphs)))

    # Also save train/val split (normal=train, stuck=val for anomaly detection)
    train_data = [g for g in all_graphs if not g.is_anomaly]
    val_data = [g for g in all_graphs if g.is_anomaly]
    torch.save(train_data, os.path.join(args.output_dir, 'train.pt'))
    torch.save(val_data, os.path.join(args.output_dir, 'val.pt'))
    print("  train.pt: %d graphs (normal)" % len(train_data))
    print("  val.pt: %d graphs (anomaly)" % len(val_data))

    # Compute and save normalization stats from train set
    all_x = torch.cat([g.x for g in train_data], dim=0)
    all_ea = torch.cat([g.edge_attr for g in train_data], dim=0)
    norm_stats = {
        'mean': all_x.mean(dim=0),
        'std': all_x.std(dim=0).clamp(min=1e-6),
        'edge_attr_mean': all_ea.mean(dim=0),
        'edge_attr_std': all_ea.std(dim=0).clamp(min=1e-6),
    }
    torch.save(norm_stats, os.path.join(args.output_dir, 'norm_stats.pt'))
    print("  norm_stats.pt saved")


if __name__ == '__main__':
    main()
