# -*- coding: utf-8 -*-
"""
s12_czm_thermal_200 vs s12_thermal_500 データセット差分分析

なぜ同じ GAT で F1 が 0.78 vs 0.40 と倍も違うのかを調査する。
- ノード数・エッジ数の分布
- クラス分布（欠陥比率）
- 特徴量の統計（mean/std per dim）
- ラベル分布の偏り

Usage:
    python scripts/analyze_dataset_diff.py
"""

import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def analyze_dataset(name, data_dir):
    """Analyze a single dataset."""
    train_path = os.path.join(data_dir, 'train.pt')
    val_path = os.path.join(data_dir, 'val.pt')

    if not os.path.exists(train_path):
        print("  NOT FOUND: %s" % train_path)
        return None

    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)
    all_data = train_data + val_data

    print("\n" + "=" * 60)
    print("Dataset: %s" % name)
    print("=" * 60)
    print("  Graphs: %d train + %d val = %d total" % (
        len(train_data), len(val_data), len(all_data)))

    # Node/edge counts
    node_counts = [d.x.shape[0] for d in all_data]
    edge_counts = [d.edge_index.shape[1] for d in all_data]
    feat_dim = all_data[0].x.shape[1]
    edge_dim = all_data[0].edge_attr.shape[1] if all_data[0].edge_attr is not None else 0

    print("  Node features: %d dims" % feat_dim)
    print("  Edge features: %d dims" % edge_dim)
    print("  Nodes/graph: min=%d, max=%d, mean=%.0f, std=%.0f" % (
        min(node_counts), max(node_counts), np.mean(node_counts), np.std(node_counts)))
    print("  Edges/graph: min=%d, max=%d, mean=%.0f" % (
        min(edge_counts), max(edge_counts), np.mean(edge_counts)))

    # Label distribution
    all_labels = torch.cat([d.y for d in all_data])
    num_classes = int(all_labels.max().item()) + 1
    total_nodes = all_labels.numel()
    print("  Total nodes: %d" % total_nodes)
    print("  Num classes: %d" % num_classes)

    for c in range(num_classes):
        n_c = (all_labels == c).sum().item()
        print("    Class %d: %d nodes (%.4f%%)" % (c, n_c, 100.0 * n_c / total_nodes))

    defect_mask = all_labels > 0
    n_defect = defect_mask.sum().item()
    defect_ratio = n_defect / total_nodes
    print("  Defect ratio: %.4f%% (%d / %d)" % (defect_ratio * 100, n_defect, total_nodes))

    # Per-graph defect ratio
    per_graph_defect = []
    for d in all_data:
        n_def = (d.y > 0).sum().item()
        per_graph_defect.append(n_def / d.y.shape[0])
    print("  Per-graph defect ratio: min=%.4f%%, max=%.4f%%, mean=%.4f%%, std=%.4f%%" % (
        min(per_graph_defect) * 100, max(per_graph_defect) * 100,
        np.mean(per_graph_defect) * 100, np.std(per_graph_defect) * 100))

    # Feature statistics (raw, unnormalized)
    all_x = torch.cat([d.x for d in all_data], dim=0)
    print("\n  Feature statistics (per dim):")
    print("  %-4s  %-10s  %-10s  %-10s  %-10s" % ("Dim", "Mean", "Std", "Min", "Max"))
    for i in range(feat_dim):
        col = all_x[:, i]
        print("  %-4d  %-10.4f  %-10.4f  %-10.4f  %-10.4f" % (
            i, col.mean().item(), col.std().item(), col.min().item(), col.max().item()))

    # Defect node vs healthy node feature comparison
    if n_defect > 0:
        healthy_x = all_x[~defect_mask]
        defect_x = all_x[defect_mask]
        print("\n  Feature diff (defect_mean - healthy_mean):")
        print("  %-4s  %-12s  %-12s  %-12s" % ("Dim", "Healthy Mean", "Defect Mean", "Δ"))
        for i in range(feat_dim):
            h_mean = healthy_x[:, i].mean().item()
            d_mean = defect_x[:, i].mean().item()
            delta = d_mean - h_mean
            marker = " ***" if abs(delta) > 0.5 * max(abs(h_mean), 1e-6) else ""
            print("  %-4d  %-12.4f  %-12.4f  %+.4f%s" % (i, h_mean, d_mean, delta, marker))

    return {
        'n_graphs': len(all_data),
        'feat_dim': feat_dim,
        'mean_nodes': np.mean(node_counts),
        'defect_ratio': defect_ratio,
        'num_classes': num_classes,
    }


def main():
    datasets = {
        's12_czm_thermal_200': os.path.join(PROJECT_ROOT, 'data/processed_s12_czm_thermal_200'),
        's12_thermal_500': os.path.join(PROJECT_ROOT, 'data/processed_s12_thermal_500'),
    }

    # Also check for other datasets
    data_root = os.path.join(PROJECT_ROOT, 'data')
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, 'train.pt')):
            if entry not in [k.replace('/', '') for k in datasets]:
                pass  # Only analyze the two main ones

    results = {}
    for name, path in datasets.items():
        results[name] = analyze_dataset(name, path)

    # Comparison summary
    if all(v is not None for v in results.values()):
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        for key in ['n_graphs', 'feat_dim', 'mean_nodes', 'defect_ratio', 'num_classes']:
            vals = {k: v[key] for k, v in results.items()}
            print("  %-15s: %s" % (key, '  |  '.join(
                '%s=%.4f' % (k, v) if isinstance(v, float) else '%s=%s' % (k, v)
                for k, v in vals.items())))


if __name__ == '__main__':
    main()
