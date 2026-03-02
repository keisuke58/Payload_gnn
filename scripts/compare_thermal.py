#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
熱 vs 非熱モデル比較スクリプト

3つの比較モードをサポート:
  1. メトリクス比較: 2つの学習ランのbest_model.ptを比較
  2. 学習曲線比較: training_log.csvを重ねてプロット
  3. データセット統計比較: processed data (train.pt) の特徴量分布を比較

Usage:
  # モード1: 2つのランを比較
  python scripts/compare_thermal.py \\
      --thermal_run runs/gat_thermal_xxx \\
      --baseline_run runs/gat_20260228_032142

  # モード2: データセット統計も比較
  python scripts/compare_thermal.py \\
      --thermal_run runs/gat_thermal_xxx \\
      --baseline_run runs/gat_20260228_032142 \\
      --thermal_data data/processed_realistic_25mm \\
      --baseline_data data/processed_25mm_100

  # モード3: データセットのみ比較 (学習前でもOK)
  python scripts/compare_thermal.py \\
      --thermal_data data/processed_realistic_25mm \\
      --baseline_data data/processed_25mm_100
"""

import argparse
import csv
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_checkpoint_metrics(run_dir):
    """best_model.pt からメトリクスを読み込む"""
    import torch
    path = os.path.join(run_dir, 'best_model.pt')
    if not os.path.exists(path):
        print("  WARNING: %s not found" % path)
        return None
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    metrics = ckpt.get('val_metrics', {})
    metrics['epoch'] = ckpt.get('epoch', -1)
    metrics['args'] = ckpt.get('args', {})
    return metrics


def load_training_log(run_dir):
    """training_log.csv を読み込む"""
    path = os.path.join(run_dir, 'training_log.csv')
    if not os.path.exists(path):
        print("  WARNING: %s not found" % path)
        return None
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_dataset_stats(data_dir):
    """train.pt から特徴量統計を計算"""
    import torch
    train_path = os.path.join(data_dir, 'train.pt')
    if not os.path.exists(train_path):
        print("  WARNING: %s not found" % train_path)
        return None
    data_list = torch.load(train_path, map_location='cpu', weights_only=False)

    # 全ノード特徴量を結合
    all_x = torch.cat([d.x for d in data_list], dim=0)
    all_y = torch.cat([d.y for d in data_list], dim=0)

    n_total = all_y.numel()
    n_defect = (all_y > 0).sum().item()

    stats = {
        'n_samples': len(data_list),
        'n_nodes_total': n_total,
        'n_defect': n_defect,
        'defect_ratio': n_defect / max(n_total, 1),
        'feature_dim': all_x.shape[1],
        'feature_mean': all_x.mean(dim=0).numpy(),
        'feature_std': all_x.std(dim=0).numpy(),
        'feature_min': all_x.min(dim=0).values.numpy(),
        'feature_max': all_x.max(dim=0).values.numpy(),
    }

    # 欠陥ノードと健全ノードの特徴量差
    defect_mask = all_y > 0
    if defect_mask.any():
        stats['defect_mean'] = all_x[defect_mask].mean(dim=0).numpy()
        stats['healthy_mean'] = all_x[~defect_mask].mean(dim=0).numpy()

    return stats


# 34次元特徴量の名前 (build_graph.py 準拠)
FEATURE_NAMES = [
    'x', 'y', 'z',                          # Position (3)
    'nx', 'ny', 'nz',                       # Normal (3)
    'k1', 'k2', 'H', 'K',                   # Curvature (4)
    'ux', 'uy', 'uz', 'u_mag',              # Displacement (4)
    'temp',                                   # Temperature (1)
    's11', 's22', 's12', 'smises', 'σ1+σ2',  # Stress (5)
    'thermal_smises',                         # Thermal stress (1)
    'le11', 'le22', 'le12',                  # Strain (3)
    'fiber_x', 'fiber_y', 'fiber_z',        # Fiber orient (3)
    'layup_0', 'layup_45', 'layup_-45', 'layup_90',  # Layup (4)
    'circ_angle',                            # Circumferential angle (1)
    'is_boundary', 'is_loaded',              # Boundary flags (2)
]


def print_metrics_comparison(thermal_m, baseline_m):
    """2つのランのメトリクスを比較表示"""
    print("\n" + "=" * 65)
    print(" Model Performance Comparison")
    print("=" * 65)

    # アーキテクチャ情報
    t_args = thermal_m.get('args', {})
    b_args = baseline_m.get('args', {})
    print("  Thermal : arch=%s, hidden=%s, layers=%s" % (
        t_args.get('arch', '?'), t_args.get('hidden', '?'), t_args.get('layers', '?')))
    print("  Baseline: arch=%s, hidden=%s, layers=%s" % (
        b_args.get('arch', '?'), b_args.get('hidden', '?'), b_args.get('layers', '?')))
    print()

    keys = [
        ('f1', 'F1 Score', True),
        ('auc', 'ROC-AUC', True),
        ('precision', 'Precision', True),
        ('recall', 'Recall', True),
        ('accuracy', 'Accuracy', True),
        ('loss', 'Val Loss', False),
        ('epoch', 'Best Epoch', None),
    ]

    print("%-15s %10s %10s %10s" % ("Metric", "Thermal", "Baseline", "Delta"))
    print("-" * 50)
    for key, label, higher_better in keys:
        t_val = thermal_m.get(key, 0.0)
        b_val = baseline_m.get(key, 0.0)
        delta = t_val - b_val
        if key == 'epoch':
            print("%-15s %10d %10d %+10d" % (label, int(t_val), int(b_val), int(delta)))
        else:
            marker = ''
            if higher_better is not None and abs(delta) > 0.001:
                won = (delta > 0) == higher_better
                marker = ' ✓' if won else ''
            print("%-15s %10.4f %10.4f %+10.4f%s" % (label, t_val, b_val, delta, marker))

    print("=" * 65)


def print_dataset_comparison(thermal_s, baseline_s):
    """2つのデータセットの統計を比較表示"""
    print("\n" + "=" * 65)
    print(" Dataset Statistics Comparison")
    print("=" * 65)

    print("%-20s %15s %15s" % ("", "Thermal", "Baseline"))
    print("-" * 55)
    print("%-20s %15d %15d" % ("Samples", thermal_s['n_samples'], baseline_s['n_samples']))
    print("%-20s %15d %15d" % ("Total nodes", thermal_s['n_nodes_total'], baseline_s['n_nodes_total']))
    print("%-20s %15d %15d" % ("Defect nodes", thermal_s['n_defect'], baseline_s['n_defect']))
    print("%-20s %14.4f%% %14.4f%%" % (
        "Defect ratio",
        thermal_s['defect_ratio'] * 100,
        baseline_s['defect_ratio'] * 100))
    print("%-20s %15d %15d" % ("Feature dim", thermal_s['feature_dim'], baseline_s['feature_dim']))
    print()

    # 特徴量ごとの比較 (mean ± std)
    n_feat = min(thermal_s['feature_dim'], baseline_s['feature_dim'], len(FEATURE_NAMES))
    print("Feature-wise statistics (mean ± std):")
    print("%-18s %22s %22s" % ("Feature", "Thermal", "Baseline"))
    print("-" * 65)

    # 大きく異なる特徴量をハイライト
    notable = []
    for i in range(n_feat):
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else 'feat_%d' % i
        t_mean, t_std = thermal_s['feature_mean'][i], thermal_s['feature_std'][i]
        b_mean, b_std = baseline_s['feature_mean'][i], baseline_s['feature_std'][i]
        print("%-18s %10.4f ± %-8.4f %10.4f ± %-8.4f" % (
            name, t_mean, t_std, b_mean, b_std))

        # 差が大きい特徴量を記録
        combined_std = max(t_std, b_std, 1e-6)
        diff_ratio = abs(t_mean - b_mean) / combined_std
        if diff_ratio > 0.5:
            notable.append((name, diff_ratio, t_mean, b_mean))

    if notable:
        notable.sort(key=lambda x: -x[1])
        print("\n  Notable differences (|delta_mean| / max_std > 0.5):")
        for name, ratio, t_val, b_val in notable[:10]:
            print("    %-18s ratio=%.2f  (thermal=%.4f, baseline=%.4f)" % (
                name, ratio, t_val, b_val))

    # 欠陥/健全ノードの特徴量差
    if 'defect_mean' in thermal_s and 'defect_mean' in baseline_s:
        print("\n  Defect-vs-Healthy feature gap (top discriminative features):")
        print("  %-18s %12s %12s" % ("Feature", "Thermal gap", "Baseline gap"))
        print("  " + "-" * 45)
        gaps = []
        for i in range(n_feat):
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else 'feat_%d' % i
            t_gap = abs(thermal_s['defect_mean'][i] - thermal_s['healthy_mean'][i])
            b_gap = abs(baseline_s['defect_mean'][i] - baseline_s['healthy_mean'][i])
            gaps.append((name, t_gap, b_gap))

        gaps.sort(key=lambda x: -(x[1] + x[2]))
        for name, t_gap, b_gap in gaps[:10]:
            marker = '  ← thermal stronger' if t_gap > b_gap * 1.2 else ''
            marker = '  ← baseline stronger' if b_gap > t_gap * 1.2 else marker
            print("  %-18s %12.6f %12.6f%s" % (name, t_gap, b_gap, marker))

    print("=" * 65)


def plot_training_curves(thermal_log, baseline_log, output_dir):
    """学習曲線をプロット"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Thermal vs Baseline Training Comparison', fontsize=14)

    metrics = [
        ('val_f1', 'Validation F1', axes[0, 0]),
        ('val_loss', 'Validation Loss', axes[0, 1]),
        ('train_f1', 'Train F1', axes[1, 0]),
        ('train_loss', 'Train Loss', axes[1, 1]),
    ]

    for key, title, ax in metrics:
        if thermal_log:
            epochs_t = [r['epoch'] for r in thermal_log]
            vals_t = [r.get(key, 0) for r in thermal_log]
            ax.plot(epochs_t, vals_t, label='Thermal', color='tab:red', linewidth=1.5)
        if baseline_log:
            epochs_b = [r['epoch'] for r in baseline_log]
            vals_b = [r.get(key, 0) for r in baseline_log]
            ax.plot(epochs_b, vals_b, label='Baseline', color='tab:blue', linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'thermal_vs_baseline_curves.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved: %s" % out_path)


def plot_feature_comparison(thermal_s, baseline_s, output_dir):
    """特徴量分布の比較プロット"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    n_feat = min(thermal_s['feature_dim'], baseline_s['feature_dim'], len(FEATURE_NAMES))

    # 特徴量ごとの mean 差分バーチャート
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 1. Mean comparison
    x = np.arange(n_feat)
    width = 0.35
    names = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else 'f%d' % i for i in range(n_feat)]
    axes[0].bar(x - width/2, thermal_s['feature_mean'][:n_feat], width, label='Thermal', color='tab:red', alpha=0.7)
    axes[0].bar(x + width/2, baseline_s['feature_mean'][:n_feat], width, label='Baseline', color='tab:blue', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0].set_title('Feature Mean Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. Defect-vs-healthy gap comparison
    if 'defect_mean' in thermal_s and 'defect_mean' in baseline_s:
        t_gaps = np.abs(thermal_s['defect_mean'][:n_feat] - thermal_s['healthy_mean'][:n_feat])
        b_gaps = np.abs(baseline_s['defect_mean'][:n_feat] - baseline_s['healthy_mean'][:n_feat])
        axes[1].bar(x - width/2, t_gaps, width, label='Thermal gap', color='tab:red', alpha=0.7)
        axes[1].bar(x + width/2, b_gaps, width, label='Baseline gap', color='tab:blue', alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        axes[1].set_title('Defect vs Healthy Feature Gap')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'thermal_vs_baseline_features.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved: %s" % out_path)


def main():
    parser = argparse.ArgumentParser(description='Thermal vs Baseline comparison')
    parser.add_argument('--thermal_run', type=str, default=None,
                        help='Thermal model run dir (e.g., runs/gat_thermal_xxx)')
    parser.add_argument('--baseline_run', type=str, default=None,
                        help='Baseline model run dir (e.g., runs/gat_20260228_032142)')
    parser.add_argument('--thermal_data', type=str, default=None,
                        help='Thermal processed data dir (e.g., data/processed_realistic_25mm)')
    parser.add_argument('--baseline_data', type=str, default=None,
                        help='Baseline processed data dir (e.g., data/processed_25mm_100)')
    parser.add_argument('--output_dir', type=str, default='runs/comparison',
                        help='Output dir for plots')
    parser.add_argument('--no_plot', action='store_true', default=False,
                        help='Skip plot generation')
    args = parser.parse_args()

    # Resolve paths
    for attr in ['thermal_run', 'baseline_run', 'thermal_data', 'baseline_data', 'output_dir']:
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            setattr(args, attr, os.path.join(PROJECT_ROOT, val))

    if not args.thermal_run and not args.thermal_data:
        parser.print_help()
        print("\nERROR: --thermal_run か --thermal_data のどちらかが必要です")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Metrics comparison ---
    if args.thermal_run and args.baseline_run:
        print("Loading model checkpoints...")
        thermal_m = load_checkpoint_metrics(args.thermal_run)
        baseline_m = load_checkpoint_metrics(args.baseline_run)
        if thermal_m and baseline_m:
            print_metrics_comparison(thermal_m, baseline_m)

        # Training curves
        thermal_log = load_training_log(args.thermal_run)
        baseline_log = load_training_log(args.baseline_run)
        if not args.no_plot and (thermal_log or baseline_log):
            print("\nPlotting training curves...")
            plot_training_curves(thermal_log, baseline_log, args.output_dir)

    # --- Dataset comparison ---
    if args.thermal_data and args.baseline_data:
        print("\nLoading dataset statistics...")
        thermal_s = load_dataset_stats(args.thermal_data)
        baseline_s = load_dataset_stats(args.baseline_data)
        if thermal_s and baseline_s:
            print_dataset_comparison(thermal_s, baseline_s)
            if not args.no_plot:
                print("\nPlotting feature comparison...")
                plot_feature_comparison(thermal_s, baseline_s, args.output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
