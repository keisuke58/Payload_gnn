#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特徴量重要度分析 (Permutation Importance)

学習済みモデルに対し、各特徴量を1つずつシャッフルして
F1スコアの低下量で重要度を測定する。

Usage:
  python scripts/feature_importance.py \
      --checkpoint runs/gat_20260228_032142/best_model.pt \
      --data_dir data/processed_25mm_100

  # 繰り返し回数指定 (精度向上、時間増)
  python scripts/feature_importance.py \
      --checkpoint runs/gat_xxx/best_model.pt \
      --data_dir data/processed_realistic_25mm \
      --n_repeats 5

  # プロットなし (テキスト出力のみ)
  python scripts/feature_importance.py \
      --checkpoint runs/gat_xxx/best_model.pt \
      --data_dir data/processed_realistic_25mm \
      --no_plot
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from models import build_model

# 34次元特徴量の名前 (build_graph.py 準拠)
FEATURE_NAMES = [
    'x', 'y', 'z',
    'nx', 'ny', 'nz',
    'k1', 'k2', 'H', 'K',
    'ux', 'uy', 'uz', 'u_mag',
    'temp',
    's11', 's22', 's12', 'smises', 'sigma_sum',
    'thermal_smises',
    'le11', 'le22', 'le12',
    'fiber_x', 'fiber_y', 'fiber_z',
    'layup_0', 'layup_45', 'layup_m45', 'layup_90',
    'circ_angle',
    'is_boundary', 'is_loaded',
]

# 特徴量グループ
FEATURE_GROUPS = {
    'Position':     [0, 1, 2],
    'Normal':       [3, 4, 5],
    'Curvature':    [6, 7, 8, 9],
    'Displacement': [10, 11, 12, 13],
    'Temperature':  [14],
    'Stress':       [15, 16, 17, 18, 19],
    'ThermalStress':[20],
    'Strain':       [21, 22, 23],
    'FiberOrient':  [24, 25, 26],
    'Layup':        [27, 28, 29, 30],
    'CircAngle':    [31],
    'Boundary':     [32, 33],
}


@torch.no_grad()
def evaluate_f1(model, data_list, device):
    """全データに対するF1を計算"""
    model.eval()
    all_preds, all_targets = [], []
    for data in data_list:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch if hasattr(data, 'batch') else None)
        preds = out.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(data.y.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return f1_score(targets, preds, zero_division=0)


@torch.no_grad()
def permutation_importance(model, data_list, device, n_repeats=3):
    """
    Permutation importance: 各特徴量をシャッフルしてF1低下を測定

    Returns:
        base_f1: シャッフルなしのF1
        importances: dict[feat_idx] = {'mean': float, 'std': float, 'drops': list}
    """
    base_f1 = evaluate_f1(model, data_list, device)
    n_features = data_list[0].x.shape[1]

    importances = {}
    for feat_idx in range(n_features):
        drops = []
        for _ in range(n_repeats):
            # 各グラフの該当特徴量をシャッフル
            shuffled_data = []
            for data in data_list:
                d = data.clone()
                d = d.to(device)
                # ノード間でシャッフル
                perm = torch.randperm(d.x.shape[0], device=device)
                d.x = d.x.clone()
                d.x[:, feat_idx] = d.x[perm, feat_idx]
                shuffled_data.append(d)

            shuffled_f1 = evaluate_f1(model, shuffled_data, device)
            drops.append(base_f1 - shuffled_f1)

        importances[feat_idx] = {
            'mean': float(np.mean(drops)),
            'std': float(np.std(drops)),
            'drops': drops,
        }

    return base_f1, importances


@torch.no_grad()
def group_importance(model, data_list, device, n_repeats=3):
    """グループ単位の重要度 (グループ内全特徴量を同時シャッフル)"""
    base_f1 = evaluate_f1(model, data_list, device)
    n_features = data_list[0].x.shape[1]

    group_results = {}
    for group_name, indices in FEATURE_GROUPS.items():
        valid_indices = [i for i in indices if i < n_features]
        if not valid_indices:
            continue

        drops = []
        for _ in range(n_repeats):
            shuffled_data = []
            for data in data_list:
                d = data.clone()
                d = d.to(device)
                d.x = d.x.clone()
                perm = torch.randperm(d.x.shape[0], device=device)
                for idx in valid_indices:
                    d.x[:, idx] = d.x[perm, idx]
                shuffled_data.append(d)
            shuffled_f1 = evaluate_f1(model, shuffled_data, device)
            drops.append(base_f1 - shuffled_f1)

        group_results[group_name] = {
            'mean': float(np.mean(drops)),
            'std': float(np.std(drops)),
            'indices': valid_indices,
        }

    return group_results


def print_results(base_f1, importances, group_results):
    """結果をテーブル表示"""
    n_features = len(importances)

    print("\n" + "=" * 60)
    print(" Permutation Feature Importance")
    print(" Base F1: %.4f" % base_f1)
    print("=" * 60)

    # 個別特徴量 (重要度降順)
    sorted_feats = sorted(importances.items(), key=lambda x: -x[1]['mean'])

    print("\n--- Per-Feature Importance (F1 drop when shuffled) ---")
    print("%-5s %-18s %10s %10s %8s" % ("Rank", "Feature", "F1 drop", "± std", "Impact"))
    print("-" * 55)
    for rank, (idx, imp) in enumerate(sorted_feats, 1):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else 'feat_%d' % idx
        # Impact rating
        if imp['mean'] > 0.01:
            impact = '★★★'
        elif imp['mean'] > 0.005:
            impact = '★★'
        elif imp['mean'] > 0.001:
            impact = '★'
        elif imp['mean'] > 0:
            impact = '·'
        else:
            impact = '-'
        print("%-5d %-18s %+10.5f %10.5f %8s" % (
            rank, name, imp['mean'], imp['std'], impact))

    # 不要特徴量候補
    negligible = [(idx, imp) for idx, imp in sorted_feats if imp['mean'] <= 0.0]
    if negligible:
        print("\n  Candidate features for removal (F1 drop <= 0):")
        for idx, imp in negligible:
            name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else 'feat_%d' % idx
            print("    %s (drop=%.5f)" % (name, imp['mean']))

    # グループ重要度
    if group_results:
        print("\n--- Group Importance ---")
        print("%-15s %10s %10s %6s" % ("Group", "F1 drop", "± std", "#feat"))
        print("-" * 45)
        sorted_groups = sorted(group_results.items(), key=lambda x: -x[1]['mean'])
        for name, g in sorted_groups:
            print("%-15s %+10.5f %10.5f %6d" % (
                name, g['mean'], g['std'], len(g['indices'])))

    print("=" * 60)


def plot_importance(importances, group_results, output_dir):
    """重要度を棒グラフでプロット"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    n_features = len(importances)

    # --- Per-feature bar chart ---
    sorted_feats = sorted(importances.items(), key=lambda x: -x[1]['mean'])
    names = []
    means = []
    stds = []
    for idx, imp in sorted_feats:
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else 'feat_%d' % idx
        names.append(name)
        means.append(imp['mean'])
        stds.append(imp['std'])

    fig, ax = plt.subplots(figsize=(12, max(6, n_features * 0.3)))
    y_pos = np.arange(len(names))
    colors = ['tab:red' if m > 0.01 else 'tab:orange' if m > 0.005
              else 'tab:blue' if m > 0 else 'tab:gray' for m in means]
    ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('F1 Score Drop (higher = more important)')
    ax.set_title('Permutation Feature Importance')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved: %s" % out_path)

    # --- Group bar chart ---
    if group_results:
        sorted_groups = sorted(group_results.items(), key=lambda x: -x[1]['mean'])
        g_names = [g[0] for g in sorted_groups]
        g_means = [g[1]['mean'] for g in sorted_groups]
        g_stds = [g[1]['std'] for g in sorted_groups]

        fig, ax = plt.subplots(figsize=(10, 5))
        x_pos = np.arange(len(g_names))
        colors = ['tab:red' if m > 0.01 else 'tab:orange' if m > 0.005
                  else 'tab:blue' if m > 0 else 'tab:gray' for m in g_means]
        ax.bar(x_pos, g_means, yerr=g_stds, color=colors, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(g_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('F1 Score Drop')
        ax.set_title('Feature Group Importance')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        out_path = os.path.join(output_dir, 'feature_group_importance.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print("  Saved: %s" % out_path)


def main():
    parser = argparse.ArgumentParser(description='Feature importance analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Processed data dir (with val.pt and norm_stats.pt)')
    parser.add_argument('--n_repeats', type=int, default=3,
                        help='Number of shuffle repeats per feature')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'],
                        help='Which split to evaluate on')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir for plots (default: same as checkpoint dir)')
    parser.add_argument('--no_plot', action='store_true', default=False)
    args = parser.parse_args()

    # Resolve paths
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(PROJECT_ROOT, args.checkpoint)
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint)
    elif not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(PROJECT_ROOT, args.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    print("Loading checkpoint: %s" % args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})

    in_channels = ckpt.get('in_channels', 34)
    edge_attr_dim = ckpt.get('edge_attr_dim', 0)
    arch = ckpt_args.get('arch', 'gat')
    hidden = ckpt_args.get('hidden', 128)
    layers = ckpt_args.get('layers', 4)
    dropout = ckpt_args.get('dropout', 0.1)
    num_classes = ckpt_args.get('num_classes', 2)
    use_residual = ckpt_args.get('residual', False)

    # Auto-detect num_classes from state_dict
    state = ckpt.get('model_state_dict', ckpt)
    for key in state:
        if 'head' in key and key.endswith('.weight'):
            num_classes = max(num_classes, state[key].shape[0])

    model = build_model(
        arch, in_channels, edge_attr_dim,
        hidden_channels=hidden, num_layers=layers,
        dropout=dropout, num_classes=num_classes,
        use_residual=use_residual,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Model: %s | in=%d | hidden=%d | layers=%d | classes=%d" % (
        arch.upper(), in_channels, hidden, layers, num_classes))

    # Load data
    data_path = os.path.join(args.data_dir, '%s.pt' % args.split)
    print("Loading data: %s" % data_path)
    data_list = torch.load(data_path, map_location='cpu', weights_only=False)
    print("  %d graphs loaded" % len(data_list))

    # Normalize features
    norm_path = os.path.join(args.data_dir, 'norm_stats.pt')
    if os.path.exists(norm_path):
        norm_stats = torch.load(norm_path, map_location='cpu', weights_only=False)
        x_mean = norm_stats['mean']
        x_std = norm_stats['std']
        for d in data_list:
            d.x = (d.x - x_mean) / x_std
        print("  Normalized with norm_stats.pt")

    # Run permutation importance
    print("\nRunning permutation importance (n_repeats=%d)..." % args.n_repeats)
    t0 = time.time()
    base_f1, importances = permutation_importance(
        model, data_list, device, n_repeats=args.n_repeats)
    elapsed = time.time() - t0
    print("  Done in %.1fs" % elapsed)

    # Run group importance
    print("Running group importance...")
    group_results = group_importance(
        model, data_list, device, n_repeats=args.n_repeats)

    # Results
    print_results(base_f1, importances, group_results)

    # Plots
    if not args.no_plot:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_importance(importances, group_results, args.output_dir)

    # Save raw results
    import json
    result_path = os.path.join(args.output_dir, 'feature_importance.json')
    result = {
        'base_f1': base_f1,
        'n_repeats': args.n_repeats,
        'features': {},
        'groups': {},
    }
    for idx, imp in importances.items():
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else 'feat_%d' % idx
        result['features'][name] = {'mean': imp['mean'], 'std': imp['std']}
    for gname, g in group_results.items():
        result['groups'][gname] = {'mean': g['mean'], 'std': g['std']}

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print("  Saved: %s" % result_path)


if __name__ == '__main__':
    main()
