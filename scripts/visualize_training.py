#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 可視化スクリプト — 学習結果の図表生成

Roadmap Phase 4 対応:
  Fig 2: Training curves (Loss, F1 vs epoch)
  Fig 3: Confusion Matrix
  Fig 5: モデル比較バーチャート
  Table 1: モデル比較テーブル (F1, AUC, Precision, Recall, Params, Time)

Usage:
  # 単一 run の学習曲線 + Confusion Matrix
  python scripts/visualize_training.py --run_dir runs/gat_20260301_... --data_dir data/processed_realistic_25mm

  # 複数 run のモデル比較
  python scripts/visualize_training.py --compare runs/gcn_... runs/gat_... runs/gin_... runs/sage_...

  # Confusion Matrix のみ (チェックポイントから推論)
  python scripts/visualize_training.py --run_dir runs/gat_... --data_dir data/processed_realistic_25mm --confusion_only
"""

import os
import sys
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Consistent style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

COLORS = {
    'gcn': '#2563EB',
    'gat': '#DC2626',
    'gin': '#059669',
    'sage': '#7C3AED',
    'pointnet': '#D97706',
}

DEFECT_TYPE_NAMES = [
    'healthy', 'debonding', 'fod', 'impact',
    'delamination', 'inner_debond', 'thermal_prog', 'acoustic_fat',
]


# =========================================================================
# Data loading
# =========================================================================
def load_training_log(run_dir):
    """Load training_log.csv as dict of lists."""
    log_path = os.path.join(run_dir, 'training_log.csv')
    if not os.path.exists(log_path):
        raise FileNotFoundError("No training_log.csv in %s" % run_dir)
    data = {}
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return data


def load_checkpoint(run_dir):
    """Load best_model.pt checkpoint."""
    import torch
    ckpt_path = os.path.join(run_dir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        return None
    return torch.load(ckpt_path, weights_only=False)


def detect_arch(run_dir):
    """Detect architecture from run directory name."""
    name = os.path.basename(run_dir).lower()
    for arch in ['gcn', 'gat', 'gin', 'sage', 'pointnet']:
        if name.startswith(arch):
            return arch
    return 'unknown'


# =========================================================================
# Fig 2: Training Curves
# =========================================================================
def plot_training_curves(run_dir, out_dir=None):
    """Plot loss and F1 curves vs epoch."""
    data = load_training_log(run_dir)
    arch = detect_arch(run_dir)
    epochs = data['epoch']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    color = COLORS.get(arch, '#333333')

    # (a) Loss
    ax = axes[0]
    ax.plot(epochs, data['train_loss'], '-', color=color, alpha=0.8, label='Train')
    ax.plot(epochs, data['val_loss'], '--', color=color, alpha=0.6, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) F1
    ax = axes[1]
    ax.plot(epochs, data['train_f1'], '-', color=color, alpha=0.8, label='Train')
    ax.plot(epochs, data['val_f1'], '--', color=color, alpha=0.6, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('(b) F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # (c) AUC
    ax = axes[2]
    if 'val_auc' in data:
        ax.plot(epochs, data['val_auc'], '-', color=color, alpha=0.8, label='Val AUC')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('(c) AUC-ROC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.suptitle('%s — Training Curves' % arch.upper(), fontsize=14, fontweight='bold')
    plt.tight_layout()

    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, 'figures', 'training')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'training_curves_%s.png' % arch)
    plt.savefig(out_path)
    print("Saved: %s" % out_path)
    plt.close()
    return out_path


# =========================================================================
# Fig 2b: Multi-model Training Curves Overlay
# =========================================================================
def plot_training_curves_multi(run_dirs, out_dir=None):
    """Overlay training curves of multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for run_dir in run_dirs:
        data = load_training_log(run_dir)
        arch = detect_arch(run_dir)
        color = COLORS.get(arch, '#333333')
        epochs = data['epoch']

        axes[0].plot(epochs, data['val_loss'], '-', color=color, label=arch.upper(), lw=1.5)
        axes[1].plot(epochs, data['val_f1'], '-', color=color, label=arch.upper(), lw=1.5)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('(a) Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation F1')
    axes[1].set_title('(b) Validation F1')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    fig.suptitle('GNN Model Comparison — Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, 'figures', 'training')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'training_curves_comparison.png')
    plt.savefig(out_path)
    print("Saved: %s" % out_path)
    plt.close()
    return out_path


# =========================================================================
# Fig 3: Confusion Matrix
# =========================================================================
def plot_confusion_matrix(run_dir, data_dir, split='val', out_dir=None):
    """Generate confusion matrix from checkpoint + data."""
    import torch
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    ckpt = load_checkpoint(run_dir)
    if ckpt is None:
        print("No checkpoint found in %s" % run_dir)
        return None

    arch = detect_arch(run_dir)
    args_dict = ckpt.get('args', {})
    in_channels = ckpt['in_channels']
    edge_attr_dim = ckpt['edge_attr_dim']

    # Load data
    data_path = os.path.join(data_dir, '%s.pt' % split)
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)
    eval_data = torch.load(data_path, weights_only=False)

    # Detect num_classes from checkpoint (not data, which may have fewer classes)
    head_key = [k for k in ckpt['model_state_dict'] if 'head' in k and 'weight' in k]
    if head_key:
        num_classes = ckpt['model_state_dict'][head_key[-1]].shape[0]
    else:
        all_labels = torch.cat([d.y for d in eval_data])
        num_classes = max(int(all_labels.max().item()) + 1, 2)

    # Build model and load weights
    data_in_channels = eval_data[0].x.shape[1]
    if data_in_channels != in_channels:
        print("WARNING: Data features (%d) != checkpoint features (%d). "
              "Use matching data_dir." % (data_in_channels, in_channels))
        return None

    from models import build_model
    model = build_model(
        arch,
        in_channels, edge_attr_dim,
        hidden_channels=args_dict.get('hidden', 128),
        num_layers=args_dict.get('layers', 4),
        dropout=args_dict.get('dropout', 0.1),
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Inference
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data in eval_data:
            out = model(data.x, data.edge_index, data.edge_attr, None)
            preds = out.argmax(dim=1)
            all_preds.append(preds)
            all_targets.append(data.y)

    preds_np = torch.cat(all_preds).numpy()
    targets_np = torch.cat(all_targets).numpy()

    # Labels
    labels_present = sorted(set(targets_np.tolist()) | set(preds_np.tolist()))
    display_labels = [DEFECT_TYPE_NAMES[i] if i < len(DEFECT_TYPE_NAMES)
                      else 'class_%d' % i for i in labels_present]

    # Plot
    cm = confusion_matrix(targets_np, preds_np, labels=labels_present)
    fig, ax = plt.subplots(figsize=(8, 7))

    # Normalize for display but show counts
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('%s — Confusion Matrix (%s set)\n'
                 'Values: count (row-normalized %%)' % (arch.upper(), split),
                 fontsize=12)

    # Annotate cells
    for i in range(len(display_labels)):
        for j in range(len(display_labels)):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, '%d\n(%.1f%%)' % (count, pct),
                    ha='center', va='center', fontsize=8, color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Row-normalized')
    plt.tight_layout()

    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, 'figures', 'training')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'confusion_matrix_%s.png' % arch)
    plt.savefig(out_path)
    print("Saved: %s" % out_path)
    plt.close()
    return out_path


# =========================================================================
# Fig 5 + Table 1: Model Comparison
# =========================================================================
def compare_models(run_dirs, out_dir=None):
    """Generate comparison bar chart and table from multiple runs."""
    import torch

    results = []
    for run_dir in run_dirs:
        arch = detect_arch(run_dir)
        ckpt = load_checkpoint(run_dir)
        log = load_training_log(run_dir)

        if ckpt is None:
            continue

        val_m = ckpt.get('val_metrics', {})
        args_dict = ckpt.get('args', {})
        n_params = sum(p.numel() for p in
                       torch.load(os.path.join(run_dir, 'best_model.pt'),
                                  weights_only=False)['model_state_dict'].values())

        results.append({
            'arch': arch.upper(),
            'f1': val_m.get('f1', 0),
            'auc': val_m.get('auc', 0),
            'precision': val_m.get('precision', 0),
            'recall': val_m.get('recall', 0),
            'accuracy': val_m.get('accuracy', 0),
            'params': n_params,
            'best_epoch': ckpt.get('epoch', 0),
            'loss': args_dict.get('loss', 'unknown'),
        })

    if not results:
        print("No valid runs to compare.")
        return

    # Sort by F1
    results.sort(key=lambda r: r['f1'], reverse=True)

    # === Table 1: Print ===
    print("\n" + "=" * 90)
    print("Table 1: Model Comparison")
    print("=" * 90)
    print("%-8s  %8s  %8s  %8s  %8s  %8s  %10s  %6s" % (
        'Model', 'F1', 'AUC', 'Prec', 'Recall', 'Acc', 'Params', 'Epoch'))
    print("-" * 90)
    for r in results:
        print("%-8s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %10s  %6d" % (
            r['arch'], r['f1'], r['auc'], r['precision'], r['recall'],
            r['accuracy'], '{:,}'.format(r['params']), r['best_epoch']))
    print("=" * 90)

    # === Fig 5: Bar chart ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    archs = [r['arch'] for r in results]
    colors = [COLORS.get(r['arch'].lower(), '#333333') for r in results]
    x = np.arange(len(archs))

    # (a) F1 / AUC / Precision / Recall
    ax = axes[0]
    width = 0.2
    metrics_to_plot = ['f1', 'auc', 'precision', 'recall']
    metric_labels = ['F1', 'AUC', 'Precision', 'Recall']
    metric_colors = ['#2563EB', '#DC2626', '#059669', '#D97706']

    for i, (metric, label, mc) in enumerate(zip(metrics_to_plot, metric_labels, metric_colors)):
        vals = [r[metric] for r in results]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=mc, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_ylabel('Score')
    ax.set_title('(a) Classification Metrics')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, axis='y')

    # (b) Parameter count
    ax = axes[1]
    params_k = [r['params'] / 1000 for r in results]
    bars = ax.bar(x, params_k, color=colors, alpha=0.85)
    for bar, pk in zip(bars, params_k):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                '%.0fK' % pk, ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_ylabel('Parameters (K)')
    ax.set_title('(b) Model Size')
    ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('GNN Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, 'figures', 'training')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'model_comparison.png')
    plt.savefig(out_path)
    print("Saved: %s" % out_path)
    plt.close()
    return out_path


# =========================================================================
# Defect Probability Map (3D)
# =========================================================================
def plot_defect_probability_map(run_dir, data_dir, sample_idx=0, split='val',
                                out_dir=None):
    """Plot 3D defect probability heatmap on the fairing surface."""
    import torch

    ckpt = load_checkpoint(run_dir)
    if ckpt is None:
        print("No checkpoint found.")
        return None

    arch = detect_arch(run_dir)
    args_dict = ckpt.get('args', {})
    in_channels = ckpt['in_channels']
    edge_attr_dim = ckpt['edge_attr_dim']

    # Load data
    data_path = os.path.join(data_dir, '%s.pt' % split)
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)
    eval_data = torch.load(data_path, weights_only=False)

    if sample_idx >= len(eval_data):
        print("Sample index %d out of range (max %d)" % (sample_idx, len(eval_data) - 1))
        return None

    graph = eval_data[sample_idx]
    head_key = [k for k in ckpt['model_state_dict'] if 'head' in k and 'weight' in k]
    if head_key:
        num_classes = ckpt['model_state_dict'][head_key[-1]].shape[0]
    else:
        all_labels = torch.cat([d.y for d in eval_data])
        num_classes = max(int(all_labels.max().item()) + 1, 2)

    data_in_channels = eval_data[0].x.shape[1]
    if data_in_channels != in_channels:
        print("WARNING: Data features (%d) != checkpoint features (%d). "
              "Use matching data_dir." % (data_in_channels, in_channels))
        return None

    # Build model
    from models import build_model
    model = build_model(
        arch, in_channels, edge_attr_dim,
        hidden_channels=args_dict.get('hidden', 128),
        num_layers=args_dict.get('layers', 4),
        dropout=args_dict.get('dropout', 0.1),
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Inference
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index, graph.edge_attr, None)
        if num_classes == 2:
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        else:
            # Max probability of any defect class
            probs = 1.0 - torch.softmax(logits, dim=1)[:, 0].numpy()

    # Positions
    pos = graph.pos.numpy() if hasattr(graph, 'pos') and graph.pos is not None else graph.x[:, :3].numpy()
    true_labels = graph.y.numpy()

    # Plot
    fig = plt.figure(figsize=(16, 6))

    # (a) Predicted probability
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=probs, cmap='hot_r', s=1, alpha=0.6, vmin=0, vmax=1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('(a) Predicted Defect Probability')
    fig.colorbar(sc1, ax=ax1, shrink=0.6, label='P(defect)')

    # (b) Ground truth
    ax2 = fig.add_subplot(122, projection='3d')
    gt_colors = np.where(true_labels > 0, 1.0, 0.0)
    sc2 = ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=gt_colors, cmap='hot_r', s=1, alpha=0.6, vmin=0, vmax=1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('(b) Ground Truth')
    fig.colorbar(sc2, ax=ax2, shrink=0.6, label='Defect')

    fig.suptitle('%s — Defect Probability Map (sample %d)' % (arch.upper(), sample_idx),
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, 'figures', 'training')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'defect_probability_map_%s_%04d.png' % (arch, sample_idx))
    plt.savefig(out_path)
    print("Saved: %s" % out_path)
    plt.close()
    return out_path


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--run_dir', type=str, help='Single run directory')
    parser.add_argument('--data_dir', type=str, help='Data dir for confusion matrix / defect map')
    parser.add_argument('--compare', nargs='+', help='Multiple run dirs for comparison')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--confusion_only', action='store_true')
    parser.add_argument('--defect_map', action='store_true',
                        help='Generate defect probability map')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index for defect map')
    args = parser.parse_args()

    if args.compare:
        plot_training_curves_multi(args.compare, args.out_dir)
        compare_models(args.compare, args.out_dir)
    elif args.run_dir:
        if not args.confusion_only:
            plot_training_curves(args.run_dir, args.out_dir)
        if args.data_dir:
            plot_confusion_matrix(args.run_dir, args.data_dir, out_dir=args.out_dir)
        if args.defect_map and args.data_dir:
            plot_defect_probability_map(
                args.run_dir, args.data_dir,
                sample_idx=args.sample_idx, out_dir=args.out_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
