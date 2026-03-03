#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文品質の学習結果可視化 — S12 CZM Thermal 200 Dataset

Generates 6 publication-quality figures:
  Fig 1: 5-Fold CV Training Curves (Val Loss + Val F1, mean±σ band)
  Fig 2: Focal Loss γ Sensitivity Analysis (bar chart)
  Fig 3: 5-Fold CV Box Plot (F1 distribution)
  Fig 4: Confusion Matrix (best fold model)
  Fig 5: Defect Probability Map (3D, predicted vs GT)
  Fig 6: ROC Curve (best fold model)

Usage:
  python scripts/generate_paper_figures.py
  python scripts/generate_paper_figures.py --out_dir wiki_repo/images/training
  python scripts/generate_paper_figures.py --fig 1 2 3  # specific figures only
"""

import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# ==========================================================================
# Style settings (publication quality)
# ==========================================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

GAT_COLOR = '#DC2626'
GAT_COLOR_LIGHT = '#FCA5A5'
FOLD_COLORS = ['#2563EB', '#DC2626', '#059669', '#7C3AED', '#D97706']
METRIC_COLORS = {
    'f1': '#2563EB',
    'auc': '#DC2626',
    'precision': '#059669',
    'recall': '#D97706',
}

# ==========================================================================
# Run directories (update if paths change)
# ==========================================================================
FOLD_DIRS = [
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_160640_fold0'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_161051_fold1'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_161330_fold2'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_161936_fold3'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_162502_fold4'),
]

GAMMA_RUNS = {
    1.0: os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_155223'),
    2.0: os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_155538'),
    3.0: os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_155845'),
    5.0: os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_160150'),
}

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_s12_czm_thermal_200_binary')


# ==========================================================================
# Data loading
# ==========================================================================
def load_training_log(run_dir):
    """Load training_log.csv as dict of numpy arrays."""
    log_path = os.path.join(run_dir, 'training_log.csv')
    data = {}
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in data.items()}


def load_checkpoint(run_dir):
    """Load best_model.pt checkpoint."""
    import torch
    ckpt_path = os.path.join(run_dir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        return None
    return torch.load(ckpt_path, weights_only=False, map_location='cpu')


def get_best_f1(run_dir):
    """Get best val F1 from training log."""
    data = load_training_log(run_dir)
    return float(np.max(data['val_f1']))


def get_best_epoch(run_dir):
    """Get epoch of best val F1."""
    data = load_training_log(run_dir)
    return int(data['epoch'][np.argmax(data['val_f1'])])


# ==========================================================================
# Fig 1: 5-Fold CV Training Curves
# ==========================================================================
def fig1_cv_training_curves(out_dir):
    """5-fold CV training curves with mean±σ band."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Collect all fold data, pad to same length
    all_loss, all_f1 = [], []
    max_epochs = 0
    fold_data = []
    for fd in FOLD_DIRS:
        d = load_training_log(fd)
        fold_data.append(d)
        max_epochs = max(max_epochs, len(d['epoch']))

    for i, d in enumerate(fold_data):
        epochs = d['epoch'].astype(int)
        n = len(epochs)

        # (a) Val Loss — individual fold lines
        axes[0].plot(epochs, d['val_loss'], '-', color=FOLD_COLORS[i],
                     alpha=0.35, lw=1, label='Fold %d' % (i + 1) if i == 0 else None)
        # (b) Val F1
        axes[1].plot(epochs, d['val_f1'], '-', color=FOLD_COLORS[i],
                     alpha=0.35, lw=1)

        # Pad for mean calculation
        loss_padded = np.full(max_epochs, np.nan)
        f1_padded = np.full(max_epochs, np.nan)
        loss_padded[:n] = d['val_loss']
        f1_padded[:n] = d['val_f1']
        all_loss.append(loss_padded)
        all_f1.append(f1_padded)

        # Mark early stopping point
        best_ep = int(d['epoch'][np.argmax(d['val_f1'])])
        best_f1 = np.max(d['val_f1'])
        axes[1].plot(best_ep, best_f1, 'o', color=FOLD_COLORS[i],
                     markersize=5, alpha=0.7)

    # Mean ± σ bands
    all_loss = np.array(all_loss)
    all_f1 = np.array(all_f1)
    ep_range = np.arange(1, max_epochs + 1)

    for metric_arr, ax in [(all_loss, axes[0]), (all_f1, axes[1])]:
        mean = np.nanmean(metric_arr, axis=0)
        std = np.nanstd(metric_arr, axis=0)
        # Only plot where at least 3 folds have data
        valid = np.sum(~np.isnan(metric_arr), axis=0) >= 3
        ep_valid = ep_range[valid]
        mean_valid = mean[valid]
        std_valid = std[valid]

        ax.plot(ep_valid, mean_valid, '-', color=GAT_COLOR, lw=2.5,
                label='Mean', zorder=10)
        ax.fill_between(ep_valid, mean_valid - std_valid, mean_valid + std_valid,
                         color=GAT_COLOR_LIGHT, alpha=0.3, label='±1σ', zorder=5)

    # Formatting
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('(a) Validation Loss')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation F1 Score')
    axes[1].set_title('(b) Validation F1 Score')
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend(['Fold (individual)', 'Best epoch', 'Mean', '±1σ'],
                   fontsize=9, loc='lower right')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('GAT 5-Fold Cross-Validation — Training Curves (S12 Thermal, N=200)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'fig1_cv_training_curves.png')
    plt.savefig(out_path)
    print("Saved: %s" % out_path)
    plt.close()
    return out_path


# ==========================================================================
# Fig 2: Focal Loss γ Sensitivity
# ==========================================================================
def fig2_gamma_sensitivity(out_dir):
    """Bar chart of γ vs Best Val F1."""
    gammas = sorted(GAMMA_RUNS.keys())
    f1_scores = [get_best_f1(GAMMA_RUNS[g]) for g in gammas]
    best_idx = np.argmax(f1_scores)

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ['#9CA3AF'] * len(gammas)  # gray
    colors[best_idx] = GAT_COLOR  # highlight best

    bars = ax.bar(range(len(gammas)), f1_scores, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5)

    # Value labels
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        weight = 'bold' if i == best_idx else 'normal'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                '%.4f' % f1, ha='center', va='bottom', fontsize=11,
                fontweight=weight)

    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels(['γ = %.1f' % g for g in gammas])
    ax.set_ylabel('Best Validation F1 Score')
    ax.set_title('Focal Loss γ Sensitivity Analysis', fontsize=13, fontweight='bold')
    ax.set_ylim(0.75, 0.86)
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig2_gamma_sensitivity.png')
    plt.savefig(out_path)
    print("Saved: %s" % out_path)
    plt.close()
    return out_path


# ==========================================================================
# Fig 3: 5-Fold CV Box Plot
# ==========================================================================
def fig3_cv_boxplot(out_dir):
    """Box plot + strip plot of fold F1 scores."""
    f1_scores = [get_best_f1(fd) for fd in FOLD_DIRS]
    best_epochs = [get_best_epoch(fd) for fd in FOLD_DIRS]
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    fig, (ax_box, ax_table) = plt.subplots(1, 2, figsize=(10, 4.5),
                                            gridspec_kw={'width_ratios': [2, 1]})

    # Box plot
    bp = ax_box.boxplot(f1_scores, vert=True, patch_artist=True,
                        widths=0.5, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor=GAT_COLOR,
                                       markeredgecolor='white', markersize=8),
                        medianprops=dict(color='black', linewidth=2),
                        boxprops=dict(facecolor=GAT_COLOR_LIGHT, edgecolor=GAT_COLOR,
                                      linewidth=1.5),
                        whiskerprops=dict(color=GAT_COLOR, linewidth=1.5),
                        capprops=dict(color=GAT_COLOR, linewidth=1.5))

    # Individual fold points (strip/swarm)
    jitter = np.random.RandomState(42).uniform(-0.08, 0.08, len(f1_scores))
    for i, (f1, j) in enumerate(zip(f1_scores, jitter)):
        ax_box.plot(1 + j, f1, 'o', color=FOLD_COLORS[i], markersize=10,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=10)

    # Mean ± std annotation
    ax_box.axhline(mean_f1, color=GAT_COLOR, linestyle='--', alpha=0.5, lw=1)
    ax_box.text(1.4, mean_f1, 'μ = %.4f ± %.4f' % (mean_f1, std_f1),
                fontsize=10, color=GAT_COLOR, va='center')

    ax_box.set_ylabel('Best Validation F1 Score')
    ax_box.set_title('5-Fold Cross-Validation Results', fontsize=13, fontweight='bold')
    ax_box.set_xticks([1])
    ax_box.set_xticklabels(['GAT (Focal, γ=1.0)'])
    ax_box.set_ylim(0.78, 0.92)
    ax_box.grid(True, alpha=0.2, axis='y')
    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)

    # Table
    ax_table.axis('off')
    table_data = [['Fold', 'F1 Score', 'Best Epoch']]
    for i, (f1, ep) in enumerate(zip(f1_scores, best_epochs)):
        table_data.append(['Fold %d' % (i + 1), '%.4f' % f1, '%d' % ep])
    table_data.append(['', '', ''])
    table_data.append(['Mean', '%.4f' % mean_f1, ''])
    table_data.append(['Std', '%.4f' % std_f1, ''])

    table = ax_table.table(cellText=table_data, loc='center',
                           cellLoc='center', colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Header row style
    for j in range(3):
        table[0, j].set_facecolor('#374151')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Mean/Std rows
    for row in [len(f1_scores) + 2, len(f1_scores) + 3]:
        for j in range(3):
            table[row, j].set_facecolor('#FEE2E2')
            table[row, j].set_text_props(fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig3_cv_boxplot.png')
    plt.savefig(out_path)
    print("Saved: %s" % out_path)
    plt.close()
    return out_path


# ==========================================================================
# Fig 4: Confusion Matrix
# ==========================================================================
def fig4_confusion_matrix(out_dir):
    """Confusion matrix from best fold model."""
    import torch
    from sklearn.metrics import confusion_matrix

    # Find best fold
    f1_scores = [get_best_f1(fd) for fd in FOLD_DIRS]
    best_fold = int(np.argmax(f1_scores))
    best_dir = FOLD_DIRS[best_fold]
    print("Best fold: %d (F1=%.4f) from %s" % (best_fold, f1_scores[best_fold],
                                                 os.path.basename(best_dir)))

    ckpt = load_checkpoint(best_dir)
    if ckpt is None:
        print("No checkpoint found.")
        return None

    args_dict = ckpt.get('args', {})
    in_channels = ckpt['in_channels']
    edge_attr_dim = ckpt['edge_attr_dim']

    # Detect num_classes
    head_key = [k for k in ckpt['model_state_dict'] if 'head' in k and 'weight' in k]
    num_classes = ckpt['model_state_dict'][head_key[-1]].shape[0] if head_key else 2

    # Load val data
    val_data = torch.load(os.path.join(DATA_DIR, 'val.pt'), weights_only=False)

    from models import build_model
    model = build_model(
        'gat', in_channels, edge_attr_dim,
        hidden_channels=args_dict.get('hidden', 128),
        num_layers=args_dict.get('layers', 4),
        dropout=args_dict.get('dropout', 0.1),
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Norm stats
    norm_path = os.path.join(DATA_DIR, 'norm_stats.pt')
    if os.path.exists(norm_path):
        ns = torch.load(norm_path, weights_only=False)
        node_mean, node_std = ns['mean'], ns['std']
    else:
        node_mean = node_std = None

    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for data in val_data:
            x = data.x
            if node_mean is not None:
                x = (x - node_mean) / node_std.clamp(min=1e-8)
            out = model(x, data.edge_index, data.edge_attr, None)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            all_preds.append(preds)
            all_targets.append(data.y)
            all_probs.append(probs[:, 1] if num_classes == 2 else 1 - probs[:, 0])

    preds_np = torch.cat(all_preds).numpy()
    targets_np = torch.cat(all_targets).numpy()
    probs_np = torch.cat(all_probs).numpy()

    # Save probs for ROC (Fig 6)
    _cache['probs'] = probs_np
    _cache['targets'] = targets_np

    # Confusion matrix
    cm = confusion_matrix(targets_np, preds_np, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    labels = ['Healthy', 'Defect']

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)

    # Annotate
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, '%s\n(%.1f%%)' % ('{:,}'.format(count), pct),
                    ha='center', va='center', fontsize=14, color=text_color,
                    fontweight='bold')

    fig.colorbar(im, ax=ax, shrink=0.8, label='Row-normalized')
    ax.set_title('Confusion Matrix — GAT (Best Fold %d, F1=%.3f)' % (
        best_fold + 1, f1_scores[best_fold]), fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig4_confusion_matrix.png')
    plt.savefig(out_path)
    print("Saved: %s" % out_path)
    plt.close()
    return out_path


# ==========================================================================
# Fig 5: Defect Inference Results (real-scale fairing)
# ==========================================================================
def _load_model_and_predict(val_data, best_dir, node_mean, node_std):
    """Load model and run inference on all val samples."""
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score

    ckpt = load_checkpoint(best_dir)
    args_dict = ckpt.get('args', {})
    in_channels = ckpt['in_channels']
    edge_attr_dim = ckpt['edge_attr_dim']
    head_key = [k for k in ckpt['model_state_dict'] if 'head' in k and 'weight' in k]
    num_classes = ckpt['model_state_dict'][head_key[-1]].shape[0] if head_key else 2

    from models import build_model
    model = build_model(
        'gat', in_channels, edge_attr_dim,
        hidden_channels=args_dict.get('hidden', 128),
        num_layers=args_dict.get('layers', 4),
        dropout=args_dict.get('dropout', 0.1),
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    results = []
    with torch.no_grad():
        for i, graph in enumerate(val_data):
            x = graph.x
            if node_mean is not None:
                x = (x - node_mean) / node_std.clamp(min=1e-8)
            logits = model(x, graph.edge_index, graph.edge_attr, None)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds = (probs > 0.5).astype(int)
            targets = graph.y.numpy()
            pos = graph.x[:, :3].numpy()  # real mm coords

            n_defect = int((targets > 0).sum())
            if n_defect > 0:
                f1 = f1_score(targets > 0, preds, zero_division=0)
                prec = precision_score(targets > 0, preds, zero_division=0)
                rec = recall_score(targets > 0, preds, zero_division=0)
            else:
                f1 = prec = rec = 0.0

            results.append({
                'idx': i, 'pos': pos, 'probs': probs,
                'targets': targets, 'preds': preds,
                'n_defect': n_defect, 'f1': f1,
                'precision': prec, 'recall': rec,
            })
    return results


def _plot_fairing_panel(ax, pos, values, cmap, vmin, vmax, title, cbar_label):
    """Plot a single fairing panel in real-scale (vertical, Y=axial up)."""
    x, y_axial, z = pos[:, 0], pos[:, 1], pos[:, 2]
    # Arc coordinate: circumferential position
    theta = np.arctan2(z, x)
    arc = 2600.0 * theta  # arc length [mm]

    sc = ax.scatter(arc, y_axial, c=values, cmap=cmap, s=0.3,
                    alpha=0.8, vmin=vmin, vmax=vmax, edgecolors='none',
                    rasterized=True)
    ax.set_xlabel('Arc Length [mm]', fontsize=9)
    ax.set_ylabel('Axial Position [mm]', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=8)
    return sc


def fig5_defect_probability_map(out_dir):
    """Multi-sample inference results on real-scale fairing."""
    import torch

    f1_scores = [get_best_f1(fd) for fd in FOLD_DIRS]
    best_fold = int(np.argmax(f1_scores))
    best_dir = FOLD_DIRS[best_fold]
    print("Using best fold %d (F1=%.4f)" % (best_fold, f1_scores[best_fold]))

    val_data = torch.load(os.path.join(DATA_DIR, 'val.pt'), weights_only=False)

    norm_path = os.path.join(DATA_DIR, 'norm_stats.pt')
    ns = torch.load(norm_path, weights_only=False)
    node_mean, node_std = ns['mean'], ns['std']

    # Run inference on all val samples
    results = _load_model_and_predict(val_data, best_dir, node_mean, node_std)

    # Pick 3 representative samples: large/medium/small defect
    defective = [r for r in results if r['n_defect'] > 20]
    defective.sort(key=lambda r: r['n_defect'], reverse=True)

    # Large, Medium, Small
    picks = []
    if len(defective) >= 3:
        picks = [defective[0], defective[len(defective) // 2], defective[-1]]
    else:
        picks = defective[:3]

    n_samples = len(picks)
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 4.5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, 2)

    for row, r in enumerate(picks):
        pos = r['pos']
        probs = r['probs']
        targets = r['targets']

        # (left) Predicted P(defect)
        vmax_pred = max(0.3, np.percentile(probs, 99.5))
        sc1 = _plot_fairing_panel(
            axes[row, 0], pos, probs, 'hot_r', 0, vmax_pred,
            'Predicted P(defect)', 'P(defect)')
        cb1 = fig.colorbar(sc1, ax=axes[row, 0], shrink=0.8, pad=0.02)
        cb1.set_label('P(defect)', fontsize=8)
        cb1.ax.tick_params(labelsize=7)

        # (right) Ground Truth
        gt = np.where(targets > 0, 1.0, 0.0)
        sc2 = _plot_fairing_panel(
            axes[row, 1], pos, gt, 'hot_r', 0, 1,
            'Ground Truth', 'Defect')
        cb2 = fig.colorbar(sc2, ax=axes[row, 1], shrink=0.8, pad=0.02)
        cb2.set_label('Defect', fontsize=8)
        cb2.ax.tick_params(labelsize=7)

        # Metrics annotation
        metrics_text = ('val[%d]  %d defect nodes\n'
                        'F1=%.3f  Prec=%.3f  Rec=%.3f') % (
            r['idx'], r['n_defect'], r['f1'], r['precision'], r['recall'])
        axes[row, 0].text(0.02, 0.97, metrics_text, transform=axes[row, 0].transAxes,
                          fontsize=8, va='top', ha='left',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    alpha=0.85, edgecolor='#9CA3AF'))

        # Highlight defect region with circle
        defect_mask = targets > 0
        if defect_mask.any():
            theta = np.arctan2(pos[defect_mask, 2], pos[defect_mask, 0])
            arc_def = 2600.0 * theta
            y_def = pos[defect_mask, 1]
            for ax in [axes[row, 0], axes[row, 1]]:
                ax.plot(arc_def.mean(), y_def.mean(), 'o', color='none',
                        markeredgecolor='#22C55E', markeredgewidth=1.5,
                        markersize=20)

    fig.suptitle('GAT Defect Localization — S12 Thermal 1/12 Sector (Real Scale)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'fig5_defect_probability_map.png')
    plt.savefig(out_path)
    print("Saved: %s" % out_path)
    plt.close()
    return out_path


# ==========================================================================
# Fig 6: ROC Curve
# ==========================================================================
def fig6_roc_curve(out_dir):
    """ROC curve from best fold model predictions."""
    from sklearn.metrics import roc_curve, auc

    if 'probs' not in _cache or 'targets' not in _cache:
        print("Run fig4 first to generate predictions.")
        return None

    probs = _cache['probs']
    targets = _cache['targets']

    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, color=GAT_COLOR, lw=2.5,
            label='GAT (AUC = %.4f)' % roc_auc)
    ax.plot([0, 1], [0, 1], '--', color='#9CA3AF', lw=1.5, label='Random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — GAT (S12 Thermal, N=200)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig6_roc_curve.png')
    plt.savefig(out_path)
    print("Saved: %s" % out_path)
    plt.close()
    return out_path


# ==========================================================================
# Main
# ==========================================================================
_cache = {}  # shared data between fig4 → fig6

def main():
    parser = argparse.ArgumentParser(description='Generate paper-quality figures')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'wiki_repo', 'images', 'training'))
    parser.add_argument('--fig', nargs='+', type=int, default=None,
                        help='Generate specific figures (1-6). Default: all')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    figs_to_gen = args.fig or [1, 2, 3, 4, 5, 6]

    generators = {
        1: ('5-Fold CV Training Curves', fig1_cv_training_curves),
        2: ('Focal Loss γ Sensitivity', fig2_gamma_sensitivity),
        3: ('5-Fold CV Box Plot', fig3_cv_boxplot),
        4: ('Confusion Matrix', fig4_confusion_matrix),
        5: ('Defect Probability Map', fig5_defect_probability_map),
        6: ('ROC Curve', fig6_roc_curve),
    }

    print("=" * 60)
    print("Generating %d figures → %s" % (len(figs_to_gen), args.out_dir))
    print("=" * 60)

    for fig_num in figs_to_gen:
        if fig_num in generators:
            name, func = generators[fig_num]
            print("\n--- Fig %d: %s ---" % (fig_num, name))
            try:
                func(args.out_dir)
            except Exception as e:
                print("ERROR in Fig %d: %s" % (fig_num, e))
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done. Figures saved to: %s" % args.out_dir)
    print("=" * 60)


if __name__ == '__main__':
    main()
