# -*- coding: utf-8 -*-
"""Evaluate PRAD anomaly detection performance.

Usage:
    python src/prad/eval_prad.py \
        --checkpoint checkpoints/prad_mae_sage.pt \
        --data_dir data/processed_s12_mixed_400 \
        --output_dir figures/prad
"""

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from prad.anomaly_score import load_model_and_score
from prad.graphmae import PIGraphMAE
from prad import STRESS_DIMS, PHYSICS_DIMS


def compute_metrics(scores, labels, thresholds=None):
    """Compute anomaly detection metrics.

    Args:
        scores: (N,) numpy array of anomaly scores.
        labels: (N,) numpy array of ground truth (0=healthy, >0=defect).

    Returns:
        dict with ROC-AUC, PR-AUC, best F1, threshold, precision, recall.
    """
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 precision_recall_curve, f1_score,
                                 roc_curve)

    binary_labels = (labels > 0).astype(int)

    # Edge case: all same label
    if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
        print("  WARNING: all labels are the same (%d defect / %d total)" %
              (binary_labels.sum(), len(binary_labels)))
        return {
            'roc_auc': 0.0, 'pr_auc': 0.0, 'best_f1': 0.0,
            'best_threshold': 0.0, 'precision': 0.0, 'recall': 0.0,
            'n_defect': int(binary_labels.sum()),
            'n_total': len(binary_labels),
        }

    roc_auc = roc_auc_score(binary_labels, scores)
    pr_auc = average_precision_score(binary_labels, scores)

    # Find best F1 threshold
    precisions, recalls, thresholds_pr = precision_recall_curve(
        binary_labels, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = f1s.argmax()
    best_f1 = f1s[best_idx]
    best_threshold = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) \
        else thresholds_pr[-1]

    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'best_f1': float(best_f1),
        'best_threshold': float(best_threshold),
        'precision': float(precisions[best_idx]),
        'recall': float(recalls[best_idx]),
        'n_defect': int(binary_labels.sum()),
        'n_total': len(binary_labels),
    }


def plot_roc_pr(scores, labels, output_dir, prefix='prad'):
    """Generate ROC and PR curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve
    except ImportError:
        print("  matplotlib/sklearn not available, skipping plots")
        return

    binary_labels = (labels > 0).astype(int)
    if binary_labels.sum() == 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(binary_labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, 'b-', lw=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('PRAD ROC Curve')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, '%s_roc.png' % prefix),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # PR
    prec, rec, _ = precision_recall_curve(binary_labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, 'r-', lw=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PRAD Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, '%s_pr.png' % prefix),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Score distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    healthy = scores[binary_labels == 0]
    defect = scores[binary_labels == 1]
    ax.hist(healthy, bins=100, alpha=0.6, label='Healthy', color='blue',
            density=True)
    if len(defect) > 0:
        ax.hist(defect, bins=100, alpha=0.6, label='Defect', color='red',
                density=True)
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title('PRAD Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, '%s_score_dist.png' % prefix),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("  Plots saved to %s/" % output_dir)


def plot_residual_tsne(residuals, labels, output_dir, prefix='prad',
                       max_samples=5000):
    """t-SNE of residual patterns for multi-class separation analysis."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("  sklearn/matplotlib not available, skipping t-SNE")
        return

    # Subsample for speed
    N = len(labels)
    if N > max_samples:
        idx = np.random.choice(N, max_samples, replace=False)
        residuals = residuals[idx]
        labels = labels[idx]

    # Use physics-relevant dims only
    phys_idx = PHYSICS_DIMS
    R = residuals[:, phys_idx] if residuals.shape[1] > max(phys_idx) \
        else residuals

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = tsne.fit_transform(R)

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_labels), 2)))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = 'Healthy' if lbl == 0 else 'Defect-%d' % lbl
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[colors[i]], s=3, alpha=0.5, label=name)
    ax.set_title('t-SNE of Residual Patterns (Physics Dims)')
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, '%s_tsne.png' % prefix),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  t-SNE plot saved")


def grid_search_scoring(checkpoint_path, data_dir, device='cpu'):
    """Grid search over scoring hyperparameters.

    Searches alpha (cosine vs L1 blend), smooth_rounds, and smooth_alpha
    to find the best combination for F1 and PR-AUC.
    """
    from prad.train_mae import load_data
    from prad.anomaly_score import compute_anomaly_scores

    ckpt = torch.load(checkpoint_path, map_location=device,
                      weights_only=False)
    args = ckpt['args']
    model = PIGraphMAE(
        encoder_arch=args['encoder_arch'],
        in_channels=args.get('in_channels', 34),
        hidden_channels=args['hidden'],
        num_layers=args['num_layers'],
        dropout=args['dropout'],
        mask_ratio=args['mask_ratio'],
        lambda_physics=args['lambda_physics'],
        decoder_type=args.get('decoder_type', 'mlp'),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    _, val_data = load_data(data_dir, device)

    # Grid
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    smooth_configs = [(0, 1.0), (1, 0.7), (1, 0.5), (2, 0.7)]

    print("\n  Grid Search: alpha x smoothing")
    print("  " + "-" * 72)
    print("  %-6s %-10s %-10s | %-8s %-8s %-8s" % (
        'alpha', 'sm_rounds', 'sm_alpha', 'ROC-AUC', 'PR-AUC', 'F1'))
    print("  " + "-" * 72)

    best_f1 = 0
    best_config = {}

    for alpha in alphas:
        for sm_rounds, sm_alpha in smooth_configs:
            all_scores, all_labels = [], []
            for data in val_data:
                scores, _ = compute_anomaly_scores(
                    model, data, alpha=alpha,
                    smooth_rounds=sm_rounds, smooth_alpha=sm_alpha)
                all_scores.append(scores.cpu())
                all_labels.append(data.y.cpu())

            scores_np = torch.cat(all_scores).numpy()
            labels_np = torch.cat(all_labels).numpy()
            metrics = compute_metrics(scores_np, labels_np)

            marker = ''
            if metrics['best_f1'] > best_f1:
                best_f1 = metrics['best_f1']
                best_config = {
                    'alpha': alpha, 'smooth_rounds': sm_rounds,
                    'smooth_alpha': sm_alpha, 'metrics': metrics,
                }
                marker = ' *'

            print("  %-6.1f %-10d %-10.1f | %-8.4f %-8.4f %-8.4f%s" % (
                alpha, sm_rounds, sm_alpha,
                metrics['roc_auc'], metrics['pr_auc'],
                metrics['best_f1'], marker))

    print("  " + "-" * 72)
    print("\n  Best config: alpha=%.1f, smooth_rounds=%d, smooth_alpha=%.1f" % (
        best_config['alpha'], best_config['smooth_rounds'],
        best_config['smooth_alpha']))
    m = best_config['metrics']
    print("  ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f  P=%.4f  R=%.4f" % (
        m['roc_auc'], m['pr_auc'], m['best_f1'],
        m['precision'], m['recall']))

    return best_config


def main():
    parser = argparse.ArgumentParser(description='Evaluate PRAD')
    parser.add_argument('--checkpoint',
                        default='checkpoints/prad_mae_sage.pt')
    parser.add_argument('--data_dir',
                        default='data/processed_s12_mixed_400')
    parser.add_argument('--output_dir', default='figures/prad')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Cosine vs L1 blend (1.0=cosine only)')
    parser.add_argument('--smooth_rounds', type=int, default=1)
    parser.add_argument('--smooth_alpha', type=float, default=0.7)
    parser.add_argument('--grid_search', action='store_true',
                        help='Grid search scoring hyperparameters')
    parser.add_argument('--stress_weight', type=float, default=2.0)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 60)
    print("PRAD Evaluation")
    print("=" * 60)
    print("  checkpoint: %s" % args.checkpoint)
    print("  data_dir:   %s" % args.data_dir)
    print()

    if args.grid_search:
        best = grid_search_scoring(args.checkpoint, args.data_dir,
                                   device=device)
        args.alpha = best['alpha']
        args.smooth_rounds = best['smooth_rounds']
        args.smooth_alpha = best['smooth_alpha']
        print()

    # Score validation data
    val_scores, val_labels, val_residuals = load_model_and_score(
        args.checkpoint, args.data_dir, device=device,
        alpha=args.alpha, smooth_rounds=args.smooth_rounds,
        smooth_alpha=args.smooth_alpha)

    # Concatenate all graphs
    scores_np = torch.cat(val_scores).numpy()
    labels_np = torch.cat(val_labels).numpy()
    residuals_np = torch.cat(val_residuals).numpy()

    print("  Total nodes:  %d" % len(scores_np))
    print("  Defect nodes: %d (%.2f%%)" % (
        (labels_np > 0).sum(),
        100.0 * (labels_np > 0).sum() / len(labels_np)))
    print("  Scoring: alpha=%.1f, smooth_rounds=%d, smooth_alpha=%.1f" % (
        args.alpha, args.smooth_rounds, args.smooth_alpha))
    print()

    # Compute metrics
    metrics = compute_metrics(scores_np, labels_np)
    print("Results:")
    print("  ROC-AUC:    %.4f" % metrics['roc_auc'])
    print("  PR-AUC:     %.4f" % metrics['pr_auc'])
    print("  Best F1:    %.4f" % metrics['best_f1'])
    print("  Threshold:  %.4f" % metrics['best_threshold'])
    print("  Precision:  %.4f" % metrics['precision'])
    print("  Recall:     %.4f" % metrics['recall'])

    # Plots
    plot_roc_pr(scores_np, labels_np, args.output_dir)
    plot_residual_tsne(residuals_np, labels_np, args.output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
