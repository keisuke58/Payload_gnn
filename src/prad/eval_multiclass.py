# -*- coding: utf-8 -*-
"""PRAD Multi-class evaluation — unsupervised defect type separation.

The MAE was trained on binary labels (healthy vs defect), but the
residual patterns should naturally cluster by defect type because
different defects produce different physical signatures.

This script evaluates:
1. Binary anomaly detection (ROC/PR/F1) as sanity check
2. Per-type detection rates (which defect types are easiest/hardest)
3. t-SNE of residual patterns colored by defect type
4. k-means clustering on residuals → NMI/ARI vs true types
5. Per-type residual heatmaps (which dimensions deviate most)

Usage:
    python src/prad/eval_multiclass.py \
        --checkpoint checkpoints/prad_mae_sage.pt \
        --data_dir data/processed_s12_czm_thermal_200 \
        --output_dir figures/prad_multiclass
"""

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from prad.graphmae import PIGraphMAE
from prad.anomaly_score import score_dataset
from prad.eval_prad import compute_metrics
from prad.train_mae import load_data
from prad import PHYSICS_DIMS

DEFECT_TYPE_NAMES = [
    'healthy', 'debonding', 'fod', 'impact',
    'delamination', 'inner_debond', 'thermal_prog', 'acoustic_fat',
]


def per_type_metrics(scores, labels):
    """Compute detection metrics per defect type."""
    results = {}
    binary = (labels > 0).astype(int)

    # Overall binary
    overall = compute_metrics(scores, labels)
    results['overall'] = overall

    # Per type: type vs healthy
    unique_types = np.unique(labels)
    for t in unique_types:
        if t == 0:
            continue
        # Select healthy + this type only
        mask = (labels == 0) | (labels == t)
        sub_scores = scores[mask]
        sub_labels = (labels[mask] > 0).astype(int)
        if sub_labels.sum() == 0:
            continue
        m = compute_metrics(sub_scores, sub_labels)
        m['n_defect_nodes'] = int((labels == t).sum())
        name = DEFECT_TYPE_NAMES[t] if t < len(DEFECT_TYPE_NAMES) else \
            'type_%d' % t
        results[name] = m

    return results


def cluster_residuals(residuals, labels, n_clusters=7):
    """K-means clustering on residual patterns, evaluate vs true types."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import (normalized_mutual_info_score,
                                 adjusted_rand_score)

    # Only defect nodes
    defect_mask = labels > 0
    if defect_mask.sum() < n_clusters:
        return {}

    R = residuals[defect_mask]
    true_types = labels[defect_mask]

    # Use physics-relevant dims
    phys_idx = PHYSICS_DIMS
    R_phys = R[:, phys_idx] if R.shape[1] > max(phys_idx) else R

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_clusters = km.fit_predict(R_phys)

    nmi = normalized_mutual_info_score(true_types, pred_clusters)
    ari = adjusted_rand_score(true_types, pred_clusters)

    return {'nmi': nmi, 'ari': ari, 'n_clusters': n_clusters,
            'n_defect_nodes': int(defect_mask.sum())}


def plot_tsne_multiclass(residuals, labels, output_dir, max_samples=8000):
    """t-SNE colored by defect type (not just binary)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("  matplotlib/sklearn not available")
        return

    # Subsample
    N = len(labels)
    if N > max_samples:
        # Stratified: keep all defect, subsample healthy
        defect_mask = labels > 0
        n_defect = defect_mask.sum()
        n_healthy_sample = min(max_samples - n_defect, (labels == 0).sum())
        healthy_idx = np.where(labels == 0)[0]
        healthy_idx = np.random.choice(healthy_idx, n_healthy_sample,
                                       replace=False)
        defect_idx = np.where(defect_mask)[0]
        idx = np.concatenate([healthy_idx, defect_idx])
        residuals = residuals[idx]
        labels = labels[idx]

    # Physics dims
    phys_idx = PHYSICS_DIMS
    R = residuals[:, phys_idx] if residuals.shape[1] > max(phys_idx) \
        else residuals

    print("  Running t-SNE on %d nodes..." % len(labels))
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = tsne.fit_transform(R)

    os.makedirs(output_dir, exist_ok=True)

    # Plot: defect types only (excluding healthy for clarity)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: binary (healthy vs defect)
    ax = axes[0]
    binary = (labels > 0).astype(int)
    for lbl, name, color in [(0, 'Healthy', 'blue'), (1, 'Defect', 'red')]:
        mask = binary == lbl
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, s=3, alpha=0.3, label=name)
    ax.set_title('Binary (Healthy vs Defect)')
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)

    # Right: multi-class
    ax = axes[1]
    unique_types = np.unique(labels)
    cmap = plt.cm.Set1(np.linspace(0, 1, max(len(unique_types), 8)))
    for i, t in enumerate(unique_types):
        mask = labels == t
        name = DEFECT_TYPE_NAMES[t] if t < len(DEFECT_TYPE_NAMES) else \
            'type_%d' % t
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap[i]], s=3 if t == 0 else 8,
                   alpha=0.2 if t == 0 else 0.6, label=name)
    ax.set_title('Multi-class (7 Defect Types)')
    ax.legend(markerscale=5, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('PRAD Residual t-SNE — Unsupervised Defect Type Separation',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'prad_multiclass_tsne.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: prad_multiclass_tsne.png")


def plot_residual_heatmap(residuals, labels, output_dir):
    """Per-type mean residual heatmap (which dimensions deviate most)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    unique_types = sorted([t for t in np.unique(labels) if t > 0])
    if not unique_types:
        return

    # Compute mean residual per type per dimension
    n_dims = residuals.shape[1]
    n_types = len(unique_types)
    heatmap = np.zeros((n_types, n_dims))

    for i, t in enumerate(unique_types):
        mask = labels == t
        heatmap[i] = residuals[mask].mean(axis=0)

    # Normalize per dimension (z-score across types)
    dim_mean = heatmap.mean(axis=0, keepdims=True)
    dim_std = heatmap.std(axis=0, keepdims=True) + 1e-8
    heatmap_norm = (heatmap - dim_mean) / dim_std

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(heatmap_norm, aspect='auto', cmap='RdBu_r',
                   vmin=-2, vmax=2)
    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Defect Type')
    type_names = [DEFECT_TYPE_NAMES[t] if t < len(DEFECT_TYPE_NAMES)
                  else 'type_%d' % t for t in unique_types]
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(type_names)
    ax.set_title('Residual Pattern by Defect Type (z-scored)')
    fig.colorbar(im, ax=ax, label='z-score')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'prad_residual_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: prad_residual_heatmap.png")


def main():
    parser = argparse.ArgumentParser(
        description='PRAD Multi-class Evaluation')
    parser.add_argument('--checkpoint',
                        default='checkpoints/prad_mae_sage.pt')
    parser.add_argument('--data_dir',
                        default='data/processed_s12_czm_thermal_200')
    parser.add_argument('--output_dir', default='figures/prad_multiclass')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--smooth_rounds', type=int, default=1)
    parser.add_argument('--smooth_alpha', type=float, default=0.5)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 60)
    print("PRAD Multi-class Evaluation")
    print("=" * 60)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device,
                      weights_only=False)
    mae_args = ckpt['args']
    model = PIGraphMAE(
        encoder_arch=mae_args['encoder_arch'],
        hidden_channels=mae_args['hidden'],
        num_layers=mae_args['num_layers'],
        dropout=mae_args['dropout'],
        mask_ratio=mae_args['mask_ratio'],
        lambda_physics=mae_args['lambda_physics'],
        decoder_type=mae_args.get('decoder_type', 'mlp'),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print("  Model: %s" % args.checkpoint)

    # Load multi-class data
    _, val_data = load_data(args.data_dir, device)
    all_labels = torch.cat([d.y for d in val_data])
    unique = all_labels.unique().tolist()
    print("  Data: %d val graphs, %d classes %s" % (
        len(val_data), len(unique), unique))
    for c in sorted(unique):
        name = DEFECT_TYPE_NAMES[c] if c < len(DEFECT_TYPE_NAMES) \
            else 'type_%d' % c
        n = (all_labels == c).sum().item()
        print("    %s: %d nodes" % (name, n))

    # Score all validation graphs
    print("\n  Scoring...")
    scores_list, labels_list, residuals_list = score_dataset(
        model, val_data, alpha=args.alpha,
        smooth_rounds=args.smooth_rounds,
        smooth_alpha=args.smooth_alpha)

    scores_np = torch.cat(scores_list).numpy()
    labels_np = torch.cat(labels_list).numpy()
    residuals_np = torch.cat(residuals_list).numpy()

    # 1. Per-type detection metrics
    print("\n--- Per-Type Detection ---")
    type_metrics = per_type_metrics(scores_np, labels_np)

    print("  %-18s  ROC-AUC  PR-AUC  F1      N_defect" % "Type")
    print("  " + "-" * 62)
    for name in ['overall'] + [DEFECT_TYPE_NAMES[t] for t in sorted(unique)
                                if t > 0]:
        if name not in type_metrics:
            continue
        m = type_metrics[name]
        n = m.get('n_defect_nodes', m.get('n_defect', 0))
        print("  %-18s  %.4f   %.4f  %.4f   %d" % (
            name, m['roc_auc'], m['pr_auc'], m['best_f1'], n))

    # 2. Clustering
    print("\n--- Unsupervised Clustering ---")
    n_types = len([t for t in unique if t > 0])
    cluster_result = cluster_residuals(residuals_np, labels_np,
                                       n_clusters=n_types)
    if cluster_result:
        print("  K-means (k=%d):" % cluster_result['n_clusters'])
        print("    NMI: %.4f" % cluster_result['nmi'])
        print("    ARI: %.4f" % cluster_result['ari'])
        print("    Defect nodes: %d" % cluster_result['n_defect_nodes'])
    else:
        print("  Not enough defect nodes for clustering")

    # 3. Plots
    print("\n--- Generating Plots ---")
    plot_tsne_multiclass(residuals_np, labels_np, args.output_dir)
    plot_residual_heatmap(residuals_np, labels_np, args.output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
