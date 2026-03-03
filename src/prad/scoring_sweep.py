# -*- coding: utf-8 -*-
"""Scoring strategy sweep — find optimal anomaly scoring for F1 improvement.

Tests multiple scoring strategies on the existing MAE checkpoint without
retraining. This is the fastest way to improve F1 since the MAE already
achieves ROC 0.992.

Strategies tested:
1. Baseline: cosine + L1(s12, le12) with current best params
2. Extended L1: cosine + L1(all physics dims)
3. Full mechanical: cosine + L1(physics + mechanical dims)
4. Mahalanobis distance on physics dims
5. Isolation Forest on full residual pattern
6. Per-graph normalized scoring
7. Percentile-based thresholding

Usage:
    python src/prad/scoring_sweep.py \
        --checkpoint checkpoints/prad_mae_sage.pt \
        --data_dir data/processed_s12_czm_thermal_200
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from prad.graphmae import PIGraphMAE
from prad.eval_prad import compute_metrics
from prad.train_mae import load_data
from prad import (PHYSICS_DIMS, MECHANICAL_DIMS, STRESS_DIMS,
                  STRAIN_DIMS)

# High-signal dims from anomaly_score.py diagnostic
HIGH_SIGNAL_DIMS = [17, 23]  # s12, le12

try:
    from torch_scatter import scatter_mean
except ImportError:
    from torch_geometric.utils import scatter
    def scatter_mean(src, index, dim=0, dim_size=None):
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')

# Additional high-signal dim candidates from physics analysis
EXTENDED_SIGNAL_DIMS = STRESS_DIMS + STRAIN_DIMS  # s11,s22,s12 + le11,le22,le12
ALL_PHYSICS_MECH_DIMS = PHYSICS_DIMS + MECHANICAL_DIMS


def _minmax_normalize(x):
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < 1e-10:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def _graph_smooth(scores, edge_index, rounds=1, alpha=0.7):
    N = scores.size(0)
    src, dst = edge_index[0], edge_index[1]
    for _ in range(rounds):
        neighbor_mean = scatter_mean(scores[src], dst, dim=0, dim_size=N)
        scores = alpha * scores + (1.0 - alpha) * neighbor_mean
    return scores


def get_residuals(model, data_list):
    """Get reconstruction residuals and reconstructions for all graphs."""
    model.eval()
    all_residuals = []
    all_labels = []
    all_edge_indices = []
    all_x = []
    all_x_recon = []

    with torch.no_grad():
        for data in data_list:
            x_recon = model.reconstruct(data)
            residual = (data.x - x_recon).abs()
            all_residuals.append(residual.cpu())
            all_labels.append(data.y.cpu())
            all_edge_indices.append(data.edge_index.cpu())
            all_x.append(data.x.cpu())
            all_x_recon.append(x_recon.cpu())

    return all_residuals, all_labels, all_edge_indices, all_x, all_x_recon


def score_strategy_baseline(residuals, x_list, x_recon_list, edge_indices,
                             alpha=0.7, smooth_rounds=1, smooth_alpha=0.7):
    """Original scoring: cosine + L1(s12, le12)."""
    all_scores = []
    for res, x, xr, ei in zip(residuals, x_list, x_recon_list, edge_indices):
        # Cosine distance using actual x_recon
        cos_sim = F.cosine_similarity(x, xr, dim=1)
        cos_dist = 1.0 - cos_sim

        # L1 on high-signal dims
        valid_dims = [d for d in [17, 23] if d < res.size(1)]
        if valid_dims:
            l1 = res[:, valid_dims].mean(dim=1)
        else:
            l1 = res.mean(dim=1)

        cos_norm = _minmax_normalize(cos_dist)
        l1_norm = _minmax_normalize(l1)
        scores = alpha * cos_norm + (1.0 - alpha) * l1_norm

        if smooth_rounds > 0:
            scores = _graph_smooth(scores, ei, rounds=smooth_rounds,
                                   alpha=smooth_alpha)
        all_scores.append(scores)
    return all_scores


def score_strategy_extended_dims(residuals, x_list, x_recon_list,
                                  edge_indices, dims,
                                  alpha=0.7, smooth_rounds=1, smooth_alpha=0.7):
    """Cosine + L1 on extended dimension set."""
    all_scores = []
    for res, x, xr, ei in zip(residuals, x_list, x_recon_list, edge_indices):
        cos_sim = F.cosine_similarity(x, xr, dim=1)
        cos_dist = 1.0 - cos_sim

        valid_dims = [d for d in dims if d < res.size(1)]
        if valid_dims:
            l1 = res[:, valid_dims].mean(dim=1)
        else:
            l1 = res.mean(dim=1)

        cos_norm = _minmax_normalize(cos_dist)
        l1_norm = _minmax_normalize(l1)
        scores = alpha * cos_norm + (1.0 - alpha) * l1_norm

        if smooth_rounds > 0:
            scores = _graph_smooth(scores, ei, rounds=smooth_rounds,
                                   alpha=smooth_alpha)
        all_scores.append(scores)
    return all_scores


def score_strategy_mahalanobis(residuals, labels, dims=None):
    """Mahalanobis distance on residual patterns (fitted on healthy nodes)."""
    # Concatenate all healthy node residuals for covariance estimation
    all_healthy_res = []
    for res, lbl in zip(residuals, labels):
        healthy_mask = lbl == 0
        if dims:
            valid = [d for d in dims if d < res.size(1)]
            all_healthy_res.append(res[healthy_mask][:, valid].numpy())
        else:
            all_healthy_res.append(res[healthy_mask].numpy())
    healthy_cat = np.concatenate(all_healthy_res, axis=0)

    # Subsample for covariance estimation (memory)
    if len(healthy_cat) > 50000:
        idx = np.random.choice(len(healthy_cat), 50000, replace=False)
        healthy_cat = healthy_cat[idx]

    mean = healthy_cat.mean(axis=0)
    cov = np.cov(healthy_cat, rowvar=False)
    # Regularize covariance
    cov += np.eye(cov.shape[0]) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # Score each node
    all_scores = []
    for res, lbl in zip(residuals, labels):
        if dims:
            valid = [d for d in dims if d < res.size(1)]
            R = res[:, valid].numpy()
        else:
            R = res.numpy()
        diff = R - mean
        # Mahalanobis: sqrt((x-mu)^T Sigma^-1 (x-mu))
        mahal = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        all_scores.append(torch.from_numpy(mahal.astype(np.float32)))
    return all_scores


def score_strategy_isolation_forest(residuals, labels, dims=None,
                                     contamination=0.01):
    """Isolation Forest on residual patterns."""
    from sklearn.ensemble import IsolationForest

    # Concatenate all residuals
    all_res_list = []
    for res in residuals:
        if dims:
            valid = [d for d in dims if d < res.size(1)]
            all_res_list.append(res[:, valid].numpy())
        else:
            all_res_list.append(res.numpy())
    all_res = np.concatenate(all_res_list, axis=0)

    # Subsample for training (Isolation Forest is memory-intensive)
    n_total = len(all_res)
    max_train = 100000
    if n_total > max_train:
        train_idx = np.random.choice(n_total, max_train, replace=False)
        train_data = all_res[train_idx]
    else:
        train_data = all_res

    print("    Training IsolationForest on %d samples..." % len(train_data))
    iforest = IsolationForest(
        contamination=contamination, random_state=42, n_jobs=-1)
    iforest.fit(train_data)

    # Score all nodes (negative anomaly score → higher = more anomalous)
    raw_scores = -iforest.score_samples(all_res)

    # Split back into per-graph scores
    all_scores = []
    offset = 0
    for res in residuals:
        n = res.size(0)
        scores = raw_scores[offset:offset + n]
        all_scores.append(torch.from_numpy(scores.astype(np.float32)))
        offset += n
    return all_scores


def score_strategy_per_graph_norm(residuals, x_list, x_recon_list,
                                   edge_indices, dims,
                                   alpha=0.7, smooth_rounds=1,
                                   smooth_alpha=0.7):
    """Per-graph normalization: scores normalized within each graph."""
    all_scores = []
    for res, x, xr, ei in zip(residuals, x_list, x_recon_list, edge_indices):
        cos_sim = F.cosine_similarity(x, xr, dim=1)
        cos_dist = 1.0 - cos_sim

        valid_dims = [d for d in dims if d < res.size(1)]
        if valid_dims:
            l1 = res[:, valid_dims].mean(dim=1)
        else:
            l1 = res.mean(dim=1)

        # Per-graph z-score normalization
        cos_z = (cos_dist - cos_dist.mean()) / (cos_dist.std() + 1e-8)
        l1_z = (l1 - l1.mean()) / (l1.std() + 1e-8)

        scores = alpha * cos_z + (1.0 - alpha) * l1_z

        if smooth_rounds > 0:
            scores = _graph_smooth(scores, ei, rounds=smooth_rounds,
                                   alpha=smooth_alpha)
        all_scores.append(scores)
    return all_scores


def score_strategy_max_dim(residuals, edge_indices, dims,
                            smooth_rounds=1, smooth_alpha=0.7):
    """Max residual across physics dims (catches any single-dim anomaly)."""
    all_scores = []
    for res, ei in zip(residuals, edge_indices):
        valid_dims = [d for d in dims if d < res.size(1)]
        if valid_dims:
            scores = res[:, valid_dims].max(dim=1).values
        else:
            scores = res.max(dim=1).values

        scores = _minmax_normalize(scores)
        if smooth_rounds > 0:
            scores = _graph_smooth(scores, ei, rounds=smooth_rounds,
                                   alpha=smooth_alpha)
        all_scores.append(scores)
    return all_scores


def evaluate_scores(all_scores, all_labels, name):
    """Evaluate a scoring strategy."""
    scores_np = torch.cat(all_scores).numpy()
    labels_np = torch.cat(all_labels).numpy()
    m = compute_metrics(scores_np, labels_np)
    return m


def main():
    parser = argparse.ArgumentParser(
        description='PRAD Scoring Strategy Sweep')
    parser.add_argument('--checkpoint',
                        default='checkpoints/prad_mae_sage.pt')
    parser.add_argument('--data_dir',
                        default='data/processed_s12_czm_thermal_200')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 70)
    print("PRAD Scoring Strategy Sweep")
    print("=" * 70)

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
    print("  Model: %s (epoch %d)" % (args.checkpoint, ckpt.get('epoch', -1)))

    # Load data
    _, val_data = load_data(args.data_dir, device)
    all_labels_cat = torch.cat([d.y for d in val_data])
    print("  Data: %d val graphs, %d nodes, %d defect (%.2f%%)" % (
        len(val_data), len(all_labels_cat),
        (all_labels_cat > 0).sum().item(),
        100.0 * (all_labels_cat > 0).sum().item() / len(all_labels_cat)))

    # Get residuals (once, reuse for all strategies)
    print("\n  Computing residuals...")
    residuals, labels, edge_indices, x_list, x_recon_list = \
        get_residuals(model, val_data)
    print("  Done.\n")

    # ---- Sweep strategies ----
    results = {}

    # 1. Baseline (current best)
    print("--- Strategy 1: Baseline (cosine + L1(s12,le12)) ---")
    for alpha in [0.5, 0.7, 0.8, 0.9, 1.0]:
        for sr, sa in [(0, 1.0), (1, 0.5), (1, 0.7), (2, 0.5)]:
            scores = score_strategy_baseline(
                residuals, x_list, x_recon_list, edge_indices,
                alpha=alpha, smooth_rounds=sr, smooth_alpha=sa)
            name = "baseline_a%.1f_sr%d_sa%.1f" % (alpha, sr, sa)
            m = evaluate_scores(scores, labels, name)
            results[name] = m

    # Find best baseline
    best_baseline = max(
        [(k, v) for k, v in results.items() if k.startswith('baseline')],
        key=lambda x: x[1]['best_f1'])
    print("  Best baseline: %s  F1=%.4f  ROC=%.4f  PR=%.4f" % (
        best_baseline[0], best_baseline[1]['best_f1'],
        best_baseline[1]['roc_auc'], best_baseline[1]['pr_auc']))

    # 2. Extended dims: all stress + strain
    print("\n--- Strategy 2: Extended dims (stress + strain) ---")
    for alpha in [0.5, 0.7, 0.8]:
        for sr, sa in [(1, 0.5), (1, 0.7)]:
            scores = score_strategy_extended_dims(
                residuals, x_list, x_recon_list, edge_indices,
                EXTENDED_SIGNAL_DIMS,
                alpha=alpha, smooth_rounds=sr, smooth_alpha=sa)
            name = "ext_stress_strain_a%.1f_sr%d_sa%.1f" % (alpha, sr, sa)
            m = evaluate_scores(scores, labels, name)
            results[name] = m

    # 3. All physics + mechanical
    print("\n--- Strategy 3: All physics + mechanical dims ---")
    for alpha in [0.5, 0.7, 0.8]:
        scores = score_strategy_extended_dims(
            residuals, x_list, x_recon_list, edge_indices,
            ALL_PHYSICS_MECH_DIMS,
            alpha=alpha, smooth_rounds=1, smooth_alpha=0.7)
        name = "all_phys_mech_a%.1f" % alpha
        m = evaluate_scores(scores, labels, name)
        results[name] = m

    # 4. Mahalanobis distance
    print("\n--- Strategy 4: Mahalanobis distance ---")
    for dims_name, dims in [
        ('physics', PHYSICS_DIMS),
        ('stress_strain', EXTENDED_SIGNAL_DIMS),
        ('all_phys_mech', ALL_PHYSICS_MECH_DIMS),
    ]:
        scores = score_strategy_mahalanobis(residuals, labels, dims)
        # Apply smoothing
        for sr, sa in [(0, 1.0), (1, 0.7)]:
            smoothed = []
            for s, ei in zip(scores, edge_indices):
                if sr > 0:
                    s = _graph_smooth(s, ei, rounds=sr, alpha=sa)
                smoothed.append(s)
            name = "mahal_%s_sr%d" % (dims_name, sr)
            m = evaluate_scores(smoothed, labels, name)
            results[name] = m

    # 5. Per-graph normalized
    print("\n--- Strategy 5: Per-graph z-score normalization ---")
    for dims_name, dims in [
        ('s12_le12', [17, 23]),
        ('stress_strain', EXTENDED_SIGNAL_DIMS),
    ]:
        for alpha in [0.5, 0.7]:
            scores = score_strategy_per_graph_norm(
                residuals, x_list, x_recon_list, edge_indices, dims,
                alpha=alpha, smooth_rounds=1, smooth_alpha=0.7)
            name = "pergraph_%s_a%.1f" % (dims_name, alpha)
            m = evaluate_scores(scores, labels, name)
            results[name] = m

    # 6. Max residual across dims
    print("\n--- Strategy 6: Max residual across dims ---")
    for dims_name, dims in [
        ('physics', PHYSICS_DIMS),
        ('stress_strain', EXTENDED_SIGNAL_DIMS),
        ('all_phys_mech', ALL_PHYSICS_MECH_DIMS),
    ]:
        scores = score_strategy_max_dim(
            residuals, edge_indices, dims,
            smooth_rounds=1, smooth_alpha=0.7)
        name = "maxdim_%s" % dims_name
        m = evaluate_scores(scores, labels, name)
        results[name] = m

    # 7. Isolation Forest (slower but powerful)
    print("\n--- Strategy 7: Isolation Forest ---")
    for dims_name, dims in [
        ('stress_strain', EXTENDED_SIGNAL_DIMS),
        ('all_phys_mech', ALL_PHYSICS_MECH_DIMS),
    ]:
        scores_raw = score_strategy_isolation_forest(
            residuals, labels, dims, contamination=0.01)
        # With and without smoothing
        for sr, sa in [(0, 1.0), (1, 0.7)]:
            smoothed = []
            for s, ei in zip(scores_raw, edge_indices):
                if sr > 0:
                    s = _graph_smooth(s, ei, rounds=sr, alpha=sa)
                smoothed.append(s)
            name = "iforest_%s_sr%d" % (dims_name, sr)
            m = evaluate_scores(smoothed, labels, name)
            results[name] = m

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SCORING STRATEGY SWEEP RESULTS")
    print("=" * 70)
    print("  %-40s  ROC-AUC  PR-AUC  F1      P       R" % "Strategy")
    print("  " + "-" * 85)

    # Sort by F1
    sorted_results = sorted(results.items(), key=lambda x: -x[1]['best_f1'])
    for name, m in sorted_results[:20]:
        print("  %-40s  %.4f   %.4f  %.4f  %.4f  %.4f" % (
            name, m['roc_auc'], m['pr_auc'], m['best_f1'],
            m['precision'], m['recall']))

    print("\n  " + "-" * 85)
    best = sorted_results[0]
    print("  BEST: %s" % best[0])
    print("    ROC-AUC: %.4f" % best[1]['roc_auc'])
    print("    PR-AUC:  %.4f" % best[1]['pr_auc'])
    print("    F1:      %.4f" % best[1]['best_f1'])
    print("    P:       %.4f" % best[1]['precision'])
    print("    R:       %.4f" % best[1]['recall'])

    # Compare to known baseline
    print("\n  Current baseline (alpha=0.7, sr=1, sa=0.7): F1=0.775")
    print("  Improvement: %+.4f" % (best[1]['best_f1'] - 0.775))


if __name__ == '__main__':
    main()
