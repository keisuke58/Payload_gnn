# -*- coding: utf-8 -*-
"""Anomaly scoring for PRAD.

Combines cosine distance (direction) with per-dimension L1 residual
(magnitude) and graph-spatial smoothing for robust defect detection.
"""

import sys
import os

import torch
import torch.nn.functional as F
import numpy as np

try:
    from torch_scatter import scatter_mean
except ImportError:
    from torch_geometric.utils import scatter
    def scatter_mean(src, index, dim=0, dim_size=None):
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prad import PHYSICS_DIMS, MECHANICAL_DIMS, STRESS_DIMS
from prad.graphmae import PIGraphMAE

# High-signal dimensions identified by diagnostic (defect/healthy ratio > 1.5)
HIGH_SIGNAL_DIMS = [17, 23]  # s12, le12 (ratio 2.55x)


def compute_anomaly_scores(model, data, alpha=0.7, smooth_rounds=1,
                           smooth_alpha=0.7, stress_weight=2.0):
    """Ensemble anomaly scoring: cosine distance + targeted L1 + smoothing.

    Combines:
    - Cosine distance (captures direction mismatch across all dims)
    - Weighted L1 on high-signal dimensions (s12, le12 where defects are 2.5x)
    - Graph spatial smoothing (reduces isolated false positives)

    Args:
        model: trained PIGraphMAE.
        data: PyG Data object with .x, .edge_index, .edge_attr.
        alpha: blend weight for cosine vs L1 (1.0 = cosine only).
        smooth_rounds: number of graph smoothing iterations.
        smooth_alpha: blending for smoothing (1.0 = no smoothing).
        stress_weight: unused (kept for API compatibility).

    Returns:
        scores: (N,) anomaly scores per node (higher = more anomalous).
        residual: (N, D) full residual vector per node.
    """
    model.eval()
    with torch.no_grad():
        x_recon = model.reconstruct(data)

    residual = (data.x - x_recon).abs()  # (N, D)

    # Component 1: Cosine distance (direction mismatch)
    cos_sim = F.cosine_similarity(data.x, x_recon, dim=1)  # (N,)
    cos_dist = 1.0 - cos_sim

    # Component 2: L1 on high-signal dimensions only
    valid_dims = [d for d in HIGH_SIGNAL_DIMS if d < residual.size(1)]
    if valid_dims:
        stress_l1 = residual[:, valid_dims].mean(dim=1)  # (N,)
    else:
        stress_l1 = residual.mean(dim=1)

    # Normalize each to [0, 1] range for fair blending
    cos_norm = _minmax_normalize(cos_dist)
    l1_norm = _minmax_normalize(stress_l1)

    # Ensemble blend
    scores = alpha * cos_norm + (1.0 - alpha) * l1_norm

    # Graph spatial smoothing
    if smooth_rounds > 0 and hasattr(data, 'edge_index'):
        scores = _graph_smooth(scores, data.edge_index,
                               rounds=smooth_rounds, alpha=smooth_alpha)

    return scores, residual


def _minmax_normalize(x):
    """Min-max normalize to [0, 1]."""
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < 1e-10:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def _graph_smooth(scores, edge_index, rounds=1, alpha=0.7):
    """Smooth scores over the graph to reduce isolated false positives.

    smoothed = alpha * score_i + (1 - alpha) * mean(score_j for j in N(i))
    """
    N = scores.size(0)
    src, dst = edge_index[0], edge_index[1]

    for _ in range(rounds):
        neighbor_mean = scatter_mean(scores[src], dst, dim=0, dim_size=N)
        scores = alpha * scores + (1.0 - alpha) * neighbor_mean

    return scores


def score_dataset(model, data_list, alpha=0.7, smooth_rounds=1,
                  smooth_alpha=0.7, stress_weight=2.0):
    """Score all graphs in a dataset.

    Args:
        model: trained PIGraphMAE.
        data_list: list of PyG Data objects.
        alpha: cosine vs L1 blend weight.
        smooth_rounds: graph smoothing iterations.
        smooth_alpha: smoothing blend weight.
        stress_weight: unused (API compat).

    Returns:
        all_scores: list of (N_i,) tensors.
        all_labels: list of (N_i,) tensors (ground truth).
        all_residuals: list of (N_i, D) tensors.
    """
    all_scores = []
    all_labels = []
    all_residuals = []

    for data in data_list:
        scores, residual = compute_anomaly_scores(
            model, data, alpha=alpha, smooth_rounds=smooth_rounds,
            smooth_alpha=smooth_alpha)
        all_scores.append(scores.cpu())
        all_labels.append(data.y.cpu())
        all_residuals.append(residual.cpu())

    return all_scores, all_labels, all_residuals


def load_model_and_score(checkpoint_path, data_dir, device='cpu',
                         alpha=0.7, smooth_rounds=1, smooth_alpha=0.7,
                         stress_weight=2.0):
    """Load checkpoint and score validation data.

    Args:
        checkpoint_path: path to prad_mae_*.pt checkpoint.
        data_dir: path to processed data directory.
        device: torch device.
        alpha: cosine vs L1 blend weight.
        smooth_rounds: graph smoothing iterations.
        smooth_alpha: smoothing blend weight.
        stress_weight: unused (API compat).

    Returns:
        val_scores, val_labels, val_residuals
    """
    from prad.train_mae import load_data

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

    return score_dataset(model, val_data, alpha=alpha,
                         smooth_rounds=smooth_rounds,
                         smooth_alpha=smooth_alpha)
