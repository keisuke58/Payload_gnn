# -*- coding: utf-8 -*-
"""Residual-based anomaly scoring for PRAD.

Uses the trained PI-GraphMAE to reconstruct node features, then
computes per-node anomaly scores from reconstruction residuals.
Defect nodes will have high residuals because the model learned
"healthy physics" and defects break that expectation.
"""

import sys
import os

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prad import PHYSICS_DIMS, MECHANICAL_DIMS, STRESS_DIMS
from prad.graphmae import PIGraphMAE


def compute_anomaly_scores(model, data, stress_weight=2.0):
    """Compute per-node anomaly scores from cosine distance.

    Uses cosine distance (1 - cos_sim) between original and reconstructed
    features as the anomaly score. This aligns with the training objective
    (scaled cosine error) and is much more discriminative than L1 residual
    for detecting defect nodes.

    Args:
        model: trained PIGraphMAE.
        data: PyG Data object with .x, .edge_index, .edge_attr.
        stress_weight: unused (kept for API compatibility).

    Returns:
        scores: (N,) anomaly scores per node (higher = more anomalous).
        residual: (N, D) full residual vector per node.
    """
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        x_recon = model.reconstruct(data)

    residual = (data.x - x_recon).abs()  # (N, D)

    # Cosine distance: matches the training loss function
    cos_sim = F.cosine_similarity(data.x, x_recon, dim=1)  # (N,)
    scores = 1.0 - cos_sim  # higher = more anomalous

    return scores, residual


def score_dataset(model, data_list, stress_weight=2.0):
    """Score all graphs in a dataset.

    Args:
        model: trained PIGraphMAE.
        data_list: list of PyG Data objects.
        stress_weight: extra weight for stress dimensions.

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
            model, data, stress_weight=stress_weight)
        all_scores.append(scores.cpu())
        all_labels.append(data.y.cpu())
        all_residuals.append(residual.cpu())

    return all_scores, all_labels, all_residuals


def load_model_and_score(checkpoint_path, data_dir, device='cpu',
                         stress_weight=2.0):
    """Load checkpoint and score validation data.

    Args:
        checkpoint_path: path to prad_mae_*.pt checkpoint.
        data_dir: path to processed data directory.
        device: torch device.
        stress_weight: extra weight for stress dimensions.

    Returns:
        val_scores, val_labels, val_residuals
    """
    from prad.train_mae import load_data

    ckpt = torch.load(checkpoint_path, map_location=device,
                      weights_only=False)
    args = ckpt['args']

    # Rebuild model from checkpoint args
    model = PIGraphMAE(
        encoder_arch=args['encoder_arch'],
        in_channels=args.get('in_channels', 34),
        hidden_channels=args['hidden'],
        num_layers=args['num_layers'],
        dropout=args['dropout'],
        mask_ratio=args['mask_ratio'],
        lambda_physics=args['lambda_physics'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    _, val_data = load_data(data_dir, device)

    return score_dataset(model, val_data, stress_weight=stress_weight)
