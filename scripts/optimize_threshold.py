#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_threshold.py — 学習済みGNNモデルの閾値最適化

ベストモデルをロードし、val データに対して閾値をスイープして
F1 を最大化する最適閾値を見つける。

Usage:
    python scripts/optimize_threshold.py \
        --checkpoint runs/sweep_binary/.../best_model.pt \
        --data_dir data/processed_s12_czm_96_binary
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from models import build_model
from torch_geometric.loader import DataLoader


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})
    in_channels = ckpt['in_channels']
    edge_attr_dim = ckpt.get('edge_attr_dim', 0)

    # Auto-detect num_classes from checkpoint
    state = ckpt['model_state_dict']
    # head.3.weight shape is (num_classes, hidden//2)
    for k, v in state.items():
        if 'head' in k and 'weight' in k and v.dim() == 2:
            last_weight = v
    num_classes = last_weight.shape[0]

    model = build_model(
        args.get('arch', 'sage'),
        in_channels,
        edge_attr_dim,
        hidden_channels=args.get('hidden', 128),
        num_layers=args.get('layers', 4),
        dropout=0.0,  # no dropout at inference
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def collect_predictions(model, data_list, device, norm_stats=None):
    """Run forward pass on all graphs and collect probabilities + labels."""
    # Apply normalization
    if norm_stats is not None:
        x_mean = norm_stats['mean']
        x_std = norm_stats['std']
        for d in data_list:
            d.x = (d.x - x_mean) / x_std

    # Normalize edge_attr
    edge_attrs = [d.edge_attr for d in data_list if d.edge_attr is not None]
    if edge_attrs:
        ea_cat = torch.cat(edge_attrs, dim=0)
        ea_mean = ea_cat.mean(dim=0)
        ea_std = ea_cat.std(dim=0)
        ea_std[ea_std < 1e-8] = 1.0
        for d in data_list:
            if d.edge_attr is not None:
                d.edge_attr = (d.edge_attr - ea_mean) / ea_std

    loader = DataLoader(data_list, batch_size=4, shuffle=False)
    all_probs = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_targets.append(batch.y.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_targets)


def sweep_thresholds(probs, targets, n_steps=99):
    """Sweep thresholds and find optimal F1."""
    thresholds = np.linspace(0.01, 0.99, n_steps)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        prec = precision_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, zero_division=0)
        results.append({'threshold': t, 'f1': f1, 'precision': prec, 'recall': rec})

    return results


def main():
    parser = argparse.ArgumentParser(description='Optimize decision threshold for binary GNN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data dir with val.pt and norm_stats.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model, ckpt = load_model(args.checkpoint, device)
    train_args = ckpt.get('args', {})
    print("Model: %s | Checkpoint F1: %.4f" % (
        train_args.get('arch', '?'), ckpt.get('val_f1', 0)))

    # Load data
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    val_data = torch.load(os.path.join(data_dir, 'val.pt'), weights_only=False)

    # Load norm stats
    norm_path = os.path.join(data_dir, 'norm_stats.pt')
    norm_stats = torch.load(norm_path, weights_only=False) if os.path.exists(norm_path) else None

    print("Val: %d graphs" % len(val_data))

    # Collect predictions
    probs, targets = collect_predictions(model, val_data, device, norm_stats)
    print("Nodes: %d | Defect: %d (%.2f%%)" % (
        len(targets), targets.sum(), 100 * targets.mean()))

    # AUC
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = 0.0
    print("AUC: %.4f" % auc)

    # Threshold sweep
    results = sweep_thresholds(probs, targets)

    # Find best
    best = max(results, key=lambda r: r['f1'])
    print("\n=== Threshold Sweep Results ===")
    print("Best threshold: %.2f" % best['threshold'])
    print("  F1:        %.4f" % best['f1'])
    print("  Precision: %.4f" % best['precision'])
    print("  Recall:    %.4f" % best['recall'])

    # Default (0.5) comparison
    default = [r for r in results if abs(r['threshold'] - 0.5) < 0.02][0]
    print("\nDefault (0.50):")
    print("  F1:        %.4f" % default['f1'])
    print("  Precision: %.4f" % default['precision'])
    print("  Recall:    %.4f" % default['recall'])

    # Top-5 thresholds
    top5 = sorted(results, key=lambda r: -r['f1'])[:5]
    print("\nTop-5 thresholds:")
    for r in top5:
        print("  t=%.2f → F1=%.4f  P=%.4f  R=%.4f" % (
            r['threshold'], r['f1'], r['precision'], r['recall']))

    # Classification report at best threshold
    preds_best = (probs >= best['threshold']).astype(int)
    print("\nClassification Report (threshold=%.2f):" % best['threshold'])
    print(classification_report(targets, preds_best,
                                target_names=['Healthy', 'Defect'], zero_division=0))


if __name__ == '__main__':
    main()
