#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Dataset Evaluation — Test trained GNN on external SHM datasets.

Evaluates a pretrained model on different datasets to measure generalization.
This is key for demonstrating foundation-model-like capability.

Usage:
  python src/eval_cross_dataset.py \
      --model runs/verify_realistic/gat_20260301_010732/best_model.pt \
      --datasets data/processed_external/ogw3.pt data/processed_separation_15/val.pt \
      --output results/cross_dataset_eval.json
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint."""
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    args = ckpt.get('args', {})
    in_channels = ckpt.get('in_channels', 34)
    edge_attr_dim = ckpt.get('edge_attr_dim', 5)

    # Import model builder
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import build_model

    arch = args.get('arch', 'gat')
    hidden = args.get('hidden', 64)
    layers = args.get('layers', 3)
    dropout = args.get('dropout', 0.1)
    # Detect num_classes from checkpoint
    head_key = [k for k in ckpt['model_state_dict'] if 'head' in k and 'weight' in k]
    if head_key:
        num_classes = ckpt['model_state_dict'][head_key[-1]].shape[0]
    else:
        num_classes = 2

    model = build_model(arch, in_channels, edge_attr_dim,
                         hidden_channels=hidden, num_layers=layers,
                         dropout=dropout, num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    print("Model: %s | Params: %d | In: %d | Edge: %d" % (
        arch, sum(p.numel() for p in model.parameters()),
        in_channels, edge_attr_dim))

    return model, in_channels, edge_attr_dim


def pad_features(data, target_node_dim, target_edge_dim):
    """Pad node/edge features to match model input dimensions."""
    x = data.x
    if x.shape[1] < target_node_dim:
        pad = torch.zeros(x.shape[0], target_node_dim - x.shape[1])
        x = torch.cat([x, pad], dim=1)
    elif x.shape[1] > target_node_dim:
        x = x[:, :target_node_dim]

    edge_attr = data.edge_attr
    if edge_attr is not None:
        if edge_attr.shape[1] < target_edge_dim:
            pad = torch.zeros(edge_attr.shape[0],
                             target_edge_dim - edge_attr.shape[1])
            edge_attr = torch.cat([edge_attr, pad], dim=1)
        elif edge_attr.shape[1] > target_edge_dim:
            edge_attr = edge_attr[:, :target_edge_dim]
    else:
        edge_attr = torch.zeros(data.edge_index.shape[1], target_edge_dim)

    return x, edge_attr


def evaluate_dataset(model, dataset_path, in_channels, edge_attr_dim,
                      device='cpu'):
    """Evaluate model on a dataset and return metrics."""
    data_list = torch.load(dataset_path, weights_only=False, map_location=device)
    if not isinstance(data_list, list):
        data_list = [data_list]

    results = []
    all_probs = []
    all_labels = []

    for data in data_list:
        x, edge_attr = pad_features(data, in_channels, edge_attr_dim)
        x = x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = edge_attr.to(device)
        y = data.y.to(device) if data.y is not None else None

        with torch.no_grad():
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            out = model(x, edge_index, edge_attr, batch)
            probs = F.softmax(out, dim=1)

        case_name = getattr(data, 'case_name', 'unknown')
        n_nodes = x.shape[0]

        if y is not None and y.max() > 0:
            y_np = y.cpu().numpy()
            probs_np = probs[:, 1].cpu().numpy()
            pred = (probs_np > 0.5).astype(int)

            # Handle single-class edge case
            if len(np.unique(y_np)) > 1:
                auc = roc_auc_score(y_np, probs_np)
            else:
                auc = float('nan')

            f1 = f1_score(y_np, pred, zero_division=0)
            prec = precision_score(y_np, pred, zero_division=0)
            rec = recall_score(y_np, pred, zero_division=0)

            # Threshold-optimized F1
            best_f1, best_t = 0, 0.5
            for t in np.arange(0.01, 1.0, 0.01):
                pred_t = (probs_np > t).astype(int)
                f1_t = f1_score(y_np, pred_t, zero_division=0)
                if f1_t > best_f1:
                    best_f1, best_t = f1_t, t

            all_probs.extend(probs_np.tolist())
            all_labels.extend(y_np.tolist())
        else:
            auc = f1 = prec = rec = best_f1 = float('nan')
            best_t = 0.5
            # For healthy-only data, check prediction distribution
            probs_np = probs[:, 1].cpu().numpy()

        result = {
            'case': case_name,
            'nodes': n_nodes,
            'auc': float(auc) if not np.isnan(auc) else None,
            'f1': float(f1) if not np.isnan(f1) else None,
            'precision': float(prec) if not np.isnan(prec) else None,
            'recall': float(rec) if not np.isnan(rec) else None,
            'opt_f1': float(best_f1) if not np.isnan(best_f1) else None,
            'opt_threshold': float(best_t),
            'mean_p_defect': float(probs_np.mean()),
            'max_p_defect': float(probs_np.max()),
        }
        results.append(result)
        print("  %s: %d nodes | AUC=%s F1=%s OptF1=%s | P(defect): mean=%.4f max=%.4f" % (
            case_name, n_nodes,
            "%.4f" % auc if not np.isnan(auc) else "N/A",
            "%.4f" % f1 if not np.isnan(f1) else "N/A",
            "%.4f" % best_f1 if not np.isnan(best_f1) else "N/A",
            probs_np.mean(), probs_np.max()))

    # Aggregate metrics
    aggregate = {}
    if all_labels and len(np.unique(all_labels)) > 1:
        aggregate['auc'] = float(roc_auc_score(all_labels, all_probs))
        pred_all = (np.array(all_probs) > 0.5).astype(int)
        aggregate['f1'] = float(f1_score(all_labels, pred_all, zero_division=0))
        aggregate['n_total'] = len(all_labels)
        aggregate['n_defect'] = int(sum(all_labels))

    return {
        'dataset': os.path.basename(dataset_path),
        'per_case': results,
        'aggregate': aggregate,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Cross-dataset evaluation of trained GNN')
    parser.add_argument('--model', required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Paths to dataset .pt files')
    parser.add_argument('--output', default='results/cross_dataset_eval.json',
                        help='Output JSON path')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print("=" * 60)
    print("Cross-Dataset Evaluation")
    print("=" * 60)

    model, in_ch, edge_dim = load_model(args.model, args.device)

    all_results = {
        'model_path': args.model,
        'in_channels': in_ch,
        'edge_attr_dim': edge_dim,
        'evaluations': [],
    }

    for ds_path in args.datasets:
        print("\n--- %s ---" % os.path.basename(ds_path))
        if not os.path.exists(ds_path):
            print("  SKIP: file not found")
            continue
        result = evaluate_dataset(model, ds_path, in_ch, edge_dim, args.device)
        all_results['evaluations'].append(result)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n\nResults saved: %s" % args.output)

    # Print summary table
    print("\n" + "=" * 60)
    print("%-25s %6s %6s %6s" % ("Dataset/Case", "AUC", "F1", "OptF1"))
    print("-" * 60)
    for ev in all_results['evaluations']:
        for case in ev['per_case']:
            print("%-25s %6s %6s %6s" % (
                case['case'][:25],
                "%.3f" % case['auc'] if case['auc'] else "N/A",
                "%.3f" % case['f1'] if case['f1'] else "N/A",
                "%.3f" % case['opt_f1'] if case['opt_f1'] else "N/A",
            ))
    print("=" * 60)


if __name__ == '__main__':
    main()
