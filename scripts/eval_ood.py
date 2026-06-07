#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ood.py — Evaluate a trained GNN on the OOD test split.

Usage:
  python scripts/eval_ood.py \
    --run_dir runs/ood_lgsta_ood_p75 \
    --ood_dir data/ood_p75 \
    --arch lgsta

Output:
  runs/ood_lgsta_ood_p75/ood_results.json  (ID val + OOD test metrics)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from torch_geometric.loader import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _find_best_ckpt(run_dir):
    candidates = [
        os.path.join(run_dir, 'best_model.pt'),
    ]
    # check timestamped subdirs
    for d in sorted(os.listdir(run_dir)):
        cand = os.path.join(run_dir, d, 'best_model.pt')
        if os.path.exists(cand):
            candidates.append(cand)
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError('No best_model.pt found in ' + run_dir)


def threshold_opt_f1(labels, probs_positive):
    """Find threshold that maximises macro-F1 on a binary defect/healthy split."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.05, 0.95, 37):
        preds = (probs_positive >= thr).astype(int)
        f1 = f1_score(labels, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_f1, best_thr


def evaluate_split(model, data_list, norm_stats, device, num_classes=5, tag='split'):
    """Run inference and compute metrics on a list of PyG Data objects."""
    model.eval()

    mean = norm_stats['mean'].to(device)
    std  = norm_stats['std'].to(device)

    all_labels, all_preds, all_probs = [], [], []
    all_def_preds, all_def_labels = [], []   # defect-only binary
    centroid_errors = []

    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.x = (batch.x - mean) / std

            try:
                out = model(batch)
            except TypeError:
                out = model(batch.x, batch.edge_index,
                            getattr(batch, 'edge_attr', None),
                            getattr(batch, 'batch', None))

            if isinstance(out, tuple):
                logits, centroid_pred = out[0], out[1] if len(out) > 1 else None
            else:
                logits, centroid_pred = out, None

            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            y = batch.y.cpu().numpy()
            p = preds.cpu().numpy()
            pr = probs.cpu().numpy()

            all_labels.extend(y)
            all_preds.extend(p)
            all_probs.append(pr)

            # Binary defect detection
            all_def_labels.extend((y > 0).astype(int))
            all_def_preds.extend((p > 0).astype(int))

            # Centroid RMSE (if multitask)
            if centroid_pred is not None:
                def_nodes = (batch.y > 0)
                if def_nodes.sum() > 0:
                    gt_centroid = batch.pos[def_nodes].mean(dim=0).cpu()
                    pred_c = centroid_pred.squeeze().cpu()
                    err = (gt_centroid - pred_c).norm().item()
                    centroid_errors.append(err)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.vstack(all_probs)
    all_def_labels = np.array(all_def_labels)
    all_def_preds  = np.array(all_def_preds)

    # Metrics
    macro_f1 = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    opt_f1, opt_thr = threshold_opt_f1(
        all_def_labels,
        1.0 - all_probs[:, 0]   # prob of being defect (not class-0)
    )
    try:
        prob_def = 1.0 - all_probs[:, 0]
        auc = float(roc_auc_score(all_def_labels, prob_def))
    except ValueError:
        auc = float('nan')

    centroid_rmse = float(np.mean(centroid_errors)) if centroid_errors else None

    per_class = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class[c] = float(f1_score(all_labels == c, all_preds == c,
                                          average='binary', zero_division=0))

    print(f'  [{tag}] macro_F1={macro_f1:.4f}  opt_F1={opt_f1:.4f}(thr={opt_thr:.2f})'
          f'  AUC={auc:.4f}' +
          (f'  centroid_RMSE={centroid_rmse:.1f}mm' if centroid_rmse else ''))

    return {
        'macro_f1': macro_f1,
        'opt_f1':   opt_f1,
        'opt_threshold': opt_thr,
        'auc':      auc,
        'centroid_rmse_mm': centroid_rmse,
        'per_class_f1': per_class,
        'n_samples': len(data_list),
        'n_nodes':   int(len(all_labels)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True,
                        help='Path to run directory (contains best_model.pt or timestamped subdir)')
    parser.add_argument('--ood_dir', required=True,
                        help='Path to OOD data directory (contains val.pt, ood.pt, norm_stats.pt)')
    parser.add_argument('--arch', default='lgsta')
    parser.add_argument('--num_classes', type=int, default=5)
    args = parser.parse_args()

    run_dir = os.path.join(PROJECT_ROOT, args.run_dir) if not os.path.isabs(args.run_dir) else args.run_dir
    ood_dir = os.path.join(PROJECT_ROOT, args.ood_dir) if not os.path.isabs(args.ood_dir) else args.ood_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Load checkpoint
    ckpt_path = _find_best_ckpt(run_dir)
    print('Loading checkpoint:', ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

    # Build model from train.py (import)
    from src.train import build_model
    sample_data = torch.load(os.path.join(ood_dir, 'val.pt'),
                             map_location='cpu', weights_only=False)[0]
    num_features = sample_data.x.shape[1]

    # Parse args for model construction (use defaults + arch override)
    import argparse as _ap
    fake = _ap.Namespace(
        arch=args.arch, num_classes=args.num_classes,
        hidden=128, layers=4, dropout=0.1,
        task='multitask', reg_weight=0.5,
        edge_features=False, use_edge_attr=False,
    )
    model = build_model(fake, num_features).to(device)
    model.load_state_dict(state, strict=False)
    print(f'Model: {args.arch} loaded ({sum(p.numel() for p in model.parameters())} params)')

    # Load data splits
    val_data = torch.load(os.path.join(ood_dir, 'val.pt'),  map_location='cpu', weights_only=False)
    ood_data = torch.load(os.path.join(ood_dir, 'ood.pt'),  map_location='cpu', weights_only=False)
    norm_stats = torch.load(os.path.join(ood_dir, 'norm_stats.pt'), map_location='cpu', weights_only=False)

    print(f'\nID val: {len(val_data)} samples | OOD test: {len(ood_data)} samples')
    print('\n--- Evaluation ---')
    id_metrics  = evaluate_split(model, val_data,  norm_stats, device, args.num_classes, tag='ID-val')
    ood_metrics = evaluate_split(model, ood_data,  norm_stats, device, args.num_classes, tag='OOD-test')

    # OOD degradation
    delta_optf1 = ood_metrics['opt_f1'] - id_metrics['opt_f1']
    delta_auc   = ood_metrics['auc']    - id_metrics['auc']
    print(f'\n  ΔOptF1 (OOD-ID): {delta_optf1:+.4f}')
    print(f'  ΔAUC   (OOD-ID): {delta_auc:+.4f}')

    results = {
        'arch':    args.arch,
        'run_dir': run_dir,
        'ood_dir': ood_dir,
        'id_val':  id_metrics,
        'ood_test': ood_metrics,
        'delta': {'opt_f1': delta_optf1, 'auc': delta_auc},
    }
    out_path = os.path.join(run_dir, 'ood_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
