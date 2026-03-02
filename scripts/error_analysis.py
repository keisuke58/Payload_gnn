#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
error_analysis.py — Binary GNN モデルのエラー分析

欠陥タイプ別・グラフ別の検出性能を詳細分析する。
8クラスデータから元の欠陥タイプを復元し、binary予測と照合。

Usage:
    python scripts/error_analysis.py \
        --checkpoint runs/s12_binary/sage_.../best_model.pt \
        --binary_dir data/processed_s12_czm_96_binary \
        --multiclass_dir data/processed_s12_czm_96
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from models import build_model
from torch_geometric.loader import DataLoader

DEFECT_NAMES = {
    0: 'healthy',
    1: 'debonding',
    2: 'fod',
    3: 'impact',
    4: 'delamination',
    5: 'inner_debond',
    6: 'thermal_progression',
    7: 'acoustic_fatigue',
}


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})
    in_channels = ckpt['in_channels']
    edge_attr_dim = ckpt.get('edge_attr_dim', 0)

    state = ckpt['model_state_dict']
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
        dropout=0.0,
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_graph(model, data, device, norm_stats, ea_mean, ea_std):
    """Run inference on a single graph, return probs and targets."""
    d = data.clone()
    if norm_stats is not None:
        d.x = (d.x - norm_stats['mean']) / norm_stats['std']
    if d.edge_attr is not None and ea_mean is not None:
        d.edge_attr = (d.edge_attr - ea_mean) / ea_std
    d = d.to(device)
    out = model(d.x, d.edge_index, d.edge_attr,
                torch.zeros(d.x.size(0), dtype=torch.long, device=device))
    probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    return probs


def compute_edge_stats(data_list):
    """Compute edge_attr normalization stats."""
    edge_attrs = [d.edge_attr for d in data_list if d.edge_attr is not None]
    if not edge_attrs:
        return None, None
    ea_cat = torch.cat(edge_attrs, dim=0)
    ea_mean = ea_cat.mean(dim=0)
    ea_std = ea_cat.std(dim=0)
    ea_std[ea_std < 1e-8] = 1.0
    return ea_mean, ea_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--binary_dir', required=True)
    parser.add_argument('--multiclass_dir', default=None,
                        help='8-class data dir (optional, for defect type info)')
    parser.add_argument('--defect_meta', default=None,
                        help='JSON with per-graph defect type (alternative to multiclass_dir)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Decision threshold (default: auto-optimize)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve paths
    for attr in ['checkpoint', 'binary_dir']:
        v = getattr(args, attr)
        if not os.path.isabs(v):
            setattr(args, attr, os.path.join(PROJECT_ROOT, v))
    if args.multiclass_dir and not os.path.isabs(args.multiclass_dir):
        args.multiclass_dir = os.path.join(PROJECT_ROOT, args.multiclass_dir)
    if args.defect_meta and not os.path.isabs(args.defect_meta):
        args.defect_meta = os.path.join(PROJECT_ROOT, args.defect_meta)

    # Load model
    model, ckpt = load_model(args.checkpoint, device)
    print("Model: %s | Best val F1: %.4f" % (
        ckpt.get('args', {}).get('arch', '?'), ckpt.get('val_f1', 0)))

    # Load binary val data (for predictions)
    bin_val = torch.load(os.path.join(args.binary_dir, 'val.pt'), weights_only=False)

    norm_path = os.path.join(args.binary_dir, 'norm_stats.pt')
    norm_stats = torch.load(norm_path, weights_only=False) if os.path.exists(norm_path) else None

    # Load defect type info
    import json as _json
    defect_meta = None
    if args.defect_meta and os.path.exists(args.defect_meta):
        with open(args.defect_meta) as f:
            defect_meta = _json.load(f)
    elif args.multiclass_dir:
        mc_val = torch.load(os.path.join(args.multiclass_dir, 'val.pt'), weights_only=False)
        defect_meta = []
        for i, md in enumerate(mc_val):
            mc_labels = md.y.numpy()
            defect_mask = mc_labels > 0
            n_defect = int(defect_mask.sum())
            primary = int(np.bincount(mc_labels[defect_mask]).argmax()) if n_defect > 0 else 0
            defect_meta.append({'idx': i, 'primary_defect_type': primary, 'n_defect': n_defect})
        del mc_val

    print("Val graphs: %d" % len(bin_val))

    # Edge normalization stats from training data
    bin_train = torch.load(os.path.join(args.binary_dir, 'train.pt'), weights_only=False)
    ea_mean, ea_std = compute_edge_stats(bin_train)
    del bin_train

    # --- Per-graph analysis ---
    print("\n" + "=" * 70)
    print(" PER-GRAPH ANALYSIS")
    print("=" * 70)

    graph_results = []
    for i, bd in enumerate(bin_val):
        probs = predict_graph(model, bd, device, norm_stats, ea_mean, ea_std)
        binary_labels = bd.y.numpy()

        n_defect = int((binary_labels > 0).sum())

        # Get defect type from metadata
        if defect_meta and i < len(defect_meta):
            primary_type = defect_meta[i]['primary_defect_type']
        else:
            primary_type = 0 if n_defect == 0 else -1

        graph_results.append({
            'idx': i,
            'defect_type': primary_type,
            'defect_name': DEFECT_NAMES.get(primary_type, 'unknown'),
            'n_nodes': len(binary_labels),
            'n_defect': int(n_defect),
            'probs': probs,
            'targets': binary_labels,
        })

    # Find optimal threshold if not specified
    if args.threshold is None:
        all_probs = np.concatenate([g['probs'] for g in graph_results])
        all_targets = np.concatenate([g['targets'] for g in graph_results])
        best_f1, best_t = 0, 0.5
        for t in np.linspace(0.01, 0.99, 99):
            preds = (all_probs >= t).astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(all_targets, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        threshold = best_t
        print("Auto-optimized threshold: %.2f (F1=%.4f)" % (threshold, best_f1))
    else:
        threshold = args.threshold
        print("Using threshold: %.2f" % threshold)

    # Per-graph metrics
    from sklearn.metrics import f1_score, precision_score, recall_score
    print("\n%-4s %-22s %6s %6s %8s %8s %8s" % (
        '#', 'Defect Type', 'Nodes', 'Defect', 'F1', 'Prec', 'Recall'))
    print("-" * 70)

    for g in graph_results:
        preds = (g['probs'] >= threshold).astype(int)
        if g['n_defect'] > 0:
            f1 = f1_score(g['targets'], preds, zero_division=0)
            prec = precision_score(g['targets'], preds, zero_division=0)
            rec = recall_score(g['targets'], preds, zero_division=0)
        else:
            # Healthy graph: compute FP rate
            fp = preds.sum()
            f1 = 1.0 if fp == 0 else 0.0
            prec = 1.0 if fp == 0 else 0.0
            rec = 1.0  # no defects to miss
        g['f1'] = f1
        g['prec'] = prec
        g['rec'] = rec
        g['preds'] = preds

        marker = " ***" if f1 < 0.5 and g['n_defect'] > 0 else ""
        print("%-4d %-22s %6d %6d %8.4f %8.4f %8.4f%s" % (
            g['idx'], g['defect_name'], g['n_nodes'], g['n_defect'],
            f1, prec, rec, marker))

    # --- Per-defect-type summary ---
    print("\n" + "=" * 70)
    print(" PER-DEFECT-TYPE SUMMARY")
    print("=" * 70)

    type_stats = defaultdict(lambda: {
        'graphs': 0, 'total_defect': 0, 'tp': 0, 'fp': 0, 'fn': 0,
        'f1_list': [], 'probs_defect': [], 'probs_healthy': []
    })

    for g in graph_results:
        dt = g['defect_name']
        s = type_stats[dt]
        s['graphs'] += 1
        s['total_defect'] += g['n_defect']
        preds = g['preds']
        targets = g['targets']
        s['tp'] += int(((preds == 1) & (targets == 1)).sum())
        s['fp'] += int(((preds == 1) & (targets == 0)).sum())
        s['fn'] += int(((preds == 0) & (targets == 1)).sum())
        if g['n_defect'] > 0:
            s['f1_list'].append(g['f1'])
            s['probs_defect'].extend(g['probs'][targets == 1].tolist())
        s['probs_healthy'].extend(g['probs'][targets == 0].tolist())

    print("\n%-22s %6s %8s %8s %8s %8s %8s %10s %10s" % (
        'Type', 'Graphs', 'Defect', 'TP', 'FP', 'FN', 'F1(avg)',
        'P(defect)', 'P(healthy)'))
    print("-" * 100)

    for dtype in ['debonding', 'fod', 'impact', 'delamination',
                   'inner_debond', 'thermal_progression', 'acoustic_fatigue', 'healthy']:
        s = type_stats.get(dtype)
        if s is None:
            continue
        f1_avg = np.mean(s['f1_list']) if s['f1_list'] else 0.0
        p_def = np.mean(s['probs_defect']) if s['probs_defect'] else 0.0
        p_hlt = np.mean(s['probs_healthy']) if s['probs_healthy'] else 0.0
        print("%-22s %6d %8d %8d %8d %8d %8.4f %10.4f %10.4f" % (
            dtype, s['graphs'], s['total_defect'],
            s['tp'], s['fp'], s['fn'], f1_avg, p_def, p_hlt))

    # --- Hardest graphs ---
    print("\n" + "=" * 70)
    print(" HARDEST GRAPHS (lowest F1, defect only)")
    print("=" * 70)
    defect_graphs = [g for g in graph_results if g['n_defect'] > 0]
    worst = sorted(defect_graphs, key=lambda g: g['f1'])[:5]
    for g in worst:
        preds = g['preds']
        targets = g['targets']
        tp = ((preds == 1) & (targets == 1)).sum()
        fp = ((preds == 1) & (targets == 0)).sum()
        fn = ((preds == 0) & (targets == 1)).sum()
        avg_prob_defect = g['probs'][targets == 1].mean()
        max_prob_defect = g['probs'][targets == 1].max()
        print("  Graph %d (%s): F1=%.4f | defect=%d, TP=%d, FP=%d, FN=%d" % (
            g['idx'], g['defect_name'], g['f1'], g['n_defect'], tp, fp, fn))
        print("    Defect node probs: mean=%.4f, max=%.4f" % (avg_prob_defect, max_prob_defect))

    # --- False positive analysis ---
    print("\n" + "=" * 70)
    print(" FALSE POSITIVE ANALYSIS")
    print("=" * 70)
    total_fp = sum(s['fp'] for s in type_stats.values())
    total_nodes = sum(g['n_nodes'] for g in graph_results)
    total_healthy_nodes = total_nodes - sum(g['n_defect'] for g in graph_results)
    print("Total FP: %d / %d healthy nodes (%.4f%%)" % (
        total_fp, total_healthy_nodes, 100 * total_fp / total_healthy_nodes))

    # FP spatial distribution (using node features: x,y,z are first 3 dims)
    fp_positions = []
    for g in graph_results:
        fp_mask = (g['preds'] == 1) & (g['targets'] == 0)
        if fp_mask.any():
            # Get original positions from data
            bin_data = bin_val[g['idx']]
            fp_pos = bin_data.x[fp_mask, :3].numpy()  # x, y, z (pre-norm)
            fp_positions.append(fp_pos)

    if fp_positions:
        fp_all = np.concatenate(fp_positions, axis=0)
        print("FP spatial stats (pre-normalization):")
        print("  x: [%.1f, %.1f], mean=%.1f" % (fp_all[:, 0].min(), fp_all[:, 0].max(), fp_all[:, 0].mean()))
        print("  y: [%.1f, %.1f], mean=%.1f" % (fp_all[:, 1].min(), fp_all[:, 1].max(), fp_all[:, 1].mean()))
        print("  z: [%.1f, %.1f], mean=%.1f" % (fp_all[:, 2].min(), fp_all[:, 2].max(), fp_all[:, 2].mean()))

    # --- Probability distribution ---
    print("\n" + "=" * 70)
    print(" PROBABILITY DISTRIBUTION")
    print("=" * 70)
    all_probs = np.concatenate([g['probs'] for g in graph_results])
    all_targets = np.concatenate([g['targets'] for g in graph_results])

    def_probs = all_probs[all_targets == 1]
    hlt_probs = all_probs[all_targets == 0]

    print("Defect nodes (n=%d):" % len(def_probs))
    for pct in [10, 25, 50, 75, 90]:
        print("  P%d = %.4f" % (pct, np.percentile(def_probs, pct)))

    print("Healthy nodes (n=%d):" % len(hlt_probs))
    for pct in [90, 95, 99, 99.9]:
        print("  P%.1f = %.6f" % (pct, np.percentile(hlt_probs, pct)))

    # Separation score
    separation = def_probs.mean() - hlt_probs.mean()
    print("\nSeparation (mean_defect - mean_healthy): %.4f" % separation)
    print("Overlap: defect min=%.4f vs healthy P99.9=%.6f" % (
        def_probs.min(), np.percentile(hlt_probs, 99.9)))


if __name__ == '__main__':
    main()
