#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ensemble_inference.py — 5-Fold アンサンブル + 閾値最適化 + グラフ後処理

3つの改善を統合:
  1. 5-Fold アンサンブル: 5モデルの P(defect) を平均
  2. 閾値最適化: recall-aware な最適閾値をスイープで決定
  3. グラフベース後処理: k-hop 近傍のラベル伝搬で孤立誤検出除去・見逃し補完

Usage:
    python scripts/ensemble_inference.py
    python scripts/ensemble_inference.py --target_recall 0.90
"""

import argparse
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from models import build_model

# 5-fold model directories — DW=5+Residual (recall-optimized)
FOLD_DIRS = [
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_185224_fold0'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_185615_fold1'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_190406_fold2'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_190839_fold3'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_191500_fold4'),
]
# Baseline model directories (for comparison)
BASELINE_FOLD_DIRS = [
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_160640_fold0'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_161051_fold1'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_161330_fold2'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_161936_fold3'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_162502_fold4'),
]
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_s12_czm_thermal_200_binary')


def load_fold_model(fold_dir):
    """Load a single fold model."""
    ckpt_path = os.path.join(fold_dir, 'best_model.pt')
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    args_dict = ckpt.get('args', {})
    in_channels = ckpt['in_channels']
    edge_attr_dim = ckpt['edge_attr_dim']

    head_key = [k for k in ckpt['model_state_dict'] if 'head' in k and 'weight' in k]
    num_classes = ckpt['model_state_dict'][head_key[-1]].shape[0] if head_key else 2

    model = build_model(
        'gat', in_channels, edge_attr_dim,
        hidden_channels=args_dict.get('hidden', 128),
        num_layers=args_dict.get('layers', 4),
        dropout=0.0,  # inference: no dropout
        num_classes=num_classes,
        use_residual=args_dict.get('residual', False),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


@torch.no_grad()
def ensemble_predict(models, val_data, node_mean, node_std):
    """Run ensemble inference: average P(defect) across all models.

    Returns per-graph list of dicts with: probs, targets, pos, edge_index.
    """
    results = []
    for gi, graph in enumerate(val_data):
        x = graph.x.clone()
        if node_mean is not None:
            x = (x - node_mean) / node_std.clamp(min=1e-8)

        # Average across models
        prob_sum = np.zeros(graph.x.shape[0], dtype=np.float64)
        for model in models:
            logits = model(x, graph.edge_index, graph.edge_attr, None)
            probs = F.softmax(logits, dim=1)[:, 1].numpy()
            prob_sum += probs
        avg_probs = (prob_sum / len(models)).astype(np.float32)

        results.append({
            'idx': gi,
            'probs': avg_probs,
            'targets': graph.y.numpy(),
            'pos': graph.x[:, :3].numpy(),
            'edge_index': graph.edge_index.numpy(),
        })
    return results


def graph_postprocess(probs, edge_index, threshold, min_cluster=3, propagate_iters=2):
    """Graph-based post-processing of predictions.

    1. Initial classification at given threshold
    2. Remove isolated defect predictions (< min_cluster connected defect neighbors)
    3. Propagate defect labels to high-probability neighbors of defect clusters

    Args:
        probs: P(defect) array, shape (N,)
        edge_index: (2, E) array
        threshold: classification threshold
        min_cluster: minimum connected defect nodes to keep
        propagate_iters: iterations of label propagation

    Returns:
        refined_preds: post-processed binary predictions
    """
    N = len(probs)
    preds = (probs >= threshold).astype(int)

    # Build adjacency list
    adj = [[] for _ in range(N)]
    src, dst = edge_index[0], edge_index[1]
    for s, d in zip(src, dst):
        adj[s].append(d)

    # Step 1: Remove isolated defect nodes
    # A defect node is "isolated" if fewer than min_cluster of its neighbors are also defect
    refined = preds.copy()
    defect_nodes = np.where(preds == 1)[0]
    for node in defect_nodes:
        n_defect_neighbors = sum(1 for nb in adj[node] if preds[nb] == 1)
        if n_defect_neighbors < min_cluster:
            refined[node] = 0  # Remove isolated false positive

    # Step 2: Propagate defect labels to high-probability neighbors
    # If a healthy node has many defect neighbors AND high P(defect), mark it as defect
    propagation_threshold = threshold * 0.6  # lower threshold for propagation
    for _ in range(propagate_iters):
        new_defects = []
        for node in range(N):
            if refined[node] == 1:
                continue  # already defect
            if probs[node] < propagation_threshold:
                continue  # too low probability
            n_defect_neighbors = sum(1 for nb in adj[node] if refined[nb] == 1)
            n_neighbors = len(adj[node])
            if n_neighbors > 0 and n_defect_neighbors / n_neighbors >= 0.3:
                new_defects.append(node)
        for node in new_defects:
            refined[node] = 1

    return refined


def sweep_thresholds(all_probs, all_targets, n_steps=199):
    """Sweep thresholds on concatenated predictions."""
    thresholds = np.linspace(0.01, 0.99, n_steps)
    results = []
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_targets, preds, zero_division=0)
        prec = precision_score(all_targets, preds, zero_division=0)
        rec = recall_score(all_targets, preds, zero_division=0)
        results.append({'threshold': t, 'f1': f1, 'precision': prec, 'recall': rec})
    return results


def sweep_with_postprocess(graph_results, n_steps=99):
    """Sweep thresholds WITH graph post-processing applied."""
    thresholds = np.linspace(0.05, 0.70, n_steps)
    results = []
    for t in thresholds:
        all_preds, all_targets = [], []
        for r in graph_results:
            refined = graph_postprocess(r['probs'], r['edge_index'], t)
            all_preds.append(refined)
            all_targets.append(r['targets'])
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        prec = precision_score(all_targets, all_preds, zero_division=0)
        rec = recall_score(all_targets, all_preds, zero_division=0)
        results.append({'threshold': t, 'f1': f1, 'precision': prec, 'recall': rec})
    return results


def print_comparison(label, prec, rec, f1):
    """Print a row of the comparison table."""
    print("  %-35s  P=%.4f  R=%.4f  F1=%.4f" % (label, prec, rec, f1))


def main():
    parser = argparse.ArgumentParser(description='Ensemble + threshold + post-processing')
    parser.add_argument('--target_recall', type=float, default=0.90,
                        help='Minimum target recall (default: 0.90)')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Save detailed results JSON')
    args = parser.parse_args()

    print("=" * 70)
    print("Ensemble Inference: 5-Fold + Threshold Optimization + Post-Processing")
    print("=" * 70)

    # Load all 5 fold models
    print("\nLoading 5 fold models...")
    models = []
    for i, fd in enumerate(FOLD_DIRS):
        model = load_fold_model(fd)
        models.append(model)
        print("  Fold %d: %s" % (i, os.path.basename(fd)))

    # Load val data
    print("\nLoading val data...")
    val_data = torch.load(os.path.join(DATA_DIR, 'val.pt'), weights_only=False)
    ns = torch.load(os.path.join(DATA_DIR, 'norm_stats.pt'), weights_only=False)
    node_mean, node_std = ns['mean'], ns['std']
    print("  %d graphs" % len(val_data))

    # ===== Step 1: Single model baseline (best fold, threshold=0.5) =====
    print("\n--- Baseline: Single Model (Best Fold), threshold=0.5 ---")
    best_fold_results = ensemble_predict([models[1]], val_data, node_mean, node_std)
    probs_single = np.concatenate([r['probs'] for r in best_fold_results])
    targets_all = np.concatenate([r['targets'] for r in best_fold_results])
    preds_baseline = (probs_single >= 0.5).astype(int)

    baseline_f1 = f1_score(targets_all, preds_baseline, zero_division=0)
    baseline_prec = precision_score(targets_all, preds_baseline, zero_division=0)
    baseline_rec = recall_score(targets_all, preds_baseline, zero_division=0)

    print_comparison("Baseline (single, t=0.5)", baseline_prec, baseline_rec, baseline_f1)

    # ===== Step 2: Ensemble (5-fold average, threshold=0.5) =====
    print("\n--- Step 1: 5-Fold Ensemble ---")
    ensemble_results = ensemble_predict(models, val_data, node_mean, node_std)
    probs_ensemble = np.concatenate([r['probs'] for r in ensemble_results])

    preds_ens_05 = (probs_ensemble >= 0.5).astype(int)
    ens_f1 = f1_score(targets_all, preds_ens_05, zero_division=0)
    ens_prec = precision_score(targets_all, preds_ens_05, zero_division=0)
    ens_rec = recall_score(targets_all, preds_ens_05, zero_division=0)
    ens_auc = roc_auc_score(targets_all, probs_ensemble)

    print_comparison("Ensemble (5-fold, t=0.5)", ens_prec, ens_rec, ens_f1)
    print("  AUC: %.4f" % ens_auc)

    # ===== Step 3: Threshold optimization (ensemble, no post-processing) =====
    print("\n--- Step 2: Threshold Optimization (Ensemble) ---")
    sweep_results = sweep_thresholds(probs_ensemble, targets_all)

    # Best F1
    best_f1_result = max(sweep_results, key=lambda r: r['f1'])
    print_comparison("Best F1 (t=%.3f)" % best_f1_result['threshold'],
                     best_f1_result['precision'], best_f1_result['recall'],
                     best_f1_result['f1'])

    # Best threshold meeting recall target
    recall_candidates = [r for r in sweep_results if r['recall'] >= args.target_recall]
    if recall_candidates:
        best_recall_f1 = max(recall_candidates, key=lambda r: r['f1'])
        print_comparison("Best F1 @ Recall>=%.0f%% (t=%.3f)" % (
            args.target_recall * 100, best_recall_f1['threshold']),
            best_recall_f1['precision'], best_recall_f1['recall'],
            best_recall_f1['f1'])
    else:
        print("  WARNING: No threshold achieves recall >= %.2f" % args.target_recall)
        best_recall_f1 = min(sweep_results, key=lambda r: abs(r['recall'] - args.target_recall))

    # ===== Step 4: Graph post-processing + threshold =====
    print("\n--- Step 3: Graph Post-Processing + Threshold ---")
    pp_results = sweep_with_postprocess(ensemble_results)

    best_pp_f1 = max(pp_results, key=lambda r: r['f1'])
    print_comparison("Best F1 w/ postprocess (t=%.3f)" % best_pp_f1['threshold'],
                     best_pp_f1['precision'], best_pp_f1['recall'],
                     best_pp_f1['f1'])

    pp_recall_cands = [r for r in pp_results if r['recall'] >= args.target_recall]
    if pp_recall_cands:
        best_pp_recall = max(pp_recall_cands, key=lambda r: r['f1'])
        print_comparison("Best F1 @ Recall>=%.0f%% w/ pp (t=%.3f)" % (
            args.target_recall * 100, best_pp_recall['threshold']),
            best_pp_recall['precision'], best_pp_recall['recall'],
            best_pp_recall['f1'])
    else:
        best_pp_recall = min(pp_results, key=lambda r: abs(r['recall'] - args.target_recall))
        print_comparison("Closest to Recall>=%.0f%% w/ pp (t=%.3f)" % (
            args.target_recall * 100, best_pp_recall['threshold']),
            best_pp_recall['precision'], best_pp_recall['recall'],
            best_pp_recall['f1'])

    # ===== Summary comparison =====
    print("\n" + "=" * 70)
    print("SUMMARY — Before vs After")
    print("=" * 70)
    print("%-40s  Prec     Recall   F1" % "Method")
    print("-" * 70)
    print_comparison("(A) Single model, t=0.50", baseline_prec, baseline_rec, baseline_f1)
    print_comparison("(B) Ensemble, t=0.50", ens_prec, ens_rec, ens_f1)
    print_comparison("(C) Ensemble, optimal t (F1)", best_f1_result['precision'],
                     best_f1_result['recall'], best_f1_result['f1'])
    if recall_candidates:
        print_comparison("(D) Ensemble, t @ Rec>=%.0f%%" % (args.target_recall * 100),
                         best_recall_f1['precision'], best_recall_f1['recall'],
                         best_recall_f1['f1'])
    print_comparison("(E) Ensemble + PostProc, best F1", best_pp_f1['precision'],
                     best_pp_f1['recall'], best_pp_f1['f1'])
    if pp_recall_cands:
        print_comparison("(F) Ensemble + PP, Rec>=%.0f%%" % (args.target_recall * 100),
                         best_pp_recall['precision'], best_pp_recall['recall'],
                         best_pp_recall['f1'])

    # Per-sample analysis with best config
    print("\n--- Per-Sample Results (Ensemble + PostProcess, t=%.3f) ---" %
          best_pp_f1['threshold'])
    optimal_t = best_pp_f1['threshold']
    for r in ensemble_results:
        refined = graph_postprocess(r['probs'], r['edge_index'], optimal_t)
        targets = r['targets']
        n_defect = int((targets > 0).sum())
        if n_defect > 0:
            f1 = f1_score(targets > 0, refined, zero_division=0)
            prec = precision_score(targets > 0, refined, zero_division=0)
            rec = recall_score(targets > 0, refined, zero_division=0)
            print("  val[%2d]  %4d defect nodes  F1=%.3f  P=%.3f  R=%.3f" % (
                r['idx'], n_defect, f1, prec, rec))

    # Save results
    save_path = args.save_results or os.path.join(PROJECT_ROOT, 'runs', 'ensemble_results.json')
    output = {
        'baseline': {'precision': baseline_prec, 'recall': baseline_rec, 'f1': baseline_f1},
        'ensemble_t05': {'precision': ens_prec, 'recall': ens_rec, 'f1': ens_f1, 'auc': ens_auc},
        'ensemble_best_f1': best_f1_result,
        'ensemble_pp_best_f1': best_pp_f1,
        'target_recall': args.target_recall,
    }
    if recall_candidates:
        output['ensemble_recall_target'] = best_recall_f1
    if pp_recall_cands:
        output['ensemble_pp_recall_target'] = best_pp_recall

    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print("\nResults saved: %s" % save_path)


if __name__ == '__main__':
    main()
