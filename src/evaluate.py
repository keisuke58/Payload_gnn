# -*- coding: utf-8 -*-
"""
Evaluation & Visualization for Fairing Defect Localization

Features:
- Quantitative metrics (Accuracy, F1, Precision, Recall, AUC, IoU)
- Localization error (mm)
- Architecture comparison table
- Defect prediction heatmaps (3D shell plot)
- Attention weight visualization (GAT)
- Confusion matrix
- OOD generalization evaluation

Usage:
    python src/evaluate.py --checkpoint runs/gat_xxx/best_model.pt --data_dir dataset/processed
    python src/evaluate.py --compare_dir runs/ --data_dir dataset/processed
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report,
)

from models import build_model


# =========================================================================
# Metrics
# =========================================================================
def compute_full_metrics(logits, targets, pos):
    """Compute full evaluation metrics including localization error and IoU."""
    preds = logits.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    pos_np = pos.cpu().numpy()

    # Classification metrics
    acc = (preds == targets_np).mean()
    f1 = f1_score(targets_np, preds, zero_division=0)
    prec = precision_score(targets_np, preds, zero_division=0)
    rec = recall_score(targets_np, preds, zero_division=0)
    try:
        auc = roc_auc_score(targets_np, probs)
    except ValueError:
        auc = 0.0

    # IoU (Intersection over Union)
    intersection = ((preds == 1) & (targets_np == 1)).sum()
    union = ((preds == 1) | (targets_np == 1)).sum()
    iou = intersection / max(union, 1)

    # Localization error (distance between predicted and true defect centroids)
    true_defect_mask = targets_np == 1
    pred_defect_mask = preds == 1

    if true_defect_mask.sum() > 0 and pred_defect_mask.sum() > 0:
        true_centroid = pos_np[true_defect_mask].mean(axis=0)
        pred_centroid = pos_np[pred_defect_mask].mean(axis=0)
        loc_error = np.linalg.norm(true_centroid - pred_centroid)
    else:
        loc_error = float('inf')

    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'auc': float(auc),
        'iou': float(iou),
        'localization_error_mm': float(loc_error),
        'n_true_defect': int(true_defect_mask.sum()),
        'n_pred_defect': int(pred_defect_mask.sum()),
    }


# =========================================================================
# Evaluation
# =========================================================================
@torch.no_grad()
def evaluate_dataset(model, data_list, device, label='test'):
    """Evaluate model on a dataset, return per-sample and aggregate metrics."""
    model.eval()
    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    sample_results = []
    all_logits, all_targets, all_pos = [], [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        metrics = compute_full_metrics(out, batch.y, batch.pos)
        sample_results.append(metrics)

        all_logits.append(out.cpu())
        all_targets.append(batch.y.cpu())
        all_pos.append(batch.pos.cpu())

    # Aggregate
    agg = {}
    for key in sample_results[0]:
        vals = [r[key] for r in sample_results if r[key] != float('inf')]
        if vals:
            agg[key + '_mean'] = float(np.mean(vals))
            agg[key + '_std'] = float(np.std(vals))

    # Global confusion matrix
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    preds_all = logits_cat.argmax(dim=1).numpy()
    targets_all = targets_cat.numpy()
    cm = confusion_matrix(targets_all, preds_all, labels=[0, 1])

    agg['confusion_matrix'] = cm.tolist()
    agg['classification_report'] = classification_report(
        targets_all, preds_all, target_names=['Healthy', 'Defect'],
        zero_division=0)

    print("\n=== %s Results ===" % label.upper())
    print("  F1:    %.4f +/- %.4f" % (agg.get('f1_mean', 0), agg.get('f1_std', 0)))
    print("  AUC:   %.4f +/- %.4f" % (agg.get('auc_mean', 0), agg.get('auc_std', 0)))
    print("  IoU:   %.4f +/- %.4f" % (agg.get('iou_mean', 0), agg.get('iou_std', 0)))
    print("  LocErr: %.1f +/- %.1f mm" %
          (agg.get('localization_error_mm_mean', 0),
           agg.get('localization_error_mm_std', 0)))
    print("  Confusion Matrix:\n", np.array(cm))
    print(agg['classification_report'])

    return agg, sample_results


# =========================================================================
# Visualization
# =========================================================================
def plot_prediction_heatmap(data, logits, save_path):
    """Plot 3D defect prediction heatmap on the shell surface."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available, skipping visualization.")
        return

    pos = data.pos.cpu().numpy()
    targets = data.y.cpu().numpy()
    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    fig = plt.figure(figsize=(18, 6))

    # Ground truth
    ax1 = fig.add_subplot(131, projection='3d')
    sc1 = ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=targets, cmap='RdYlBu_r', s=1, vmin=0, vmax=1)
    ax1.set_title('Ground Truth')
    plt.colorbar(sc1, ax=ax1, shrink=0.5)

    # Prediction probability
    ax2 = fig.add_subplot(132, projection='3d')
    sc2 = ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=probs, cmap='RdYlBu_r', s=1, vmin=0, vmax=1)
    ax2.set_title('Prediction Probability')
    plt.colorbar(sc2, ax=ax2, shrink=0.5)

    # Error (FP=red, FN=blue, correct=gray)
    preds = (probs > 0.5).astype(int)
    error_map = np.zeros(len(pos))
    error_map[(preds == 1) & (targets == 0)] = 1.0   # FP
    error_map[(preds == 0) & (targets == 1)] = -1.0   # FN

    ax3 = fig.add_subplot(133, projection='3d')
    sc3 = ax3.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=error_map, cmap='coolwarm', s=1, vmin=-1, vmax=1)
    ax3.set_title('Error (Red=FP, Blue=FN)')
    plt.colorbar(sc3, ax=ax3, shrink=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved heatmap: %s" % save_path)


def plot_training_curve(log_path, save_path):
    """Plot training curves from CSV log."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import csv as csv_module
    except ImportError:
        return

    epochs, train_loss, val_loss = [], [], []
    train_f1, val_f1 = [], []

    with open(log_path, 'r') as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            train_f1.append(float(row['train_f1']))
            val_f1.append(float(row['val_f1']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, label='Train')
    ax1.plot(epochs, val_loss, label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_f1, label='Train')
    ax2.plot(epochs, val_f1, label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =========================================================================
# Architecture comparison
# =========================================================================
def compare_architectures(compare_dir, data_dir, device):
    """Load all checkpoints in compare_dir and generate comparison table."""
    results = {}
    test_data = torch.load(os.path.join(data_dir, 'test.pt'), weights_only=False)

    for run_name in sorted(os.listdir(compare_dir)):
        ckpt_path = os.path.join(compare_dir, run_name, 'best_model.pt')
        if not os.path.exists(ckpt_path):
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        run_args = argparse.Namespace(**ckpt['args'])

        sample = test_data[0]
        in_ch = sample.x.shape[1]
        edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0

        model = build_model(
            run_args.arch, in_ch, edge_dim,
            hidden_channels=run_args.hidden, num_layers=run_args.layers,
            dropout=0.0, num_classes=2,
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        print("\n--- Evaluating: %s (%s) ---" % (run_name, run_args.arch.upper()))
        agg, _ = evaluate_dataset(model, test_data, device, label=run_name)
        results[run_name] = {
            'arch': run_args.arch,
            'f1': agg.get('f1_mean', 0),
            'auc': agg.get('auc_mean', 0),
            'iou': agg.get('iou_mean', 0),
            'loc_err': agg.get('localization_error_mm_mean', 0),
        }

    # Print comparison table
    print("\n" + "=" * 75)
    print("%-20s %-8s %-8s %-8s %-8s %-12s" %
          ('Run', 'Arch', 'F1', 'AUC', 'IoU', 'LocErr(mm)'))
    print("-" * 75)
    for name, r in sorted(results.items(), key=lambda x: -x[1]['f1']):
        print("%-20s %-8s %-8.4f %-8.4f %-8.4f %-12.1f" %
              (name[:20], r['arch'], r['f1'], r['auc'], r['iou'], r['loc_err']))
    print("=" * 75)

    return results


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate GNN defect localization')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to best_model.pt checkpoint')
    parser.add_argument('--data_dir', type=str, default='dataset/processed')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--compare_dir', type=str, default=None,
                        help='Directory containing multiple runs for comparison')
    parser.add_argument('--eval_ood', action='store_true',
                        help='Also evaluate on OOD test set')
    parser.add_argument('--num_vis', type=int, default=5,
                        help='Number of heatmap visualizations to generate')
    parser.add_argument('--bayes_enable', action='store_true',
                        help='Enable Bayesian aggregation with MC Dropout')
    parser.add_argument('--bayes_T', type=int, default=20,
                        help='Number of MC Dropout forward passes')
    parser.add_argument('--bayes_prior_p', type=float, default=0.02,
                        help='Beta prior mean for defect occurrence')
    parser.add_argument('--bayes_strength', type=float, default=10.0,
                        help='Pseudo-count strength for Beta prior')
    parser.add_argument('--uncertainty', action='store_true',
                        help='Run MC Dropout uncertainty quantification')
    parser.add_argument('--mc_T', type=int, default=30,
                        help='MC Dropout forward passes (default: 30)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Architecture comparison mode
    if args.compare_dir:
        results = compare_architectures(args.compare_dir, args.data_dir, device)
        with open(os.path.join(args.output_dir, 'comparison.json'), 'w') as f:
            json.dump(results, f, indent=2)
        return

    # Single checkpoint evaluation
    if not args.checkpoint:
        print("Error: provide --checkpoint or --compare_dir")
        return

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    run_args = argparse.Namespace(**ckpt['args'])

    # Load test data
    test_data = torch.load(os.path.join(args.data_dir, 'test.pt'),
                           weights_only=False)
    print("Test samples: %d" % len(test_data))

    # Build model
    sample = test_data[0]
    in_ch = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0

    model = build_model(
        run_args.arch, in_ch, edge_dim,
        hidden_channels=run_args.hidden, num_layers=run_args.layers,
        dropout=getattr(run_args, 'dropout', 0.0), num_classes=2,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print("Loaded %s model from epoch %d (val_f1=%.4f)" %
          (run_args.arch.upper(), ckpt['epoch'], ckpt['val_f1']))

    # Evaluate on test set (standard)
    test_agg, test_samples = evaluate_dataset(model, test_data, device, label='test')
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_agg, f, indent=2, default=str)

    # Bayesian aggregation (optional)
    if args.bayes_enable:
        def mc_dropout_probs(model, data, T):
            model.train()
            probs_list = []
            for _ in range(T):
                out = model(data.x, data.edge_index, data.edge_attr)
                probs = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
                probs_list.append(probs)
            return np.stack(probs_list, axis=0)

        def beta_posterior_mean(k, T, prior_p, strength):
            alpha0 = prior_p * strength
            beta0 = (1.0 - prior_p) * strength
            return (alpha0 + k) / (alpha0 + beta0 + T)

        model.eval()
        bayes_sample_results = []
        for data in test_data:
            data = data.to(device)
            probs_mc = mc_dropout_probs(model, data, args.bayes_T)
            k = (probs_mc > 0.5).sum(axis=0)
            post_mean = beta_posterior_mean(k, args.bayes_T, args.bayes_prior_p, args.bayes_strength)
            logits_bayes = torch.from_numpy(
                np.stack([1.0 - post_mean, post_mean], axis=1)
            ).to(device)
            metrics = compute_full_metrics(logits_bayes, data.y, data.pos)
            bayes_sample_results.append(metrics)

        bayes_agg = {}
        for key in bayes_sample_results[0]:
            vals = [r[key] for r in bayes_sample_results if r[key] != float('inf')]
            if vals:
                bayes_agg[key + '_mean'] = float(np.mean(vals))
                bayes_agg[key + '_std'] = float(np.std(vals))
        logits_cat = []
        targets_cat = []
        for i, data in enumerate(test_data):
            data = data.to(device)
            probs_mc = mc_dropout_probs(model, data, args.bayes_T)
            k = (probs_mc > 0.5).sum(axis=0)
            post_mean = beta_posterior_mean(k, args.bayes_T, args.bayes_prior_p, args.bayes_strength)
            preds = (post_mean > 0.5).astype(int)
            logits_cat.append(torch.from_numpy(preds))
            targets_cat.append(data.y.cpu())
        preds_all = torch.cat(logits_cat, dim=0).numpy()
        targets_all = torch.cat(targets_cat, dim=0).numpy()
        cm = confusion_matrix(targets_all, preds_all, labels=[0, 1])
        bayes_agg['confusion_matrix'] = cm.tolist()
        bayes_agg['classification_report'] = classification_report(
            targets_all, preds_all, target_names=['Healthy', 'Defect'], zero_division=0)
        print("\n=== BAYES Results ===")
        print("  F1:    %.4f +/- %.4f" % (bayes_agg.get('f1_mean', 0), bayes_agg.get('f1_std', 0)))
        print("  AUC:   %.4f +/- %.4f" % (bayes_agg.get('auc_mean', 0), bayes_agg.get('auc_std', 0)))
        print("  IoU:   %.4f +/- %.4f" % (bayes_agg.get('iou_mean', 0), bayes_agg.get('iou_std', 0)))
        print("  LocErr: %.1f +/- %.1f mm" %
              (bayes_agg.get('localization_error_mm_mean', 0),
               bayes_agg.get('localization_error_mm_std', 0)))
        print("  Confusion Matrix:\n", np.array(cm))
        print(bayes_agg['classification_report'])
        with open(os.path.join(args.output_dir, 'test_results_bayes.json'), 'w') as f:
            json.dump(bayes_agg, f, indent=2, default=str)

    # Uncertainty quantification via MC Dropout
    if args.uncertainty:
        from uncertainty import (
            mc_dropout_predict, uncertainty_quality_metrics,
            plot_uncertainty_map, plot_calibration_curve,
        )
        print("\n=== MC Dropout Uncertainty (T=%d) ===" % args.mc_T)

        unc_dir = os.path.join(args.output_dir, 'uncertainty')
        os.makedirs(unc_dir, exist_ok=True)

        all_probs, all_std, all_targets_unc = [], [], []
        for i, data in enumerate(test_data):
            data = data.to(device)
            mean_p, std_p, entropy = mc_dropout_predict(
                model, data.x, data.edge_index, data.edge_attr,
                T=args.mc_T)
            all_probs.append(mean_p)
            all_std.append(std_p)
            all_targets_unc.append(data.y.cpu().numpy())

            if i < 5:
                pos = data.pos.cpu().numpy() if data.pos is not None else data.x[:, :3].cpu().numpy()
                plot_uncertainty_map(
                    pos, mean_p, std_p, data.y.cpu().numpy(),
                    os.path.join(unc_dir, 'mc_uncertainty_%03d.png' % i))

        probs_cat = np.concatenate(all_probs)
        std_cat = np.concatenate(all_std)
        targets_cat_unc = np.concatenate(all_targets_unc)

        uq_metrics = uncertainty_quality_metrics(probs_cat, std_cat, targets_cat_unc)
        print("  ECE: %.4f" % uq_metrics['ece'])
        if 'uncertainty_auroc' in uq_metrics:
            print("  Uncertainty AUROC: %.4f" % uq_metrics['uncertainty_auroc'])
        if 'uncertainty_separation' in uq_metrics:
            print("  Separation (incorrect-correct): %.4f" % uq_metrics['uncertainty_separation'])
        for k, v in uq_metrics.items():
            if k.startswith('accuracy_reject'):
                print("  %s: %.4f" % (k, v))
        print("  Mean uncertainty: %.4f" % std_cat.mean())

        plot_calibration_curve(probs_cat, targets_cat_unc,
                               os.path.join(unc_dir, 'calibration_mc.png'))

        with open(os.path.join(unc_dir, 'uncertainty_metrics.json'), 'w') as f:
            json.dump(uq_metrics, f, indent=2, default=float)
        print("  Saved to: %s" % unc_dir)

    # OOD evaluation
    if args.eval_ood:
        ood_path = os.path.join(args.data_dir, 'ood.pt')
        if os.path.exists(ood_path):
            ood_data = torch.load(ood_path, weights_only=False)
            print("\nOOD samples: %d" % len(ood_data))
            ood_agg, _ = evaluate_dataset(model, ood_data, device, label='ood')
            with open(os.path.join(args.output_dir, 'ood_results.json'), 'w') as f:
                json.dump(ood_agg, f, indent=2, default=str)

    # Visualizations
    model.eval()
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    for i in range(min(args.num_vis, len(test_data))):
        data = test_data[i].to(device)
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.edge_attr)
        save_path = os.path.join(vis_dir, 'heatmap_sample_%03d.png' % i)
        plot_prediction_heatmap(data, logits, save_path)

    # Training curve
    log_dir = os.path.dirname(args.checkpoint)
    log_path = os.path.join(log_dir, 'training_log.csv')
    if os.path.exists(log_path):
        plot_training_curve(log_path, os.path.join(args.output_dir, 'training_curve.png'))

    print("\nResults saved to %s" % args.output_dir)


if __name__ == '__main__':
    main()
