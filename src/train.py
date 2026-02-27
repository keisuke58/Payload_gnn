# -*- coding: utf-8 -*-
"""
Training Script for Fairing Defect Localization GNN

Features:
- Focal Loss for class imbalance (defect nodes << healthy nodes)
- Cosine Annealing LR scheduler
- Early stopping
- K-Fold cross-validation
- TensorBoard / CSV logging
- Model checkpointing

Usage:
    python src/train.py --arch gat --data_dir dataset/processed --epochs 200
    python src/train.py --arch gat --cross_val 5  # 5-fold CV
"""

import os
import json
import time
import argparse
import csv as csv_module
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from models import build_model


# =========================================================================
# Focal Loss
# =========================================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (defect nodes << healthy)."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        # alpha weighting: higher weight for minority class (defect=1)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * ce
        return loss.mean()


# =========================================================================
# Metrics
# =========================================================================
def compute_metrics(logits, targets):
    """Compute classification metrics."""
    preds = logits.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    acc = (preds == targets_np).mean()
    f1 = f1_score(targets_np, preds, zero_division=0)
    prec = precision_score(targets_np, preds, zero_division=0)
    rec = recall_score(targets_np, preds, zero_division=0)

    try:
        auc = roc_auc_score(targets_np, probs)
    except ValueError:
        auc = 0.0

    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'auc': float(auc),
    }


# =========================================================================
# Training loop
# =========================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(out.detach())
        all_targets.append(batch.y.detach())

    avg_loss = total_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_cat, targets_cat)
    metrics['loss'] = avg_loss
    return metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y)

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(out)
        all_targets.append(batch.y)

    avg_loss = total_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_cat, targets_cat)
    metrics['loss'] = avg_loss
    return metrics


# =========================================================================
# Single training run
# =========================================================================
def train(args, train_data, val_data, fold=None):
    """Execute a single training run."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Model
    sample = train_data[0]
    in_channels = sample.x.shape[1]
    edge_attr_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0

    model = build_model(
        args.arch, in_channels, edge_attr_dim,
        hidden_channels=args.hidden, num_layers=args.layers,
        dropout=args.dropout, num_classes=2,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model: %s | Params: %d | Device: %s" % (args.arch.upper(), param_count, device))

    # Optimizer, scheduler, criterion
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    # Logging
    fold_str = '_fold%d' % fold if fold is not None else ''
    run_name = '%s_%s%s' % (args.arch, datetime.now().strftime('%Y%m%d_%H%M%S'), fold_str)
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv_module.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_f1', 'train_acc',
                         'val_loss', 'val_f1', 'val_acc', 'val_auc', 'lr'])

    # Training
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]['lr']

        # Log
        with open(log_path, 'a', newline='') as f:
            writer = csv_module.writer(f)
            writer.writerow([epoch, '%.6f' % train_m['loss'], '%.4f' % train_m['f1'],
                             '%.4f' % train_m['accuracy'],
                             '%.6f' % val_m['loss'], '%.4f' % val_m['f1'],
                             '%.4f' % val_m['accuracy'], '%.4f' % val_m['auc'],
                             '%.2e' % lr])

        if epoch % args.log_every == 0 or epoch == 1:
            print("  Epoch %3d/%d | Train F1=%.4f Loss=%.4f | "
                  "Val F1=%.4f AUC=%.4f Loss=%.4f | LR=%.2e | %.1fs" %
                  (epoch, args.epochs, train_m['f1'], train_m['loss'],
                   val_m['f1'], val_m['auc'], val_m['loss'], lr, elapsed))

        # Checkpoint best
        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_metrics': val_m,
                'args': vars(args),
                'in_channels': in_channels,
                'edge_attr_dim': edge_attr_dim,
            }, os.path.join(run_dir, 'best_model.pt'))
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print("  Early stopping at epoch %d (patience=%d)" %
                  (epoch, args.patience))
            break

    print("  Best Val F1: %.4f" % best_val_f1)
    return run_dir, best_val_f1


# =========================================================================
# K-Fold Cross-Validation
# =========================================================================
def cross_validate(args, all_data):
    """Run K-fold cross-validation."""
    k = args.cross_val
    n = len(all_data)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // k

    results = []
    for fold in range(k):
        print("\n===== Fold %d / %d =====" % (fold + 1, k))
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k - 1 else n

        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]

        run_dir, best_f1 = train(args, train_data, val_data, fold=fold)
        results.append({'fold': fold, 'best_val_f1': best_f1, 'run_dir': run_dir})

    # Summary
    f1_scores = [r['best_val_f1'] for r in results]
    print("\n===== Cross-Validation Summary =====")
    print("F1 scores: %s" % ['%.4f' % f for f in f1_scores])
    print("Mean F1: %.4f +/- %.4f" % (np.mean(f1_scores), np.std(f1_scores)))

    summary_path = os.path.join(args.output_dir, 'cv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'folds': results,
            'mean_f1': float(np.mean(f1_scores)),
            'std_f1': float(np.std(f1_scores)),
        }, f, indent=2)

    return results


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='Train GNN for defect localization')
    # Data
    parser.add_argument('--data_dir', type=str, default='dataset/processed')
    parser.add_argument('--output_dir', type=str, default='runs')
    # Model
    parser.add_argument('--arch', type=str, default='gat',
                        choices=['gcn', 'gat', 'gin', 'sage'])
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    # Cross-validation
    parser.add_argument('--cross_val', type=int, default=0,
                        help='Number of folds for CV (0=disabled)')
    # Misc
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data from %s ..." % args.data_dir)
    train_data = torch.load(os.path.join(args.data_dir, 'train.pt'),
                            weights_only=False)
    val_data = torch.load(os.path.join(args.data_dir, 'val.pt'),
                          weights_only=False)
    print("Train: %d samples | Val: %d samples" % (len(train_data), len(val_data)))

    if args.cross_val > 1:
        # Merge train + val for CV
        all_data = train_data + val_data
        print("Cross-validation mode: %d folds, %d total samples" %
              (args.cross_val, len(all_data)))
        cross_validate(args, all_data)
    else:
        train(args, train_data, val_data)

    print("\nDone.")


if __name__ == '__main__':
    main()
