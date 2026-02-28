# -*- coding: utf-8 -*-
"""
Training Script for Fairing Defect Localization GNN

Features:
- Focal Loss for class imbalance (defect nodes << healthy nodes)
- Defect-centric sub-graph sampling (--sampler defect_centric)
- Focal Loss gamma grid search (--gamma_search)
- Cosine Annealing LR scheduler
- Early stopping
- K-Fold cross-validation
- TensorBoard / CSV logging
- Model checkpointing

Usage:
    python src/train.py --arch gat --data_dir dataset/processed --epochs 200
    python src/train.py --arch gat --cross_val 5  # 5-fold CV
    python src/train.py --sampler defect_centric --loss focal --focal_gamma 3.0
    python src/train.py --loss focal --gamma_search --epochs 20

Multi-GPU (4x GPU DataParallel):
    torchrun --nproc_per_node=4 src/train.py --multi_gpu --arch gat --batch_size 4
"""

import os
import sys
import json
import time
import argparse
import csv as csv_module
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from torch.utils.tensorboard import SummaryWriter

from models import build_model
from subgraph_sampler import DefectCentricSampler


# =========================================================================
# Focal Loss
# =========================================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance. Supports multi-class."""

    def __init__(self, alpha=None, gamma=2.0, num_classes=2):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float))
        else:
            # Scalar alpha: binary-compatible (alpha for class=1, 1-alpha for class=0)
            if num_classes == 2:
                self.register_buffer('alpha', torch.tensor([1 - alpha, alpha], dtype=torch.float))
            else:
                self.alpha = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal
        return focal.mean()


# =========================================================================
# Metrics
# =========================================================================
DEFECT_TYPE_NAMES = [
    'healthy', 'debonding', 'fod', 'impact',
    'delamination', 'inner_debond', 'thermal_progression', 'acoustic_fatigue',
]


def compute_metrics(logits, targets, num_classes=2):
    """Compute classification metrics (binary or multi-class)."""
    preds = logits.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()

    acc = (preds == targets_np).mean()

    if num_classes == 2:
        # Binary metrics (backward compatible)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        f1 = f1_score(targets_np, preds, zero_division=0)
        prec = precision_score(targets_np, preds, zero_division=0)
        rec = recall_score(targets_np, preds, zero_division=0)
        try:
            auc = roc_auc_score(targets_np, probs)
        except ValueError:
            auc = 0.0
    else:
        # Multi-class: macro averages
        f1 = f1_score(targets_np, preds, average='macro', zero_division=0)
        prec = precision_score(targets_np, preds, average='macro', zero_division=0)
        rec = recall_score(targets_np, preds, average='macro', zero_division=0)
        try:
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            auc = roc_auc_score(targets_np, probs, multi_class='ovr', average='macro')
        except ValueError:
            auc = 0.0

    result = {
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'auc': float(auc),
    }

    # Per-class F1 for multi-class monitoring
    if num_classes > 2:
        f1_per = f1_score(targets_np, preds, average=None, zero_division=0)
        for c in range(min(len(f1_per), num_classes)):
            name = DEFECT_TYPE_NAMES[c] if c < len(DEFECT_TYPE_NAMES) else 'class_%d' % c
            result['f1_%s' % name] = float(f1_per[c])

    return result


# =========================================================================
# Training loop
# =========================================================================
def train_epoch(model, loader, optimizer, criterion, device, num_classes=2):
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
    metrics = compute_metrics(logits_cat, targets_cat, num_classes=num_classes)
    metrics['loss'] = avg_loss
    return metrics


def train_epoch_subgraph(model, train_data, sampler, optimizer, criterion,
                         device, num_classes=2):
    """Training epoch with defect-centric sub-graph sampling."""
    model.train()
    total_loss = 0.0
    total_nodes = 0
    all_logits, all_targets = [], []

    for graph in train_data:
        subgraphs = sampler.sample(graph)
        for sg in subgraphs:
            sg = sg.to(device)
            optimizer.zero_grad()
            out = model(sg.x, sg.edge_index, sg.edge_attr, None)
            loss = criterion(out, sg.y)
            loss.backward()
            optimizer.step()

            n = sg.y.shape[0]
            total_loss += loss.item() * n
            total_nodes += n
            all_logits.append(out.detach())
            all_targets.append(sg.y.detach())

    avg_loss = total_loss / max(total_nodes, 1)
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_cat, targets_cat, num_classes=num_classes)
    metrics['loss'] = avg_loss
    return metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, num_classes=2):
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
    metrics = compute_metrics(logits_cat, targets_cat, num_classes=num_classes)
    metrics['loss'] = avg_loss
    return metrics


# =========================================================================
# Single training run
# =========================================================================
def train(args, train_data, val_data, fold=None):
    """Execute a single training run."""
    multi_gpu = getattr(args, 'multi_gpu', False) and dist.is_initialized()
    rank = dist.get_rank() if multi_gpu else 0
    world_size = dist.get_world_size() if multi_gpu else 1
    is_main = (rank == 0)

    if multi_gpu:
        device = torch.device('cuda', rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loaders
    if multi_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                  sampler=train_sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Model — auto-detect num_classes from labels
    sample = train_data[0]
    in_channels = sample.x.shape[1]
    edge_attr_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0

    all_labels = torch.cat([d.y for d in train_data])
    num_classes = int(all_labels.max().item()) + 1
    num_classes = max(num_classes, 2)  # At least binary

    model = build_model(
        args.arch, in_channels, edge_attr_dim,
        hidden_channels=args.hidden, num_layers=args.layers,
        dropout=args.dropout, num_classes=num_classes,
    ).to(device)

    if multi_gpu:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    raw_model = model.module if multi_gpu else model
    param_count = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        gpu_info = "GPU x%d" % world_size if multi_gpu else str(device)
        print("Model: %s | Params: %d | Classes: %d | Device: %s" % (
            args.arch.upper(), param_count, num_classes, gpu_info))

    # Compute class distribution
    n_total = all_labels.numel()
    class_counts = {}
    for c in range(num_classes):
        n_c = (all_labels == c).sum().item()
        class_counts[c] = n_c
        name = DEFECT_TYPE_NAMES[c] if c < len(DEFECT_TYPE_NAMES) else 'class_%d' % c
        if is_main:
            print("  Class %d (%s): %d nodes (%.4f%%)" % (c, name, n_c, 100.0 * n_c / max(n_total, 1)))

    # Optimizer, scheduler, criterion
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    if args.loss == 'weighted_ce':
        # Inverse frequency class weights (generalized for K classes)
        weights = []
        for c in range(num_classes):
            n_c = max(class_counts.get(c, 1), 1)
            weights.append(n_total / (num_classes * n_c))
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        if is_main:
            print("Loss: WeightedCE — weights=%s" % ['%.2f' % w for w in weights])
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        # Focal loss with per-class alpha (inverse frequency)
        alpha_vec = []
        for c in range(num_classes):
            n_c = max(class_counts.get(c, 1), 1)
            alpha_vec.append(n_total / (num_classes * n_c))
        # Normalize to sum to num_classes
        alpha_sum = sum(alpha_vec)
        alpha_vec = [a * num_classes / alpha_sum for a in alpha_vec]
        if is_main:
            print("Loss: FocalLoss — alpha=%s, gamma=%.1f" % (
                ['%.3f' % a for a in alpha_vec], args.focal_gamma))
        criterion = FocalLoss(alpha=alpha_vec, gamma=args.focal_gamma, num_classes=num_classes).to(device)

    # Sub-graph sampler
    use_subgraph = getattr(args, 'sampler', 'full_graph') == 'defect_centric'
    sampler = None
    if use_subgraph:
        sampler = DefectCentricSampler(
            num_hops=args.subgraph_hops,
            healthy_ratio=args.healthy_ratio,
        )
        if is_main:
            print("Sampler: DefectCentricSampler (hops=%d, healthy_ratio=%d)" % (
                args.subgraph_hops, args.healthy_ratio))

    # Logging (main process only)
    fold_str = '_fold%d' % fold if fold is not None else ''
    gpu_suffix = '_ddp%d' % world_size if multi_gpu else ''
    run_name = '%s_%s%s%s' % (args.arch, datetime.now().strftime('%Y%m%d_%H%M%S'), fold_str, gpu_suffix)
    run_dir = os.path.join(args.output_dir, run_name)
    tb_writer = None
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, 'training_log.csv')
        with open(log_path, 'w', newline='') as f:
            writer = csv_module.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_f1', 'train_acc',
                             'val_loss', 'val_f1', 'val_acc', 'val_auc', 'lr'])
        # TensorBoard
        tb_writer = SummaryWriter(log_dir=run_dir)
        tb_writer.add_text('config', json.dumps(vars(args), indent=2))

    # Training
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if multi_gpu:
            train_sampler.set_epoch(epoch)
        if use_subgraph:
            train_m = train_epoch_subgraph(
                model, train_data, sampler, optimizer, criterion,
                device, num_classes=num_classes)
        else:
            train_m = train_epoch(model, train_loader, optimizer, criterion, device, num_classes=num_classes)
        val_m = eval_epoch(model, val_loader, criterion, device, num_classes=num_classes)
        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]['lr']

        if is_main:
            # Log CSV
            with open(log_path, 'a', newline='') as f:
                writer = csv_module.writer(f)
                writer.writerow([epoch, '%.6f' % train_m['loss'], '%.4f' % train_m['f1'],
                                 '%.4f' % train_m['accuracy'],
                                 '%.6f' % val_m['loss'], '%.4f' % val_m['f1'],
                                 '%.4f' % val_m['accuracy'], '%.4f' % val_m['auc'],
                                 '%.2e' % lr])

            # Log TensorBoard
            if tb_writer is not None:
                tb_writer.add_scalars('loss', {'train': train_m['loss'], 'val': val_m['loss']}, epoch)
                tb_writer.add_scalars('f1', {'train': train_m['f1'], 'val': val_m['f1']}, epoch)
                tb_writer.add_scalars('accuracy', {'train': train_m['accuracy'], 'val': val_m['accuracy']}, epoch)
                tb_writer.add_scalar('val/auc', val_m['auc'], epoch)
                tb_writer.add_scalar('val/precision', val_m['precision'], epoch)
                tb_writer.add_scalar('val/recall', val_m['recall'], epoch)
                tb_writer.add_scalar('lr', lr, epoch)
                if num_classes > 2:
                    for c in range(num_classes):
                        name = DEFECT_TYPE_NAMES[c] if c < len(DEFECT_TYPE_NAMES) else 'class_%d' % c
                        key = 'f1_%s' % name
                        if key in val_m:
                            tb_writer.add_scalar('val_f1_class/%s' % name, val_m[key], epoch)

            if epoch % args.log_every == 0 or epoch == 1:
                msg = ("  Epoch %3d/%d | Train F1=%.4f Loss=%.4f | "
                       "Val F1=%.4f AUC=%.4f Loss=%.4f | LR=%.2e | %.1fs" %
                       (epoch, args.epochs, train_m['f1'], train_m['loss'],
                        val_m['f1'], val_m['auc'], val_m['loss'], lr, elapsed))
                # Per-class F1 for multi-class
                if num_classes > 2:
                    parts = []
                    for c in range(num_classes):
                        name = DEFECT_TYPE_NAMES[c] if c < len(DEFECT_TYPE_NAMES) else 'c%d' % c
                        key = 'f1_%s' % name
                        parts.append('%s=%.3f' % (name[:3], val_m.get(key, 0.0)))
                    msg += '\n    Per-class: ' + ' | '.join(parts)
                print(msg)

            # Checkpoint best
            if val_m['f1'] > best_val_f1:
                best_val_f1 = val_m['f1']
                patience_counter = 0
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'val_metrics': val_m,
                    'args': vars(args),
                    'in_channels': in_channels,
                    'edge_attr_dim': edge_attr_dim,
                }
                torch.save(save_dict, os.path.join(run_dir, 'best_model.pt'))
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= args.patience:
                print("  Early stopping at epoch %d (patience=%d)" %
                      (epoch, args.patience))
                break
        else:
            # Non-main ranks: track patience for synchronized early stopping
            if val_m['f1'] > best_val_f1:
                best_val_f1 = val_m['f1']
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                break

    if is_main:
        print("  Best Val F1: %.4f" % best_val_f1)

        # Load best model and print final per-class metrics
        if num_classes > 2:
            ckpt = torch.load(os.path.join(run_dir, 'best_model.pt'), weights_only=False)
            best_m = ckpt.get('val_metrics', {})
            print("  Per-class F1 (best):")
            for c in range(num_classes):
                name = DEFECT_TYPE_NAMES[c] if c < len(DEFECT_TYPE_NAMES) else 'class_%d' % c
                key = 'f1_%s' % name
                print("    %s: %.4f" % (name, best_m.get(key, 0.0)))

        if tb_writer is not None:
            tb_writer.add_hparams(
                {'arch': args.arch, 'lr': args.lr, 'hidden': args.hidden,
                 'layers': args.layers, 'dropout': args.dropout,
                 'loss': args.loss, 'batch_size': args.batch_size},
                {'hparam/best_val_f1': best_val_f1})
            tb_writer.close()

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
    parser.add_argument('--data_dir', type=str, default='data/processed_50mm_100',
                        help='Dir with train.pt, val.pt (from prepare_ml_data.py)')
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
    parser.add_argument('--loss', type=str, default='weighted_ce',
                        choices=['weighted_ce', 'focal'],
                        help='Loss function (default: weighted_ce)')
    parser.add_argument('--focal_alpha', type=float, default=0.0,
                        help='Focal loss alpha (0=auto from class ratio)')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--gamma_search', action='store_true', default=False,
                        help='Grid search gamma in {1.0, 2.0, 3.0, 5.0}')
    # Sub-graph sampling
    parser.add_argument('--sampler', type=str, default='full_graph',
                        choices=['full_graph', 'defect_centric'],
                        help='Sampling strategy (default: full_graph)')
    parser.add_argument('--subgraph_hops', type=int, default=4,
                        help='k-hop expansion for defect_centric sampler')
    parser.add_argument('--healthy_ratio', type=int, default=5,
                        help='Extra healthy nodes per defect node')
    # Cross-validation
    parser.add_argument('--cross_val', type=int, default=0,
                        help='Number of folds for CV (0=disabled)')
    # Multi-GPU
    parser.add_argument('--multi_gpu', action='store_true', default=False,
                        help='Enable DDP multi-GPU (use with torchrun)')
    # Misc
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # DDP init
    if args.multi_gpu:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        if rank == 0:
            print("DDP: %d GPUs initialized" % dist.get_world_size())
    else:
        rank = 0

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    train_path = os.path.join(data_dir, 'train.pt')
    val_path = os.path.join(data_dir, 'val.pt')

    if not os.path.exists(train_path):
        print("Data not found: %s" % train_path)
        return

    # Load data
    if rank == 0:
        print("Loading data from %s..." % args.data_dir)
    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)
    if rank == 0:
        print("Train: %d samples | Val: %d samples" % (len(train_data), len(val_data)))

    if args.gamma_search:
        # Grid search over gamma values
        args.loss = 'focal'
        gammas = [1.0, 2.0, 3.0, 5.0]
        print("\n===== Gamma Grid Search =====")
        print("Gammas: %s | Epochs: %d" % (gammas, args.epochs))
        results = []
        for gamma in gammas:
            print("\n--- gamma=%.1f ---" % gamma)
            args.focal_gamma = gamma
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            _, best_f1 = train(args, train_data, val_data)
            results.append((gamma, best_f1))
        print("\n===== Gamma Search Results =====")
        for gamma, f1 in sorted(results, key=lambda x: -x[1]):
            print("  gamma=%.1f → Val F1=%.4f" % (gamma, f1))
        best_gamma, best_f1 = max(results, key=lambda x: x[1])
        print("Best: gamma=%.1f (F1=%.4f)" % (best_gamma, best_f1))
    elif args.cross_val > 1:
        # Merge train + val for CV
        all_data = train_data + val_data
        print("Cross-validation mode: %d folds, %d total samples" %
              (args.cross_val, len(all_data)))
        cross_validate(args, all_data)
    else:
        train(args, train_data, val_data)

    if rank == 0:
        print("\nDone.")

    if args.multi_gpu:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
