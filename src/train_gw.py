# -*- coding: utf-8 -*-
"""
GW フェアリング グラフ分類学習 — train_gw.py

センサ時刻歴から構築したグラフ（ノード=センサ）の二値分類:
- 0: Healthy（健全）
- 1: Defect（欠陥あり）

クラスバランス: WeightedRandomSampler / Focal Loss / Oversampling + Augmentation

Usage:
    python src/train_gw.py --data_dir data/processed_gw_comprehensive --arch gat --epochs 200
    python src/train_gw.py --data_dir data/processed_gw_comprehensive --loss focal --oversample 10
"""

import os
import sys
import json
import time
import copy
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data

from models import build_model


# =========================================================================
# Graph-level model wrapper
# =========================================================================
class GraphLevelWrapper(nn.Module):
    """GNN encoder + pooling + MLP for graph-level classification."""

    def __init__(self, encoder, hidden_channels, num_classes=2, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        # Use both mean and max pooling (2x hidden dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder.encode(x, edge_index, edge_attr)
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        g = torch.cat([g_mean, g_max], dim=1)
        return self.classifier(g)


def build_graph_level_model(arch, in_channels, edge_attr_dim=0,
                             hidden=128, num_classes=2, num_layers=3):
    base = build_model(arch, in_channels, edge_attr_dim=edge_attr_dim,
                      hidden_channels=hidden, num_classes=num_classes,
                      num_layers=num_layers)
    return GraphLevelWrapper(base, hidden, num_classes)


# =========================================================================
# Data augmentation
# =========================================================================
def augment_graph(data, noise_std=0.01, drop_node_prob=0.1):
    """Augment a single graph by adding noise and optionally dropping nodes."""
    d = data.clone()
    # Feature noise
    d.x = d.x + torch.randn_like(d.x) * noise_std * d.x.abs().mean()
    # Random node masking (zero out some features)
    if drop_node_prob > 0 and d.num_nodes > 2:
        mask = torch.rand(d.num_nodes) > drop_node_prob
        if mask.sum() >= 2:
            d.x = d.x * mask.float().unsqueeze(1)
    return d


def oversample_minority(train_data, factor=10, noise_std=0.02):
    """Oversample defect class with augmentation."""
    defect = [d for d in train_data if d.graph_y.item() == 1]
    healthy = [d for d in train_data if d.graph_y.item() == 0]
    if not defect:
        return train_data

    augmented = []
    for _ in range(factor):
        for d in defect:
            aug = augment_graph(d, noise_std=noise_std)
            aug.graph_y = d.graph_y.clone()
            if hasattr(d, 'name'):
                aug.name = d.name + '_aug'
            augmented.append(aug)

    result = healthy + defect + augmented
    n_h = sum(1 for d in result if d.graph_y.item() == 0)
    n_d = sum(1 for d in result if d.graph_y.item() == 1)
    print("  After oversampling: %d healthy, %d defect (factor=%d)" % (n_h, n_d, factor))
    return result


# =========================================================================
# Focal Loss
# =========================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

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
def compute_metrics(logits, targets):
    preds = logits.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    acc = float((preds == targets_np).mean())

    tp = int(((preds == 1) & (targets_np == 1)).sum())
    fp = int(((preds == 1) & (targets_np == 0)).sum())
    fn = int(((preds == 0) & (targets_np == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # AUC
    pos = probs[targets_np == 1]
    neg = probs[targets_np == 0]
    if len(pos) > 0 and len(neg) > 0:
        auc = 0.0
        for p in pos:
            auc += (neg < p).sum() + 0.5 * (neg == p).sum()
        auc = float(auc / (len(pos) * len(neg)))
    else:
        auc = 0.0

    return {"accuracy": acc, "f1": float(f1), "precision": float(prec),
            "recall": float(rec), "auc": auc}


# =========================================================================
# Training / Eval
# =========================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        ea = getattr(batch, 'edge_attr', None)
        out = model(batch.x, batch.edge_index, ea, batch.batch)
        targets = getattr(batch, 'graph_y', batch.y)
        loss = criterion(out, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(out.detach())
        all_targets.append(targets.detach())

    avg_loss = total_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_cat, targets_cat)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        ea = getattr(batch, 'edge_attr', None)
        out = model(batch.x, batch.edge_index, ea, batch.batch)
        targets = getattr(batch, 'graph_y', batch.y)
        loss = criterion(out, targets)
        total_loss += loss.item() * batch.num_graphs
        all_logits.append(out)
        all_targets.append(targets)

    avg_loss = total_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_cat, targets_cat)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def test_evaluation(model, test_data, device):
    """Per-sample test evaluation with detailed output."""
    model.eval()
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    results = []
    for batch in test_loader:
        batch = batch.to(device)
        ea = getattr(batch, 'edge_attr', None)
        out = model(batch.x, batch.edge_index, ea, batch.batch)
        prob = F.softmax(out, dim=1)[0, 1].item()
        true = batch.graph_y.item()
        pred = 1 if prob > 0.5 else 0
        name = batch.name[0] if hasattr(batch, 'name') else 'unknown'
        results.append({
            'name': name, 'true': true, 'pred': pred,
            'prob_defect': round(prob, 4),
            'correct': pred == true,
        })
    return results


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='GW graph-level classification')
    parser.add_argument('--data_dir', type=str, default='data/processed_gw_comprehensive')
    parser.add_argument('--arch', type=str, default='gat',
                        choices=['gcn', 'gat', 'gin', 'sage'])
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='focal', choices=['ce', 'focal'])
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--class_weight', action='store_true',
                        help='Use class weights for CE/Focal')
    parser.add_argument('--oversample', type=int, default=0,
                        help='Oversample defect by this factor (0=off)')
    parser.add_argument('--aug_noise', type=float, default=0.02,
                        help='Augmentation noise std for oversampled data')
    parser.add_argument('--weighted_sampler', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_dir', type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    if not os.path.exists(data_dir):
        print("ERROR: data_dir not found: %s" % data_dir)
        sys.exit(1)

    train_data = torch.load(os.path.join(data_dir, 'train.pt'), weights_only=False)
    val_data = torch.load(os.path.join(data_dir, 'val.pt'), weights_only=False)
    test_path = os.path.join(data_dir, 'test.pt')
    test_data = torch.load(test_path, weights_only=False) if os.path.exists(test_path) else []

    # Set graph_y
    for d in train_data + val_data + test_data:
        if not hasattr(d, 'graph_y'):
            d.graph_y = d.y.squeeze(0) if d.y.dim() > 0 else d.y

    n_healthy = sum(1 for d in train_data if d.graph_y.item() == 0)
    n_defect = len(train_data) - n_healthy
    print("Train: %d (%d healthy, %d defect) | Val: %d | Test: %d" % (
        len(train_data), n_healthy, n_defect, len(val_data), len(test_data)))

    # Oversample minority class
    if args.oversample > 0:
        train_data = oversample_minority(train_data, factor=args.oversample,
                                          noise_std=args.aug_noise)
        n_healthy = sum(1 for d in train_data if d.graph_y.item() == 0)
        n_defect = len(train_data) - n_healthy

    in_channels = train_data[0].x.shape[1]
    ea = getattr(train_data[0], 'edge_attr', None)
    edge_attr_dim = ea.shape[1] if ea is not None else 0
    print("Features: %d | Device: %s" % (in_channels, 'cuda' if torch.cuda.is_available() else 'cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_graph_level_model(args.arch, in_channels, edge_attr_dim,
                                     hidden=args.hidden, num_layers=args.num_layers)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model: %s, %d params" % (args.arch, n_params))

    # Loss
    if args.loss == 'focal':
        alpha = None
        if args.class_weight and n_healthy > 0 and n_defect > 0:
            alpha = torch.tensor([1.0 / n_healthy, 1.0 / n_defect])
            alpha = alpha / alpha.sum()
            alpha = alpha.to(device)
        criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma)
    else:
        if args.class_weight and n_healthy > 0 and n_defect > 0:
            w = torch.tensor([1.0 / n_healthy, 1.0 / n_defect], dtype=torch.float)
            w = w / w.sum()
            criterion = nn.CrossEntropyLoss(weight=w.to(device))
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.weighted_sampler:
        labels = [d.graph_y.item() for d in train_data]
        class_counts = np.bincount(labels)
        sample_weights = 1.0 / class_counts[labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights))
        train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    run_dir = args.run_dir or os.path.join(PROJECT_ROOT, 'runs/gw_%s_%s' % (
        args.arch, datetime.now().strftime('%Y%m%d_%H%M%S')))
    os.makedirs(run_dir, exist_ok=True)

    t0 = time.time()
    best_val_f1 = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pt'))

        history.append({'epoch': epoch, 'train': train_m, 'val': val_m})

        if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = time.time() - t0
            print("Ep %3d | trn loss %.4f f1 %.3f | val loss %.4f f1 %.3f rec %.3f auc %.3f | %.0fs" % (
                epoch, train_m['loss'], train_m['f1'],
                val_m['loss'], val_m['f1'], val_m['recall'], val_m['auc'],
                elapsed))

    print("\nBest val F1: %.3f @ epoch %d" % (best_val_f1, best_epoch))

    # Load best model for test evaluation
    if test_data:
        best_path = os.path.join(run_dir, 'best_model.pt')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, weights_only=True, map_location=device))

        test_results = test_evaluation(model, test_data, device)
        print("\n=== TEST RESULTS ===")
        for r in test_results:
            status = 'OK' if r['correct'] else 'WRONG'
            print("  %-50s true=%d pred=%d prob=%.4f [%s]" % (
                r['name'], r['true'], r['pred'], r['prob_defect'], status))

        tp = sum(1 for r in test_results if r['pred'] == 1 and r['true'] == 1)
        fp = sum(1 for r in test_results if r['pred'] == 1 and r['true'] == 0)
        fn = sum(1 for r in test_results if r['pred'] == 0 and r['true'] == 1)
        tn = sum(1 for r in test_results if r['pred'] == 0 and r['true'] == 0)
        test_acc = (tp + tn) / max(len(test_results), 1)
        print("  TP=%d FP=%d FN=%d TN=%d | Accuracy=%.3f" % (tp, fp, fn, tn, test_acc))

    # Save results
    results = {
        'args': vars(args),
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch,
        'n_params': n_params,
        'total_time_s': time.time() - t0,
    }
    if test_data:
        results['test'] = test_results
        results['test_accuracy'] = test_acc
    with open(os.path.join(run_dir, 'results.json'), 'w') as fp:
        json.dump(results, fp, indent=2)

    print("\nSaved: %s" % run_dir)


if __name__ == '__main__':
    main()
