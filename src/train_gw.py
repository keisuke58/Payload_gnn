# -*- coding: utf-8 -*-
"""
GW フェアリング グラフ分類学習 — train_gw.py

センサ時刻歴から構築したグラフ（ノード=センサ）の二値分類:
- 0: Healthy（健全）
- 1: Defect（欠陥あり）

クラスバランス: WeightedRandomSampler または Focal Loss で対応

Usage:
    python src/train_gw.py --data_dir data/processed_gw_100 --arch gat --epochs 200
    python src/train_gw.py --data_dir data/processed_gw_100 --loss focal --focal_gamma 2.0
"""

import os
import sys
import json
import time
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
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from models import build_model


# =========================================================================
# Graph-level model wrapper
# =========================================================================
class GraphLevelWrapper(nn.Module):
    """GNN encoder + global_mean_pool + MLP for graph-level classification."""

    def __init__(self, encoder, hidden_channels, num_classes=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.pool = global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder.encode(x, edge_index, edge_attr)
        g = self.pool(h, batch)
        return self.classifier(g)


def build_graph_level_model(arch, in_channels, edge_attr_dim=0, hidden=128, num_classes=2):
    """Build GNN encoder and wrap for graph-level output."""
    base = build_model(arch, in_channels, edge_attr_dim=edge_attr_dim,
                      hidden_channels=hidden, num_classes=num_classes, num_layers=3)
    return GraphLevelWrapper(base, hidden, num_classes)


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

    acc = (preds == targets_np).mean()
    f1 = f1_score(targets_np, preds, zero_division=0)
    prec = precision_score(targets_np, preds, zero_division=0)
    rec = recall_score(targets_np, preds, zero_division=0)
    try:
        auc = roc_auc_score(targets_np, probs)
    except ValueError:
        auc = 0.0

    return {"accuracy": float(acc), "f1": float(f1), "precision": float(prec),
            "recall": float(rec), "auc": float(auc)}


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


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='GW graph-level classification')
    parser.add_argument('--data_dir', type=str, default='data/processed_gw_100')
    parser.add_argument('--arch', type=str, default='gat', choices=['gcn', 'gat', 'gin', 'sage'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss', type=str, default='focal', choices=['ce', 'focal'])
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--class_weight', action='store_true',
                        help='Use class weights for CE (inverse freq)')
    parser.add_argument('--weighted_sampler', action='store_true',
                        help='Use WeightedRandomSampler for class balance')
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

    # Set graph_y from y (GW data has y as graph-level already)
    for d in train_data + val_data:
        if not hasattr(d, 'graph_y'):
            d.graph_y = d.y.squeeze(0) if d.y.dim() > 0 else d.y

    n_train = len(train_data)
    n_val = len(val_data)
    n_healthy = sum(1 for d in train_data if d.graph_y.item() == 0)
    n_defect = n_train - n_healthy
    print("Train: %d (%d healthy, %d defect) | Val: %d" % (n_train, n_healthy, n_defect, n_val))

    in_channels = train_data[0].x.shape[1]
    ea = getattr(train_data[0], 'edge_attr', None)
    edge_attr_dim = ea.shape[1] if ea is not None else 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_graph_level_model(args.arch, in_channels, edge_attr_dim, num_classes=2)
    model = model.to(device)

    if args.loss == 'focal':
        criterion = FocalLoss(gamma=args.focal_gamma)
    else:
        if args.class_weight and n_healthy > 0 and n_defect > 0:
            w_healthy = 1.0 / n_healthy
            w_defect = 1.0 / n_defect
            weights = torch.tensor([w_healthy, w_defect], dtype=torch.float)
            weights = weights / weights.sum()
            criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.weighted_sampler:
        labels = [d.graph_y.item() for d in train_data]
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    run_dir = args.run_dir or os.path.join(PROJECT_ROOT, 'runs/gw_%s_%s' % (
        args.arch, datetime.now().strftime('%Y%m%d_%H%M%S')))
    os.makedirs(run_dir, exist_ok=True)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        if val_m['f1'] > best_val:
            best_val = val_m['f1']
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pt'))

        if epoch % 20 == 0 or epoch == 1:
            print("Epoch %3d | train loss %.4f f1 %.3f | val loss %.4f f1 %.3f auc %.3f" % (
                epoch, train_m['loss'], train_m['f1'], val_m['loss'], val_m['f1'], val_m['auc']))

    print("\nDone. Best val F1: %.3f | run_dir: %s" % (best_val, run_dir))


if __name__ == '__main__':
    main()
