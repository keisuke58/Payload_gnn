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
# Boundary-aware weighting
# =========================================================================
def find_boundary_nodes(edge_index, y):
    """Find healthy nodes adjacent to defect nodes (boundary transition zone)."""
    src, dst = edge_index[0], edge_index[1]
    defect_mask = y > 0
    # Healthy src connected to defect dst
    boundary_mask = (~defect_mask[src]) & defect_mask[dst]
    return src[boundary_mask].unique()


def build_node_weights(data, boundary_weight=1.0, defect_weight=1.0):
    """Build per-node weight tensor.

    Args:
        data: graph data with .y and .edge_index
        boundary_weight: weight for healthy nodes adjacent to defect nodes
        defect_weight: weight for defect nodes themselves (>1 = penalize misses more)
    """
    w = torch.ones(data.y.shape[0], dtype=torch.float, device=data.y.device)
    if defect_weight > 1.0:
        defect_mask = data.y > 0
        w[defect_mask] = defect_weight
    if boundary_weight > 1.0:
        boundary_idx = find_boundary_nodes(data.edge_index, data.y)
        w[boundary_idx] = boundary_weight
    return w


# =========================================================================
# Graph augmentation (DropEdge / Feature Noise)
# =========================================================================
def drop_edge(edge_index, edge_attr, drop_rate):
    """Randomly drop edges during training."""
    if drop_rate <= 0:
        return edge_index, edge_attr
    num_edges = edge_index.shape[1]
    mask = torch.rand(num_edges, device=edge_index.device) >= drop_rate
    new_edge_index = edge_index[:, mask]
    new_edge_attr = edge_attr[mask] if edge_attr is not None else None
    return new_edge_index, new_edge_attr


def add_feature_noise(x, noise_std):
    """Add Gaussian noise to node features during training."""
    if noise_std <= 0:
        return x
    return x + torch.randn_like(x) * noise_std


def mask_features(x, mask_rate):
    """Randomly zero out feature dimensions during training.

    Args:
        x: (N, D) node features
        mask_rate: probability of masking each element
    """
    if mask_rate <= 0:
        return x
    mask = torch.rand_like(x) >= mask_rate
    return x * mask.float()


def node_drop(data, drop_rate):
    """Randomly drop nodes and their edges during training.

    Useful for large graphs to create varied subgraph views.
    Preserves at least 50% of nodes to maintain graph structure.
    """
    if drop_rate <= 0:
        return data
    n_nodes = data.x.size(0)
    keep_mask = torch.rand(n_nodes, device=data.x.device) >= drop_rate
    # Ensure at least 50% kept
    if keep_mask.sum() < n_nodes // 2:
        return data

    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    # Remap node indices
    remap = torch.full((n_nodes,), -1, dtype=torch.long, device=data.x.device)
    remap[keep_idx] = torch.arange(keep_idx.size(0), device=data.x.device)

    # Filter edges
    src, dst = data.edge_index
    edge_mask = keep_mask[src] & keep_mask[dst]
    new_edge_index = remap[data.edge_index[:, edge_mask]]

    data = data.clone()
    data.x = data.x[keep_idx]
    data.y = data.y[keep_idx]
    data.edge_index = new_edge_index
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[edge_mask]
    if hasattr(data, 'pos') and data.pos is not None:
        data.pos = data.pos[keep_idx]
    return data


def circumferential_flip(data):
    """Apply circumferential flip exploiting fairing axial symmetry.

    Negates x,z position/normal/fiber components and circumferential angle.
    Feature dims (from build_graph.py):
      0,2: x,z | 3,5: nx,nz | 24,26: fiber_x,z | 31: θ
    """
    x = data.x.clone()
    for dim in (0, 2, 3, 5, 24, 26, 31):
        x[:, dim] = -x[:, dim]
    data = data.clone()
    data.x = x
    # Also flip edge relative position (dx, dz) — dims 0, 2 of edge_attr
    if data.edge_attr is not None:
        ea = data.edge_attr.clone()
        ea[:, 0] = -ea[:, 0]
        if ea.shape[1] > 2:
            ea[:, 2] = -ea[:, 2]
        data.edge_attr = ea
    return data


# =========================================================================
# Physics-informed loss functions
# =========================================================================
def spatial_smoothness_loss(logits, edge_index):
    """Laplacian smoothness on P(defect) between adjacent nodes.

    Penalizes difference in defect probability between neighbors,
    reducing isolated false positive predictions.
    """
    probs = F.softmax(logits, dim=1)[:, 1]
    src, dst = edge_index[0], edge_index[1]
    diff = probs[src] - probs[dst]
    return (diff ** 2).mean()


def stress_gradient_loss(logits, x, edge_index, stress_dim=18):
    """Penalize defect predictions at nodes with low stress gradient.

    Physical rationale: debonding defects cause local stress concentrations.
    If stress is uniform around a node, it is unlikely a defect boundary.
    """
    probs = F.softmax(logits, dim=1)[:, 1]
    stress = x[:, stress_dim]

    src, dst = edge_index[0], edge_index[1]
    stress_diff = torch.abs(stress[src] - stress[dst])

    # Max stress gradient per node
    max_grad = torch.zeros(x.shape[0], device=x.device)
    max_grad.scatter_reduce_(0, src, stress_diff, reduce='amax', include_self=False)

    # Normalize to [0, 1]
    grad_max = max_grad.max()
    max_grad_norm = max_grad / (grad_max + 1e-8) if grad_max > 0 else max_grad

    # Penalize: high P(defect) at low-gradient nodes
    penalty = probs * (1.0 - max_grad_norm)
    return penalty.mean()


def connected_component_penalty(logits, edge_index):
    """Penalize isolated defect predictions (neighbors not predicted as defect).

    Differentiable approximation: for each node, compute average P(defect)
    of its neighbors. Penalize when a node has high P(defect) but neighbors
    do not.
    """
    probs = F.softmax(logits, dim=1)[:, 1]
    src, dst = edge_index[0], edge_index[1]

    neighbor_prob_sum = torch.zeros(logits.shape[0], device=logits.device)
    degree = torch.zeros(logits.shape[0], device=logits.device)
    neighbor_prob_sum.scatter_add_(0, src, probs[dst])
    degree.scatter_add_(0, src, torch.ones_like(probs[dst]))

    avg_neighbor_prob = neighbor_prob_sum / (degree + 1e-8)

    # Node has high P(defect) but neighbors have low P(defect)
    isolation_score = probs * (1.0 - avg_neighbor_prob)
    return isolation_score.mean()


# =========================================================================
# Metrics
# =========================================================================
DEFECT_TYPE_NAMES = [
    'healthy', 'debonding', 'fod', 'impact',
    'delamination', 'inner_debond', 'thermal_progression', 'acoustic_fatigue',
]


def optimize_binary_threshold(probs_defect, targets_binary):
    """Find optimal threshold for binary (healthy vs defect) F1.

    Args:
        probs_defect: (N,) P(defect) scores
        targets_binary: (N,) 0=healthy, 1=defect

    Returns:
        best_f1, best_threshold, best_precision, best_recall
    """
    best_f1, best_t, best_p, best_r = 0.0, 0.5, 0.0, 0.0
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (probs_defect > t).astype(int)
        f1 = f1_score(targets_binary, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_p = precision_score(targets_binary, preds, zero_division=0)
            best_r = recall_score(targets_binary, preds, zero_division=0)
    return float(best_f1), float(best_t), float(best_p), float(best_r)


def compute_metrics(logits, targets, num_classes=2):
    """Compute classification metrics (binary or multi-class).

    Always includes threshold-optimized binary F1 (healthy vs any defect).
    """
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

    # Threshold-optimized binary F1 (healthy vs any defect)
    probs_all = F.softmax(logits, dim=1).detach().cpu().numpy()
    probs_defect = 1.0 - probs_all[:, 0]  # P(any defect)
    targets_bin = (targets_np > 0).astype(int)
    if targets_bin.sum() > 0:
        opt_f1, opt_t, opt_p, opt_r = optimize_binary_threshold(probs_defect, targets_bin)
        result['opt_f1'] = opt_f1
        result['opt_threshold'] = opt_t
        result['opt_precision'] = opt_p
        result['opt_recall'] = opt_r

    return result


# =========================================================================
# Training loop
# =========================================================================
def _compute_weighted_loss(criterion, out, y, node_weights):
    """Compute per-node loss weighted by node_weights."""
    if isinstance(criterion, FocalLoss):
        ce = F.cross_entropy(out, y, reduction='none')
        pt = torch.exp(-ce)
        per_node_loss = (1 - pt) ** criterion.gamma * ce
        if criterion.alpha is not None:
            alpha_t = criterion.alpha[y]
            per_node_loss = alpha_t * per_node_loss
    else:
        weight = criterion.weight if hasattr(criterion, 'weight') and criterion.weight is not None else None
        per_node_loss = F.cross_entropy(out, y, weight=weight, reduction='none')
    return (per_node_loss * node_weights).mean()


def train_epoch(model, loader, optimizer, criterion, device, num_classes=2,
                boundary_weight=1.0, defect_weight=1.0,
                drop_edge_rate=0.0, feature_noise_std=0.0,
                feature_mask_rate=0.0, flip_prob=0.0,
                node_drop_rate=0.0,
                lambda_smooth=0.0, lambda_stress=0.0, lambda_connected=0.0,
                stress_dim=18):
    model.train()
    total_loss = 0.0
    total_physics = {'smooth': 0.0, 'stress': 0.0, 'connected': 0.0}
    all_logits, all_targets = [], []
    use_node_weights = (boundary_weight > 1.0 or defect_weight > 1.0)
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # Augmentation: node drop
        if node_drop_rate > 0:
            batch = node_drop(batch, node_drop_rate)
        # Augmentation: circumferential flip
        if flip_prob > 0 and torch.rand(1).item() < flip_prob:
            batch = circumferential_flip(batch)
        # Augmentation: feature noise + masking
        x = batch.x
        if feature_noise_std > 0:
            x = add_feature_noise(x, feature_noise_std)
        if feature_mask_rate > 0:
            x = mask_features(x, feature_mask_rate)
        ei, ea = drop_edge(batch.edge_index, batch.edge_attr, drop_edge_rate) if drop_edge_rate > 0 else (batch.edge_index, batch.edge_attr)
        out = model(x, ei, ea, batch.batch)
        if use_node_weights:
            node_w = build_node_weights(batch, boundary_weight, defect_weight)
            loss = _compute_weighted_loss(criterion, out, batch.y, node_w)
        else:
            loss = criterion(out, batch.y)
        # Physics-informed losses
        if lambda_smooth > 0:
            l_s = spatial_smoothness_loss(out, ei)
            loss = loss + lambda_smooth * l_s
            total_physics['smooth'] += l_s.item()
        if lambda_stress > 0:
            l_st = stress_gradient_loss(out, x, ei, stress_dim=stress_dim)
            loss = loss + lambda_stress * l_st
            total_physics['stress'] += l_st.item()
        if lambda_connected > 0:
            l_c = connected_component_penalty(out, ei)
            loss = loss + lambda_connected * l_c
            total_physics['connected'] += l_c.item()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(out.detach())
        all_targets.append(batch.y.detach())
        n_batches += 1

    avg_loss = total_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_cat, targets_cat, num_classes=num_classes)
    metrics['loss'] = avg_loss
    # Average physics losses for logging
    for k, v in total_physics.items():
        if v > 0:
            metrics['physics_%s' % k] = v / n_batches
    return metrics


def train_epoch_subgraph(model, train_data, sampler, optimizer, criterion,
                         device, num_classes=2, drop_edge_rate=0.0,
                         feature_noise_std=0.0,
                         boundary_weight=1.0, defect_weight=1.0,
                         feature_mask_rate=0.0,
                         lambda_smooth=0.0, lambda_stress=0.0,
                         lambda_connected=0.0, stress_dim=18):
    """Training epoch with defect-centric sub-graph sampling."""
    model.train()
    total_loss = 0.0
    total_nodes = 0
    total_physics = {'smooth': 0.0, 'stress': 0.0, 'connected': 0.0}
    all_logits, all_targets = [], []
    use_node_weights = (boundary_weight > 1.0 or defect_weight > 1.0)
    n_batches = 0

    for graph in train_data:
        subgraphs = sampler.sample(graph)
        for sg in subgraphs:
            sg = sg.to(device)
            optimizer.zero_grad()
            x = sg.x
            if feature_noise_std > 0:
                x = add_feature_noise(x, feature_noise_std)
            if feature_mask_rate > 0:
                x = mask_features(x, feature_mask_rate)
            ei, ea = drop_edge(sg.edge_index, sg.edge_attr, drop_edge_rate) if drop_edge_rate > 0 else (sg.edge_index, sg.edge_attr)
            out = model(x, ei, ea, None)
            if use_node_weights:
                node_w = build_node_weights(sg, boundary_weight, defect_weight)
                loss = _compute_weighted_loss(criterion, out, sg.y, node_w)
            else:
                loss = criterion(out, sg.y)
            # Physics-informed losses
            if lambda_smooth > 0:
                l_s = spatial_smoothness_loss(out, ei)
                loss = loss + lambda_smooth * l_s
                total_physics['smooth'] += l_s.item()
            if lambda_stress > 0:
                l_st = stress_gradient_loss(out, x, ei, stress_dim=stress_dim)
                loss = loss + lambda_stress * l_st
                total_physics['stress'] += l_st.item()
            if lambda_connected > 0:
                l_c = connected_component_penalty(out, ei)
                loss = loss + lambda_connected * l_c
                total_physics['connected'] += l_c.item()
            loss.backward()
            optimizer.step()

            n = sg.y.shape[0]
            total_loss += loss.item() * n
            total_nodes += n
            all_logits.append(out.detach())
            all_targets.append(sg.y.detach())
            n_batches += 1

    avg_loss = total_loss / max(total_nodes, 1)
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_cat, targets_cat, num_classes=num_classes)
    metrics['loss'] = avg_loss
    for k, v in total_physics.items():
        if v > 0:
            metrics['physics_%s' % k] = v / max(n_batches, 1)
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
        use_residual=getattr(args, 'residual', False),
    ).to(device)

    # Transfer learning: load pretrained weights
    pretrained_path = getattr(args, 'pretrained', None)
    if pretrained_path and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        # Handle num_classes mismatch: skip head layers
        model_state = model.state_dict()
        loaded = {}
        skipped = []
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                loaded[k] = v
            else:
                skipped.append(k)
        model_state.update(loaded)
        model.load_state_dict(model_state)
        if is_main:
            print("Pretrained: loaded %d/%d params from %s" % (
                len(loaded), len(state), pretrained_path))
            if skipped:
                print("  Skipped (shape mismatch): %s" % skipped)

    # Freeze early layers for fine-tuning
    freeze_n = getattr(args, 'freeze_layers', 0)
    if freeze_n > 0:
        frozen = 0
        for name, param in model.named_parameters():
            if 'convs.' in name:
                layer_idx = int(name.split('convs.')[1].split('.')[0])
                if layer_idx < freeze_n:
                    param.requires_grad = False
                    frozen += 1
            elif 'norms.' in name:
                layer_idx = int(name.split('norms.')[1].split('.')[0])
                if layer_idx < freeze_n:
                    param.requires_grad = False
                    frozen += 1
        if is_main:
            print("Frozen: %d params (first %d conv layers)" % (frozen, freeze_n))

    if multi_gpu:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    raw_model = model.module if multi_gpu else model
    param_count = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        gpu_info = "GPU x%d" % world_size if multi_gpu else str(device)
        res_str = " | Residual: ON" if getattr(args, 'residual', False) else ""
        print("Model: %s | Params: %d | Classes: %d | Device: %s%s" % (
            args.arch.upper(), param_count, num_classes, gpu_info, res_str))

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

    # Augmentation / weighting info
    if is_main:
        aug_parts = []
        if getattr(args, 'drop_edge', 0) > 0:
            aug_parts.append("DropEdge=%.2f" % args.drop_edge)
        if getattr(args, 'feature_noise', 0) > 0:
            aug_parts.append("FeatureNoise=%.3f" % args.feature_noise)
        if getattr(args, 'feature_mask', 0) > 0:
            aug_parts.append("FeatureMask=%.2f" % args.feature_mask)
        if getattr(args, 'augment_flip', 0) > 0:
            aug_parts.append("CircumFlip=%.2f" % args.augment_flip)
        if getattr(args, 'node_drop', 0) > 0:
            aug_parts.append("NodeDrop=%.2f" % args.node_drop)
        if aug_parts:
            print("Augmentation: %s" % ', '.join(aug_parts))
        weight_parts = []
        if getattr(args, 'boundary_weight', 1.0) > 1.0:
            weight_parts.append("boundary=%.1f" % args.boundary_weight)
        if getattr(args, 'defect_weight', 1.0) > 1.0:
            weight_parts.append("defect=%.1f" % args.defect_weight)
        if weight_parts:
            print("Node weights: %s" % ', '.join(weight_parts))
        # Physics-informed loss info
        physics_parts = []
        if getattr(args, 'physics_lambda_smooth', 0) > 0:
            physics_parts.append("smooth=%.3f" % args.physics_lambda_smooth)
        if getattr(args, 'physics_lambda_stress', 0) > 0:
            physics_parts.append("stress=%.3f" % args.physics_lambda_stress)
        if getattr(args, 'physics_lambda_connected', 0) > 0:
            physics_parts.append("connected=%.3f" % args.physics_lambda_connected)
        if physics_parts:
            print("Physics loss: %s (stress_dim=%d)" % (
                ', '.join(physics_parts), getattr(args, 'stress_dim', 18)))

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
                             'val_loss', 'val_f1', 'val_acc', 'val_auc', 'lr',
                             'val_opt_f1', 'val_opt_threshold'])
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
        de_rate = getattr(args, 'drop_edge', 0.0)
        fn_std = getattr(args, 'feature_noise', 0.0)
        fm_rate = getattr(args, 'feature_mask', 0.0)
        flip_p = getattr(args, 'augment_flip', 0.0)
        nd_rate = getattr(args, 'node_drop', 0.0)
        bw = getattr(args, 'boundary_weight', 1.0)
        dw = getattr(args, 'defect_weight', 1.0)
        ls = getattr(args, 'physics_lambda_smooth', 0.0)
        lst = getattr(args, 'physics_lambda_stress', 0.0)
        lc = getattr(args, 'physics_lambda_connected', 0.0)
        sd = getattr(args, 'stress_dim', 18)
        if use_subgraph:
            train_m = train_epoch_subgraph(
                model, train_data, sampler, optimizer, criterion,
                device, num_classes=num_classes,
                drop_edge_rate=de_rate, feature_noise_std=fn_std,
                boundary_weight=bw, defect_weight=dw,
                feature_mask_rate=fm_rate,
                lambda_smooth=ls, lambda_stress=lst,
                lambda_connected=lc, stress_dim=sd)
        else:
            train_m = train_epoch(model, train_loader, optimizer, criterion, device,
                                   num_classes=num_classes,
                                   boundary_weight=bw, defect_weight=dw,
                                   drop_edge_rate=de_rate, feature_noise_std=fn_std,
                                   feature_mask_rate=fm_rate, flip_prob=flip_p,
                                   node_drop_rate=nd_rate,
                                   lambda_smooth=ls, lambda_stress=lst,
                                   lambda_connected=lc, stress_dim=sd)
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
                                 '%.2e' % lr,
                                 '%.4f' % val_m.get('opt_f1', 0.0),
                                 '%.3f' % val_m.get('opt_threshold', 0.5)])

            # Log TensorBoard
            if tb_writer is not None:
                tb_writer.add_scalars('loss', {'train': train_m['loss'], 'val': val_m['loss']}, epoch)
                tb_writer.add_scalars('f1', {'train': train_m['f1'], 'val': val_m['f1']}, epoch)
                tb_writer.add_scalars('accuracy', {'train': train_m['accuracy'], 'val': val_m['accuracy']}, epoch)
                tb_writer.add_scalar('val/auc', val_m['auc'], epoch)
                tb_writer.add_scalar('val/precision', val_m['precision'], epoch)
                tb_writer.add_scalar('val/recall', val_m['recall'], epoch)
                tb_writer.add_scalar('lr', lr, epoch)
                if 'opt_f1' in val_m:
                    tb_writer.add_scalar('val/opt_f1', val_m['opt_f1'], epoch)
                    tb_writer.add_scalar('val/opt_threshold', val_m['opt_threshold'], epoch)
                # Physics-informed loss components
                for pk in ('physics_smooth', 'physics_stress', 'physics_connected'):
                    if pk in train_m:
                        tb_writer.add_scalar('physics/%s' % pk, train_m[pk], epoch)
                if num_classes > 2:
                    for c in range(num_classes):
                        name = DEFECT_TYPE_NAMES[c] if c < len(DEFECT_TYPE_NAMES) else 'class_%d' % c
                        key = 'f1_%s' % name
                        if key in val_m:
                            tb_writer.add_scalar('val_f1_class/%s' % name, val_m[key], epoch)

            if epoch % args.log_every == 0 or epoch == 1:
                opt_str = ""
                if 'opt_f1' in val_m:
                    opt_str = " | OptF1=%.4f(t=%.2f)" % (val_m['opt_f1'], val_m['opt_threshold'])
                msg = ("  Epoch %3d/%d | Train F1=%.4f Loss=%.4f | "
                       "Val F1=%.4f AUC=%.4f%s Loss=%.4f | LR=%.2e | %.1fs" %
                       (epoch, args.epochs, train_m['f1'], train_m['loss'],
                        val_m['f1'], val_m['auc'], opt_str, val_m['loss'], lr, elapsed))
                # Per-class F1 for multi-class
                if num_classes > 2:
                    parts = []
                    for c in range(num_classes):
                        name = DEFECT_TYPE_NAMES[c] if c < len(DEFECT_TYPE_NAMES) else 'c%d' % c
                        key = 'f1_%s' % name
                        parts.append('%s=%.3f' % (name[:3], val_m.get(key, 0.0)))
                    msg += '\n    Per-class: ' + ' | '.join(parts)
                print(msg)

            # Checkpoint best (use opt_f1 if available, else f1)
            current_f1 = val_m.get('opt_f1', val_m['f1'])
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
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
            current_f1 = val_m.get('opt_f1', val_m['f1'])
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                break

    if is_main:
        print("  Best Val OptF1: %.4f" % best_val_f1)

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
    parser.add_argument('--residual', action='store_true', default=False,
                        help='Add residual (skip) connections to GNN layers')
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
    # Graph augmentation
    parser.add_argument('--drop_edge', type=float, default=0.0,
                        help='DropEdge rate: fraction of edges to drop per epoch (0=off)')
    parser.add_argument('--feature_noise', type=float, default=0.0,
                        help='Gaussian noise std added to node features per epoch (0=off)')
    parser.add_argument('--feature_mask', type=float, default=0.0,
                        help='Random feature masking rate (0=off)')
    parser.add_argument('--augment_flip', type=float, default=0.0,
                        help='Probability of circumferential flip per batch (0=off)')
    parser.add_argument('--node_drop', type=float, default=0.0,
                        help='Node drop rate per batch (0=off, 0.1=drop 10%% of nodes)')
    # Physics-informed loss
    parser.add_argument('--physics_lambda_smooth', type=float, default=0.0,
                        help='Laplacian smoothness loss weight (0=off)')
    parser.add_argument('--physics_lambda_stress', type=float, default=0.0,
                        help='Stress gradient consistency loss weight (0=off)')
    parser.add_argument('--physics_lambda_connected', type=float, default=0.0,
                        help='Connected component penalty weight (0=off)')
    parser.add_argument('--stress_dim', type=int, default=18,
                        help='Feature dim index for von Mises stress (default: 18)')
    # Node-level loss weighting
    parser.add_argument('--boundary_weight', type=float, default=1.0,
                        help='Weight multiplier for boundary nodes (healthy adjacent to defect)')
    parser.add_argument('--defect_weight', type=float, default=1.0,
                        help='Weight multiplier for defect nodes (>1 = penalize false negatives more)')
    # Transfer learning
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint (best_model.pt) for fine-tuning')
    parser.add_argument('--freeze_layers', type=int, default=0,
                        help='Freeze first N conv layers during fine-tuning')
    # Cross-validation
    parser.add_argument('--cross_val', type=int, default=0,
                        help='Number of folds for CV (0=disabled)')
    # Multi-GPU
    parser.add_argument('--multi_gpu', action='store_true', default=False,
                        help='Enable DDP multi-GPU (use with torchrun)')
    # Misc
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_normalize', action='store_true', default=False,
                        help='Skip feature normalization (debug only)')
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

    # --- Feature normalization ---
    if not args.no_normalize:
        norm_path = os.path.join(data_dir, 'norm_stats.pt')
        if os.path.exists(norm_path):
            norm_stats = torch.load(norm_path, weights_only=False)
            x_mean = norm_stats['mean']
            x_std = norm_stats['std']
            # Normalize node features for all samples
            for d in train_data + val_data:
                d.x = (d.x - x_mean) / x_std
            if rank == 0:
                print("Normalized node features: %d dims (from norm_stats.pt)" % x_mean.shape[0])
        else:
            if rank == 0:
                print("WARNING: norm_stats.pt not found — training without normalization")

        # Normalize edge_attr (compute stats from train data)
        edge_attrs = [d.edge_attr for d in train_data if d.edge_attr is not None]
        if edge_attrs:
            ea_cat = torch.cat(edge_attrs, dim=0)
            ea_mean = ea_cat.mean(dim=0)
            ea_std = ea_cat.std(dim=0)
            ea_std[ea_std < 1e-8] = 1.0
            for d in train_data + val_data:
                if d.edge_attr is not None:
                    d.edge_attr = (d.edge_attr - ea_mean) / ea_std
            if rank == 0:
                print("Normalized edge features: %d dims (computed from train)" % ea_mean.shape[0])
    else:
        if rank == 0:
            print("Normalization SKIPPED (--no_normalize)")

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
