# -*- coding: utf-8 -*-
"""Domain-Adversarial Neural Network (DANN) for GNN-SHM.

Implements gradient reversal + domain discriminator to learn
domain-invariant node representations across:
  - Different mesh resolutions (25mm vs 50mm)
  - Different defect types (CZM debonding, thermal, etc.)
  - Different structures (fairing vs flat plate)

Reference: Ganin et al. "Domain-Adversarial Training of Neural Networks" (JMLR 2016)

Usage:
    python src/prad/domain_adapt.py \
        --source_dir data/processed_s12_czm_thermal_200_binary \
        --target_dir data/processed_25mm_100 \
        --arch sage --epochs 100 --device cuda
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from models import build_model


# ====================================================================
# Gradient Reversal Layer
# ====================================================================
class GradientReversal(Function):
    """Reverse gradients during backward pass (DANN core).

    Forward:  identity
    Backward: negate gradients, scaled by lambda
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)


# ====================================================================
# Domain Discriminator
# ====================================================================
class DomainDiscriminator(nn.Module):
    """MLP domain classifier operating on node embeddings."""

    def __init__(self, hidden_channels, num_domains=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, num_domains),
        )

    def forward(self, h, lambda_=1.0):
        """Forward with gradient reversal.

        Args:
            h: (N, hidden) node embeddings from GNN encoder
            lambda_: gradient reversal strength (annealed during training)
        """
        h_rev = grad_reverse(h, lambda_)
        return self.net(h_rev)


# ====================================================================
# DANN-GNN Model
# ====================================================================
class DANNGNN(nn.Module):
    """GNN with domain-adversarial training.

    Components:
        1. GNN encoder (shared) → node embeddings
        2. Task head → defect classification (2-class)
        3. Domain discriminator (with GRL) → domain classification
    """

    def __init__(self, arch, in_channels, hidden_channels=128,
                 num_layers=4, dropout=0.1, num_classes=2,
                 num_domains=2, edge_attr_dim=0):
        super().__init__()

        self.encoder = build_model(
            arch, in_channels=in_channels,
            edge_attr_dim=edge_attr_dim,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Task classification head (reuse encoder's head)
        # The encoder already has a .head, so we use encoder.forward()
        # for task prediction

        # Domain discriminator (separate)
        self.domain_disc = DomainDiscriminator(hidden_channels, num_domains)

    def forward(self, x, edge_index, edge_attr=None, lambda_=1.0):
        """Forward pass returning both task logits and domain logits.

        Returns:
            task_out: (N, num_classes) task classification logits
            domain_out: (N, num_domains) domain classification logits
        """
        # Shared encoder
        h = self.encoder.encode(x, edge_index, edge_attr)

        # Task head
        task_out = self.encoder.head(h)

        # Domain discriminator (with gradient reversal)
        domain_out = self.domain_disc(h, lambda_=lambda_)

        return task_out, domain_out

    def predict(self, x, edge_index, edge_attr=None):
        """Task prediction only (for evaluation)."""
        h = self.encoder.encode(x, edge_index, edge_attr)
        return self.encoder.head(h)


# ====================================================================
# Lambda Annealing Schedule
# ====================================================================
def compute_lambda(epoch, max_epochs, gamma=10.0):
    """Sigmoid annealing for gradient reversal strength.

    Starts near 0, ramps up to 1 over training.
    From Ganin et al. (2016): lambda = 2/(1+exp(-gamma*p)) - 1
    """
    p = epoch / max_epochs
    return 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p)).item()) - 1.0


# ====================================================================
# Training Loop
# ====================================================================
def train_dann(args):
    """Train GNN with domain-adversarial adaptation."""
    device = torch.device(args.device)
    print("=" * 60)
    print("Domain-Adversarial GNN Training")
    print("=" * 60)

    # Load source domain (labeled)
    print("\nLoading source domain: %s" % args.source_dir)
    src_train = torch.load(os.path.join(args.source_dir, 'train.pt'),
                           weights_only=False)
    src_val = torch.load(os.path.join(args.source_dir, 'val.pt'),
                         weights_only=False)

    # Load normalization stats from source
    norm_path = os.path.join(args.source_dir, 'norm_stats.pt')
    norm_stats = None
    if os.path.exists(norm_path):
        norm_stats = torch.load(norm_path, weights_only=False)

    # Load target domain
    print("Loading target domain: %s" % args.target_dir)
    tgt_train = torch.load(os.path.join(args.target_dir, 'train.pt'),
                           weights_only=False)
    tgt_val = torch.load(os.path.join(args.target_dir, 'val.pt'),
                         weights_only=False)

    # Normalize all data with source stats
    if norm_stats is not None:
        x_mean = norm_stats['mean'].to(device)
        x_std = norm_stats['std'].to(device)
        x_std[x_std < 1e-8] = 1.0
        for d in src_train + src_val + tgt_train + tgt_val:
            d.x = (d.x.to(device) - x_mean) / x_std
            d.edge_index = d.edge_index.to(device)
            if d.edge_attr is not None:
                d.edge_attr = d.edge_attr.to(device)
            d.y = d.y.to(device)
    else:
        for d in src_train + src_val + tgt_train + tgt_val:
            d.x = d.x.to(device)
            d.edge_index = d.edge_index.to(device)
            if d.edge_attr is not None:
                d.edge_attr = d.edge_attr.to(device)
            d.y = d.y.to(device)

    in_channels = src_train[0].x.size(1)
    print("  Source: %d train, %d val" % (len(src_train), len(src_val)))
    print("  Target: %d train, %d val" % (len(tgt_train), len(tgt_val)))
    print("  Features: %d" % in_channels)

    # Build model
    edge_attr_dim = 0
    if args.arch == 'gat' and src_train[0].edge_attr is not None:
        edge_attr_dim = src_train[0].edge_attr.size(1)

    model = DANNGNN(
        arch=args.arch,
        in_channels=in_channels,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=args.num_classes,
        num_domains=2,
        edge_attr_dim=edge_attr_dim,
    ).to(device)

    # Load MAE pretrained encoder weights if provided
    if hasattr(args, 'pretrained') and args.pretrained:
        print("  Loading MAE pretrained: %s" % args.pretrained)
        ckpt = torch.load(args.pretrained, weights_only=False,
                          map_location=device)
        if 'encoder_state_dict' in ckpt:
            enc_state = ckpt['encoder_state_dict']
        else:
            enc_state = {k.replace('encoder.', ''): v
                         for k, v in ckpt['model_state_dict'].items()
                         if k.startswith('encoder.')}
        missing, unexpected = model.encoder.load_state_dict(
            enc_state, strict=False)
        print("  Loaded %d keys (missing=%d, unexpected=%d)" % (
            len(enc_state), len(missing), len(unexpected)))

    n_params = sum(p.numel() for p in model.parameters())
    print("  DANN-%s parameters: %d (%.1fK)" % (args.arch.upper(),
                                                  n_params, n_params / 1000))

    # Class weights for imbalanced task labels
    all_y = torch.cat([d.y for d in src_train])
    n_pos = (all_y > 0).sum().item()
    n_neg = len(all_y) - n_pos
    pos_weight = min(n_neg / max(n_pos, 1), 10.0)
    task_weights = torch.tensor([1.0, pos_weight], device=device)
    print("  Task class weights: [1.0, %.2f]" % pos_weight)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir,
                              'dann_%s.pt' % args.arch)
    best_auroc = 0.0
    patience_counter = 0

    print("\nTraining (lambda: sigmoid annealing)...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lambda_ = compute_lambda(epoch, args.epochs)

        # === Train ===
        model.train()
        total_task_loss = 0.0
        total_domain_loss = 0.0
        n_batches = 0

        # Interleave source and target graphs
        n_src = len(src_train)
        n_tgt = len(tgt_train)
        n_iter = max(n_src, n_tgt)

        src_perm = torch.randperm(n_src)
        tgt_perm = torch.randperm(n_tgt)

        for i in range(n_iter):
            optimizer.zero_grad()

            # Source graph: task loss + domain loss
            src_idx = src_perm[i % n_src]
            src_data = src_train[src_idx]
            task_out, domain_out = model(
                src_data.x, src_data.edge_index,
                getattr(src_data, 'edge_attr', None),
                lambda_=lambda_)

            # Task loss (source only — labeled)
            task_loss = F.cross_entropy(task_out, src_data.y,
                                        weight=task_weights)

            # Domain loss (source = 0)
            src_domain_labels = torch.zeros(src_data.x.size(0),
                                            dtype=torch.long, device=device)
            domain_loss_src = F.cross_entropy(domain_out, src_domain_labels)

            # Target graph: domain loss only (no task labels needed)
            tgt_idx = tgt_perm[i % n_tgt]
            tgt_data = tgt_train[tgt_idx]
            _, domain_out_tgt = model(
                tgt_data.x, tgt_data.edge_index,
                getattr(tgt_data, 'edge_attr', None),
                lambda_=lambda_)

            # Domain loss (target = 1)
            tgt_domain_labels = torch.ones(tgt_data.x.size(0),
                                           dtype=torch.long, device=device)
            domain_loss_tgt = F.cross_entropy(domain_out_tgt, tgt_domain_labels)

            # Combined loss
            domain_loss = 0.5 * (domain_loss_src + domain_loss_tgt)
            loss = task_loss + args.lambda_domain * domain_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_task_loss += task_loss.item()
            total_domain_loss += domain_loss.item()
            n_batches += 1

        avg_task = total_task_loss / max(n_batches, 1)
        avg_domain = total_domain_loss / max(n_batches, 1)

        # === Evaluate on source val (standard) ===
        from sklearn.metrics import roc_auc_score
        model.eval()
        src_probs, src_labels = [], []
        with torch.no_grad():
            for data in src_val:
                out = model.predict(data.x, data.edge_index,
                                    getattr(data, 'edge_attr', None))
                probs = F.softmax(out, dim=1)[:, 1]
                src_probs.append(probs.cpu())
                src_labels.append(data.y.cpu())

        src_probs = torch.cat(src_probs).numpy()
        src_labels = torch.cat(src_labels).numpy()
        try:
            src_auroc = roc_auc_score(src_labels, src_probs)
        except ValueError:
            src_auroc = 0.0

        # === Evaluate on target val (transfer) ===
        tgt_probs, tgt_labels = [], []
        with torch.no_grad():
            for data in tgt_val:
                out = model.predict(data.x, data.edge_index,
                                    getattr(data, 'edge_attr', None))
                probs = F.softmax(out, dim=1)[:, 1]
                tgt_probs.append(probs.cpu())
                tgt_labels.append(data.y.cpu())

        tgt_probs = torch.cat(tgt_probs).numpy()
        tgt_labels = torch.cat(tgt_labels).numpy()
        try:
            tgt_auroc = roc_auc_score(tgt_labels, tgt_probs)
        except ValueError:
            tgt_auroc = 0.0

        scheduler.step()
        dt = time.time() - t0

        # Track best by target AUROC (main goal: improve transfer)
        improved = ''
        metric = tgt_auroc if tgt_auroc > 0 else src_auroc
        if metric > best_auroc:
            best_auroc = metric
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'src_auroc': src_auroc,
                'tgt_auroc': tgt_auroc,
                'args': vars(args),
            }, ckpt_path)
            improved = ' *'
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            print("  Epoch %3d/%d  task=%.4f  dom=%.4f  "
                  "src_AUROC=%.4f  tgt_AUROC=%.4f  "
                  "lam=%.3f  %.1fs%s" % (
                      epoch, args.epochs, avg_task, avg_domain,
                      src_auroc, tgt_auroc, lambda_, dt, improved))

        if patience_counter >= args.patience:
            print("\n  Early stopping at epoch %d" % epoch)
            break

    print("\nBest AUROC (target): %.4f" % best_auroc)
    print("Checkpoint: %s" % ckpt_path)
    return ckpt_path


# ====================================================================
# Baseline: No adaptation (source only)
# ====================================================================
def train_baseline(args):
    """Train standard GNN on source only, evaluate on target."""
    device = torch.device(args.device)
    print("=" * 60)
    print("Baseline (No Adaptation) — %s" % args.arch.upper())
    print("=" * 60)

    # Load source
    src_train = torch.load(os.path.join(args.source_dir, 'train.pt'),
                           weights_only=False)
    src_val = torch.load(os.path.join(args.source_dir, 'val.pt'),
                         weights_only=False)

    norm_path = os.path.join(args.source_dir, 'norm_stats.pt')
    norm_stats = None
    if os.path.exists(norm_path):
        norm_stats = torch.load(norm_path, weights_only=False)

    # Load target
    tgt_val = torch.load(os.path.join(args.target_dir, 'val.pt'),
                         weights_only=False)

    # Normalize
    if norm_stats is not None:
        x_mean = norm_stats['mean'].to(device)
        x_std = norm_stats['std'].to(device)
        x_std[x_std < 1e-8] = 1.0
        for d in src_train + src_val + tgt_val:
            d.x = (d.x.to(device) - x_mean) / x_std
            d.edge_index = d.edge_index.to(device)
            if d.edge_attr is not None:
                d.edge_attr = d.edge_attr.to(device)
            d.y = d.y.to(device)
    else:
        for d in src_train + src_val + tgt_val:
            d.x = d.x.to(device)
            d.edge_index = d.edge_index.to(device)
            if d.edge_attr is not None:
                d.edge_attr = d.edge_attr.to(device)
            d.y = d.y.to(device)

    in_channels = src_train[0].x.size(1)
    edge_attr_dim = 0
    if args.arch == 'gat' and src_train[0].edge_attr is not None:
        edge_attr_dim = src_train[0].edge_attr.size(1)

    model = build_model(
        args.arch, in_channels=in_channels,
        edge_attr_dim=edge_attr_dim,
        hidden_channels=args.hidden,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # Class weights
    all_y = torch.cat([d.y for d in src_train])
    n_pos = (all_y > 0).sum().item()
    n_neg = len(all_y) - n_pos
    pos_weight = min(n_neg / max(n_pos, 1), 10.0)
    task_weights = torch.tensor([1.0, pos_weight], device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_src_auroc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for data in src_train:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out, data.y, weight=task_weights)
            loss.backward()
            optimizer.step()

        # Eval
        from sklearn.metrics import roc_auc_score
        model.eval()
        src_probs, src_labels = [], []
        tgt_probs, tgt_labels = [], []
        with torch.no_grad():
            for data in src_val:
                out = model(data.x, data.edge_index)
                src_probs.append(F.softmax(out, dim=1)[:, 1].cpu())
                src_labels.append(data.y.cpu())
            for data in tgt_val:
                out = model(data.x, data.edge_index)
                tgt_probs.append(F.softmax(out, dim=1)[:, 1].cpu())
                tgt_labels.append(data.y.cpu())

        try:
            src_auroc = roc_auc_score(
                torch.cat(src_labels).numpy(),
                torch.cat(src_probs).numpy())
        except ValueError:
            src_auroc = 0.0
        try:
            tgt_auroc = roc_auc_score(
                torch.cat(tgt_labels).numpy(),
                torch.cat(tgt_probs).numpy())
        except ValueError:
            tgt_auroc = 0.0

        scheduler.step()
        if src_auroc > best_src_auroc:
            best_src_auroc = src_auroc

        if epoch % 10 == 0 or epoch == 1:
            print("  Epoch %3d/%d  src_AUROC=%.4f  tgt_AUROC=%.4f" % (
                epoch, args.epochs, src_auroc, tgt_auroc))

    print("\nBaseline results:")
    print("  Source AUROC: %.4f" % best_src_auroc)
    print("  Target AUROC: %.4f (zero-shot transfer)" % tgt_auroc)
    return src_auroc, tgt_auroc


# ====================================================================
# Main
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Domain-Adversarial GNN for cross-domain SHM')
    parser.add_argument('--source_dir',
                        default='data/processed_s12_czm_thermal_200_binary',
                        help='Source domain data directory')
    parser.add_argument('--target_dir',
                        default='data/processed_25mm_100',
                        help='Target domain data directory')
    parser.add_argument('--arch', default='sage',
                        choices=['gcn', 'gat', 'gin', 'sage', 'gps'])
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lambda_domain', type=float, default=0.1,
                        help='Weight for domain adversarial loss')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--pretrained', default='',
                        help='Path to MAE pretrained encoder checkpoint')
    parser.add_argument('--run_baseline', action='store_true',
                        help='Also run baseline (no adaptation) for comparison')

    args = parser.parse_args()

    if args.run_baseline:
        print("\n--- BASELINE (No Adaptation) ---")
        baseline_src, baseline_tgt = train_baseline(args)

    print("\n--- DANN (Domain Adaptation) ---")
    train_dann(args)


if __name__ == '__main__':
    main()
