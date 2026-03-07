# -*- coding: utf-8 -*-
"""
Foundation Model Pre-training: Masked Feature Reconstruction for GNN-SHM

Self-supervised pre-training on unlabeled graph data.
Masks random node features and trains the GNN encoder to reconstruct them,
forcing the model to learn physical relationships (stress ↔ displacement ↔ geometry).

After pre-training, use the saved encoder weights for fine-tuning:
    python src/train.py --pretrained runs/pretrain_XXX/best_encoder.pt --freeze_layers 2

Usage:
    python src/pretrain_foundation.py --data_dir data/processed_s12_thermal_500
    python src/pretrain_foundation.py --data_dir data/processed_s12_thermal_500 --mask_ratio 0.3 --epochs 100

Architecture:
    [Input (34-dim)] → [Mask 20%] → GNN Encoder → [128-dim embeddings] → Decoder → [Reconstruct masked dims]

The decoder is discarded after pre-training; only the encoder is kept.
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
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from models import build_model


class MaskedFeatureDecoder(nn.Module):
    """Lightweight MLP decoder for reconstructing masked node features."""

    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, h):
        return self.net(h)


def mask_node_features(x, mask_ratio, mask_token=None):
    """Mask random feature dimensions per node.

    Args:
        x: (N, D) node features
        mask_ratio: fraction of dimensions to mask per node
        mask_token: (D,) learnable mask token, or None for zeros

    Returns:
        x_masked: (N, D) with masked dims replaced
        mask: (N, D) boolean, True = masked
    """
    N, D = x.shape
    num_mask = max(1, int(D * mask_ratio))

    # Per-node random dimension masking
    mask = torch.zeros(N, D, dtype=torch.bool, device=x.device)
    for i in range(N):
        dims = torch.randperm(D, device=x.device)[:num_mask]
        mask[i, dims] = True

    x_masked = x.clone()
    if mask_token is not None:
        x_masked[mask] = mask_token.expand(N, -1)[mask]
    else:
        x_masked[mask] = 0.0

    return x_masked, mask


def mask_node_features_fast(x, mask_ratio):
    """Vectorized version — masks the same dims for all nodes in a batch.

    Much faster than per-node masking. Sufficient for pre-training since
    the GNN encoder aggregates neighbor information anyway.

    Returns:
        x_masked: (N, D) with masked dims zeroed out
        mask: (D,) boolean, True = masked dimension
    """
    N, D = x.shape
    num_mask = max(1, int(D * mask_ratio))
    dims = torch.randperm(D, device=x.device)[:num_mask]
    mask = torch.zeros(D, dtype=torch.bool, device=x.device)
    mask[dims] = True

    x_masked = x.clone()
    x_masked[:, mask] = 0.0

    return x_masked, mask


def pretrain_epoch(model, decoder, loader, optimizer, device, mask_ratio,
                   mask_strategy='batch'):
    """One epoch of masked feature reconstruction."""
    model.train()
    decoder.train()
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        x_orig = batch.x

        # Mask features
        if mask_strategy == 'batch':
            x_masked, mask = mask_node_features_fast(x_orig, mask_ratio)
        else:
            x_masked, mask = mask_node_features(x_orig, mask_ratio)

        optimizer.zero_grad()

        # Encode masked input
        h = model.encode(x_masked, batch.edge_index, batch.edge_attr)

        # Decode to reconstruct
        x_pred = decoder(h)

        # Loss: only on masked positions
        if mask_strategy == 'batch':
            # mask is (D,) — same dims for all nodes
            loss = F.mse_loss(x_pred[:, mask], x_orig[:, mask])
        else:
            # mask is (N, D)
            loss = F.mse_loss(x_pred[mask], x_orig[mask])

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_orig.shape[0]
        total_nodes += x_orig.shape[0]

    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def eval_pretrain(model, decoder, loader, device, mask_ratio):
    """Evaluate reconstruction loss on validation set."""
    model.eval()
    decoder.eval()
    total_loss = 0.0
    total_nodes = 0

    # Fixed mask dims for reproducible eval
    sample = next(iter(loader))
    D = sample.x.shape[1]
    num_mask = max(1, int(D * mask_ratio))
    torch.manual_seed(0)
    mask_dims = torch.randperm(D)[:num_mask]
    mask = torch.zeros(D, dtype=torch.bool)
    mask[mask_dims] = True
    mask = mask.to(device)

    for batch in loader:
        batch = batch.to(device)
        x_orig = batch.x

        x_masked = x_orig.clone()
        x_masked[:, mask] = 0.0

        h = model.encode(x_masked, batch.edge_index, batch.edge_attr)
        x_pred = decoder(h)

        loss = F.mse_loss(x_pred[:, mask], x_orig[:, mask])
        total_loss += loss.item() * x_orig.shape[0]
        total_nodes += x_orig.shape[0]

    return total_loss / max(total_nodes, 1)


def main():
    parser = argparse.ArgumentParser(
        description='Foundation model pre-training (masked feature reconstruction)')
    parser.add_argument('--data_dir', type=str,
                        default='data/processed_s12_thermal_500')
    parser.add_argument('--output_dir', type=str, default='runs')

    # Model (same as train.py for compatibility)
    parser.add_argument('--arch', type=str, default='gat',
                        choices=['gcn', 'gat', 'gin', 'sage'])
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--residual', action='store_true', default=False)

    # Pre-training
    parser.add_argument('--mask_ratio', type=float, default=0.20,
                        help='Fraction of node features to mask (default: 0.20)')
    parser.add_argument('--mask_strategy', type=str, default='batch',
                        choices=['batch', 'per_node'],
                        help='batch: same dims masked for all nodes (fast); '
                             'per_node: random dims per node (slower)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=20)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=5)
    parser.add_argument('--no_normalize', action='store_true', default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Load data ----
    data_dir = (os.path.join(PROJECT_ROOT, args.data_dir)
                if not os.path.isabs(args.data_dir) else args.data_dir)
    train_path = os.path.join(data_dir, 'train.pt')
    val_path = os.path.join(data_dir, 'val.pt')

    if not os.path.exists(train_path):
        print("Data not found: %s" % train_path)
        sys.exit(1)

    print("Loading data from %s..." % args.data_dir)
    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)
    print("Train: %d graphs | Val: %d graphs" % (len(train_data), len(val_data)))

    # ---- Normalize ----
    if not args.no_normalize:
        norm_path = os.path.join(data_dir, 'norm_stats.pt')
        if os.path.exists(norm_path):
            norm_stats = torch.load(norm_path, weights_only=False)
            x_mean = norm_stats['mean']
            x_std = norm_stats['std']
            for d in train_data + val_data:
                d.x = (d.x - x_mean) / x_std
            print("Normalized node features: %d dims" % x_mean.shape[0])

        edge_attrs = [d.edge_attr for d in train_data if d.edge_attr is not None]
        if edge_attrs:
            ea_cat = torch.cat(edge_attrs, dim=0)
            ea_mean = ea_cat.mean(dim=0)
            ea_std = ea_cat.std(dim=0)
            ea_std[ea_std < 1e-8] = 1.0
            for d in train_data + val_data:
                if d.edge_attr is not None:
                    d.edge_attr = (d.edge_attr - ea_mean) / ea_std
            print("Normalized edge features: %d dims" % ea_mean.shape[0])

    # ---- DataLoaders ----
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # ---- Model ----
    sample = train_data[0]
    in_channels = sample.x.shape[1]
    edge_attr_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0

    # Build GNN encoder (num_classes=2 is dummy, head won't be used)
    model = build_model(
        args.arch, in_channels, edge_attr_dim,
        hidden_channels=args.hidden, num_layers=args.layers,
        dropout=args.dropout, num_classes=2,
        use_residual=args.residual,
    ).to(device)

    # Decoder: hidden_channels → in_channels (reconstruct original features)
    decoder = MaskedFeatureDecoder(args.hidden, in_channels).to(device)

    total_params = (sum(p.numel() for p in model.parameters()) +
                    sum(p.numel() for p in decoder.parameters()))
    print("Encoder: %s-%d (%d hidden) | Decoder: MLP" % (
        args.arch.upper(), args.layers, args.hidden))
    print("Total params: %d | Mask ratio: %.0f%% (%s)" % (
        total_params, args.mask_ratio * 100, args.mask_strategy))
    print("Device: %s" % device)

    # ---- Optimizer ----
    params = list(model.parameters()) + list(decoder.parameters())
    optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- Logging ----
    run_name = 'pretrain_%s_%s' % (args.arch, datetime.now().strftime('%Y%m%d_%H%M%S'))
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, 'pretrain_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv_module.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])

    # Save config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\nPre-training → %s" % run_dir)
    print("=" * 60)

    # ---- Training loop ----
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = pretrain_epoch(
            model, decoder, train_loader, optimizer, device,
            args.mask_ratio, args.mask_strategy)
        val_loss = eval_pretrain(
            model, decoder, val_loader, device, args.mask_ratio)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # Log CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv_module.writer(f)
            writer.writerow([epoch, '%.6f' % train_loss, '%.6f' % val_loss,
                             '%.2e' % lr])

        if epoch % args.log_every == 0 or epoch == 1:
            print("  Epoch %3d/%d | Train MSE=%.6f | Val MSE=%.6f | LR=%.2e | %.1fs" %
                  (epoch, args.epochs, train_loss, val_loss, lr, elapsed))

        # Checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save full checkpoint (encoder + decoder)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args),
                'in_channels': in_channels,
                'edge_attr_dim': edge_attr_dim,
            }, os.path.join(run_dir, 'best_model.pt'))

            # Save encoder-only (for fine-tuning with train.py --pretrained)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args),
                'in_channels': in_channels,
                'edge_attr_dim': edge_attr_dim,
            }, os.path.join(run_dir, 'best_encoder.pt'))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("  Early stopping at epoch %d (patience=%d)" %
                  (epoch, args.patience))
            break

    print("=" * 60)
    print("Best Val MSE: %.6f" % best_val_loss)
    print("Encoder saved: %s" % os.path.join(run_dir, 'best_encoder.pt'))
    print("\nTo fine-tune:")
    print("  python src/train.py \\")
    print("    --data_dir %s \\" % args.data_dir)
    print("    --pretrained %s \\" % os.path.join(run_dir, 'best_encoder.pt'))
    print("    --arch %s --hidden %d --layers %d" % (
        args.arch, args.hidden, args.layers))
    print("\nDone.")


if __name__ == '__main__':
    main()
