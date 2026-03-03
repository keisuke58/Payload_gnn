# -*- coding: utf-8 -*-
"""Train PI-GraphMAE — Self-supervised pre-training on FEM mesh graphs.

Usage:
    python src/prad/train_mae.py \
        --data_dir data/processed_s12_mixed_400 \
        --encoder_arch sage --epochs 200 --lr 1e-3 \
        --mask_ratio 0.5 --lambda_physics 0.1
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

# Allow imports from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from prad.graphmae import PIGraphMAE


def load_data(data_dir, device='cpu', healthy_only_train=False):
    """Load PyG data and normalization stats.

    Args:
        data_dir: path to processed data directory.
        device: torch device.
        healthy_only_train: if True, filter training data to only include
            graphs with no defect nodes (y==0 for all nodes). This is
            critical for anomaly detection: the model must learn only
            "healthy physics" so defects produce high reconstruction error.

    Returns:
        train_data: list of PyG Data objects.
        val_data: list of PyG Data objects.
    """
    train_data = torch.load(os.path.join(data_dir, 'train.pt'),
                            weights_only=False)
    val_data = torch.load(os.path.join(data_dir, 'val.pt'),
                          weights_only=False)

    # Apply normalization
    norm_path = os.path.join(data_dir, 'norm_stats.pt')
    if os.path.exists(norm_path):
        stats = torch.load(norm_path, weights_only=False)
        x_mean = stats['mean'].to(device)
        x_std = stats['std'].to(device)
        x_std[x_std < 1e-8] = 1.0
        for d in train_data + val_data:
            d.x = (d.x.to(device) - x_mean) / x_std
            d.edge_index = d.edge_index.to(device)
            if d.edge_attr is not None:
                d.edge_attr = d.edge_attr.to(device)
            d.y = d.y.to(device)
    else:
        for d in train_data + val_data:
            d.x = d.x.to(device)
            d.edge_index = d.edge_index.to(device)
            if d.edge_attr is not None:
                d.edge_attr = d.edge_attr.to(device)
            d.y = d.y.to(device)

    # Filter training data to healthy-only graphs
    if healthy_only_train:
        n_before = len(train_data)
        train_data = [d for d in train_data if (d.y == 0).all()]
        print("  healthy filter: %d → %d train graphs" % (
            n_before, len(train_data)))

    return train_data, val_data


def train_epoch(model, data_list, optimizer):
    """Train one epoch over all graphs."""
    model.train()
    total_loss = 0.0
    n = 0
    for data in data_list:
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, data_list):
    """Evaluate: reconstruction loss + cosine similarity on val set."""
    model.eval()
    total_loss = 0.0
    total_cos_sim = 0.0
    n = 0
    for data in data_list:
        loss, x_recon = model(data, return_recon=True)
        total_loss += loss.item()

        # Cosine similarity (all nodes, not just masked)
        cos_sim = torch.nn.functional.cosine_similarity(
            x_recon, data.x, dim=1).mean().item()
        total_cos_sim += cos_sim
        n += 1

    return total_loss / max(n, 1), total_cos_sim / max(n, 1)


def main():
    parser = argparse.ArgumentParser(
        description='PI-GraphMAE self-supervised pre-training')
    parser.add_argument('--data_dir',
                        default='data/processed_s12_mixed_400')
    parser.add_argument('--encoder_arch', default='sage',
                        choices=['gcn', 'gat', 'gin', 'sage'])
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--lambda_physics', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--log_dir', default='runs/prad_mae')
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--healthy_only', action='store_true',
                        help='Train only on healthy graphs (no defect nodes). '
                             'Note: loss is always computed on healthy nodes '
                             'only (via model). This flag filters whole graphs.')
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 60)
    print("PI-GraphMAE Pre-training")
    print("=" * 60)
    print("  data_dir:       %s" % args.data_dir)
    print("  encoder:        %s" % args.encoder_arch)
    print("  hidden:         %d" % args.hidden)
    print("  mask_ratio:     %.2f" % args.mask_ratio)
    print("  lambda_physics: %.3f" % args.lambda_physics)
    print("  healthy_only:   %s" % args.healthy_only)
    print("  device:         %s" % device)
    print()

    # Load data
    train_data, val_data = load_data(args.data_dir, device,
                                     healthy_only_train=args.healthy_only)
    print("  train graphs: %d" % len(train_data))
    print("  val graphs:   %d" % len(val_data))
    if train_data:
        print("  nodes/graph:  %d" % train_data[0].x.size(0))
        print("  features:     %d" % train_data[0].x.size(1))

    # Build model
    model = PIGraphMAE(
        encoder_arch=args.encoder_arch,
        in_channels=train_data[0].x.size(1),
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        lambda_physics=args.lambda_physics,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("  parameters:   %d (%.1fK)" % (n_params, n_params / 1000))

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Logging
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    ckpt_name = 'prad_mae_%s.pt' % args.encoder_arch
    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)

    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_data, optimizer)
        val_loss, val_cos_sim = eval_epoch(model, val_data)
        scheduler.step()
        dt = time.time() - t0

        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('metric/val_cos_sim', val_cos_sim, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        improved = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_cos_sim': val_cos_sim,
                'args': vars(args),
            }, ckpt_path)
            improved = ' *'
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or improved:
            print("  Epoch %3d/%d  train=%.4f  val=%.4f  cos_sim=%.4f  "
                  "%.1fs%s" % (epoch, args.epochs, train_loss, val_loss,
                               val_cos_sim, dt, improved))

        if patience_counter >= args.patience:
            print("\n  Early stopping at epoch %d (patience=%d)" %
                  (epoch, args.patience))
            break

    writer.close()
    print("\nBest val loss: %.4f" % best_val_loss)
    print("Checkpoint: %s" % ckpt_path)


if __name__ == '__main__':
    main()
