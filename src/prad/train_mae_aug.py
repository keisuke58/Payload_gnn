# -*- coding: utf-8 -*-
"""Train PI-GraphMAE with data augmentation for improved anomaly detection.

Augmentation strategies applied during training:
1. DropEdge: Randomly remove edges (improves GNN generalization)
2. Feature noise: Multiplicative + additive noise on node features
3. Variable mask ratio: Random mask ratio per graph (0.3-0.7)

These augmentations make the MAE more robust to normal variation,
so it produces higher residuals only for genuine anomalies.

Usage:
    python src/prad/train_mae_aug.py \
        --data_dir data/processed_s12_mixed_400 \
        --encoder_arch sage --epochs 200 --lr 1e-3 \
        --decoder_type bottleneck --augment
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from prad.graphmae import PIGraphMAE
from prad.anomaly_score import score_dataset
from prad.eval_prad import compute_metrics
from prad.train_mae import load_data, eval_epoch


def drop_edge(edge_index, edge_attr=None, drop_ratio=0.1):
    """Randomly drop edges for augmentation.

    DropEdge (Rong et al., 2020) improves GNN generalization by
    preventing over-smoothing and reducing co-adaptation.
    """
    n_edges = edge_index.size(1)
    n_keep = max(1, int(n_edges * (1.0 - drop_ratio)))
    perm = torch.randperm(n_edges, device=edge_index.device)[:n_keep]
    new_ei = edge_index[:, perm]
    new_ea = edge_attr[perm] if edge_attr is not None else None
    return new_ei, new_ea


def augment_features(x, mult_std=0.01, add_std_ratio=0.01):
    """Add calibrated noise to node features during training.

    Simulates sensor noise and calibration drift to make the MAE
    robust to normal physical variation.
    """
    # Multiplicative: x * (1 + η), η ~ N(0, mult_std²)
    eta = torch.randn_like(x) * mult_std
    x_aug = x * (1.0 + eta)

    # Additive: per-feature scaled noise
    feat_std = x.std(dim=0, keepdim=True)
    feat_std[feat_std < 1e-8] = 1.0
    eps = torch.randn_like(x) * (feat_std * add_std_ratio)
    x_aug = x_aug + eps

    # Don't corrupt boundary flags (dims 32, 33)
    if x.size(1) > 33:
        x_aug[:, 32] = x[:, 32]
        x_aug[:, 33] = x[:, 33]

    return x_aug


def train_epoch_aug(model, data_list, optimizer, augment=True,
                    drop_edge_ratio=0.1, feat_noise_mult=0.01,
                    feat_noise_add=0.01, variable_mask=True):
    """Train one epoch with optional augmentation."""
    model.train()
    total_loss = 0.0
    n = 0

    perm = torch.randperm(len(data_list))
    for idx in perm:
        data = data_list[idx]

        if augment:
            # Augment features
            orig_x = data.x
            data.x = augment_features(
                data.x, mult_std=feat_noise_mult,
                add_std_ratio=feat_noise_add)

            # DropEdge
            orig_ei = data.edge_index
            orig_ea = getattr(data, 'edge_attr', None)
            if drop_edge_ratio > 0:
                data.edge_index, data.edge_attr = drop_edge(
                    data.edge_index, orig_ea, drop_ratio=drop_edge_ratio)

            # Variable mask ratio
            if variable_mask:
                # Random mask ratio in [0.3, 0.7] per graph
                mr = 0.3 + 0.4 * torch.rand(1).item()
                model.mask_ratio = mr

        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1

        if augment:
            # Restore original data (data objects are shared references)
            data.x = orig_x
            data.edge_index = orig_ei
            data.edge_attr = orig_ea
            model.mask_ratio = 0.5  # restore default

    return total_loss / max(n, 1)


def evaluate_anomaly(model, val_data, alpha=0.7, smooth_rounds=1,
                     smooth_alpha=0.7):
    """Quick anomaly detection evaluation."""
    scores_list, labels_list, _ = score_dataset(
        model, val_data, alpha=alpha,
        smooth_rounds=smooth_rounds, smooth_alpha=smooth_alpha)
    scores_np = torch.cat(scores_list).numpy()
    labels_np = torch.cat(labels_list).numpy()
    return compute_metrics(scores_np, labels_np)


def main():
    parser = argparse.ArgumentParser(
        description='PI-GraphMAE with data augmentation')
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
    parser.add_argument('--log_dir', default='runs/prad_mae_aug')
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--decoder_type', default='bottleneck',
                        choices=['mlp', 'bottleneck'])
    parser.add_argument('--output_dir', default='figures/prad_aug')
    # Augmentation params
    parser.add_argument('--augment', action='store_true',
                        help='Enable training augmentation')
    parser.add_argument('--drop_edge_ratio', type=float, default=0.1,
                        help='DropEdge ratio (default: 10%%)')
    parser.add_argument('--feat_noise_mult', type=float, default=0.01,
                        help='Feature multiplicative noise std (default: 1%%)')
    parser.add_argument('--feat_noise_add', type=float, default=0.01,
                        help='Feature additive noise std ratio (default: 1%%)')
    parser.add_argument('--variable_mask', action='store_true',
                        help='Randomize mask ratio per graph (0.3-0.7)')
    # Pre-trained checkpoint (warm start)
    parser.add_argument('--pretrained', default=None,
                        help='Warm start from existing checkpoint')
    # Scoring params
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--smooth_rounds', type=int, default=1)
    parser.add_argument('--smooth_alpha', type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 60)
    print("PI-GraphMAE Training with Augmentation")
    print("=" * 60)
    print("  data_dir:       %s" % args.data_dir)
    print("  encoder:        %s" % args.encoder_arch)
    print("  decoder:        %s" % args.decoder_type)
    print("  hidden:         %d" % args.hidden)
    print("  augment:        %s" % args.augment)
    if args.augment:
        print("  drop_edge:      %.2f" % args.drop_edge_ratio)
        print("  feat_noise:     mult=%.3f  add=%.3f" % (
            args.feat_noise_mult, args.feat_noise_add))
        print("  variable_mask:  %s" % args.variable_mask)
    print("  device:         %s" % device)
    print()

    # Load data
    train_data, val_data = load_data(args.data_dir, device)
    print("  train graphs: %d" % len(train_data))
    print("  val graphs:   %d" % len(val_data))
    if train_data:
        print("  nodes/graph:  %d" % train_data[0].x.size(0))
        print("  features:     %d" % train_data[0].x.size(1))

    # Node stats
    all_labels = torch.cat([d.y for d in val_data])
    n_defect = (all_labels > 0).sum().item()
    n_total = len(all_labels)
    print("  val defect:   %d / %d (%.2f%%)" % (
        n_defect, n_total, 100.0 * n_defect / n_total))

    # Build model
    model = PIGraphMAE(
        encoder_arch=args.encoder_arch,
        in_channels=train_data[0].x.size(1),
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        lambda_physics=args.lambda_physics,
        decoder_type=args.decoder_type,
    ).to(device)

    # Warm start from pre-trained
    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print("  Warm start: %s (epoch %d)" % (
            args.pretrained, ckpt.get('epoch', -1)))

    n_params = sum(p.numel() for p in model.parameters())
    print("  parameters:   %d (%.1fK)" % (n_params, n_params / 1000))

    # Baseline evaluation (before training)
    print("\n  Baseline evaluation:")
    baseline = evaluate_anomaly(
        model, val_data, alpha=args.alpha,
        smooth_rounds=args.smooth_rounds, smooth_alpha=args.smooth_alpha)
    print("    ROC=%.4f  PR=%.4f  F1=%.4f" % (
        baseline['roc_auc'], baseline['pr_auc'], baseline['best_f1']))

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
    best_f1 = 0.0
    patience_counter = 0
    f1_patience_counter = 0
    suffix = '_aug' if args.augment else ''
    dec_str = '_%s' % args.decoder_type if args.decoder_type != 'mlp' else ''
    ckpt_name = 'prad_mae_%s%s%s.pt' % (args.encoder_arch, dec_str, suffix)
    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
    # Separate F1-best checkpoint (primary selection criterion)
    ckpt_f1_path = ckpt_path.replace('.pt', '_f1best.pt')

    print("\nTraining...")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch_aug(
            model, train_data, optimizer,
            augment=args.augment,
            drop_edge_ratio=args.drop_edge_ratio,
            feat_noise_mult=args.feat_noise_mult,
            feat_noise_add=args.feat_noise_add,
            variable_mask=args.variable_mask)
        val_loss, val_cos_sim = eval_epoch(model, val_data)
        scheduler.step()
        dt = time.time() - t0

        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('metric/val_cos_sim', val_cos_sim, epoch)

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
            elapsed = time.time() - t_start
            print("  Epoch %3d/%d  train=%.4f  val=%.4f  cos=%.4f  "
                  "%.1fs  (%.0fs)%s" % (
                      epoch, args.epochs, train_loss, val_loss,
                      val_cos_sim, dt, elapsed, improved))

        # Evaluate anomaly detection every 10 epochs (more frequent)
        if epoch % 10 == 0:
            m = evaluate_anomaly(
                model, val_data, alpha=args.alpha,
                smooth_rounds=args.smooth_rounds,
                smooth_alpha=args.smooth_alpha)
            writer.add_scalar('anomaly/roc_auc', m['roc_auc'], epoch)
            writer.add_scalar('anomaly/pr_auc', m['pr_auc'], epoch)
            writer.add_scalar('anomaly/f1', m['best_f1'], epoch)
            print("         → ROC=%.4f  PR=%.4f  F1=%.4f" % (
                m['roc_auc'], m['pr_auc'], m['best_f1']))
            if m['best_f1'] > best_f1:
                best_f1 = m['best_f1']
                f1_patience_counter = 0
                # Save F1-best checkpoint separately
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'val_cos_sim': val_cos_sim,
                    'best_f1': best_f1,
                    'args': vars(args),
                }, ckpt_f1_path)
                print("         → F1-best saved! (%.4f)" % best_f1)
            else:
                f1_patience_counter += 1

        # Early stop: if F1 hasn't improved for 5 eval cycles (50 epochs)
        if f1_patience_counter >= 5 and epoch >= 60:
            print("\n  F1-based early stopping at epoch %d "
                  "(no F1 improvement for %d eval cycles)" % (
                      epoch, f1_patience_counter))
            break

        if patience_counter >= args.patience:
            print("\n  Early stopping at epoch %d" % epoch)
            break

    writer.close()

    # Load F1-best model (preferred) or val-loss-best
    if os.path.exists(ckpt_f1_path):
        best_ckpt = torch.load(ckpt_f1_path, map_location=device,
                               weights_only=False)
        print("\n  Using F1-best checkpoint (epoch %d, F1=%.4f)" % (
            best_ckpt['epoch'], best_ckpt.get('best_f1', 0)))
    else:
        best_ckpt = torch.load(ckpt_path, map_location=device,
                               weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    # Final evaluation
    print("\n--- Final Evaluation ---")
    final = evaluate_anomaly(
        model, val_data, alpha=args.alpha,
        smooth_rounds=args.smooth_rounds, smooth_alpha=args.smooth_alpha)

    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print("  %-30s  ROC-AUC  PR-AUC  F1" % "")
    print("  " + "-" * 55)
    print("  %-30s  %.4f   %.4f  %.4f" % (
        "Before training",
        baseline['roc_auc'], baseline['pr_auc'], baseline['best_f1']))
    print("  %-30s  %.4f   %.4f  %.4f" % (
        "After augmented training",
        final['roc_auc'], final['pr_auc'], final['best_f1']))
    delta_f1 = final['best_f1'] - baseline['best_f1']
    print("\n  Delta F1: %+.4f" % delta_f1)
    print("  Best val loss: %.4f" % best_val_loss)
    print("  Checkpoint: %s" % ckpt_path)

    print("\nDone.")


if __name__ == '__main__':
    main()
