# -*- coding: utf-8 -*-
"""Train GPS Graph Transformer MAE — Pre-training + Fine-tuning pipeline.

Phase 1: Self-supervised pre-training with GPS+MAE on combined
         fairing + CompDam data
Phase 2: Fine-tune encoder + classifier on labeled fairing data

Usage:
    # Phase 1: Pre-training (on GPU server)
    python src/prad/train_gtmae.py pretrain \
        --data_dir data/processed_s12_mixed_400 \
        --compdam_graphs data/compdam_graphs.pt \
        --hidden 128 --num_layers 4 --heads 4 \
        --epochs 300 --lr 1e-3

    # Phase 2: Fine-tuning (uses pre-trained encoder)
    python src/prad/train_gtmae.py finetune \
        --data_dir data/processed_s12_mixed_400 \
        --pretrained checkpoints/gtmae_gps.pt \
        --epochs 100 --lr 5e-4
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from prad.graphmae import PIGraphMAE
from models import build_model


def load_fairing_data(data_dir, device='cpu'):
    """Load fairing PyG data with normalization."""
    train_data = torch.load(os.path.join(data_dir, 'train.pt'),
                            weights_only=False)
    val_data = torch.load(os.path.join(data_dir, 'val.pt'),
                          weights_only=False)

    norm_path = os.path.join(data_dir, 'norm_stats.pt')
    norm_stats = None
    if os.path.exists(norm_path):
        norm_stats = torch.load(norm_path, weights_only=False)
        x_mean = norm_stats['mean'].to(device)
        x_std = norm_stats['std'].to(device)
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

    return train_data, val_data, norm_stats


def load_compdam_data(compdam_path, device='cpu', norm_stats=None):
    """Load CompDam PyG graphs with optional normalization."""
    if not os.path.exists(compdam_path):
        print("  CompDam graphs not found: %s" % compdam_path)
        return []

    data = torch.load(compdam_path, weights_only=False)
    graphs = data['graphs']
    print("  CompDam graphs loaded: %d" % len(graphs))

    for g in graphs:
        g.x = g.x.to(device)
        g.edge_index = g.edge_index.to(device)
        if g.edge_attr is not None:
            g.edge_attr = g.edge_attr.to(device)
        g.y = g.y.to(device)

        # Apply normalization if available
        if norm_stats is not None:
            x_mean = norm_stats['mean'].to(device)
            x_std = norm_stats['std'].to(device)
            x_std[x_std < 1e-8] = 1.0
            g.x = (g.x - x_mean) / x_std

    return graphs


# ============================================================
# Phase 1: Pre-training
# ============================================================

def pretrain(args):
    """Self-supervised GPS+MAE pre-training."""
    device = torch.device(args.device)
    print("=" * 60)
    print("Phase 1: GPS Graph Transformer MAE Pre-training")
    print("=" * 60)

    # Load data
    train_data, val_data, norm_stats = load_fairing_data(
        args.data_dir, device)
    print("  Fairing: %d train, %d val" % (len(train_data), len(val_data)))

    compdam_data = []
    if args.compdam_graphs:
        compdam_data = load_compdam_data(args.compdam_graphs, device,
                                         norm_stats)

    # Combine for pre-training (all data, self-supervised)
    all_train = train_data + compdam_data
    print("  Combined pre-training data: %d graphs" % len(all_train))

    if all_train:
        print("  nodes/graph (first): %d" % all_train[0].x.size(0))
        print("  features: %d" % all_train[0].x.size(1))

    # Build GPS+MAE model
    model = PIGraphMAE(
        encoder_arch='gps',
        in_channels=all_train[0].x.size(1),
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        lambda_physics=args.lambda_physics,
        decoder_type='mlp',
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("  GPS+MAE parameters: %d (%.1fK)" % (n_params, n_params / 1000))

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Logging
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    ckpt_path = os.path.join(args.checkpoint_dir, 'gtmae_gps.pt')
    best_val_loss = float('inf')
    patience_counter = 0

    print("\nPre-training...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss = 0.0
        n = 0
        # Shuffle training data each epoch
        perm = torch.randperm(len(all_train))
        for idx in perm:
            data = all_train[idx]
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        train_loss = total_loss / max(n, 1)

        # Eval (on fairing val only)
        model.eval()
        val_loss = 0.0
        val_cos = 0.0
        n_val = 0
        with torch.no_grad():
            for data in val_data:
                loss, x_recon = model(data, return_recon=True)
                val_loss += loss.item()
                cos_sim = F.cosine_similarity(x_recon, data.x, dim=1).mean().item()
                val_cos += cos_sim
                n_val += 1
        val_loss /= max(n_val, 1)
        val_cos /= max(n_val, 1)

        scheduler.step()
        dt = time.time() - t0

        writer.add_scalar('pretrain/train_loss', train_loss, epoch)
        writer.add_scalar('pretrain/val_loss', val_loss, epoch)
        writer.add_scalar('pretrain/val_cos_sim', val_cos, epoch)

        improved = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'val_loss': val_loss,
                'val_cos_sim': val_cos,
                'args': vars(args),
            }, ckpt_path)
            improved = ' *'
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or improved:
            print("  Epoch %3d/%d  train=%.4f  val=%.4f  cos=%.4f  "
                  "%.1fs%s" % (epoch, args.epochs, train_loss, val_loss,
                               val_cos, dt, improved))

        if patience_counter >= args.patience:
            print("\n  Early stopping at epoch %d" % epoch)
            break

    writer.close()
    print("\nBest val loss: %.4f" % best_val_loss)
    print("Checkpoint: %s" % ckpt_path)
    return ckpt_path


# ============================================================
# Phase 2: Fine-tuning
# ============================================================

def finetune(args):
    """Fine-tune pre-trained GPS encoder for defect classification."""
    device = torch.device(args.device)
    print("=" * 60)
    print("Phase 2: Fine-tuning GPS encoder for defect detection")
    print("=" * 60)

    # Load data
    train_data, val_data, _ = load_fairing_data(args.data_dir, device)
    print("  Train: %d graphs, Val: %d graphs" % (len(train_data), len(val_data)))

    # Build GPS classifier model
    in_channels = train_data[0].x.size(1)
    model = build_model(
        'gps',
        in_channels=in_channels,
        hidden_channels=args.hidden,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
    ).to(device)

    # Load pre-trained encoder weights
    if args.pretrained:
        print("  Loading pre-trained: %s" % args.pretrained)
        ckpt = torch.load(args.pretrained, weights_only=False, map_location=device)
        if 'encoder_state_dict' in ckpt:
            enc_state = ckpt['encoder_state_dict']
        else:
            # Extract encoder keys from full model state
            enc_state = {k.replace('encoder.', ''): v
                         for k, v in ckpt['model_state_dict'].items()
                         if k.startswith('encoder.')}

        # Load with strict=False (head weights will be missing)
        missing, unexpected = model.load_state_dict(enc_state, strict=False)
        print("  Loaded encoder: %d keys, missing=%d (head), unexpected=%d" %
              (len(enc_state), len(missing), len(unexpected)))

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("  Total params: %d, Trainable: %d" % (n_params, n_trainable))

    # Optimizer: lower lr for encoder, higher for head
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'head' in name:
            head_params.append(param)
        else:
            encoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.lr * 0.1},  # encoder: lower lr
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Class weights for imbalanced data
    all_y = torch.cat([d.y for d in train_data])
    n_total = len(all_y)
    n_pos = (all_y > 0).sum().item()
    n_neg = n_total - n_pos
    if n_pos > 0:
        pos_weight = n_neg / n_pos
    else:
        pos_weight = 1.0
    weights = torch.tensor([1.0, min(pos_weight, 10.0)], device=device)
    print("  Class weights: [%.2f, %.2f]" % (weights[0], weights[1]))

    # Logging
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    ckpt_path = os.path.join(args.checkpoint_dir, 'gtmae_gps_finetuned.pt')
    best_auroc = 0.0
    patience_counter = 0

    print("\nFine-tuning...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss = 0.0
        n = 0
        for data in train_data:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out, data.y, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        train_loss = total_loss / max(n, 1)

        # Eval
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []
        n_val = 0
        with torch.no_grad():
            for data in val_data:
                out = model(data.x, data.edge_index)
                loss = F.cross_entropy(out, data.y, weight=weights)
                val_loss += loss.item()
                probs = F.softmax(out, dim=1)[:, 1]
                all_probs.append(probs.cpu())
                all_labels.append(data.y.cpu())
                n_val += 1
        val_loss /= max(n_val, 1)

        # AUROC
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc = 0.0

        scheduler.step()
        dt = time.time() - t0

        writer.add_scalar('finetune/train_loss', train_loss, epoch)
        writer.add_scalar('finetune/val_loss', val_loss, epoch)
        writer.add_scalar('finetune/auroc', auroc, epoch)

        improved = ''
        if auroc > best_auroc:
            best_auroc = auroc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'auroc': auroc,
                'val_loss': val_loss,
                'args': vars(args),
            }, ckpt_path)
            improved = ' *'
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or improved:
            print("  Epoch %3d/%d  train=%.4f  val=%.4f  AUROC=%.4f  "
                  "%.1fs%s" % (epoch, args.epochs, train_loss, val_loss,
                               auroc, dt, improved))

        if patience_counter >= args.patience:
            print("\n  Early stopping at epoch %d" % epoch)
            break

    writer.close()
    print("\nBest AUROC: %.4f" % best_auroc)
    print("Checkpoint: %s" % ckpt_path)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='GPS Graph Transformer MAE: Pre-train + Fine-tune')
    sub = parser.add_subparsers(dest='phase')

    # Pretrain
    p_pre = sub.add_parser('pretrain', help='Self-supervised pre-training')
    p_pre.add_argument('--data_dir', default='data/processed_s12_mixed_400')
    p_pre.add_argument('--compdam_graphs', default='',
                       help='Path to CompDam PyG graphs (optional)')
    p_pre.add_argument('--hidden', type=int, default=128)
    p_pre.add_argument('--num_layers', type=int, default=4)
    p_pre.add_argument('--heads', type=int, default=4)
    p_pre.add_argument('--dropout', type=float, default=0.1)
    p_pre.add_argument('--mask_ratio', type=float, default=0.5)
    p_pre.add_argument('--lambda_physics', type=float, default=0.1)
    p_pre.add_argument('--epochs', type=int, default=300)
    p_pre.add_argument('--lr', type=float, default=1e-3)
    p_pre.add_argument('--weight_decay', type=float, default=1e-5)
    p_pre.add_argument('--patience', type=int, default=40)
    p_pre.add_argument('--device', default='cuda' if torch.cuda.is_available()
                       else 'cpu')
    p_pre.add_argument('--log_dir', default='runs/gtmae_pretrain')
    p_pre.add_argument('--checkpoint_dir', default='checkpoints')

    # Finetune
    p_ft = sub.add_parser('finetune', help='Supervised fine-tuning')
    p_ft.add_argument('--data_dir', default='data/processed_s12_mixed_400')
    p_ft.add_argument('--pretrained', default='checkpoints/gtmae_gps.pt')
    p_ft.add_argument('--hidden', type=int, default=128)
    p_ft.add_argument('--num_layers', type=int, default=4)
    p_ft.add_argument('--num_classes', type=int, default=2)
    p_ft.add_argument('--dropout', type=float, default=0.1)
    p_ft.add_argument('--epochs', type=int, default=100)
    p_ft.add_argument('--lr', type=float, default=5e-4)
    p_ft.add_argument('--weight_decay', type=float, default=1e-5)
    p_ft.add_argument('--patience', type=int, default=30)
    p_ft.add_argument('--device', default='cuda' if torch.cuda.is_available()
                       else 'cpu')
    p_ft.add_argument('--log_dir', default='runs/gtmae_finetune')
    p_ft.add_argument('--checkpoint_dir', default='checkpoints')

    args = parser.parse_args()

    if args.phase == 'pretrain':
        pretrain(args)
    elif args.phase == 'finetune':
        finetune(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
