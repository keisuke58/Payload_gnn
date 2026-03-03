# -*- coding: utf-8 -*-
"""train_fno.py — Train FNO2d surrogate on S12 CZM stress field data.

Learns the mapping: (z_norm, theta_norm, defect_mask, temp_norm) -> smises

Usage:
  python src/train_fno.py \
    --data_dir data/fno_grids_200 \
    --epochs 500 --batch_size 16 --lr 1e-3 \
    --run_name fno_surrogate_v1
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from models_fno import FNO2d


def relative_l2_loss(pred, target):
    """Relative L2 norm loss (standard for neural operators)."""
    diff = (pred - target).reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    return (diff.norm(dim=1) / (target_flat.norm(dim=1) + 1e-8)).mean()


def main():
    parser = argparse.ArgumentParser(description='Train FNO2d surrogate')
    parser.add_argument('--data_dir', type=str, default='data/fno_grids_200')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--modes', type=int, default=12)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--run_name', type=str, default='fno_surrogate_v1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    inputs = np.load(os.path.join(args.data_dir, 'inputs.npy'))   # (N, 4, H, W)
    outputs = np.load(os.path.join(args.data_dir, 'outputs.npy'))  # (N, 1, H, W)

    # Normalize outputs (smises)
    out_mean = outputs.mean()
    out_std = outputs.std() + 1e-8
    outputs_norm = (outputs - out_mean) / out_std

    N = inputs.shape[0]
    n_val = int(N * args.val_ratio)
    n_train = N - n_val

    # Shuffle and split
    idx = np.random.permutation(N)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    x_train = torch.from_numpy(inputs[train_idx]).float()
    y_train = torch.from_numpy(outputs_norm[train_idx]).float()
    x_val = torch.from_numpy(inputs[val_idx]).float()
    y_val = torch.from_numpy(outputs_norm[val_idx]).float()
    # Keep unnormalized for evaluation
    y_val_raw = torch.from_numpy(outputs[val_idx]).float()

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_ds = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {n_train}, Val: {n_val}")
    print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}")
    print(f"Output norm: mean={out_mean:.4f}, std={out_std:.4f}")

    # Model
    model = FNO2d(modes1=args.modes, modes2=args.modes, width=args.width,
                  in_channels=4, out_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FNO2d: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    save_dir = os.path.join('runs', args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_rel_l2': []}

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = relative_l2_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.shape[0]
        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = relative_l2_loss(pred, yb)
                val_loss += loss.item() * xb.shape[0]
        val_loss /= n_val

        # Relative L2 on unnormalized
        model.eval()
        val_preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                pred = model(xb)
                pred_raw = pred.cpu() * out_std + out_mean
                val_preds.append(pred_raw)
        val_preds = torch.cat(val_preds, dim=0)
        rel_l2_raw = relative_l2_loss(val_preds, y_val_raw).item()

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rel_l2'].append(rel_l2_raw)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rel_l2': rel_l2_raw,
                'out_mean': float(out_mean),
                'out_std': float(out_std),
                'args': vars(args),
            }, os.path.join(save_dir, 'best_model.pt'))
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d} | train={train_loss:.6f} val={val_loss:.6f} "
                  f"rel_l2={rel_l2_raw:.4f} | lr={lr:.2e} | {elapsed:.0f}s")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best val rel_l2: {min(history['val_rel_l2']):.4f}")

    # Save history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f)

    # Save normalization stats
    with open(os.path.join(save_dir, 'norm_stats.json'), 'w') as f:
        json.dump({'out_mean': float(out_mean), 'out_std': float(out_std)}, f)

    print(f"Saved to: {save_dir}")


if __name__ == '__main__':
    main()
