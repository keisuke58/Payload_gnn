# -*- coding: utf-8 -*-
"""FNO → GNN Knowledge Distillation (Stage 2).

The FNO has learned the PDE solution operator in Fourier space.
This script transfers that physics knowledge into the GNN encoder
by training it to predict the FNO's stress field output.

The FNO operates on 64×64 regular grids; the GNN on irregular meshes.
Grid-to-node bilinear interpolation bridges the gap.

Usage:
    python src/prad/distill_fno2gnn.py \
        --mae_checkpoint checkpoints/prad_mae_sage.pt \
        --fno_checkpoint checkpoints/fno_best.pt \
        --data_dir data/processed_s12_mixed_400 \
        --fno_grid_dir data/fno_grids_mixed_400
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from prad.graphmae import PIGraphMAE


def interpolate_grid_to_nodes(grid, node_pos, theta_min=0.0, theta_max=30.0,
                              y_min=0.0, y_max=5000.0):
    """Bilinear interpolation from FNO 64×64 grid to graph node positions.

    The FNO grid axes are:
        - axis 0 (rows): y (axial position), normalized [0, 1]
        - axis 1 (cols): theta (circumferential), normalized [0, 1]

    Args:
        grid: (H, W) or (1, H, W) stress values on regular grid.
        node_pos: (N, 3) node positions (x, y, z) in mm.
        theta_min, theta_max: circumferential angle range (degrees).
        y_min, y_max: axial position range (mm).

    Returns:
        values: (N,) interpolated stress at each node position.
    """
    if grid.dim() == 3:
        grid = grid.squeeze(0)  # (H, W)
    H, W = grid.shape

    # Compute cylindrical coordinates from Cartesian
    x_coord = node_pos[:, 0]
    z_coord = node_pos[:, 2]
    y_coord = node_pos[:, 1]  # axial

    theta = torch.atan2(z_coord, x_coord) * 180.0 / math.pi  # degrees

    # Normalize to [0, 1]
    theta_norm = (theta - theta_min) / (theta_max - theta_min + 1e-8)
    y_norm = (y_coord - y_min) / (y_max - y_min + 1e-8)

    # Clamp to grid bounds
    theta_norm = theta_norm.clamp(0, 1)
    y_norm = y_norm.clamp(0, 1)

    # Convert to grid coordinates
    gx = theta_norm * (W - 1)  # column (theta)
    gy = y_norm * (H - 1)      # row (y)

    # Bilinear interpolation
    gx0 = gx.long().clamp(0, W - 2)
    gy0 = gy.long().clamp(0, H - 2)
    gx1 = gx0 + 1
    gy1 = gy0 + 1

    wx = gx - gx0.float()
    wy = gy - gy0.float()

    v00 = grid[gy0, gx0]
    v01 = grid[gy0, gx1]
    v10 = grid[gy1, gx0]
    v11 = grid[gy1, gx1]

    values = (v00 * (1 - wx) * (1 - wy) +
              v01 * wx * (1 - wy) +
              v10 * (1 - wx) * wy +
              v11 * wx * wy)

    return values


class StressPredictor(nn.Module):
    """GNN encoder + stress prediction head for distillation."""

    def __init__(self, mae_model):
        super().__init__()
        self.encoder = mae_model.encoder
        self.stress_head = nn.Sequential(
            nn.Linear(mae_model.hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        edge_attr = getattr(data, 'edge_attr', None)
        h = self.encoder.encode(data.x, data.edge_index, edge_attr)
        return self.stress_head(h).squeeze(-1)  # (N,)


def load_fno_teacher(fno_checkpoint, device):
    """Load trained FNO model (frozen)."""
    from models_fno import FNO2d

    fno = FNO2d(modes1=12, modes2=12, width=32,
                in_channels=4, out_channels=1)
    ckpt = torch.load(fno_checkpoint, map_location=device,
                      weights_only=False)
    if 'model_state_dict' in ckpt:
        fno.load_state_dict(ckpt['model_state_dict'])
    else:
        fno.load_state_dict(ckpt)
    fno.eval()
    fno.to(device)

    # Freeze
    for p in fno.parameters():
        p.requires_grad = False

    return fno


def main():
    parser = argparse.ArgumentParser(
        description='FNO→GNN Knowledge Distillation (PRAD Stage 2)')
    parser.add_argument('--mae_checkpoint',
                        default='checkpoints/prad_mae_sage.pt')
    parser.add_argument('--fno_checkpoint',
                        default='checkpoints/fno_best.pt')
    parser.add_argument('--data_dir',
                        default='data/processed_s12_mixed_400')
    parser.add_argument('--fno_grid_dir',
                        default='data/fno_grids_mixed_400')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--log_dir', default='runs/prad_distill')
    parser.add_argument('--output', default='checkpoints/prad_distilled.pt')
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 60)
    print("PRAD Stage 2: FNO → GNN Distillation")
    print("=" * 60)

    # Check FNO checkpoint exists
    if not os.path.exists(args.fno_checkpoint):
        print("\n  FNO checkpoint not found: %s" % args.fno_checkpoint)
        print("  Stage 2 requires a trained FNO model.")
        print("  Train FNO first: python src/train_fno.py")
        print("  Skipping distillation — Stage 1 model can still be used")
        print("  for anomaly scoring without FNO distillation.")
        return

    # Load MAE checkpoint
    ckpt = torch.load(args.mae_checkpoint, map_location=device,
                      weights_only=False)
    mae_args = ckpt['args']
    mae = PIGraphMAE(
        encoder_arch=mae_args['encoder_arch'],
        hidden_channels=mae_args['hidden'],
        num_layers=mae_args['num_layers'],
        dropout=mae_args['dropout'],
        mask_ratio=mae_args['mask_ratio'],
        lambda_physics=mae_args['lambda_physics'],
    ).to(device)
    mae.load_state_dict(ckpt['model_state_dict'])
    print("  MAE loaded: %s" % args.mae_checkpoint)

    # Build student: encoder + stress head
    student = StressPredictor(mae).to(device)

    # Load FNO teacher
    fno = load_fno_teacher(args.fno_checkpoint, device)
    print("  FNO teacher loaded: %s" % args.fno_checkpoint)

    # Load graph data
    from prad.train_mae import load_data
    train_data, val_data = load_data(args.data_dir, device)
    print("  Graphs: %d train, %d val" % (len(train_data), len(val_data)))

    # Load FNO grid data
    fno_inputs = np.load(os.path.join(args.fno_grid_dir, 'inputs.npy'))
    fno_inputs = torch.from_numpy(fno_inputs).float().to(device)
    with open(os.path.join(args.fno_grid_dir, 'meta.json')) as f:
        fno_meta = json.load(f)
    print("  FNO grids: %s" % str(fno_inputs.shape))

    # Get grid coordinate ranges from meta
    y_max = 5000.0  # default barrel height
    theta_max = 30.0  # default sector angle

    # Optimizer (only train stress_head + fine-tune encoder)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    n_graphs = min(len(train_data), fno_inputs.size(0))
    best_val_r2 = -float('inf')

    print("\nDistilling...")
    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0

        for i in range(n_graphs):
            data = train_data[i]
            grid_input = fno_inputs[i:i + 1]  # (1, 4, 64, 64)

            # FNO teacher prediction
            with torch.no_grad():
                fno_grid = fno(grid_input)  # (1, 1, 64, 64)
                fno_stress = fno_grid.squeeze(0).squeeze(0)  # (64, 64)

            # Interpolate FNO grid → graph nodes
            target = interpolate_grid_to_nodes(
                fno_stress, data.x[:, :3],
                theta_max=theta_max, y_max=y_max)

            # GNN student prediction
            pred = student(data)

            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(n_graphs, 1)
        writer.add_scalar('distill/train_loss', avg_loss, epoch)

        # Validation R²
        if epoch % 10 == 0:
            student.eval()
            all_pred, all_target = [], []
            with torch.no_grad():
                n_val = min(len(val_data), fno_inputs.size(0) - n_graphs)
                for i in range(n_val):
                    data = val_data[i]
                    gi = n_graphs + i
                    if gi >= fno_inputs.size(0):
                        break
                    grid_input = fno_inputs[gi:gi + 1]
                    fno_stress = fno(grid_input).squeeze()
                    target = interpolate_grid_to_nodes(
                        fno_stress, data.x[:, :3],
                        theta_max=theta_max, y_max=y_max)
                    pred = student(data)
                    all_pred.append(pred.cpu())
                    all_target.append(target.cpu())

            if all_pred:
                pred_cat = torch.cat(all_pred)
                tgt_cat = torch.cat(all_target)
                ss_res = ((pred_cat - tgt_cat) ** 2).sum()
                ss_tot = ((tgt_cat - tgt_cat.mean()) ** 2).sum()
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                r2 = r2.item()
                writer.add_scalar('distill/val_r2', r2, epoch)

                if r2 > best_val_r2:
                    best_val_r2 = r2
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': student.state_dict(),
                        'encoder_state_dict': student.encoder.state_dict(),
                        'val_r2': r2,
                        'mae_args': mae_args,
                    }, args.output)

                print("  Epoch %3d  loss=%.6f  R²=%.4f%s" % (
                    epoch, avg_loss, r2,
                    ' *' if r2 == best_val_r2 else ''))

    writer.close()
    print("\nBest val R²: %.4f" % best_val_r2)
    print("Checkpoint: %s" % args.output)


if __name__ == '__main__':
    main()
