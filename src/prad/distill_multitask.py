# -*- coding: utf-8 -*-
"""Multi-task Distillation: MAE reconstruction + FNO stress prediction.

Unlike sequential distillation (Stage 2 then fine-tune), this trains
the encoder with BOTH objectives simultaneously:
  L_total = L_mae + lambda_distill * L_stress

This prevents encoder representation collapse: the reconstruction
objective keeps the encoder's 34-dim feature space intact, while
the stress prediction transfers FNO physics knowledge.

Usage:
    python src/prad/distill_multitask.py \
        --mae_checkpoint checkpoints/prad_mae_sage.pt \
        --fno_checkpoint runs/fno_production/best_model.pt \
        --data_dir data/processed_s12_czm_thermal_200_binary \
        --fno_grid_dir data/fno_grids_200
"""

import argparse
import json
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
from prad.distill_fno2gnn import (
    interpolate_grid_to_nodes, load_fno_teacher, reconstruct_split_indices)
from prad.anomaly_score import score_dataset
from prad.eval_prad import compute_metrics, plot_roc_pr, plot_residual_tsne
from prad.train_mae import load_data, eval_epoch


class StressHead(nn.Module):
    """Lightweight stress prediction head attached to encoder."""

    def __init__(self, hidden_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h):
        return self.head(h).squeeze(-1)  # (N,)


def train_multitask_epoch(model, stress_head, train_data, fno_stress_cache,
                          train_idx, optimizer, lambda_distill, device):
    """One epoch of multi-task training.

    For each graph:
    1. MAE forward: mask → encode → decode → reconstruction loss
    2. Stress prediction: encode (no mask) → stress head → MSE vs FNO target
    3. Combined loss = L_mae + lambda_distill * L_stress
    """
    model.train()
    stress_head.train()

    total_mae_loss = 0.0
    total_stress_loss = 0.0
    n = len(train_data)
    perm = torch.randperm(n)

    for j in perm:
        data = train_data[j]
        fno_idx = train_idx[j]

        # --- MAE loss (masked reconstruction) ---
        mae_loss = model(data)

        # --- Stress prediction loss ---
        edge_attr = getattr(data, 'edge_attr', None)
        h = model.encoder.encode(data.x, data.edge_index, edge_attr)
        stress_pred = stress_head(h)

        # FNO target (pre-computed, interpolated to nodes)
        stress_target = fno_stress_cache[fno_idx].to(device)
        stress_loss = F.mse_loss(stress_pred, stress_target)

        # --- Combined loss ---
        loss = mae_loss + lambda_distill * stress_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(stress_head.parameters()),
            max_norm=1.0)
        optimizer.step()

        total_mae_loss += mae_loss.item()
        total_stress_loss += stress_loss.item()

    return total_mae_loss / n, total_stress_loss / n


def evaluate_anomaly(model, val_data, alpha=1.0, smooth_rounds=1,
                     smooth_alpha=0.5):
    """Quick anomaly detection evaluation."""
    scores_list, labels_list, _ = score_dataset(
        model, val_data, alpha=alpha,
        smooth_rounds=smooth_rounds, smooth_alpha=smooth_alpha)
    scores_np = torch.cat(scores_list).numpy()
    labels_np = torch.cat(labels_list).numpy()
    return compute_metrics(scores_np, labels_np)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-task Distillation: MAE + FNO stress')
    parser.add_argument('--mae_checkpoint',
                        default='checkpoints/prad_mae_sage.pt')
    parser.add_argument('--fno_checkpoint',
                        default='runs/fno_production/best_model.pt')
    parser.add_argument('--data_dir',
                        default='data/processed_s12_czm_thermal_200_binary')
    parser.add_argument('--fno_grid_dir',
                        default='data/fno_grids_200')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lambda_distill', type=float, default=0.1,
                        help='Weight for stress distillation loss '
                             '(stress targets are normalized to unit variance)')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--output', default='checkpoints/prad_mae_multitask.pt')
    parser.add_argument('--log_dir', default='runs/prad_multitask')
    parser.add_argument('--output_dir', default='figures/prad_multitask')
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    # Scoring params (best from Stage 1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--smooth_rounds', type=int, default=1)
    parser.add_argument('--smooth_alpha', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 60)
    print("PRAD: Multi-task Distillation (MAE + FNO)")
    print("=" * 60)
    print("  lambda_distill: %.4f" % args.lambda_distill)

    # ---- Load original MAE ----
    ckpt = torch.load(args.mae_checkpoint, map_location=device,
                      weights_only=False)
    mae_args = ckpt['args']
    model = PIGraphMAE(
        encoder_arch=mae_args['encoder_arch'],
        hidden_channels=mae_args['hidden'],
        num_layers=mae_args['num_layers'],
        dropout=mae_args['dropout'],
        mask_ratio=mae_args['mask_ratio'],
        lambda_physics=mae_args['lambda_physics'],
        decoder_type=mae_args.get('decoder_type', 'mlp'),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print("  MAE loaded: %s" % args.mae_checkpoint)

    # ---- Stress prediction head ----
    stress_head = StressHead(mae_args['hidden']).to(device)

    # ---- Load data ----
    train_data, val_data = load_data(args.data_dir, device)
    n_total = len(train_data) + len(val_data)
    print("  Graphs: %d train, %d val" % (len(train_data), len(val_data)))

    # ---- Load FNO teacher + grids ----
    fno, fno_out_mean, fno_out_std = load_fno_teacher(
        args.fno_checkpoint, device)
    print("  FNO teacher loaded (norm: mean=%.4f, std=%.4f)" % (
        fno_out_mean, fno_out_std))

    fno_inputs = np.load(os.path.join(args.fno_grid_dir, 'inputs.npy'))
    fno_inputs = torch.from_numpy(fno_inputs).float().to(device)
    assert fno_inputs.size(0) == n_total

    # ---- Reconstruct split indices ----
    train_idx, val_idx = reconstruct_split_indices(
        n_total, val_ratio=args.val_ratio, seed=args.split_seed)

    # ---- Pre-compute FNO stress targets (interpolated to graph nodes) ----
    print("\n  Pre-computing FNO stress targets...")
    t0 = time.time()
    with torch.no_grad():
        fno_outputs = fno(fno_inputs)  # (N, 1, 64, 64)
        if fno_out_mean is not None:
            fno_outputs = fno_outputs * fno_out_std + fno_out_mean
    fno_outputs = fno_outputs.squeeze(1)  # (N, 64, 64)

    # Get coordinate ranges
    y_max = train_data[0].pos[:, 1].max().item() + 1.0
    theta_max = 30.0

    # Pre-interpolate all FNO grids to graph node positions
    fno_stress_cache = {}
    all_targets = []
    all_indices = list(range(len(train_data))) + list(range(len(val_data)))
    all_idx_maps = list(train_idx) + list(val_idx)
    all_data_refs = list(train_data) + list(val_data)

    for k, (data_ref, fno_idx) in enumerate(
            zip(all_data_refs, all_idx_maps)):
        pos = data_ref.pos.to(device)
        target = interpolate_grid_to_nodes(
            fno_outputs[fno_idx], pos,
            theta_max=theta_max, y_max=y_max)
        all_targets.append(target)
        fno_stress_cache[fno_idx] = target.cpu()

    # Normalize stress targets to zero mean, unit std
    # This makes stress loss scale comparable to MAE cosine loss (~0.01)
    all_cat = torch.cat(all_targets)
    stress_mean = all_cat.mean().item()
    stress_std = all_cat.std().item() + 1e-8
    for fno_idx in fno_stress_cache:
        fno_stress_cache[fno_idx] = (
            fno_stress_cache[fno_idx] - stress_mean) / stress_std
    print("  Stress normalization: mean=%.2f, std=%.2f" % (
        stress_mean, stress_std))

    # Free FNO from GPU
    del fno, fno_outputs, fno_inputs, all_targets, all_cat
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("  Done (%.1fs). Cached %d normalized stress targets." % (
        time.time() - t0, len(fno_stress_cache)))

    # ---- Baseline evaluation (Stage 1 only) ----
    print("\n  Baseline (Stage 1 only):")
    baseline = evaluate_anomaly(
        model, val_data, alpha=args.alpha,
        smooth_rounds=args.smooth_rounds, smooth_alpha=args.smooth_alpha)
    print("    ROC=%.4f  PR=%.4f  F1=%.4f" % (
        baseline['roc_auc'], baseline['pr_auc'], baseline['best_f1']))

    # ---- Training ----
    all_params = list(model.parameters()) + list(stress_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    best_val_loss = float('inf')
    best_f1 = 0.0
    patience_counter = 0
    best_state = None

    print("\nTraining (multi-task)...")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        mae_loss, stress_loss = train_multitask_epoch(
            model, stress_head, train_data, fno_stress_cache,
            train_idx, optimizer, args.lambda_distill, device)
        val_loss, val_cos_sim = eval_epoch(model, val_data)
        scheduler.step()

        combined = mae_loss + args.lambda_distill * stress_loss
        writer.add_scalar('loss/mae', mae_loss, epoch)
        writer.add_scalar('loss/stress', stress_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('metric/cos_sim', val_cos_sim, epoch)

        improved = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            improved = ' *'
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or improved:
            elapsed = time.time() - t_start
            print("  Epoch %3d  mae=%.4f  stress=%.2f  val=%.4f  cos=%.4f  "
                  "(%.0fs)%s" % (
                      epoch, mae_loss, stress_loss, val_loss,
                      val_cos_sim, elapsed, improved))

        # Evaluate anomaly detection periodically
        if epoch % 20 == 0:
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

        if patience_counter >= args.patience:
            print("  Early stopping at epoch %d" % epoch)
            break

    writer.close()

    # ---- Load best state and final evaluation ----
    model.load_state_dict(best_state)

    torch.save({
        'model_state_dict': best_state,
        'args': mae_args,
        'val_loss': best_val_loss,
        'lambda_distill': args.lambda_distill,
    }, args.output)
    print("\n  Saved: %s" % args.output)

    # Final evaluation
    print("\n--- Final Evaluation ---")
    scores_list, labels_list, residuals_list = score_dataset(
        model, val_data, alpha=args.alpha,
        smooth_rounds=args.smooth_rounds,
        smooth_alpha=args.smooth_alpha)
    scores_np = torch.cat(scores_list).numpy()
    labels_np = torch.cat(labels_list).numpy()
    final = compute_metrics(scores_np, labels_np)

    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print("  %-30s  ROC-AUC  PR-AUC  F1" % "")
    print("  " + "-" * 55)
    print("  %-30s  %.4f   %.4f  %.4f" % (
        "Stage 1 only (baseline)",
        baseline['roc_auc'], baseline['pr_auc'], baseline['best_f1']))
    print("  %-30s  %.4f   %.4f  %.4f" % (
        "Multi-task (MAE + FNO)",
        final['roc_auc'], final['pr_auc'], final['best_f1']))

    delta_f1 = final['best_f1'] - baseline['best_f1']
    delta_pr = final['pr_auc'] - baseline['pr_auc']
    print("\n  Delta: F1 %+.4f, PR-AUC %+.4f" % (delta_f1, delta_pr))

    # Plots
    os.makedirs(args.output_dir, exist_ok=True)
    residuals_np = torch.cat(residuals_list).numpy()
    plot_roc_pr(scores_np, labels_np, args.output_dir,
                prefix='prad_multitask')
    plot_residual_tsne(residuals_np, labels_np, args.output_dir,
                       prefix='prad_multitask')

    print("\nDone.")


if __name__ == '__main__':
    main()
