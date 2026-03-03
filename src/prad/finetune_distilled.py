# -*- coding: utf-8 -*-
"""Fine-tune MAE with distilled encoder + evaluate anomaly detection.

Loads the Stage 1 MAE and replaces its encoder with the Stage 2
distilled encoder (which has internalized FNO physics knowledge).
Then fine-tunes the decoder to adapt to the new encoder representations.

Evaluation compares:
  A) Stage 1 only (original MAE)
  B) Distilled encoder, no fine-tune (direct swap)
  C) Distilled encoder + decoder fine-tune

Usage:
    python src/prad/finetune_distilled.py \
        --mae_checkpoint checkpoints/prad_mae_sage.pt \
        --distilled_checkpoint checkpoints/prad_distilled.pt \
        --data_dir data/processed_s12_czm_thermal_200_binary
"""

import argparse
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from prad.graphmae import PIGraphMAE
from prad.anomaly_score import score_dataset
from prad.eval_prad import compute_metrics, plot_roc_pr, plot_residual_tsne
from prad.train_mae import load_data, train_epoch, eval_epoch


def evaluate_model(model, val_data, label, alpha=1.0,
                   smooth_rounds=1, smooth_alpha=0.5):
    """Score and compute metrics."""
    scores_list, labels_list, residuals_list = score_dataset(
        model, val_data, alpha=alpha,
        smooth_rounds=smooth_rounds, smooth_alpha=smooth_alpha)

    scores_np = torch.cat(scores_list).numpy()
    labels_np = torch.cat(labels_list).numpy()

    metrics = compute_metrics(scores_np, labels_np)

    print("  [%s]" % label)
    print("    ROC-AUC: %.4f  PR-AUC: %.4f  F1: %.4f  P: %.3f  R: %.3f" % (
        metrics['roc_auc'], metrics['pr_auc'], metrics['best_f1'],
        metrics['precision'], metrics['recall']))

    return metrics, scores_np, labels_np, residuals_list


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune MAE with distilled encoder')
    parser.add_argument('--mae_checkpoint',
                        default='checkpoints/prad_mae_sage.pt')
    parser.add_argument('--distilled_checkpoint',
                        default='checkpoints/prad_distilled.pt')
    parser.add_argument('--data_dir',
                        default='data/processed_s12_czm_thermal_200_binary')
    parser.add_argument('--finetune_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--output', default='checkpoints/prad_mae_distilled.pt')
    parser.add_argument('--output_dir', default='figures/prad_distilled')
    # Best scoring params from Stage 1 grid search
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--smooth_rounds', type=int, default=1)
    parser.add_argument('--smooth_alpha', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 60)
    print("PRAD: Distilled Encoder Fine-tune & Evaluation")
    print("=" * 60)

    # ---- Load original MAE ----
    ckpt = torch.load(args.mae_checkpoint, map_location=device,
                      weights_only=False)
    mae_args = ckpt['args']

    def build_mae():
        return PIGraphMAE(
            encoder_arch=mae_args['encoder_arch'],
            hidden_channels=mae_args['hidden'],
            num_layers=mae_args['num_layers'],
            dropout=mae_args['dropout'],
            mask_ratio=mae_args['mask_ratio'],
            lambda_physics=mae_args['lambda_physics'],
            decoder_type=mae_args.get('decoder_type', 'mlp'),
        ).to(device)

    # ---- Load data ----
    train_data, val_data = load_data(args.data_dir, device)
    print("  Data: %d train, %d val" % (len(train_data), len(val_data)))

    # ---- Load distilled encoder ----
    dist_ckpt = torch.load(args.distilled_checkpoint, map_location=device,
                           weights_only=False)
    print("  Distilled encoder: R²=%.4f (epoch %d)" % (
        dist_ckpt['val_r2'], dist_ckpt['epoch']))

    # ================================================================
    # A) Baseline: original MAE (Stage 1 only)
    # ================================================================
    print("\n--- Evaluation ---")
    mae_orig = build_mae()
    mae_orig.load_state_dict(ckpt['model_state_dict'])
    metrics_a, _, _, _ = evaluate_model(
        mae_orig, val_data, "A: Stage 1 only",
        alpha=args.alpha, smooth_rounds=args.smooth_rounds,
        smooth_alpha=args.smooth_alpha)
    del mae_orig

    # ================================================================
    # B) Direct swap: distilled encoder + original decoder (no fine-tune)
    # ================================================================
    mae_swap = build_mae()
    mae_swap.load_state_dict(ckpt['model_state_dict'])
    # Replace encoder with distilled version
    mae_swap.encoder.load_state_dict(dist_ckpt['encoder_state_dict'])
    metrics_b, _, _, _ = evaluate_model(
        mae_swap, val_data, "B: Distilled encoder (no fine-tune)",
        alpha=args.alpha, smooth_rounds=args.smooth_rounds,
        smooth_alpha=args.smooth_alpha)
    del mae_swap

    # ================================================================
    # C) Distilled encoder + decoder fine-tune
    # ================================================================
    print("\n--- Fine-tuning decoder ---")
    mae_ft = build_mae()
    mae_ft.load_state_dict(ckpt['model_state_dict'])
    mae_ft.encoder.load_state_dict(dist_ckpt['encoder_state_dict'])

    # Freeze encoder, only train decoder + mask tokens
    for p in mae_ft.encoder.parameters():
        p.requires_grad = False
    trainable = (list(mae_ft.decoder.parameters()) +
                 list(mae_ft.enc_mask_token.parameters()) +
                 list(mae_ft.dec_mask_token.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.finetune_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(mae_ft, train_data, optimizer)
        val_loss, val_cos_sim = eval_epoch(mae_ft, val_data)
        scheduler.step()
        dt = time.time() - t0

        improved = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in mae_ft.state_dict().items()}
            improved = ' *'
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            print("  Epoch %3d  train=%.4f  val=%.4f  cos=%.4f  %.1fs%s" % (
                epoch, train_loss, val_loss, val_cos_sim, dt, improved))

        if patience_counter >= args.patience:
            print("  Early stopping at epoch %d" % epoch)
            break

    # Load best state
    mae_ft.load_state_dict(best_state)

    # Save checkpoint
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'model_state_dict': best_state,
        'args': mae_args,
        'val_loss': best_val_loss,
        'distilled_r2': dist_ckpt['val_r2'],
    }, args.output)
    print("  Saved: %s" % args.output)

    # Evaluate
    metrics_c, scores_c, labels_c, residuals_c = evaluate_model(
        mae_ft, val_data, "C: Distilled + fine-tuned",
        alpha=args.alpha, smooth_rounds=args.smooth_rounds,
        smooth_alpha=args.smooth_alpha)

    # Also try unfreezing encoder for a few more epochs
    print("\n--- Fine-tuning encoder + decoder ---")
    for p in mae_ft.encoder.parameters():
        p.requires_grad = True
    optimizer2 = torch.optim.AdamW(mae_ft.parameters(), lr=args.lr * 0.1)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=10)

    best_val_loss_d = best_val_loss
    best_state_d = best_state

    for epoch in range(1, 11):
        train_loss = train_epoch(mae_ft, train_data, optimizer2)
        val_loss, val_cos_sim = eval_epoch(mae_ft, val_data)
        scheduler2.step()

        improved = ''
        if val_loss < best_val_loss_d:
            best_val_loss_d = val_loss
            best_state_d = {k: v.clone() for k, v in mae_ft.state_dict().items()}
            improved = ' *'

        if epoch % 5 == 0 or epoch == 1 or improved:
            print("  Epoch %3d  train=%.4f  val=%.4f  cos=%.4f%s" % (
                epoch, train_loss, val_loss, val_cos_sim, improved))

    mae_ft.load_state_dict(best_state_d)

    metrics_d, scores_d, labels_d, residuals_d = evaluate_model(
        mae_ft, val_data, "D: Distilled + full fine-tune",
        alpha=args.alpha, smooth_rounds=args.smooth_rounds,
        smooth_alpha=args.smooth_alpha)

    # Pick the best variant for plots
    all_results = [
        ('A', metrics_a), ('B', metrics_b),
        ('C', metrics_c), ('D', metrics_d),
    ]
    best_variant = max(all_results, key=lambda x: x[1]['best_f1'])

    # Save best as the final checkpoint if better than C
    if best_variant[0] == 'D' and metrics_d['best_f1'] > metrics_c['best_f1']:
        torch.save({
            'model_state_dict': best_state_d,
            'args': mae_args,
            'val_loss': best_val_loss_d,
            'distilled_r2': dist_ckpt['val_r2'],
        }, args.output)
        print("\n  Updated checkpoint with variant D")
        final_scores, final_labels = scores_d, labels_d
        final_residuals = residuals_d
    else:
        final_scores, final_labels = scores_c, labels_c
        final_residuals = residuals_c

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("  %-35s  ROC-AUC  PR-AUC  F1" % "Variant")
    print("  " + "-" * 58)
    for name, m in all_results:
        tag = " <-- best" if name == best_variant[0] else ""
        print("  %-35s  %.4f   %.4f  %.4f%s" % (
            name, m['roc_auc'], m['pr_auc'], m['best_f1'], tag))

    # Generate plots for best variant
    import numpy as np
    residuals_np = torch.cat(final_residuals).numpy() if isinstance(
        final_residuals[0], torch.Tensor) else np.concatenate(
            [r.numpy() if hasattr(r, 'numpy') else r for r in final_residuals])
    plot_roc_pr(final_scores, final_labels, args.output_dir,
                prefix='prad_distilled')
    plot_residual_tsne(residuals_np, final_labels, args.output_dir,
                       prefix='prad_distilled')

    print("\nDone.")


if __name__ == '__main__':
    main()
