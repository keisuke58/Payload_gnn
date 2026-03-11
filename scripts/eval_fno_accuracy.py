# -*- coding: utf-8 -*-
"""Evaluate FNO surrogate accuracy: predicted vs FEM waveforms.

Usage:
  python scripts/eval_fno_accuracy.py --model_dir runs/fno_gw_XXXX \
    --data_dir abaqus_work/gw_fairing_dataset --doe doe_gw_fairing.json
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from dataset_fno_gw import GWOperatorDataset
from models_fno_gw import FNOGWSurrogate


def relative_l2_error(pred, true):
    """Per-sample relative L2 error."""
    diff = (pred - true).reshape(pred.shape[0], -1)
    true_flat = true.reshape(true.shape[0], -1)
    return (torch.norm(diff, dim=1) / (torch.norm(true_flat, dim=1) + 1e-12)).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--data_dir', default='abaqus_work/gw_fairing_dataset')
    parser.add_argument('--doe', default='doe_gw_fairing.json')
    parser.add_argument('--downsample', type=int, default=4)
    args = parser.parse_args()

    # Load dataset
    ds = GWOperatorDataset(args.data_dir, args.doe, residual=True,
                           downsample=args.downsample)
    meta = ds.get_metadata()
    print(f"Dataset: {len(ds)} samples, {meta['n_sensors']} sensors, T={meta['T']}")

    # Load model
    ckpt_path = os.path.join(args.model_dir, 'best_model.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    model = FNOGWSurrogate(
        n_sensors=meta['n_sensors'],
        n_params=meta.get('n_params', 3),
        modes=ckpt.get('modes', 64),
        width=ckpt.get('width', 64),
        n_layers=ckpt.get('n_layers', 4),
        residual=meta['residual'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Evaluate all samples
    all_rel_l2 = []
    all_peak_err = []
    all_max_abs_pred = []
    all_max_abs_true = []

    print(f"\n{'Sample':>8} {'RelL2%':>8} {'PeakErr%':>10} {'MaxTrue':>10} {'MaxPred':>10}")
    print("-" * 55)

    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            params = sample['params'].unsqueeze(0)
            healthy = sample['healthy'].unsqueeze(0)
            target = sample['target']  # residual (scattering)
            defect_full = sample['defect_full']

            # FNO predicts full waveform (healthy + residual)
            pred_full = model(params, healthy).squeeze(0)

            # Compare full waveforms
            rel_l2 = relative_l2_error(
                pred_full.unsqueeze(0), defect_full.unsqueeze(0))[0]

            # Peak amplitude comparison
            max_true = defect_full.abs().max().item()
            max_pred = pred_full.abs().max().item()
            peak_err = abs(max_pred - max_true) / (max_true + 1e-12)

            # Scattering signal comparison
            pred_scatter = pred_full - healthy.squeeze(0)
            scatter_rel_l2 = relative_l2_error(
                pred_scatter.unsqueeze(0), target.unsqueeze(0))[0]

            all_rel_l2.append(rel_l2 * 100)
            all_peak_err.append(peak_err * 100)
            all_max_abs_pred.append(max_pred)
            all_max_abs_true.append(max_true)

            print(f"{i:>8d} {rel_l2*100:>7.2f}% {peak_err*100:>9.2f}% "
                  f"{max_true:>10.4f} {max_pred:>10.4f}")

    print("-" * 55)
    print(f"{'Mean':>8} {np.mean(all_rel_l2):>7.2f}% {np.mean(all_peak_err):>9.2f}%")
    print(f"{'Median':>8} {np.median(all_rel_l2):>7.2f}% {np.median(all_peak_err):>9.2f}%")
    print(f"{'Max':>8} {np.max(all_rel_l2):>7.2f}% {np.max(all_peak_err):>9.2f}%")

    # Summary
    print(f"\n=== FNO Surrogate Accuracy Summary ===")
    print(f"  Samples: {len(ds)} (train+val)")
    print(f"  Best val loss: {ckpt.get('val_loss', 'N/A')}")
    print(f"  Mean relative L2 error: {np.mean(all_rel_l2):.2f}%")
    print(f"  Mean peak amplitude error: {np.mean(all_peak_err):.2f}%")

    quality = "EXCELLENT" if np.mean(all_rel_l2) < 5 else \
              "GOOD" if np.mean(all_rel_l2) < 10 else \
              "FAIR" if np.mean(all_rel_l2) < 20 else "POOR"
    print(f"  Quality: {quality}")
    print(f"\n  Note: Only {len(ds)} FEM samples used.")
    print(f"  With 30+ samples, expect <5% error (EXCELLENT).")


if __name__ == '__main__':
    main()
