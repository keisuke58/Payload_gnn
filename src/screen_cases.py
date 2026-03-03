# -*- coding: utf-8 -*-
"""screen_cases.py — Two-stage screening with FNO surrogate + MC-Dropout UQ.

Stage 1: FNO predicts stress field for all cases → MC-Dropout for uncertainty.
Screening: Select threshold-ambiguous cases (need full FEM).
Output: List of case IDs that require full FEM solver run.

Usage:
  python src/screen_cases.py \
    --checkpoint runs/fno_surrogate_v1/best_model.pt \
    --data_dir data/fno_grids_200 \
    --mc_samples 20 --dropout_p 0.2 \
    --k_margin 1.5 \
    --output screening_results.json
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from models_fno import FNO2d


class FNO2dDropout(FNO2d):
    """FNO2d with Dropout for MC-Dropout UQ."""
    def __init__(self, modes1=12, modes2=12, width=32,
                 in_channels=4, out_channels=1, p=0.2):
        super().__init__(modes1, modes2, width, in_channels, out_channels)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x); x2 = self.w0(x)
        x = F.gelu(x1 + x2); x = self.drop(x)
        x1 = self.conv1(x); x2 = self.w1(x)
        x = F.gelu(x1 + x2); x = self.drop(x)
        x1 = self.conv2(x); x2 = self.w2(x)
        x = F.gelu(x1 + x2); x = self.drop(x)
        x1 = self.conv3(x); x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)


def mc_predict(model, x, T=20, out_mean=0.0, out_std=1.0):
    """Run MC-Dropout inference T times, return mean and std of predictions."""
    model.train()  # Enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(T):
            pred = model(x)
            pred_raw = pred * out_std + out_mean
            preds.append(pred_raw.cpu())
    preds = torch.stack(preds, dim=0)  # (T, B, 1, H, W)
    mu = preds.mean(dim=0)             # (B, 1, H, W)
    sigma = preds.std(dim=0)           # (B, 1, H, W)
    return mu, sigma


def compute_screening_scores(mu, sigma, inputs):
    """Compute per-sample screening scores.

    Score: how different the defect region stress is from the healthy baseline.
    Uncertainty: mean MC-Dropout std in the defect region.
    """
    defect_masks = inputs[:, 2, :, :]  # (B, H, W), ch2 = defect_mask
    scores = []

    for i in range(mu.shape[0]):
        stress = mu[i, 0, :, :]      # (H, W)
        unc = sigma[i, 0, :, :]      # (H, W)
        mask = defect_masks[i]        # (H, W)

        # Global mean stress (healthy baseline proxy)
        global_mean = stress.mean().item()

        if mask.sum() > 0:
            # Defect region mean stress
            defect_mean = stress[mask > 0.01].mean().item()
            defect_unc = unc[mask > 0.01].mean().item()
            # Score = stress deviation in defect region
            score = abs(defect_mean - global_mean)
        else:
            # Healthy sample: no defect region
            score = 0.0
            defect_unc = unc.mean().item()

        scores.append({
            'score': score,
            'uncertainty': defect_unc,
            'global_stress_mean': global_mean,
            'stress_std': unc.mean().item(),
        })

    return scores


def screen(scores, k_margin=1.5, unc_percentile=70):
    """Determine which cases need full FEM.

    Cases near the threshold OR with high uncertainty → need FEM.
    """
    all_scores = np.array([s['score'] for s in scores])
    all_unc = np.array([s['uncertainty'] for s in scores])

    # Threshold: median score (adaptive)
    threshold = np.median(all_scores[all_scores > 0])
    unc_cutoff = np.percentile(all_unc, unc_percentile)

    needs_fem = []
    for i, s in enumerate(scores):
        # Near threshold
        near_threshold = abs(s['score'] - threshold) < k_margin * s['uncertainty']
        # High uncertainty
        high_unc = s['uncertainty'] > unc_cutoff
        # Healthy samples always pass (no FEM needed)
        is_healthy = s['score'] == 0.0

        needs = (near_threshold or high_unc) and not is_healthy
        needs_fem.append(needs)

    return needs_fem, threshold, unc_cutoff


def main():
    parser = argparse.ArgumentParser(description='Two-stage screening')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/fno_grids_200')
    parser.add_argument('--mc_samples', type=int, default=20)
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--k_margin', type=float, default=1.5)
    parser.add_argument('--unc_percentile', type=float, default=70)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output', type=str, default='screening_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get('args', {})
    out_mean = ckpt.get('out_mean', 0.0)
    out_std = ckpt.get('out_std', 1.0)

    # Build model with dropout
    model = FNO2dDropout(
        modes1=model_args.get('modes', 12),
        modes2=model_args.get('modes', 12),
        width=model_args.get('width', 32),
        in_channels=4, out_channels=1,
        p=args.dropout_p,
    ).to(device)

    # Load weights (strict=False because dropout layers are new)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, "
          f"val_rel_l2={ckpt.get('val_rel_l2', '?'):.4f}")

    # Load data
    inputs = np.load(os.path.join(args.data_dir, 'inputs.npy'))
    meta_path = os.path.join(args.data_dir, 'meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    sample_names = [s['sample'] for s in meta['samples']]

    # MC-Dropout inference
    x_all = torch.from_numpy(inputs).float()
    all_scores = []

    for start in range(0, len(x_all), args.batch_size):
        end = min(start + args.batch_size, len(x_all))
        xb = x_all[start:end].to(device)
        mu, sigma = mc_predict(model, xb, T=args.mc_samples,
                               out_mean=out_mean, out_std=out_std)
        batch_scores = compute_screening_scores(
            mu, sigma, x_all[start:end])
        all_scores.extend(batch_scores)

    print(f"Scored {len(all_scores)} samples")

    # Screen
    needs_fem, threshold, unc_cutoff = screen(
        all_scores, k_margin=args.k_margin,
        unc_percentile=args.unc_percentile)

    n_fem = sum(needs_fem)
    n_skip = len(needs_fem) - n_fem

    # Results
    results = {
        'total_cases': int(len(all_scores)),
        'needs_fem': int(n_fem),
        'can_skip': int(n_skip),
        'reduction_pct': 100.0 * n_skip / len(all_scores),
        'threshold': float(threshold),
        'unc_cutoff': float(unc_cutoff),
        'k_margin': args.k_margin,
        'mc_samples': args.mc_samples,
        'cases': [],
    }

    for i, (name, score, needs) in enumerate(
            zip(sample_names, all_scores, needs_fem)):
        results['cases'].append({
            'sample': name,
            'score': score['score'],
            'uncertainty': score['uncertainty'],
            'needs_fem': bool(needs),
        })

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Screening Results ===")
    print(f"Total cases:  {results['total_cases']}")
    print(f"Needs FEM:    {n_fem} ({100*n_fem/len(all_scores):.0f}%)")
    print(f"Can skip:     {n_skip} ({results['reduction_pct']:.0f}%)")
    print(f"Threshold:    {threshold:.4f}")
    print(f"Unc cutoff:   {unc_cutoff:.4f}")
    print(f"Saved to:     {args.output}")


if __name__ == '__main__':
    main()
