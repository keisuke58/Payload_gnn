# -*- coding: utf-8 -*-
"""Diagnostic: compare defect vs healthy reconstruction residuals."""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prad.anomaly_score import load_model_and_score
from prad.graphmae import PIGraphMAE
from prad.train_mae import load_data
from prad import STRESS_DIMS, PHYSICS_DIMS, DISPLACEMENT_DIMS, STRAIN_DIMS


def main():
    ckpt_path = 'checkpoints/prad_mae_sage.pt'
    data_dir = 'data/processed_s12_czm_thermal_200_binary'

    # Load model
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args']
    model = PIGraphMAE(
        encoder_arch=args['encoder_arch'],
        in_channels=args.get('in_channels', 34),
        hidden_channels=args['hidden'],
        num_layers=args['num_layers'],
        dropout=args['dropout'],
        mask_ratio=args['mask_ratio'],
        lambda_physics=args['lambda_physics'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _, val_data = load_data(data_dir, 'cpu')

    # Collect per-node diagnostics
    all_l1 = []  # |x - x_recon| per dim
    all_cos = []  # 1 - cos_sim per node
    all_labels = []

    with torch.no_grad():
        for data in val_data:
            x_recon = model.reconstruct(data)

            # Per-dim L1 residual
            l1 = (data.x - x_recon).abs()  # (N, D)
            all_l1.append(l1.numpy())

            # Per-node cosine distance
            cos = F.cosine_similarity(data.x, x_recon, dim=1)
            all_cos.append((1 - cos).numpy())

            all_labels.append(data.y.numpy())

    l1 = np.concatenate(all_l1)
    cos_dist = np.concatenate(all_cos)
    labels = np.concatenate(all_labels)

    healthy = labels == 0
    defect = labels > 0

    print("=" * 60)
    print("PRAD Diagnostic: Defect vs Healthy Residuals")
    print("=" * 60)
    print("  Total: %d nodes (%d healthy, %d defect)" % (
        len(labels), healthy.sum(), defect.sum()))

    print("\n=== Cosine Distance ===")
    print("  Healthy: mean=%.6f, std=%.6f" % (
        cos_dist[healthy].mean(), cos_dist[healthy].std()))
    print("  Defect:  mean=%.6f, std=%.6f" % (
        cos_dist[defect].mean(), cos_dist[defect].std()))

    print("\n=== Per-dim L1 Residual ===")
    dim_names = {
        0: 'x_pos', 1: 'y_pos', 2: 'z_pos',
        10: 'ux', 11: 'uy', 12: 'uz', 13: 'umag',
        14: 'temp',
        15: 's11', 16: 's22', 17: 's12', 18: 'smises', 19: 'sp_sum',
        20: 'ts_mises',
        21: 'le11', 22: 'le22', 23: 'le12',
    }
    print("Dim | Name     | Healthy mean | Defect mean  | Ratio | Note")
    print("-" * 70)
    for d in range(min(34, l1.shape[1])):
        h_m = l1[healthy, d].mean()
        d_m = l1[defect, d].mean()
        ratio = d_m / max(h_m, 1e-10)
        name = dim_names.get(d, 'dim%d' % d)
        note = ""
        if ratio > 1.5:
            note = "** HIGHER for defect"
        elif ratio < 0.67:
            note = "** LOWER for defect"
        print("%3d | %-8s | %12.6f | %12.6f | %5.2f | %s" % (
            d, name, h_m, d_m, ratio, note))

    # Stress-only score
    stress_dims = STRESS_DIMS
    stress_l1_h = l1[healthy][:, stress_dims].mean(axis=1)
    stress_l1_d = l1[defect][:, stress_dims].mean(axis=1)
    print("\n=== Stress-only L1 ===")
    print("  Healthy: mean=%.6f" % stress_l1_h.mean())
    print("  Defect:  mean=%.6f" % stress_l1_d.mean())

    # Check ROC-AUC for different scoring methods
    from sklearn.metrics import roc_auc_score
    binary = (labels > 0).astype(int)

    # Method 1: overall L1 mean
    s1 = l1.mean(axis=1)
    print("\n=== ROC-AUC by method ===")
    print("  L1 mean (all dims):     %.4f" % roc_auc_score(binary, s1))

    # Method 2: stress dims only
    s2 = l1[:, stress_dims].mean(axis=1)
    print("  L1 mean (stress only):  %.4f" % roc_auc_score(binary, s2))

    # Method 3: cosine distance
    print("  Cosine distance:        %.4f" % roc_auc_score(binary, cos_dist))

    # Method 4: max residual across dims
    s4 = l1.max(axis=1)
    print("  L1 max (all dims):      %.4f" % roc_auc_score(binary, s4))

    # Method 5: L2 norm
    s5 = np.sqrt((l1 ** 2).sum(axis=1))
    print("  L2 norm:                %.4f" % roc_auc_score(binary, s5))

    # Method 6: physics dims only
    phys_dims = PHYSICS_DIMS
    valid_phys = [d for d in phys_dims if d < l1.shape[1]]
    s6 = l1[:, valid_phys].mean(axis=1)
    print("  L1 mean (physics dims): %.4f" % roc_auc_score(binary, s6))

    # Method 7: displacement dims
    disp_dims = DISPLACEMENT_DIMS
    valid_disp = [d for d in disp_dims if d < l1.shape[1]]
    s7 = l1[:, valid_disp].mean(axis=1)
    print("  L1 disp dims:           %.4f" % roc_auc_score(binary, s7))

    # Method 8: strain dims
    strain_dims = STRAIN_DIMS
    valid_strain = [d for d in strain_dims if d < l1.shape[1]]
    s8 = l1[:, valid_strain].mean(axis=1)
    print("  L1 strain dims:         %.4f" % roc_auc_score(binary, s8))


if __name__ == '__main__':
    main()
