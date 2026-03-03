# -*- coding: utf-8 -*-
"""augment_fno_healthy.py — Generate healthy FNO grids from defective ones.

Takes existing FNO grids (200 defective), creates healthy versions by:
  1. Setting defect_mask (ch2) to 0
  2. Replacing stress in defect region with healthy baseline
  3. Adding sensor noise to all channels
  4. Combining into mixed dataset (200 defect + 200 healthy)

Usage:
  python src/augment_fno_healthy.py \
    --input data/fno_grids_200 \
    --output data/fno_grids_mixed_400 \
    --n_healthy 200
"""

import argparse
import json
import os

import numpy as np


def make_healthy_grid(inp, out, rng, noise_std=0.02):
    """Create a healthy FNO grid from a defective one.

    Args:
        inp: (4, H, W) input grid [z_norm, theta_norm, defect_mask, temp_norm]
        out: (1, H, W) output grid [smises]
        rng: numpy RandomState
        noise_std: relative noise level
    """
    inp_h = inp.copy()
    out_h = out.copy()

    # 1. Zero out defect mask
    defect_mask = inp[2] > 0.01
    inp_h[2] = 0.0

    # 2. Replace stress in defect region with healthy baseline
    if defect_mask.sum() > 0:
        # Use mean stress from non-defect cells as baseline
        healthy_cells = ~defect_mask & (out[0] > 0)
        if healthy_cells.sum() > 0:
            baseline_stress = out[0][healthy_cells].mean()
            baseline_std = out[0][healthy_cells].std()
        else:
            baseline_stress = out[0].mean()
            baseline_std = out[0].std()

        # Fill defect region with baseline + small spatial variation
        n_defect = defect_mask.sum()
        out_h[0][defect_mask] = baseline_stress + rng.normal(
            0, baseline_std * 0.1, size=n_defect).astype(np.float32)

    # 3. Add sensor noise to stress field
    stress_std = out_h[0][out_h[0] > 0].std() if (out_h[0] > 0).sum() > 0 else 1.0
    out_h[0] += rng.normal(0, stress_std * noise_std,
                           size=out_h[0].shape).astype(np.float32)
    out_h[0] = np.clip(out_h[0], 0, None)  # Stress is non-negative

    # 4. Add small noise to temperature channel (ch3)
    temp_noise = rng.normal(0, noise_std, size=inp_h[3].shape).astype(np.float32)
    inp_h[3] = np.clip(inp_h[3] + temp_noise, 0, 1)

    return inp_h, out_h


def main():
    parser = argparse.ArgumentParser(
        description="Generate healthy FNO grids via augmentation")
    parser.add_argument("--input", default="data/fno_grids_200")
    parser.add_argument("--output", default="data/fno_grids_mixed_400")
    parser.add_argument("--n_healthy", type=int, default=200)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # Load existing defective grids
    inputs = np.load(os.path.join(args.input, "inputs.npy"))    # (N, 4, H, W)
    outputs = np.load(os.path.join(args.input, "outputs.npy"))  # (N, 1, H, W)
    with open(os.path.join(args.input, "meta.json")) as f:
        meta = json.load(f)

    n_defect = inputs.shape[0]
    print("Loaded %d defective grids" % n_defect)

    # Generate healthy grids
    healthy_inputs = []
    healthy_outputs = []
    healthy_meta = []

    for i in range(args.n_healthy):
        src_idx = rng.randint(0, n_defect)
        inp_h, out_h = make_healthy_grid(
            inputs[src_idx], outputs[src_idx], rng, args.noise_std)
        healthy_inputs.append(inp_h)
        healthy_outputs.append(out_h)
        healthy_meta.append({
            "sample": "Job-S12-H%03d" % (i + 1),
            "source": meta["samples"][src_idx]["sample"],
            "is_healthy": True,
            "defect_nodes": 0,
            "defect_cells": 0,
        })

    healthy_inputs = np.stack(healthy_inputs, axis=0)
    healthy_outputs = np.stack(healthy_outputs, axis=0)

    # Combine: defective + healthy
    mixed_inputs = np.concatenate([inputs, healthy_inputs], axis=0)
    mixed_outputs = np.concatenate([outputs, healthy_outputs], axis=0)

    # Update metadata
    defect_meta = []
    for s in meta["samples"]:
        s_copy = dict(s)
        s_copy["is_healthy"] = False
        defect_meta.append(s_copy)

    mixed_meta = {
        "n_samples": len(mixed_inputs),
        "n_defect": n_defect,
        "n_healthy": args.n_healthy,
        "grid_size": meta["grid_size"],
        "norm_stats": {
            "smises_mean": float(mixed_outputs[:, 0].mean()),
            "smises_std": float(mixed_outputs[:, 0].std()),
            "smises_min": float(mixed_outputs[:, 0].min()),
            "smises_max": float(mixed_outputs[:, 0].max()),
        },
        "samples": defect_meta + healthy_meta,
    }

    # Save
    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, "inputs.npy"), mixed_inputs)
    np.save(os.path.join(args.output, "outputs.npy"), mixed_outputs)
    with open(os.path.join(args.output, "meta.json"), "w") as f:
        json.dump(mixed_meta, f, indent=2)

    print("\nMixed FNO grids:")
    print("  Defect:  %d" % n_defect)
    print("  Healthy: %d" % args.n_healthy)
    print("  Total:   %d" % len(mixed_inputs))
    print("  inputs:  %s (%.1f MB)" % (
        mixed_inputs.shape, mixed_inputs.nbytes / 1e6))
    print("  outputs: %s (%.1f MB)" % (
        mixed_outputs.shape, mixed_outputs.nbytes / 1e6))

    # Verify: defect score should differ
    defect_scores = []
    for i in range(n_defect):
        mask = mixed_inputs[i, 2] > 0.01
        if mask.sum() > 0:
            d_stress = mixed_outputs[i, 0][mask].mean()
            g_stress = mixed_outputs[i, 0][~mask].mean()
            defect_scores.append(abs(d_stress - g_stress))
    healthy_scores = []
    for i in range(n_defect, len(mixed_inputs)):
        # No defect mask, so score should be ~0
        healthy_scores.append(mixed_outputs[i, 0].std())

    print("\n  Defect score (mean): %.2f" % np.mean(defect_scores))
    print("  Healthy stress std:  %.2f" % np.mean(healthy_scores))
    print("\nSaved to: %s" % args.output)


if __name__ == "__main__":
    main()
