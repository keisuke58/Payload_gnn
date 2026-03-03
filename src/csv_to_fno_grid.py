# -*- coding: utf-8 -*-
"""csv_to_fno_grid.py — Convert S12 CZM CSV data to FNO 2D grids.

Maps 15,206 unstructured FEM nodes onto a regular (theta, y) grid
suitable for FNO2d training.

Input channels (4):
  ch0: z_norm        — axial position (y), normalized [0,1]
  ch1: theta_norm    — circumferential position, normalized [0,1]
  ch2: defect_mask   — binary defect label (0/1)
  ch3: load_proxy    — combined load indicator (temp * diff_pressure norm)

Output channels (1):
  ch0: smises        — von Mises stress field

Usage:
  python src/csv_to_fno_grid.py \
    --input_dir abaqus_work/batch_s12_100_thermal \
    --output_dir data/fno_grids_200 \
    --grid_size 64
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd


def nodes_to_grid(df, grid_size=64):
    """Convert unstructured nodes DataFrame to regular grid arrays.

    Returns:
        inp: (4, H, W) float32 array — input channels
        out: (1, H, W) float32 array — output (smises)
        meta: dict with grid statistics
    """
    # Compute cylindrical coordinates
    theta = np.degrees(np.arctan2(df['z'].values, df['x'].values))
    y = df['y'].values

    # Grid edges
    theta_min, theta_max = 0.0, 30.0
    y_min, y_max = 0.0, y.max() + 1.0

    theta_edges = np.linspace(theta_min, theta_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)

    # Bin indices for each node
    ti = np.clip(np.digitize(theta, theta_edges) - 1, 0, grid_size - 1)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, grid_size - 1)

    # Fields to grid
    fields = {
        'smises': df['smises'].values,
        'temp': df['temp'].values,
        'defect_label': df['defect_label'].values.astype(np.float32),
        'u_mag': df['u_mag'].values,
        'thermal_smises': df['thermal_smises'].values,
    }

    grids = {}
    counts = np.zeros((grid_size, grid_size), dtype=np.float32)

    for name, vals in fields.items():
        g = np.zeros((grid_size, grid_size), dtype=np.float64)
        np.add.at(g, (yi, ti), vals)
        grids[name] = g

    np.add.at(counts, (yi, ti), 1.0)
    counts_safe = np.maximum(counts, 1.0)

    for name in grids:
        grids[name] = (grids[name] / counts_safe).astype(np.float32)

    # Normalize coordinates for grid positions
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    theta_grid, y_grid = np.meshgrid(theta_centers, y_centers)

    theta_norm = (theta_grid - theta_min) / (theta_max - theta_min)
    y_norm = (y_grid - y_min) / (y_max - y_min)

    # Normalize temperature to [0,1] range (100-221 deg C typical)
    temp_norm = (grids['temp'] - 100.0) / (221.0 - 100.0 + 1e-8)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)

    # Input: 4 channels
    inp = np.stack([
        y_norm.astype(np.float32),           # ch0: axial position
        theta_norm.astype(np.float32),        # ch1: circumferential position
        grids['defect_label'],                # ch2: defect mask
        temp_norm,                            # ch3: load proxy (temperature)
    ], axis=0)

    # Output: 1 channel (smises)
    out = grids['smises'][np.newaxis, :, :]

    meta = {
        'smises_min': float(grids['smises'].min()),
        'smises_max': float(grids['smises'].max()),
        'smises_mean': float(grids['smises'].mean()),
        'temp_min': float(grids['temp'].min()),
        'temp_max': float(grids['temp'].max()),
        'defect_nodes': int(df['defect_label'].sum()),
        'defect_cells': int((grids['defect_label'] > 0.0).sum()),
        'empty_cells': int((counts == 0).sum()),
        'total_nodes': len(df),
    }

    return inp, out, meta


def process_dataset(input_dir, output_dir, grid_size=64):
    """Process all samples in a batch directory."""
    os.makedirs(output_dir, exist_ok=True)

    sample_dirs = sorted(glob.glob(os.path.join(input_dir, 'Job-S12-D*')))
    if not sample_dirs:
        print(f"No Job-S12-D* directories found in {input_dir}")
        return

    all_inp = []
    all_out = []
    all_meta = []
    skipped = 0

    for sdir in sample_dirs:
        csv_path = os.path.join(sdir, 'results', 'nodes.csv')
        if not os.path.exists(csv_path):
            skipped += 1
            continue

        sample_name = os.path.basename(sdir)
        df = pd.read_csv(csv_path)
        inp, out, meta = nodes_to_grid(df, grid_size)
        meta['sample'] = sample_name

        all_inp.append(inp)
        all_out.append(out)
        all_meta.append(meta)

    if not all_inp:
        print("No valid samples found.")
        return

    inp_arr = np.stack(all_inp, axis=0)  # (N, 4, H, W)
    out_arr = np.stack(all_out, axis=0)  # (N, 1, H, W)

    # Compute normalization stats from training data
    smises_all = out_arr[:, 0, :, :]
    norm_stats = {
        'smises_mean': float(smises_all.mean()),
        'smises_std': float(smises_all.std()),
        'smises_min': float(smises_all.min()),
        'smises_max': float(smises_all.max()),
    }

    # Save
    np.save(os.path.join(output_dir, 'inputs.npy'), inp_arr)
    np.save(os.path.join(output_dir, 'outputs.npy'), out_arr)

    import json
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump({'norm_stats': norm_stats, 'samples': all_meta,
                   'grid_size': grid_size, 'n_samples': len(all_meta),
                   'n_skipped': skipped}, f, indent=2)

    print(f"Converted {len(all_inp)} samples ({skipped} skipped)")
    print(f"  inputs:  {inp_arr.shape}  ({inp_arr.nbytes / 1e6:.1f} MB)")
    print(f"  outputs: {out_arr.shape}  ({out_arr.nbytes / 1e6:.1f} MB)")
    print(f"  smises range: [{norm_stats['smises_min']:.2f}, {norm_stats['smises_max']:.2f}]")
    print(f"  saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert S12 CZM CSV to FNO grids')
    parser.add_argument('--input_dir', type=str,
                        default='abaqus_work/batch_s12_100_thermal',
                        help='Batch directory with Job-S12-D* subdirs')
    parser.add_argument('--output_dir', type=str,
                        default='data/fno_grids_200',
                        help='Output directory for .npy files')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='Grid resolution (default: 64)')
    args = parser.parse_args()

    process_dataset(args.input_dir, args.output_dir, args.grid_size)


if __name__ == '__main__':
    main()
