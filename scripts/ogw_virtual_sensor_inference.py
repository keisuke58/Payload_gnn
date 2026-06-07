#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ogw_virtual_sensor_inference.py
Read OGW omega-stringer HDF5 data, extract virtual sensor time series,
and run MIFNO-GW inference for sim-to-real validation.

Usage:
  python scripts/ogw_virtual_sensor_inference.py \
      --zip_dir data/external/ogw_stringer \
      --model_ckpt runs/mifno_gw_v2/ckpt/best.pt \
      --freq 50kHz \
      --n_sensors 9 \
      --output_dir data/external/ogw_stringer/inference_results
"""
import argparse
import io
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

SCENARIO_LABELS = {
    'Intact': 0,
    'FirstImpact': 1,
    'SecondImpact': 1,
}
ZIP_NAMES = {
    'Intact': 'OGW_CFRP_Stringer_Wavefield_Intact.zip',
    'FirstImpact': 'OGW_CFRP_Stringer_Wavefield_FirstImpact.zip',
    'SecondImpact': 'OGW_CFRP_Stringer_Wavefield_SecondImpact.zip',
}


def read_ogw_50khz(zip_path, freq_tag='50kHz'):
    """Read wavefield + coordinates + time from zip.

    Returns:
        data   : [N_scan, T] float32  — out-of-plane displacement
        coord  : [N_scan, 2] float32  — (x, y) in meters
        time_s : [T]         float32  — time axis in seconds
    """
    import zipfile, h5py

    def _read_h5(zf, fname):
        with zf.open(fname) as f:
            buf = io.BytesIO(f.read())
        with h5py.File(buf, 'r') as h:
            k = list(h.keys())[0]
            return h[k][:]

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        data_f  = next(n for n in names if freq_tag in n and 'data_z' in n)
        coord_f = next(n for n in names if freq_tag in n and 'coordinates' in n)
        time_f  = next(n for n in names if freq_tag in n and 'time' in n)

        raw   = _read_h5(zf, data_f)   # (T, N) from file convention
        coord = _read_h5(zf, coord_f)  # (2, N)
        t     = _read_h5(zf, time_f).ravel()

    data  = raw.T.astype(np.float32)    # -> (N, T)
    coord = coord.T.astype(np.float32)  # -> (N, 2)
    return data, coord, t.astype(np.float32)


def pick_virtual_sensors(coord, n_sensors=9, strategy='grid', seed=0):
    """Choose n_sensors scan points as virtual sensors.

    strategy='grid': evenly spaced across the plate extent.
    strategy='random': random subset.

    Returns: indices [n_sensors]
    """
    N = coord.shape[0]
    if strategy == 'grid':
        # Divide plate into sqrt(n) x sqrt(n) blocks, pick centroid
        import math
        nc = int(math.ceil(math.sqrt(n_sensors)))
        x_min, x_max = coord[:, 0].min(), coord[:, 0].max()
        y_min, y_max = coord[:, 1].min(), coord[:, 1].max()
        x_edges = np.linspace(x_min, x_max, nc + 1)
        y_edges = np.linspace(y_min, y_max, nc + 1)
        indices = []
        for ix in range(nc):
            for iy in range(nc):
                mask = (
                    (coord[:, 0] >= x_edges[ix]) & (coord[:, 0] < x_edges[ix + 1]) &
                    (coord[:, 1] >= y_edges[iy]) & (coord[:, 1] < y_edges[iy + 1])
                )
                if mask.sum() == 0:
                    continue
                cx = (x_edges[ix] + x_edges[ix + 1]) / 2
                cy = (y_edges[iy] + y_edges[iy + 1]) / 2
                dists = (coord[mask, 0] - cx) ** 2 + (coord[mask, 1] - cy) ** 2
                sub_idx = np.where(mask)[0][np.argmin(dists)]
                indices.append(sub_idx)
                if len(indices) == n_sensors:
                    break
            if len(indices) == n_sensors:
                break
        # Pad if not enough blocks
        while len(indices) < n_sensors:
            rng = np.random.default_rng(seed)
            indices.append(int(rng.integers(0, N)))
        return np.array(indices[:n_sensors])
    else:
        rng = np.random.default_rng(seed)
        return rng.choice(N, size=n_sensors, replace=False)


def extract_sensor_signals(data, indices, max_len=2048):
    """Extract and z-score sensor signals.

    Returns:
        wav : [n_sensors, max_len] float32
        pos : [n_sensors]         float32 (normalised 0-1 index)
    """
    S = len(indices)
    T = data.shape[1]
    wav = data[indices, :]  # [S, T]
    if T >= max_len:
        wav = wav[:, :max_len]
    else:
        pad = np.zeros((S, max_len - T), dtype=np.float32)
        wav = np.concatenate([wav, pad], axis=1)
    # z-score per sensor
    mu  = wav.mean(axis=1, keepdims=True)
    std = wav.std(axis=1, keepdims=True) + 1e-8
    wav = (wav - mu) / std
    pos = np.linspace(0, 1, S, dtype=np.float32)
    return wav, pos


def run_inference(model, wav_np, pos_np, device):
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        wav = torch.tensor(wav_np[None], device=device)  # [1, S, T]
        pos = torch.tensor(pos_np[None], device=device)  # [1, S]
        logit = model(pos, wav)
        prob  = F.softmax(logit, dim=1)[0, 1].item()
    return prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_dir', default='data/external/ogw_stringer')
    parser.add_argument('--model_ckpt', required=True)
    parser.add_argument('--freq', default='50kHz')
    parser.add_argument('--n_sensors', type=int, default=9)
    parser.add_argument('--strategy', default='grid', choices=['grid', 'random'])
    parser.add_argument('--n_windows', type=int, default=5,
                        help='Number of random sensor windows for ensemble')
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--output_dir', default='data/external/ogw_stringer/inference_results')
    args = parser.parse_args()

    import torch
    from src.mifno_gw import MIFNO_GW

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt = torch.load(os.path.join(PROJECT_ROOT, args.model_ckpt),
                      map_location='cpu', weights_only=False)
    msd = ckpt['model']
    # Detect n_sensors from checkpoint
    n_sensors_ckpt = args.n_sensors
    model = MIFNO_GW(n_sensors=n_sensors_ckpt, seq_len=args.seq_len,
                     use_pos=False).to(device)
    model.load_state_dict(msd)
    print("Model loaded. Params:", sum(p.numel() for p in model.parameters()))

    os.makedirs(os.path.join(PROJECT_ROOT, args.output_dir), exist_ok=True)
    results = []

    for scenario, zip_name in ZIP_NAMES.items():
        zip_path = os.path.join(PROJECT_ROOT, args.zip_dir, zip_name)
        if not os.path.exists(zip_path):
            print("Skip %s (zip not found)" % scenario)
            continue
        try:
            print("Reading %s..." % scenario, end=' ', flush=True)
            data, coord, t = read_ogw_50khz(zip_path, freq_tag=args.freq)
            print("shape=%s" % str(data.shape))
        except Exception as e:
            print("ERROR:", e)
            continue

        gt = SCENARIO_LABELS[scenario]
        probs = []

        for w in range(args.n_windows):
            indices = pick_virtual_sensors(coord, args.n_sensors,
                                           strategy=args.strategy, seed=w)
            wav, pos = extract_sensor_signals(data, indices, args.seq_len)
            p = run_inference(model, wav, pos, device)
            probs.append(p)

        mean_prob = float(np.mean(probs))
        pred = 1 if mean_prob >= 0.5 else 0
        correct = (pred == gt)
        print("  %s: gt=%d  prob=%.3f±%.3f  pred=%d  %s" % (
            scenario, gt, mean_prob, np.std(probs), pred,
            'CORRECT' if correct else 'WRONG'))

        results.append({
            'scenario': scenario, 'gt': gt, 'prob_mean': mean_prob,
            'prob_std': float(np.std(probs)), 'pred': pred, 'correct': correct,
        })

    # Save results
    out_path = os.path.join(PROJECT_ROOT, args.output_dir, 'inference_%s.json' % args.freq)
    with open(out_path, 'w') as f:
        json.dump({'model': args.model_ckpt, 'freq': args.freq,
                   'n_sensors': args.n_sensors, 'n_windows': args.n_windows,
                   'results': results}, f, indent=2)
    print("\nResults saved:", out_path)
    n_correct = sum(r['correct'] for r in results)
    print("Accuracy: %d/%d" % (n_correct, len(results)))


if __name__ == '__main__':
    main()
