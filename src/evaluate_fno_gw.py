# -*- coding: utf-8 -*-
"""
FNO GW Surrogate — Multi-Metric Evaluation

Evaluates trained FNO model with comprehensive metrics:
  - Relative L2 (time domain)
  - RMSE / MAE (per sensor & global)
  - Spectral error (FFT domain, band-wise: low/mid/high)
  - Group velocity error (envelope arrival time)
  - Peak amplitude error

Reference: Lehmann et al. 2024 (MIFNO), Gao et al. 2025 (spectral)

Usage:
  python src/evaluate_fno_gw.py --checkpoint runs/fno_gw_*/best_model.pt \
    --data_dir abaqus_work/gw_fairing_dataset

  # With plots:
  python src/evaluate_fno_gw.py --checkpoint runs/fno_gw_*/best_model.pt \
    --data_dir abaqus_work/gw_fairing_dataset --plot
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_fno_gw import GWOperatorDataset
from dataset_fno_gw_augmented import AugmentedGWDataset
from models_fno_gw import FNOGWSurrogate, DualStageFNOSurrogate


# =========================================================================
# Metrics
# =========================================================================
def relative_l2(pred, target):
    """Relative L2 error (per sample, then averaged)."""
    diff = (pred - target) ** 2
    norm = target ** 2 + 1e-8
    return torch.mean(torch.sqrt(diff.sum(dim=-1) / norm.sum(dim=-1)))


def rmse(pred, target):
    """Root Mean Square Error."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred, target):
    """Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target))


def rmse_per_sensor(pred, target):
    """RMSE per sensor channel. Returns (n_sensors,) tensor."""
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=-1)).mean(dim=0)


def mae_per_sensor(pred, target):
    """MAE per sensor channel. Returns (n_sensors,) tensor."""
    return torch.mean(torch.abs(pred - target), dim=-1).mean(dim=0)


def spectral_error_banded(pred, target, n_bands=3):
    """Frequency-band-wise spectral error.

    Splits FFT into n_bands equal bands (e.g., low/mid/high)
    and computes relative L2 in each band.

    Returns:
        dict with keys 'band_0' (low), 'band_1' (mid), ..., 'overall'
    """
    pred_fft = torch.fft.rfft(pred, dim=-1).abs()
    tgt_fft = torch.fft.rfft(target, dim=-1).abs()

    n_freq = pred_fft.shape[-1]
    band_size = n_freq // n_bands

    results = {}
    for b in range(n_bands):
        start = b * band_size
        end = (b + 1) * band_size if b < n_bands - 1 else n_freq
        p_band = pred_fft[..., start:end]
        t_band = tgt_fft[..., start:end]
        diff = (p_band - t_band) ** 2
        norm = t_band ** 2 + 1e-8
        rel_l2 = torch.mean(torch.sqrt(diff.sum(dim=-1) / norm.sum(dim=-1)))
        band_names = {0: 'low_freq', 1: 'mid_freq', 2: 'high_freq'}
        name = band_names.get(b, f'band_{b}')
        results[name] = rel_l2.item()

    # Overall spectral
    diff = (pred_fft - tgt_fft) ** 2
    norm = tgt_fft ** 2 + 1e-8
    results['spectral_overall'] = torch.mean(
        torch.sqrt(diff.sum(dim=-1) / norm.sum(dim=-1))).item()

    return results


def group_velocity_error(pred, target, dt=1.0):
    """Group velocity error via envelope arrival time.

    Computes analytic signal envelope (Hilbert transform via FFT),
    finds peak envelope time for each sensor, and compares.

    Returns:
        mean_arrival_error_samples: mean absolute arrival time error (in samples)
        mean_arrival_error_frac: normalized by signal length
    """
    def envelope_peak(signal):
        """Peak of analytic signal envelope along last dim."""
        n = signal.shape[-1]
        fft = torch.fft.fft(signal, dim=-1)
        # Zero negative frequencies (Hilbert transform)
        h = torch.zeros(n, device=signal.device)
        h[0] = 1
        h[1:(n + 1) // 2] = 2
        if n % 2 == 0:
            h[n // 2] = 1
        analytic = torch.fft.ifft(fft * h, dim=-1)
        env = analytic.abs()
        return env.argmax(dim=-1)  # peak time index

    pred_peak = envelope_peak(pred).float()   # (batch, n_sensors)
    tgt_peak = envelope_peak(target).float()

    abs_err = torch.abs(pred_peak - tgt_peak)
    T = pred.shape[-1]

    return {
        'arrival_error_samples': abs_err.mean().item(),
        'arrival_error_frac': (abs_err / T).mean().item(),
    }


def peak_amplitude_error(pred, target):
    """Relative error in peak waveform amplitude per sensor."""
    pred_peak = pred.abs().max(dim=-1).values  # (batch, n_sensors)
    tgt_peak = target.abs().max(dim=-1).values
    rel_err = torch.abs(pred_peak - tgt_peak) / (tgt_peak + 1e-8)
    return {
        'peak_amp_rel_error': rel_err.mean().item(),
        'peak_amp_rel_error_std': rel_err.std().item(),
    }


# =========================================================================
# Full evaluation
# =========================================================================
def evaluate_model(model, dataloader, device, residual=True):
    """Run all metrics on dataset."""
    model.eval()

    all_pred = []
    all_target = []

    with torch.no_grad():
        for batch in dataloader:
            params = batch['params'].to(device)
            healthy = batch['healthy'].to(device)
            pred = model(params, healthy)

            if residual:
                target = batch['defect_full'].to(device)
            else:
                target = batch['target'].to(device)

            all_pred.append(pred.cpu())
            all_target.append(target.cpu())

    pred = torch.cat(all_pred, dim=0)
    target = torch.cat(all_target, dim=0)

    n_samples = pred.shape[0]
    n_sensors = pred.shape[1]
    T = pred.shape[2]

    print(f"\n{'='*60}")
    print(f"Evaluation: {n_samples} samples, {n_sensors} sensors, T={T}")
    print(f"{'='*60}")

    # Time-domain metrics
    rl2 = relative_l2(pred, target).item()
    r = rmse(pred, target).item()
    m = mae(pred, target).item()
    r_per_s = rmse_per_sensor(pred, target)
    m_per_s = mae_per_sensor(pred, target)

    print(f"\n--- Time Domain ---")
    print(f"  Relative L2:  {rl2:.6f}")
    print(f"  RMSE (global): {r:.6f}")
    print(f"  MAE (global):  {m:.6f}")
    print(f"  RMSE per sensor: {', '.join(f'{v:.6f}' for v in r_per_s.tolist())}")
    print(f"  MAE per sensor:  {', '.join(f'{v:.6f}' for v in m_per_s.tolist())}")

    # Spectral metrics
    spec = spectral_error_banded(pred, target, n_bands=3)
    print(f"\n--- Spectral (FFT) ---")
    print(f"  Overall:   {spec['spectral_overall']:.6f}")
    print(f"  Low freq:  {spec['low_freq']:.6f}")
    print(f"  Mid freq:  {spec['mid_freq']:.6f}")
    print(f"  High freq: {spec['high_freq']:.6f}")

    # Group velocity
    gv = group_velocity_error(pred, target)
    print(f"\n--- Group Velocity (Envelope) ---")
    print(f"  Arrival error: {gv['arrival_error_samples']:.2f} samples "
          f"({gv['arrival_error_frac']*100:.2f}% of T)")

    # Peak amplitude
    pa = peak_amplitude_error(pred, target)
    print(f"\n--- Peak Amplitude ---")
    print(f"  Relative error: {pa['peak_amp_rel_error']:.4f} "
          f"(std: {pa['peak_amp_rel_error_std']:.4f})")

    # Aggregate results
    results = {
        'n_samples': n_samples,
        'n_sensors': n_sensors,
        'T': T,
        'relative_l2': rl2,
        'rmse': r,
        'mae': m,
        'rmse_per_sensor': r_per_s.tolist(),
        'mae_per_sensor': m_per_s.tolist(),
        **spec,
        **gv,
        **pa,
    }

    return results, pred, target


def plot_evaluation(pred, target, results, output_dir, n_show=3):
    """Generate evaluation plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    n_samples = pred.shape[0]
    n_sensors = pred.shape[1]
    n_show = min(n_show, n_samples)

    # 1. Waveform comparison (first n_show samples)
    for si in range(n_show):
        fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 2.5 * n_sensors),
                                 sharex=True)
        if n_sensors == 1:
            axes = [axes]
        for s in range(n_sensors):
            ax = axes[s]
            ax.plot(target[si, s].numpy(), 'b-', alpha=0.8, label='FEM (target)')
            ax.plot(pred[si, s].numpy(), 'r--', alpha=0.8, label='FNO (pred)')
            ax.set_ylabel(f'Sensor {s}')
            if s == 0:
                ax.legend(loc='upper right')
        axes[-1].set_xlabel('Time step')
        fig.suptitle(f'Sample {si}: Rel L2 = {results["relative_l2"]:.4f}',
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'waveform_sample_{si:02d}.png'),
                    dpi=150)
        plt.close(fig)

    # 2. FFT comparison (first sample, all sensors)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(10, 2.5 * n_sensors),
                             sharex=True)
    if n_sensors == 1:
        axes = [axes]
    pred_fft = torch.fft.rfft(pred[0], dim=-1).abs().numpy()
    tgt_fft = torch.fft.rfft(target[0], dim=-1).abs().numpy()
    freqs = np.arange(pred_fft.shape[-1])
    for s in range(n_sensors):
        ax = axes[s]
        ax.semilogy(freqs, tgt_fft[s], 'b-', alpha=0.8, label='FEM')
        ax.semilogy(freqs, pred_fft[s], 'r--', alpha=0.8, label='FNO')
        ax.set_ylabel(f'Sensor {s}')
        if s == 0:
            ax.legend()
    axes[-1].set_xlabel('Frequency bin')
    fig.suptitle('FFT Spectrum Comparison (Sample 0)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fft_comparison.png'), dpi=150)
    plt.close(fig)

    # 3. Error heatmap (sensor x time)
    if n_samples >= 1:
        err = (pred[0] - target[0]).abs().numpy()
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(err, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Sensor')
        fig.colorbar(im, ax=ax, label='|Pred - Target|')
        fig.suptitle('Error Heatmap (Sample 0)', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'error_heatmap.png'), dpi=150)
        plt.close(fig)

    # 4. Per-sensor RMSE bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    rmse_vals = results['rmse_per_sensor']
    ax.bar(range(len(rmse_vals)), rmse_vals, color='steelblue')
    ax.set_xlabel('Sensor Index')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE per Sensor')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'rmse_per_sensor.png'), dpi=150)
    plt.close(fig)

    # 5. Spectral band error bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    bands = ['low_freq', 'mid_freq', 'high_freq']
    vals = [results[b] for b in bands]
    ax.bar(bands, vals, color=['#2196F3', '#FF9800', '#F44336'])
    ax.set_ylabel('Relative L2 Error')
    ax.set_title('Spectral Error by Frequency Band')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'spectral_band_error.png'), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {output_dir}/")


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate FNO GW Surrogate (multi-metric)')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--data_dir', default='abaqus_work/gw_fairing_dataset')
    parser.add_argument('--doe', default='doe_gw_fairing.json')
    parser.add_argument('--output', default=None,
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--plot', action='store_true',
                        help='Generate evaluation plots')
    parser.add_argument('--downsample', type=int, default=4)
    parser.add_argument('--max_timesteps', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})
    meta = ckpt.get('meta', {})

    residual = ckpt_args.get('residual', True)
    model_type = ckpt_args.get('model', 'fno')

    # Dataset
    ds = GWOperatorDataset(
        args.data_dir, args.doe,
        residual=residual,
        downsample=args.downsample,
        max_timesteps=args.max_timesteps,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    ds_meta = ds.get_metadata()
    n_params = ds_meta.get('n_params', meta.get('n_params', 3))

    # Build model
    if model_type == 'dual_fno':
        model = DualStageFNOSurrogate(
            n_sensors=ds_meta['n_sensors'],
            n_params=n_params,
            modes_low=max(ckpt_args.get('modes', 64) // 4, 8),
            modes_high=ckpt_args.get('modes', 64),
            width=ckpt_args.get('width', 64),
            n_layers_low=max(ckpt_args.get('n_layers', 4) // 2, 2),
            n_layers_high=ckpt_args.get('n_layers', 4),
            residual=residual,
        ).to(device)
    else:
        model = FNOGWSurrogate(
            n_sensors=ds_meta['n_sensors'],
            n_params=n_params,
            modes=ckpt_args.get('modes', 64),
            width=ckpt_args.get('width', 64),
            n_layers=ckpt_args.get('n_layers', 4),
            residual=residual,
        ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    n_params_model = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type}, {n_params_model:,} params")
    print(f"Checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?')})")

    # Evaluate
    results, pred, target = evaluate_model(
        model, loader, device, residual=residual)

    # Output
    if args.output is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        args.output = os.path.join(ckpt_dir, 'eval')
    os.makedirs(args.output, exist_ok=True)

    results_path = os.path.join(args.output, 'eval_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved: {results_path}")

    # Plots
    if args.plot:
        plot_evaluation(pred, target, results, args.output)


if __name__ == '__main__':
    main()
