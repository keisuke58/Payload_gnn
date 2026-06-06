# -*- coding: utf-8 -*-
"""
GW v3 Validation — FEM Result Physical Consistency Check

Compares Healthy vs Debond30/50/80 v3 FEM results:
  - Scattering signal (defect - healthy) amplitude vs defect size
  - Waveform comparison per sensor ring
  - Spectral content analysis
  - Expected trend: larger defect → stronger scattering

Usage:
  python scripts/validate_gw_v3.py \
    --data_dir abaqus_work/gw_valid_v3 \
    --output runs/validate_v3
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from dataset_fno_gw import load_sensor_csv


def compute_scattering_metrics(healthy, defect, label):
    """Compute scattering signal metrics."""
    scatter = defect - healthy  # (n_sensors, T)

    # RMS of scattering signal per sensor
    rms_per_sensor = np.sqrt(np.mean(scatter ** 2, axis=-1))

    # Peak amplitude
    peak_per_sensor = np.max(np.abs(scatter), axis=-1)

    # Energy ratio (scatter energy / healthy energy)
    healthy_energy = np.sum(healthy ** 2, axis=-1) + 1e-12
    scatter_energy = np.sum(scatter ** 2, axis=-1)
    energy_ratio = scatter_energy / healthy_energy

    # FFT analysis
    scatter_fft = np.abs(np.fft.rfft(scatter, axis=-1))
    healthy_fft = np.abs(np.fft.rfft(healthy, axis=-1))
    n_freq = scatter_fft.shape[-1]
    band_size = n_freq // 3

    spectral = {}
    for b, name in enumerate(['low', 'mid', 'high']):
        start = b * band_size
        end = (b + 1) * band_size if b < 2 else n_freq
        s_band = np.mean(scatter_fft[:, start:end])
        h_band = np.mean(healthy_fft[:, start:end]) + 1e-12
        spectral[f'{name}_freq_ratio'] = float(s_band / h_band)

    return {
        'label': label,
        'rms_mean': float(np.mean(rms_per_sensor)),
        'rms_max': float(np.max(rms_per_sensor)),
        'peak_mean': float(np.mean(peak_per_sensor)),
        'peak_max': float(np.max(peak_per_sensor)),
        'energy_ratio_mean': float(np.mean(energy_ratio)),
        'energy_ratio_max': float(np.max(energy_ratio)),
        **spectral,
        'rms_per_sensor': rms_per_sensor.tolist(),
        'peak_per_sensor': peak_per_sensor.tolist(),
    }


def plot_validation(healthy_wf, defect_wfs, defect_labels, times, positions,
                    output_dir, n_sensors_show=8):
    """Generate validation plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    n_sensors = healthy_wf.shape[0]
    n_rings = n_sensors // 8 if n_sensors % 8 == 0 else 1
    sensors_per_ring = 8 if n_rings > 1 else n_sensors

    t_ms = times * 1000  # s → ms

    # ----------------------------------------------------------------
    # 1. Waveform overlay: Healthy vs all defects (selected sensors)
    # ----------------------------------------------------------------
    # Pick sensors from different rings
    if n_rings > 1:
        show_indices = []
        for ring in [0, n_rings // 4, n_rings // 2, 3 * n_rings // 4]:
            for s_in_ring in [0, 4]:  # 0° and 180° in ring
                idx = ring * sensors_per_ring + s_in_ring
                if idx < n_sensors:
                    show_indices.append(idx)
        show_indices = show_indices[:n_sensors_show]
    else:
        show_indices = list(range(min(n_sensors_show, n_sensors)))

    fig, axes = plt.subplots(len(show_indices), 1,
                              figsize=(14, 2.5 * len(show_indices)), sharex=True)
    if len(show_indices) == 1:
        axes = [axes]

    colors = ['#2196F3', '#FF9800', '#F44336']  # D30, D50, D80
    for i, si in enumerate(show_indices):
        ax = axes[i]
        ax.plot(t_ms, healthy_wf[si], 'k-', alpha=0.7, linewidth=1.0, label='Healthy')
        for j, (dwf, dlabel) in enumerate(zip(defect_wfs, defect_labels)):
            ax.plot(t_ms, dwf[si], '--', color=colors[j], alpha=0.7,
                    linewidth=0.8, label=dlabel)
        ring_id = si // sensors_per_ring if n_rings > 1 else 0
        s_in_ring = si % sensors_per_ring if n_rings > 1 else si
        ax.set_ylabel(f'Ring {ring_id}\nSensor {s_in_ring}', fontsize=9)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=4)
    axes[-1].set_xlabel('Time [ms]')
    fig.suptitle('Waveform Comparison: Healthy vs Debonding (v3 Validation)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'waveform_overlay.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: waveform_overlay.png")

    # ----------------------------------------------------------------
    # 2. Scattering signal (defect - healthy)
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(len(show_indices), 1,
                              figsize=(14, 2.5 * len(show_indices)), sharex=True)
    if len(show_indices) == 1:
        axes = [axes]

    for i, si in enumerate(show_indices):
        ax = axes[i]
        for j, (dwf, dlabel) in enumerate(zip(defect_wfs, defect_labels)):
            scatter = dwf[si] - healthy_wf[si]
            ax.plot(t_ms, scatter, '-', color=colors[j], alpha=0.8,
                    linewidth=0.8, label=dlabel)
        ring_id = si // sensors_per_ring if n_rings > 1 else 0
        s_in_ring = si % sensors_per_ring if n_rings > 1 else si
        ax.set_ylabel(f'Ring {ring_id}\nSensor {s_in_ring}', fontsize=9)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=3)
    axes[-1].set_xlabel('Time [ms]')
    fig.suptitle('Scattering Signal (Defect - Healthy) by Defect Size', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'scattering_signal.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: scattering_signal.png")

    # ----------------------------------------------------------------
    # 3. Scattering RMS heatmap per sensor (sensor rings vs defect size)
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, len(defect_labels), figsize=(5 * len(defect_labels), 6),
                              sharey=True)
    if len(defect_labels) == 1:
        axes = [axes]

    for j, (dwf, dlabel) in enumerate(zip(defect_wfs, defect_labels)):
        scatter = dwf - healthy_wf
        # Reshape to (n_rings, sensors_per_ring, T) if possible
        if n_rings > 1:
            scatter_2d = scatter.reshape(n_rings, sensors_per_ring, -1)
            rms_2d = np.sqrt(np.mean(scatter_2d ** 2, axis=-1))  # (n_rings, 8)
        else:
            rms_2d = np.sqrt(np.mean(scatter ** 2, axis=-1, keepdims=True)).T

        ax = axes[j]
        im = ax.imshow(rms_2d, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_title(dlabel, fontsize=11)
        ax.set_xlabel('Sensor in Ring')
        if j == 0:
            ax.set_ylabel('Ring Index')
        fig.colorbar(im, ax=ax, label='RMS', shrink=0.8)

    fig.suptitle('Scattering RMS per Sensor Position', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'scattering_rms_heatmap.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: scattering_rms_heatmap.png")

    # ----------------------------------------------------------------
    # 4. Defect size vs scattering amplitude trend
    # ----------------------------------------------------------------
    radii = [30, 50, 80]
    rms_means = []
    peak_means = []
    energy_means = []

    for dwf in defect_wfs:
        scatter = dwf - healthy_wf
        rms_means.append(np.mean(np.sqrt(np.mean(scatter ** 2, axis=-1))))
        peak_means.append(np.mean(np.max(np.abs(scatter), axis=-1)))
        h_e = np.sum(healthy_wf ** 2, axis=-1) + 1e-12
        s_e = np.sum(scatter ** 2, axis=-1)
        energy_means.append(np.mean(s_e / h_e))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    ax1.plot(radii, rms_means, 'bo-', markersize=8)
    ax1.set_xlabel('Defect Radius [mm]')
    ax1.set_ylabel('Mean RMS')
    ax1.set_title('Scattering RMS vs Defect Size')
    ax1.grid(True, alpha=0.3)

    ax2.plot(radii, peak_means, 'ro-', markersize=8)
    ax2.set_xlabel('Defect Radius [mm]')
    ax2.set_ylabel('Mean Peak Amplitude')
    ax2.set_title('Scattering Peak vs Defect Size')
    ax2.grid(True, alpha=0.3)

    ax3.plot(radii, energy_means, 'go-', markersize=8)
    ax3.set_xlabel('Defect Radius [mm]')
    ax3.set_ylabel('Energy Ratio (scatter/healthy)')
    ax3.set_title('Energy Ratio vs Defect Size')
    ax3.grid(True, alpha=0.3)

    fig.suptitle('Physical Consistency: Defect Size → Scattering Amplitude', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'defect_size_trend.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: defect_size_trend.png")

    # ----------------------------------------------------------------
    # 5. FFT comparison per defect size
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(len(defect_labels), 1, figsize=(10, 3 * len(defect_labels)),
                              sharex=True)
    if len(defect_labels) == 1:
        axes = [axes]

    # Use a sensor near defect (middle ring, sensor 0)
    si_fft = n_sensors // 2
    healthy_fft_1 = np.abs(np.fft.rfft(healthy_wf[si_fft]))
    freqs = np.fft.rfftfreq(len(times), d=times[1] - times[0])
    freqs_khz = freqs / 1000

    for j, (dwf, dlabel) in enumerate(zip(defect_wfs, defect_labels)):
        ax = axes[j]
        defect_fft_1 = np.abs(np.fft.rfft(dwf[si_fft]))
        scatter_fft_1 = np.abs(np.fft.rfft(dwf[si_fft] - healthy_wf[si_fft]))

        ax.semilogy(freqs_khz, healthy_fft_1, 'k-', alpha=0.6, label='Healthy')
        ax.semilogy(freqs_khz, defect_fft_1, '-', color=colors[j], alpha=0.6, label=dlabel)
        ax.semilogy(freqs_khz, scatter_fft_1, '--', color=colors[j], alpha=0.8, label='Scatter')
        ax.set_ylabel('|FFT|')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_title(f'{dlabel} (Sensor {si_fft})', fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frequency [kHz]')
    fig.suptitle('Spectral Analysis: Healthy vs Defect vs Scattering', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'spectral_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: spectral_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='GW v3 FEM Validation')
    parser.add_argument('--data_dir', default='abaqus_work/gw_valid_v3')
    parser.add_argument('--output', default='runs/validate_v3')
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Load CSVs
    cases = {
        'Healthy': 'Job-GW-Valid-Healthy-v3_sensors.csv',
        'Debond30': 'Job-GW-Valid-Debond30-v3_sensors.csv',
        'Debond50': 'Job-GW-Valid-Debond50-v3_sensors.csv',
        'Debond80': 'Job-GW-Valid-Debond80-v3_sensors.csv',
    }

    waveforms = {}
    for label, fname in cases.items():
        path = os.path.join(data_dir, fname)
        print(f"Loading {label}: {path}")
        times, wf, positions = load_sensor_csv(path)
        if times is None:
            print(f"  ERROR: Failed to load {path}")
            return
        waveforms[label] = wf
        print(f"  Shape: {wf.shape} ({wf.shape[0]} sensors, {wf.shape[1]} timesteps)")
        print(f"  Time range: {times[0]*1000:.3f} - {times[-1]*1000:.3f} ms")

    healthy_wf = waveforms['Healthy']

    # Compute metrics
    print(f"\n{'='*60}")
    print("Physical Consistency Metrics")
    print(f"{'='*60}")

    all_metrics = []
    defect_labels = []
    defect_wfs = []

    for label in ['Debond30', 'Debond50', 'Debond80']:
        metrics = compute_scattering_metrics(healthy_wf, waveforms[label], label)
        all_metrics.append(metrics)
        defect_labels.append(label)
        defect_wfs.append(waveforms[label])

        print(f"\n--- {label} (r={label.replace('Debond', '')}mm) ---")
        print(f"  Scattering RMS  (mean): {metrics['rms_mean']:.6e}")
        print(f"  Scattering RMS  (max):  {metrics['rms_max']:.6e}")
        print(f"  Peak amplitude  (mean): {metrics['peak_mean']:.6e}")
        print(f"  Peak amplitude  (max):  {metrics['peak_max']:.6e}")
        print(f"  Energy ratio    (mean): {metrics['energy_ratio_mean']:.6e}")
        print(f"  Spectral ratio  (low):  {metrics['low_freq_ratio']:.4f}")
        print(f"  Spectral ratio  (mid):  {metrics['mid_freq_ratio']:.4f}")
        print(f"  Spectral ratio  (high): {metrics['high_freq_ratio']:.4f}")

    # Check monotonic trend
    rms_vals = [m['rms_mean'] for m in all_metrics]
    peak_vals = [m['peak_mean'] for m in all_metrics]

    print(f"\n{'='*60}")
    print("Trend Check (expected: D30 < D50 < D80)")
    print(f"{'='*60}")
    rms_ok = rms_vals[0] < rms_vals[1] < rms_vals[2]
    peak_ok = peak_vals[0] < peak_vals[1] < peak_vals[2]
    print(f"  RMS trend:  {'PASS' if rms_ok else 'FAIL'} "
          f"({rms_vals[0]:.2e} < {rms_vals[1]:.2e} < {rms_vals[2]:.2e})")
    print(f"  Peak trend: {'PASS' if peak_ok else 'FAIL'} "
          f"({peak_vals[0]:.2e} < {peak_vals[1]:.2e} < {peak_vals[2]:.2e})")

    # D80/D30 ratio (expect ~(80/30)^2 ≈ 7.1 for area-proportional scattering)
    if rms_vals[0] > 0:
        ratio_80_30 = rms_vals[2] / rms_vals[0]
        print(f"  D80/D30 RMS ratio: {ratio_80_30:.2f} "
              f"(area ratio: {(80/30)**2:.1f})")

    # Save metrics
    metrics_path = os.path.join(output_dir, 'validation_metrics.json')
    with open(metrics_path, 'w') as f:
        # Remove per-sensor arrays for readability in summary
        summary = []
        for m in all_metrics:
            s = {k: v for k, v in m.items()
                 if k not in ('rms_per_sensor', 'peak_per_sensor')}
            summary.append(s)
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # Plots
    if not args.no_plot:
        print("\nGenerating plots...")
        plot_validation(healthy_wf, defect_wfs, defect_labels, times, positions,
                        output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
