# -*- coding: utf-8 -*-
# plot_gw_freq_noise.py
# Guided wave frequency sweep analysis + noise robustness testing.
#
# Part A: DI vs Frequency plot (from frequency sweep CSVs)
# Part B: DI vs SNR plot (noise robustness on existing data)
#
# Usage:
#   python scripts/plot_gw_freq_noise.py --mode freq  [--csv_dir abaqus_work]
#   python scripts/plot_gw_freq_noise.py --mode noise [--csv_dir abaqus_work]
#   python scripts/plot_gw_freq_noise.py --mode both  [--csv_dir abaqus_work]

import sys
import os
import glob
import re
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(csv_path):
    """Load sensor CSV. Returns (times, sensor_dict, x_positions, label)."""
    times = []
    sensors = {}
    x_positions = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        n_sensors = len(header) - 1
        for i in range(n_sensors):
            sensors[i] = []

        for row in reader:
            if row[0].startswith('#'):
                x_positions = []
                for j in range(1, min(len(row), n_sensors + 1)):
                    try:
                        x_positions.append(float(row[j]))
                    except (ValueError, IndexError):
                        x_positions.append(j * 50.0)
                continue
            try:
                times.append(float(row[0]))
                for i in range(n_sensors):
                    val = row[i + 1] if i + 1 < len(row) else ''
                    sensors[i].append(float(val) if val else 0.0)
            except (ValueError, IndexError):
                continue

    basename = os.path.splitext(os.path.basename(csv_path))[0]
    label = basename.replace('_sensors', '')

    return (np.array(times),
            {k: np.array(v) for k, v in sensors.items()},
            x_positions,
            label)


def compute_mean_di(t_h, s_h, t_d, s_d):
    """Compute mean DI across sensors."""
    n_sensors = min(len(s_h), len(s_d))
    n_pts = min(len(t_h), len(t_d))
    di_list = []
    for i in range(n_sensors):
        incident_rms = np.sqrt(np.mean(s_h[i][:n_pts] ** 2))
        scattered_rms = np.sqrt(np.mean((s_d[i][:n_pts] - s_h[i][:n_pts]) ** 2))
        di = scattered_rms / incident_rms if incident_rms > 1e-20 else 0.0
        di_list.append(di)
    return np.array(di_list)


# =========================================================================
# Part A: Frequency Sweep
# =========================================================================

def plot_freq_sweep(csv_dir, out_dir):
    """Plot DI vs Frequency from frequency sweep data."""
    freqs = [25, 50, 75, 100]
    mean_dis = []
    per_sensor_dis = []

    for freq in freqs:
        h_path = os.path.join(csv_dir, 'Job-GW-Freq%dk-H_sensors.csv' % freq)
        d_path = os.path.join(csv_dir, 'Job-GW-Freq%dk-D_sensors.csv' % freq)

        if not os.path.exists(h_path) or not os.path.exists(d_path):
            # Try existing 50kHz data
            if freq == 50:
                h_path = os.path.join(csv_dir, 'Job-GW-Healthy-v2_sensors.csv')
                d_path = os.path.join(csv_dir, 'Job-GW-Debond-v2_sensors.csv')
            if not os.path.exists(h_path) or not os.path.exists(d_path):
                print("  Skip %d kHz: files not found" % freq)
                mean_dis.append(np.nan)
                per_sensor_dis.append(np.full(5, np.nan))
                continue

        t_h, s_h, _, _ = load_csv(h_path)
        t_d, s_d, _, _ = load_csv(d_path)
        di_arr = compute_mean_di(t_h, s_h, t_d, s_d)
        mean_dis.append(np.mean(di_arr))
        per_sensor_dis.append(di_arr)
        print("  %d kHz: mean DI = %.3f" % (freq, np.mean(di_arr)))

    mean_dis = np.array(mean_dis)
    per_sensor_dis = np.array(per_sensor_dis)  # (n_freq, n_sensors)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Mean DI vs Frequency
    valid = ~np.isnan(mean_dis)
    ax1.plot(np.array(freqs)[valid], mean_dis[valid], 'o-', color='red',
             linewidth=2, markersize=8)
    ax1.set_xlabel('Frequency [kHz]', fontsize=12)
    ax1.set_ylabel('Mean Damage Index', fontsize=12)
    ax1.set_title('Detection vs Frequency\n(D1: r=25mm, x=80mm)',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(freqs)

    # Right: Per-sensor DI vs Frequency
    sensor_colors = ['blue', 'orange', 'green', 'red', 'purple']
    n_s = per_sensor_dis.shape[1] if len(per_sensor_dis.shape) > 1 else 0
    for s in range(n_s):
        vals = per_sensor_dis[valid, s] if valid.any() else []
        ax2.plot(np.array(freqs)[valid], vals, 'o-', color=sensor_colors[s % 5],
                 linewidth=1.5, markersize=6, label='S%d' % s)
    ax2.set_xlabel('Frequency [kHz]', fontsize=12)
    ax2.set_ylabel('Damage Index per Sensor', fontsize=12)
    ax2.set_title('Per-Sensor Detection vs Frequency',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(freqs)

    plt.tight_layout()
    path = os.path.join(out_dir, 'gw_freq_sweep_di.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print("Saved: %s" % path)
    plt.close()

    # Print table
    print("\n--- Frequency Sweep DI ---")
    print("Freq(kHz)  Mean_DI  S0     S1     S2     S3     S4")
    for j, freq in enumerate(freqs):
        if np.isnan(mean_dis[j]):
            print("%-10d N/A" % freq)
        else:
            print("%-10d %.3f   %s" % (
                freq, mean_dis[j],
                "  ".join("%.3f" % per_sensor_dis[j, s] for s in range(n_s))))


# =========================================================================
# Part B: Noise Robustness
# =========================================================================

def add_noise_to_signal(signal, snr_db):
    """Add Gaussian noise at specified SNR (dB)."""
    signal_power = np.mean(signal ** 2)
    if signal_power < 1e-30:
        return signal.copy()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def compute_di_with_noise(t_h, s_h, t_d, s_d, snr_db, n_trials=20):
    """Compute DI with noise injection. Returns (mean_DI, std_DI)."""
    n_sensors = min(len(s_h), len(s_d))
    n_pts = min(len(t_h), len(t_d))

    trial_mean_dis = []
    for _ in range(n_trials):
        di_list = []
        for i in range(n_sensors):
            h_noisy = add_noise_to_signal(s_h[i][:n_pts], snr_db)
            d_noisy = add_noise_to_signal(s_d[i][:n_pts], snr_db)
            incident_rms = np.sqrt(np.mean(h_noisy ** 2))
            scattered_rms = np.sqrt(np.mean((d_noisy - h_noisy) ** 2))
            di = scattered_rms / incident_rms if incident_rms > 1e-20 else 0.0
            di_list.append(di)
        trial_mean_dis.append(np.mean(di_list))

    return np.mean(trial_mean_dis), np.std(trial_mean_dis)


def plot_noise_robustness(csv_dir, out_dir):
    """Plot DI vs SNR for noise robustness analysis."""
    # Use D1-D5 + healthy data
    case_files = {
        'D1 (r=25, x=80)':  ('Job-GW-Healthy-v2', 'Job-GW-Debond-v2'),
        'D2 (r=15, x=80)':  ('Job-GW-Healthy-v2', 'Job-GW-D2-r15'),
        'D4 (r=25, x=40)':  ('Job-GW-Healthy-v2', 'Job-GW-D4-near'),
        'D5 (r=25, x=120)': ('Job-GW-Healthy-v2', 'Job-GW-D5-edge'),
    }

    snr_levels = [40, 30, 25, 20, 15, 10, 5]  # dB
    colors = ['red', 'orange', 'green', 'purple']

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (case_label, (h_name, d_name)) in enumerate(case_files.items()):
        h_path = os.path.join(csv_dir, '%s_sensors.csv' % h_name)
        d_path = os.path.join(csv_dir, '%s_sensors.csv' % d_name)

        if not os.path.exists(h_path) or not os.path.exists(d_path):
            print("  Skip %s: files not found" % case_label)
            continue

        t_h, s_h, _, _ = load_csv(h_path)
        t_d, s_d, _, _ = load_csv(d_path)

        # Clean DI (no noise)
        di_clean = np.mean(compute_mean_di(t_h, s_h, t_d, s_d))

        means = [di_clean]
        stds = [0.0]
        snrs_with_clean = [50]  # treat clean as 50 dB

        for snr in snr_levels:
            m, s = compute_di_with_noise(t_h, s_h, t_d, s_d, snr)
            means.append(m)
            stds.append(s)
            snrs_with_clean.append(snr)

        means = np.array(means)
        stds = np.array(stds)

        ax.errorbar(snrs_with_clean, means, yerr=stds,
                    fmt='o-', color=colors[idx % len(colors)],
                    linewidth=2, markersize=6, capsize=3,
                    label='%s (clean=%.2f)' % (case_label, di_clean))

    # Detection threshold line
    ax.axhline(y=0.2, color='gray', ls='--', alpha=0.7, linewidth=1.5)
    ax.text(7, 0.22, 'Detection threshold (DI=0.2)', fontsize=9, color='gray')

    ax.set_xlabel('Signal-to-Noise Ratio [dB]', fontsize=12)
    ax.set_ylabel('Mean Damage Index', fontsize=12)
    ax.set_title('Noise Robustness: DI vs SNR\n'
                 '(20 Monte Carlo trials per point, Gaussian noise)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Higher SNR (better) on left
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'gw_noise_robustness.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print("Saved: %s" % path)
    plt.close()

    # Print table
    print("\n--- Noise Robustness Table ---")
    print("Case                 Clean   40dB   30dB   25dB   20dB   15dB   10dB   5dB")
    for case_label, (h_name, d_name) in case_files.items():
        h_path = os.path.join(csv_dir, '%s_sensors.csv' % h_name)
        d_path = os.path.join(csv_dir, '%s_sensors.csv' % d_name)
        if not os.path.exists(h_path) or not os.path.exists(d_path):
            continue
        t_h, s_h, _, _ = load_csv(h_path)
        t_d, s_d, _, _ = load_csv(d_path)
        di_clean = np.mean(compute_mean_di(t_h, s_h, t_d, s_d))
        vals = [di_clean]
        for snr in snr_levels:
            m, _ = compute_di_with_noise(t_h, s_h, t_d, s_d, snr)
            vals.append(m)
        print("%-20s %s" % (case_label,
                            "  ".join("%.3f" % v for v in vals)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Guided wave frequency sweep & noise robustness analysis')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['freq', 'noise', 'both'],
                        help='Analysis mode (default: both)')
    parser.add_argument('--csv_dir', type=str, default='abaqus_work',
                        help='Directory with sensor CSVs (default: abaqus_work)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory for plots')
    args = parser.parse_args()

    out_dir = args.out_dir or args.csv_dir
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(42)

    if args.mode in ('freq', 'both'):
        print("=== Part A: Frequency Sweep ===")
        plot_freq_sweep(args.csv_dir, out_dir)

    if args.mode in ('noise', 'both'):
        print("\n=== Part B: Noise Robustness ===")
        plot_noise_robustness(args.csv_dir, out_dir)
