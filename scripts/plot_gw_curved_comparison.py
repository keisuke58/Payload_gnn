# -*- coding: utf-8 -*-
# plot_gw_curved_comparison.py
# Compare flat panel vs curved fairing guided wave results.
#
# Usage: python scripts/plot_gw_curved_comparison.py

import os
import csv
import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(csv_path):
    """Load sensor CSV. Returns (times, sensor_dict, x_positions)."""
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
                        x_positions.append(j * 30.0)
                continue
            try:
                times.append(float(row[0]))
                for i in range(n_sensors):
                    val = row[i + 1] if i + 1 < len(row) else ''
                    sensors[i].append(float(val) if val else 0.0)
            except (ValueError, IndexError):
                continue

    return (np.array(times),
            {k: np.array(v) for k, v in sensors.items()},
            x_positions)


def _hilbert_envelope(signal):
    """Compute Hilbert envelope with Tukey window."""
    n = len(signal)
    win = np.ones(n)
    taper = min(50, n // 10)
    win[:taper] = 0.5 * (1 - np.cos(np.pi * np.arange(taper) / taper))
    win[-taper:] = 0.5 * (1 - np.cos(np.pi * np.arange(taper, 0, -1) / taper))
    return np.abs(hilbert(signal * win))


def plot_flat_vs_curved(flat_h_csv, flat_d_csv, curved_h_csv, curved_d_csv,
                        out_dir='.'):
    """Create comparison plots: flat panel vs curved fairing."""
    t_fh, s_fh, x_fh = load_csv(flat_h_csv)
    t_fd, s_fd, x_fd = load_csv(flat_d_csv)
    t_ch, s_ch, x_ch = load_csv(curved_h_csv)
    t_cd, s_cd, x_cd = load_csv(curved_d_csv)

    n_sensors = min(len(s_fh), len(s_ch))

    # --- Plot 1: Healthy waveform comparison ---
    fig, axes = plt.subplots(n_sensors, 1, figsize=(14, 3 * n_sensors),
                             sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for i in range(n_sensors):
        ax = axes[i]
        t_us_f = t_fh * 1e6
        t_us_c = t_ch * 1e6

        # Normalize to compare shape (different absolute values due to geometry)
        norm_f = np.max(np.abs(s_fh[i])) if np.max(np.abs(s_fh[i])) > 1e-20 else 1
        norm_c = np.max(np.abs(s_ch[i])) if np.max(np.abs(s_ch[i])) > 1e-20 else 1

        ax.plot(t_us_f, s_fh[i] / norm_f, 'b-', linewidth=0.8, alpha=0.8,
                label='Flat (U3)')
        ax.plot(t_us_c, s_ch[i] / norm_c, 'r-', linewidth=0.8, alpha=0.8,
                label='Curved (Ur)')

        offset_f = x_fh[i] if i < len(x_fh) else i * 30
        offset_c = x_ch[i] if i < len(x_ch) else i * 30
        ax.set_ylabel('Normalized', fontsize=10)
        ax.set_title('Sensor %d (flat:x=%.0f, curved:arc=%.0f mm)' % (
            i, offset_f, offset_c), fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [us]', fontsize=11)
    fig.suptitle('Healthy Panel: Flat vs Curved (R=2600mm)\n'
                 'Normalized waveform comparison',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'gw_flat_vs_curved_healthy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print("Saved: %s" % path)
    plt.close()

    # --- Plot 2: Scattered wave comparison ---
    fig, axes = plt.subplots(n_sensors, 1, figsize=(14, 3 * n_sensors),
                             sharex=True)
    if n_sensors == 1:
        axes = [axes]

    n_pts_f = min(len(t_fh), len(t_fd))
    n_pts_c = min(len(t_ch), len(t_cd))

    for i in range(n_sensors):
        ax = axes[i]
        # Flat scattered
        diff_f = s_fd[i][:n_pts_f] - s_fh[i][:n_pts_f]
        # Curved scattered
        diff_c = s_cd[i][:n_pts_c] - s_ch[i][:n_pts_c]

        norm_df = np.max(np.abs(diff_f)) if np.max(np.abs(diff_f)) > 1e-20 else 1
        norm_dc = np.max(np.abs(diff_c)) if np.max(np.abs(diff_c)) > 1e-20 else 1

        ax.plot(t_fh[:n_pts_f] * 1e6, diff_f / norm_df, 'b-', linewidth=0.8,
                alpha=0.8, label='Flat (scattered)')
        ax.plot(t_ch[:n_pts_c] * 1e6, diff_c / norm_dc, 'r-', linewidth=0.8,
                alpha=0.8, label='Curved (scattered)')

        offset_f = x_fh[i] if i < len(x_fh) else i * 30
        ax.set_ylabel('Normalized', fontsize=10)
        ax.set_title('Sensor %d: Scattered Wave (Defect-Healthy)' % i,
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [us]', fontsize=11)
    fig.suptitle('Scattered Wave: Flat vs Curved (r=25mm debonding)\n'
                 'Curvature effect on defect detection',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'gw_flat_vs_curved_scattered.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print("Saved: %s" % path)
    plt.close()

    # --- Plot 3: Damage Index comparison (bar chart) ---
    di_flat = []
    di_curved = []
    for i in range(n_sensors):
        inc_f = np.sqrt(np.mean(s_fh[i][:n_pts_f] ** 2))
        scat_f = np.sqrt(np.mean(diff_f ** 2)) if i == n_sensors - 1 else \
            np.sqrt(np.mean((s_fd[i][:n_pts_f] - s_fh[i][:n_pts_f]) ** 2))
        inc_c = np.sqrt(np.mean(s_ch[i][:n_pts_c] ** 2))
        scat_c = np.sqrt(np.mean(diff_c ** 2)) if i == n_sensors - 1 else \
            np.sqrt(np.mean((s_cd[i][:n_pts_c] - s_ch[i][:n_pts_c]) ** 2))

        # Recompute diff for correct sensor index
        diff_fi = s_fd[i][:n_pts_f] - s_fh[i][:n_pts_f]
        diff_ci = s_cd[i][:n_pts_c] - s_ch[i][:n_pts_c]
        scat_f = np.sqrt(np.mean(diff_fi ** 2))
        scat_c = np.sqrt(np.mean(diff_ci ** 2))

        di_flat.append(scat_f / inc_f if inc_f > 1e-20 else 0)
        di_curved.append(scat_c / inc_c if inc_c > 1e-20 else 0)

    x = np.arange(n_sensors)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, di_flat, width, label='Flat Panel', color='steelblue',
           edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, di_curved, width, label='Curved (R=2600mm)',
           color='indianred', edgecolor='black', linewidth=0.5)

    sensor_labels = ['S%d (arc=%d)' % (i, x_fh[i]) if i < len(x_fh)
                     else 'S%d' % i for i in range(n_sensors)]
    ax.set_xticks(x)
    ax.set_xticklabels(sensor_labels, fontsize=10)
    ax.set_ylabel('Damage Index', fontsize=11)
    ax.set_title('Damage Index: Flat vs Curved Panel (r=25mm debonding)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(out_dir, 'gw_flat_vs_curved_di.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print("Saved: %s" % path)
    plt.close()

    # Print DI table
    print("\n--- Flat vs Curved DI ---")
    print("%-10s" % "Sensor", end='')
    for i in range(n_sensors):
        print("  S%d" % i, end='')
    print("  Mean")
    print("%-10s" % "Flat", end='')
    for di in di_flat:
        print("  %.3f" % di, end='')
    print("  %.3f" % np.mean(di_flat))
    print("%-10s" % "Curved", end='')
    for di in di_curved:
        print("  %.3f" % di, end='')
    print("  %.3f" % np.mean(di_curved))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--flat_h', default='abaqus_work/Job-GW-Healthy-v2_sensors.csv')
    parser.add_argument('--flat_d', default='abaqus_work/Job-GW-Debond-v2_sensors.csv')
    parser.add_argument('--curved_h', default='abaqus_work/Job-GW-Curved-H_sensors.csv')
    parser.add_argument('--curved_d', default='abaqus_work/Job-GW-Curved-D_sensors.csv')
    parser.add_argument('--out_dir', default='wiki_repo/images/guided_wave')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_flat_vs_curved(args.flat_h, args.flat_d, args.curved_h, args.curved_d,
                        args.out_dir)
