# -*- coding: utf-8 -*-
# plot_gw_sweep.py
# Compare guided wave sensor waveforms across multiple defect configurations.
#
# Usage: python scripts/plot_gw_sweep.py abaqus_work/Job-GW-*_sensors.csv
#        python scripts/plot_gw_sweep.py --out_dir wiki_repo/images/guided_wave

import sys
import os
import glob
import re
import csv
import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

    # Extract label from filename
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    label = basename.replace('_sensors', '')

    return (np.array(times),
            {k: np.array(v) for k, v in sensors.items()},
            x_positions,
            label)


def _hilbert_envelope(signal):
    """Compute Hilbert envelope with Tukey window."""
    n = len(signal)
    win = np.ones(n)
    taper = min(50, n // 10)
    win[:taper] = 0.5 * (1 - np.cos(np.pi * np.arange(taper) / taper))
    win[-taper:] = 0.5 * (1 - np.cos(np.pi * np.arange(taper, 0, -1) / taper))
    return np.abs(hilbert(signal * win))


# Defect case metadata
CASE_META = {
    'Job-GW-Healthy-v2': {'label': 'Healthy', 'color': 'blue', 'ls': '-',
                          'defect': None},
    'Job-GW-Debond-v2':  {'label': 'D1: r=25, x=80', 'color': 'red', 'ls': '-',
                          'defect': {'x': 80, 'r': 25}},
    'Job-GW-D2-r15':     {'label': 'D2: r=15, x=80', 'color': 'orange', 'ls': '-',
                          'defect': {'x': 80, 'r': 15}},
    'Job-GW-D3-r40':     {'label': 'D3: r=40, x=80', 'color': 'darkred', 'ls': '-',
                          'defect': {'x': 80, 'r': 40}},
    'Job-GW-D4-near':    {'label': 'D4: r=25, x=40', 'color': 'green', 'ls': '-',
                          'defect': {'x': 40, 'r': 25}},
    'Job-GW-D5-edge':    {'label': 'D5: r=25, x=120', 'color': 'purple', 'ls': '-',
                          'defect': {'x': 120, 'r': 25}},
}


def _get_meta(job_name):
    """Get metadata for a job, with fallback."""
    if job_name in CASE_META:
        return CASE_META[job_name]
    return {'label': job_name, 'color': 'gray', 'ls': '-', 'defect': None}


def plot_sweep_waveforms(csv_files, out_dir='.'):
    """Plot waveform comparison for selected sensors across all cases."""
    datasets = []
    for f in csv_files:
        t, s, x, label = load_csv(f)
        datasets.append((t, s, x, label))

    if not datasets:
        print("No data loaded")
        return

    # Use sensors 2 and 3 (near/in defect zone) for main comparison
    key_sensors = [2, 3]
    n_sensors = len(datasets[0][1])

    # --- Plot 1: All sensors, all cases ---
    fig, axes = plt.subplots(n_sensors, 1, figsize=(16, 3 * n_sensors),
                             sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for t, s, x_pos, label in datasets:
        meta = _get_meta(label)
        for i in range(min(n_sensors, len(s))):
            ax = axes[i]
            t_us = t * 1e6
            ax.plot(t_us, s[i], color=meta['color'], ls=meta['ls'],
                    linewidth=0.7 if meta['defect'] else 1.2,
                    alpha=0.8, label=meta['label'])

    for i in range(n_sensors):
        ax = axes[i]
        x_mm = datasets[0][2][i] if i < len(datasets[0][2]) else i * 30
        ax.set_ylabel('U3 [mm]', fontsize=10)
        ax.set_title('Sensor %d (x = %.0f mm)' % (i, x_mm),
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=2)

    axes[-1].set_xlabel('Time [us]', fontsize=11)
    fig.suptitle('Guided Wave Sweep: All Sensors\n'
                 '(50 kHz A0 mode, 300x300 mm panel)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'gw_sweep_waveforms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print("Saved: %s" % path)
    plt.close()

    # --- Plot 2: Scattered waves (difference from Healthy) ---
    # Find healthy baseline
    healthy_data = None
    for t, s, x, label in datasets:
        if 'Healthy' in label or 'healthy' in label.lower():
            healthy_data = (t, s, x, label)
            break

    if healthy_data is None:
        print("No Healthy baseline found, skipping difference plot")
        return

    t_h, s_h, x_h, _ = healthy_data
    defect_datasets = [(t, s, x, label) for t, s, x, label in datasets
                       if label != healthy_data[3]]

    fig, axes = plt.subplots(n_sensors, 1, figsize=(16, 3 * n_sensors),
                             sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for t_d, s_d, _, label in defect_datasets:
        meta = _get_meta(label)
        n_pts = min(len(t_h), len(t_d))
        for i in range(min(n_sensors, len(s_d), len(s_h))):
            ax = axes[i]
            t_us = t_h[:n_pts] * 1e6
            diff = s_d[i][:n_pts] - s_h[i][:n_pts]
            ax.plot(t_us, diff, color=meta['color'], linewidth=0.8,
                    alpha=0.8, label=meta['label'])

    for i in range(n_sensors):
        ax = axes[i]
        x_mm = x_h[i] if i < len(x_h) else i * 30
        ax.set_ylabel('dU3 [mm]', fontsize=10)
        ax.set_title('Sensor %d (x = %.0f mm): Scattered Wave' % (i, x_mm),
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=2)

    axes[-1].set_xlabel('Time [us]', fontsize=11)
    fig.suptitle('Scattered Wave Comparison (Defect - Healthy)\n'
                 'Effect of defect size and position',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'gw_sweep_scattered.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print("Saved: %s" % path)
    plt.close()

    # --- Plot 3: Damage Index bar chart ---
    plot_damage_index(datasets, healthy_data, out_dir)


def plot_damage_index(datasets, healthy_data, out_dir='.'):
    """Compute and plot damage index (DI) for each defect case.

    DI = norm(scattered) / norm(incident) per sensor.
    """
    t_h, s_h, x_h, _ = healthy_data
    defect_datasets = [(t, s, x, label) for t, s, x, label in datasets
                       if label != healthy_data[3]]

    n_sensors = len(s_h)
    case_labels = []
    di_matrix = []  # [case][sensor]

    for t_d, s_d, _, label in defect_datasets:
        meta = _get_meta(label)
        case_labels.append(meta['label'])
        n_pts = min(len(t_h), len(t_d))
        di_row = []
        for i in range(min(n_sensors, len(s_d), len(s_h))):
            incident = np.sqrt(np.mean(s_h[i][:n_pts] ** 2))
            scattered = np.sqrt(np.mean((s_d[i][:n_pts] - s_h[i][:n_pts]) ** 2))
            di = scattered / incident if incident > 1e-20 else 0.0
            di_row.append(di)
        di_matrix.append(di_row)

    di_matrix = np.array(di_matrix)

    # Bar chart
    n_cases = len(case_labels)
    n_s = di_matrix.shape[1] if len(di_matrix) > 0 else 0
    x = np.arange(n_s)
    width = 0.8 / max(n_cases, 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [_get_meta(d[3])['color'] for d in defect_datasets]

    for j in range(n_cases):
        offset = (j - n_cases / 2 + 0.5) * width
        ax.bar(x + offset, di_matrix[j], width, label=case_labels[j],
               color=colors[j], alpha=0.85, edgecolor='black', linewidth=0.5)

    sensor_labels = ['S%d (x=%d)' % (i, x_h[i]) if i < len(x_h)
                     else 'S%d' % i for i in range(n_s)]
    ax.set_xticks(x)
    ax.set_xticklabels(sensor_labels, fontsize=10)
    ax.set_ylabel('Damage Index (RMS scattered / RMS incident)', fontsize=11)
    ax.set_title('Damage Index by Sensor and Defect Configuration',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(out_dir, 'gw_sweep_damage_index.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print("Saved: %s" % path)
    plt.close()

    # Print DI table
    print("\n--- Damage Index Table ---")
    print("%-20s" % "Case", end='')
    for i in range(n_s):
        print("  S%d" % i, end='')
    print("  Mean")
    for j in range(n_cases):
        print("%-20s" % case_labels[j], end='')
        for i in range(n_s):
            print("  %.3f" % di_matrix[j, i], end='')
        print("  %.3f" % np.mean(di_matrix[j]))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_files', nargs='*',
                        help='Sensor CSV files (glob patterns ok)')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory for plots')
    args = parser.parse_args()

    # Expand glob patterns
    csv_files = []
    for pattern in args.csv_files:
        expanded = sorted(glob.glob(pattern))
        csv_files.extend(expanded)

    if not csv_files:
        # Default: find all in abaqus_work/
        csv_files = sorted(glob.glob('abaqus_work/Job-GW-*_sensors.csv'))

    if not csv_files:
        print("No CSV files found")
        sys.exit(1)

    print("Found %d CSV files:" % len(csv_files))
    for f in csv_files:
        print("  %s" % f)

    out_dir = args.out_dir or os.path.dirname(csv_files[0]) or '.'
    os.makedirs(out_dir, exist_ok=True)

    plot_sweep_waveforms(csv_files, out_dir)
