# -*- coding: utf-8 -*-
# plot_gw_comparison.py
# Plot guided wave sensor waveforms: Healthy vs Debonding comparison.
#
# Usage: python scripts/plot_gw_comparison.py \
#            abaqus_work/Job-GW-Healthy_sensors.csv \
#            abaqus_work/Job-GW-Debond_sensors.csv

import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(csv_path):
    """Load sensor CSV data. Returns (times, sensor_dict, x_positions).

    Handles optional comment row (starts with '#') for actual X positions.
    """
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
            # Skip comment rows (actual position metadata)
            if row[0].startswith('#'):
                # Parse X positions from metadata row
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

    return (np.array(times),
            {k: np.array(v) for k, v in sensors.items()},
            x_positions)


def plot_waveforms(csv_healthy, csv_defect, output_path='gw_comparison.png'):
    """Plot healthy vs defect waveforms for each sensor."""
    t_h, s_h, x_h = load_csv(csv_healthy)
    t_d, s_d, x_d = load_csv(csv_defect)

    n_sensors = min(len(s_h), len(s_d))
    x_pos = x_h if x_h else x_d  # use whichever has positions

    fig, axes = plt.subplots(n_sensors, 1, figsize=(14, 3 * n_sensors),
                             sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for i in range(n_sensors):
        ax = axes[i]
        t_us_h = t_h * 1e6
        t_us_d = t_d * 1e6

        ax.plot(t_us_h, s_h[i], 'b-', linewidth=0.8, alpha=0.9,
                label='Healthy')
        ax.plot(t_us_d, s_d[i], 'r-', linewidth=0.8, alpha=0.7,
                label='Debonding')

        offset = x_pos[i] if i < len(x_pos) else i * 50
        ax.set_ylabel('U3 [mm]', fontsize=10)
        ax.set_title('Sensor %d (x = %.0f mm from center)' % (i, offset),
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    axes[-1].set_xlabel('Time [us]', fontsize=11)
    fig.suptitle('Guided Wave A0 Mode: Healthy vs Debonding\n'
                 '(50 kHz, 5-cycle Hanning tone burst, 300x300 mm panel)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Plot saved: %s" % output_path)
    plt.close()


def plot_difference(csv_healthy, csv_defect, output_path='gw_difference.png'):
    """Plot the difference signal (defect - healthy) at each sensor."""
    t_h, s_h, x_h = load_csv(csv_healthy)
    t_d, s_d, x_d = load_csv(csv_defect)

    n_sensors = min(len(s_h), len(s_d))
    n_points = min(len(t_h), len(t_d))
    x_pos = x_h if x_h else x_d

    fig, axes = plt.subplots(n_sensors, 1, figsize=(14, 3 * n_sensors),
                             sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for i in range(n_sensors):
        ax = axes[i]
        t_us = t_h[:n_points] * 1e6
        diff = s_d[i][:n_points] - s_h[i][:n_points]

        ax.plot(t_us, diff, 'k-', linewidth=0.8)
        ax.fill_between(t_us, diff, alpha=0.3, color='red')

        offset = x_pos[i] if i < len(x_pos) else i * 50
        ax.set_ylabel('dU3 [mm]', fontsize=10)
        ax.set_title('Sensor %d (x = %.0f mm): Scattered wave (Defect - Healthy)' %
                     (i, offset), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    axes[-1].set_xlabel('Time [us]', fontsize=11)
    fig.suptitle('Scattered Wave Field due to Debonding\n'
                 '(Difference signal: identifies defect-induced reflections)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Plot saved: %s" % output_path)
    plt.close()


def compute_and_print_velocity(csv_path, label=''):
    """Compute group velocity from peak arrival times using actual positions."""
    t, sensors, x_pos = load_csv(csv_path)
    print("\n--- Group Velocity: %s ---" % label)

    if not x_pos:
        x_pos = [0.0, 50.0, 100.0, 150.0, 200.0]

    peak_times = []
    for i in sorted(sensors.keys()):
        idx_peak = np.argmax(np.abs(sensors[i]))
        t_peak = t[idx_peak]
        u3_peak = sensors[i][idx_peak]
        offset = x_pos[i] if i < len(x_pos) else i * 50
        peak_times.append((t_peak, offset))
        print("  Sensor %d (x=%.0f mm): t_peak=%.3f us, |U3|_max=%.3e mm" %
              (i, offset, t_peak * 1e6, abs(u3_peak)))

    # Velocity from consecutive pairs (skip sensor 0 at excitation point)
    velocities = []
    for i in range(1, len(peak_times) - 1):
        dt = peak_times[i + 1][0] - peak_times[i][0]
        dx = peak_times[i + 1][1] - peak_times[i][1]
        if abs(dx) < 1.0:
            print("  Sensor %d->%d: dx~0 (same position), skip" % (i, i + 1))
            continue
        if abs(dt) > 1e-10:
            v = dx / dt / 1000.0  # m/s
            velocities.append(v)
            print("  Sensor %d->%d: dx=%.0f mm, dt=%.3f us -> v = %.0f m/s" %
                  (i, i + 1, dx, dt * 1e6, v))

    if velocities:
        v_avg = np.mean(velocities)
        print("  Average: %.0f m/s (theory ~1550 m/s, dev %.1f%%)" %
              (v_avg, abs(v_avg - 1550) / 1550 * 100))
    return velocities


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python plot_gw_comparison.py <healthy.csv> <defect.csv>")
        sys.exit(1)

    csv_h = sys.argv[1]
    csv_d = sys.argv[2]
    out_dir = os.path.dirname(csv_h) or '.'

    # Group velocity
    compute_and_print_velocity(csv_h, 'Healthy')
    compute_and_print_velocity(csv_d, 'Debonding')

    # Plots
    plot_waveforms(csv_h, csv_d,
                   os.path.join(out_dir, 'gw_comparison.png'))
    plot_difference(csv_h, csv_d,
                    os.path.join(out_dir, 'gw_difference.png'))
