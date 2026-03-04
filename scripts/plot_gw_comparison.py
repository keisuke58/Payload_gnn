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
from scipy.signal import hilbert
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


def _envelope_arrival(t, signal, threshold_frac=0.05, min_time_us=5.0):
    """Find envelope peak time and first-arrival time via Hilbert transform.

    Returns: (t_peak, t_arrival, envelope)
      t_peak: time of envelope maximum
      t_arrival: first time envelope exceeds threshold_frac * peak
                 (after min_time_us to avoid edge artifacts)
    """
    # Apply Tukey window to suppress Hilbert edge artifacts
    n = len(signal)
    win = np.ones(n)
    taper = min(50, n // 10)
    win[:taper] = 0.5 * (1 - np.cos(np.pi * np.arange(taper) / taper))
    win[-taper:] = 0.5 * (1 - np.cos(np.pi * np.arange(taper, 0, -1) / taper))

    env = np.abs(hilbert(signal * win))
    idx_peak = np.argmax(env)
    t_peak = t[idx_peak]
    env_max = env[idx_peak]

    # First arrival: envelope > threshold, after minimum time
    threshold = threshold_frac * env_max
    t_min = min_time_us * 1e-6
    arrival_indices = np.where((env > threshold) & (t > t_min))[0]
    t_arrival = t[arrival_indices[0]] if len(arrival_indices) > 0 else t_peak

    return t_peak, t_arrival, env


def compute_and_print_velocity(csv_path, label=''):
    """Compute group velocity using Hilbert envelope and actual positions.

    Two methods:
      1. Envelope peak (max of analytic signal envelope)
      2. First arrival (envelope crosses 10% of peak)
    """
    t, sensors, x_pos = load_csv(csv_path)
    print("\n--- Group Velocity: %s ---" % label)

    if not x_pos:
        x_pos = [0.0, 50.0, 100.0, 150.0, 200.0]

    peak_data = []  # (x, t_peak, t_arrival, env_max)
    for i in sorted(sensors.keys()):
        offset = x_pos[i] if i < len(x_pos) else i * 50
        t_peak, t_arrival, env = _envelope_arrival(t, sensors[i])
        env_max = np.max(env)
        peak_data.append((offset, t_peak, t_arrival, env_max))
        print("  Sensor %d (x=%.0f mm): t_peak=%.1f us, t_arrival=%.1f us, "
              "|env|_max=%.3e" %
              (i, offset, t_peak * 1e6, t_arrival * 1e6, env_max))

    # Method 1: pairwise envelope peak
    print("\n  Method 1: Envelope peak (pairwise)")
    v_peak = _compute_pairwise_velocity(peak_data, method='peak')

    # Method 2: pairwise first arrival
    print("\n  Method 2: First arrival (pairwise)")
    v_arrival = _compute_pairwise_velocity(peak_data, method='arrival')

    # Method 3: reference-based (sensor 0 → each sensor)
    print("\n  Method 3: Reference-based (sensor 0 → each)")
    v_ref = _compute_reference_velocity(peak_data)

    return v_peak, v_arrival, v_ref


def _compute_pairwise_velocity(peak_data, method='peak'):
    """Compute velocities from consecutive sensor pairs."""
    t_idx = 1 if method == 'peak' else 2  # index into peak_data tuple
    velocities = []
    for i in range(1, len(peak_data) - 1):
        dx = peak_data[i + 1][0] - peak_data[i][0]
        dt = peak_data[i + 1][t_idx] - peak_data[i][t_idx]
        if abs(dx) < 1.0:
            print("    Sensor %d->%d: dx~0, skip" % (i, i + 1))
            continue
        if abs(dt) > 1e-10:
            v = dx / dt / 1000.0  # m/s
            velocities.append(v)
            print("    Sensor %d->%d: dx=%.0f mm, dt=%.1f us -> v = %.0f m/s" %
                  (i, i + 1, dx, dt * 1e6, v))
        else:
            print("    Sensor %d->%d: dt~0 (near-field)" % (i, i + 1))

    if velocities:
        v_avg = np.mean(velocities)
        print("    Average: %.0f m/s (theory ~1550 m/s, dev %.1f%%)" %
              (v_avg, abs(v_avg - 1550) / 1550 * 100))
    return velocities


def _compute_reference_velocity(peak_data):
    """Compute velocity from sensor 0 to each distant sensor (first arrival)."""
    if len(peak_data) < 2:
        return []
    x0, _, t0_arr, _ = peak_data[0]
    velocities = []
    for i in range(1, len(peak_data)):
        xi, _, ti_arr, _ = peak_data[i]
        dx = xi - x0
        dt = ti_arr - t0_arr
        if abs(dx) < 1.0 or abs(dt) < 1e-10:
            continue
        v = dx / dt / 1000.0  # m/s
        if v > 0:
            velocities.append(v)
            print("    Sensor 0->%d: dx=%.0f mm, dt=%.1f us -> v = %.0f m/s" %
                  (i, dx, dt * 1e6, v))
    if velocities:
        v_avg = np.mean(velocities)
        print("    Average: %.0f m/s (theory ~1550 m/s, dev %.1f%%)" %
              (v_avg, abs(v_avg - 1550) / 1550 * 100))
    return velocities


def plot_envelopes(csv_path, output_path, label=''):
    """Plot signals with Hilbert envelope overlay and arrival markers."""
    t, sensors, x_pos = load_csv(csv_path)
    if not x_pos:
        x_pos = [0.0, 50.0, 100.0, 150.0, 200.0]

    n_sensors = len(sensors)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(14, 3 * n_sensors),
                             sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for i in sorted(sensors.keys()):
        ax = axes[i]
        t_us = t * 1e6
        t_peak, t_arrival, env = _envelope_arrival(t, sensors[i])

        ax.plot(t_us, sensors[i], 'b-', linewidth=0.6, alpha=0.5, label='U3')
        ax.plot(t_us, env, 'r-', linewidth=1.2, label='Envelope')
        ax.plot(t_us, -env, 'r-', linewidth=1.2, alpha=0.3)
        ax.axvline(t_peak * 1e6, color='green', linestyle='--', linewidth=1.0,
                   label='Peak (%.0f us)' % (t_peak * 1e6))
        ax.axvline(t_arrival * 1e6, color='orange', linestyle='--',
                   linewidth=1.0,
                   label='Arrival (%.0f us)' % (t_arrival * 1e6))

        offset = x_pos[i] if i < len(x_pos) else i * 50
        ax.set_ylabel('U3 [mm]', fontsize=10)
        ax.set_title('Sensor %d (x = %.0f mm)' % (i, offset),
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    axes[-1].set_xlabel('Time [us]', fontsize=11)
    fig.suptitle('Hilbert Envelope Analysis: %s\n'
                 '(green=peak, orange=first arrival @10%% threshold)' % label,
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Envelope plot saved: %s" % output_path)
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python plot_gw_comparison.py <healthy.csv> <defect.csv>")
        sys.exit(1)

    csv_h = sys.argv[1]
    csv_d = sys.argv[2]
    out_dir = os.path.dirname(csv_h) or '.'

    # Group velocity (Hilbert envelope)
    compute_and_print_velocity(csv_h, 'Healthy')
    compute_and_print_velocity(csv_d, 'Debonding')

    # Plots
    plot_waveforms(csv_h, csv_d,
                   os.path.join(out_dir, 'gw_comparison.png'))
    plot_difference(csv_h, csv_d,
                    os.path.join(out_dir, 'gw_difference.png'))

    # Envelope analysis
    plot_envelopes(csv_h, os.path.join(out_dir, 'gw_envelope_healthy.png'),
                   'Healthy')
    plot_envelopes(csv_d, os.path.join(out_dir, 'gw_envelope_debond.png'),
                   'Debonding')
