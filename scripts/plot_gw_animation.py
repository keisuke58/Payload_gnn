# -*- coding: utf-8 -*-
# plot_gw_animation.py
# Create wave propagation animation from extracted field data.
#
# Usage: python scripts/plot_gw_animation.py \
#            abaqus_work/Job-GW-Healthy \
#            abaqus_work/Job-GW-Debond
#
# Reads <prefix>_coords.csv and <prefix>_frames.csv
# Outputs: <prefix>_wave.gif and side-by-side comparison GIF

import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.animation as animation


def load_field_data(prefix):
    """Load coordinates and frame data from CSVs.

    Returns: (x, y, times, u3_frames) where u3_frames[i] = U3 at frame i
    """
    coords_path = prefix + '_coords.csv'
    frames_path = prefix + '_frames.csv'

    # Load coordinates
    x_list, y_list = [], []
    with open(coords_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            x_list.append(float(row[1]))
            y_list.append(float(row[2]))

    x = np.array(x_list)
    y = np.array(y_list)
    n_nodes = len(x)

    # Load frames
    times = []
    u3_frames = []
    with open(frames_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            times.append(float(row[0]))
            vals = [float(v) for v in row[1:n_nodes + 1]]
            u3_frames.append(np.array(vals))

    return x, y, np.array(times), u3_frames


def create_single_animation(prefix, output_path=None, fps=15):
    """Create wave propagation animation for a single model."""
    if output_path is None:
        output_path = prefix + '_wave.gif'

    print("Loading field data: %s" % prefix)
    x, y, times, u3_frames = load_field_data(prefix)
    n_frames = len(times)
    print("  %d nodes, %d frames" % (len(x), n_frames))

    # Create triangulation for contour plot
    tri = Triangulation(x, y)

    # Find global min/max for consistent colorbar
    vmax = max(np.max(np.abs(f)) for f in u3_frames)
    # Use symmetric range, skip first few frames (near-zero)
    vmax_plot = vmax * 0.3  # scale to show wave details, not just excitation peak

    model_name = os.path.basename(prefix)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    def animate(i):
        ax.clear()
        tcf = ax.tricontourf(tri, u3_frames[i], levels=50,
                             cmap='RdBu_r', vmin=-vmax_plot, vmax=vmax_plot)
        ax.set_xlim(x.min() - 5, x.max() + 5)
        ax.set_ylim(y.min() - 5, y.max() + 5)
        ax.set_aspect('equal')
        ax.set_xlabel('X [mm]', fontsize=11)
        ax.set_ylabel('Y [mm]', fontsize=11)
        ax.set_title('%s\nt = %.1f us  (U3 out-of-plane)' % (
            model_name, times[i] * 1e6), fontsize=12, fontweight='bold')
        return []

    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   interval=1000 // fps, blit=False)
    anim.save(output_path, writer='pillow', fps=fps, dpi=100)
    plt.close()
    print("Animation saved: %s (%d frames)" % (output_path, n_frames))


def create_comparison_animation(prefix_healthy, prefix_defect,
                                output_path='gw_wave_comparison.gif', fps=15):
    """Create side-by-side Healthy vs Defect animation."""
    print("Loading healthy field data...")
    x_h, y_h, t_h, u3_h = load_field_data(prefix_healthy)
    print("Loading defect field data...")
    x_d, y_d, t_d, u3_d = load_field_data(prefix_defect)

    n_frames = min(len(t_h), len(t_d))
    print("  Comparison: %d frames" % n_frames)

    tri_h = Triangulation(x_h, y_h)
    tri_d = Triangulation(x_d, y_d)

    # Global vmax
    vmax_h = max(np.max(np.abs(f)) for f in u3_h)
    vmax_d = max(np.max(np.abs(f)) for f in u3_d)
    vmax = max(vmax_h, vmax_d) * 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    def animate(i):
        ax1.clear()
        ax2.clear()

        ax1.tricontourf(tri_h, u3_h[i], levels=50,
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax1.set_xlim(x_h.min() - 5, x_h.max() + 5)
        ax1.set_ylim(y_h.min() - 5, y_h.max() + 5)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X [mm]')
        ax1.set_ylabel('Y [mm]')
        ax1.set_title('Healthy\nt = %.1f us' % (t_h[i] * 1e6),
                       fontsize=12, fontweight='bold')

        ax2.tricontourf(tri_d, u3_d[i], levels=50,
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax2.set_xlim(x_d.min() - 5, x_d.max() + 5)
        ax2.set_ylim(y_d.min() - 5, y_d.max() + 5)
        ax2.set_aspect('equal')
        ax2.set_xlabel('X [mm]')
        ax2.set_ylabel('Y [mm]')
        ax2.set_title('Debonding (x=80, r=25mm)\nt = %.1f us' % (t_d[i] * 1e6),
                       fontsize=12, fontweight='bold')

        # Draw defect circle on defect plot
        theta = np.linspace(0, 2 * np.pi, 100)
        ax2.plot(80 + 25 * np.cos(theta), 0 + 25 * np.sin(theta),
                 'g--', linewidth=2, alpha=0.7, label='Defect zone')
        ax2.legend(loc='upper left', fontsize=9)

        fig.suptitle('Guided Wave Propagation: A0 Mode (50 kHz)\n'
                     'CFRP/Al-HC Sandwich Panel (300x300 mm)',
                     fontsize=14, fontweight='bold')
        return []

    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   interval=1000 // fps, blit=False)
    anim.save(output_path, writer='pillow', fps=fps, dpi=100)
    plt.close()
    print("Comparison animation saved: %s (%d frames)" % (output_path, n_frames))


def create_snapshot_grid(prefix_healthy, prefix_defect,
                         output_path='gw_snapshots.png',
                         time_points_us=None):
    """Create a grid of snapshots at key time points."""
    x_h, y_h, t_h, u3_h = load_field_data(prefix_healthy)
    x_d, y_d, t_d, u3_d = load_field_data(prefix_defect)

    if time_points_us is None:
        time_points_us = [50, 100, 200, 300, 400, 500]

    tri_h = Triangulation(x_h, y_h)
    tri_d = Triangulation(x_d, y_d)

    vmax_h = max(np.max(np.abs(f)) for f in u3_h)
    vmax_d = max(np.max(np.abs(f)) for f in u3_d)
    vmax = max(vmax_h, vmax_d) * 0.3

    n_times = len(time_points_us)
    fig, axes = plt.subplots(2, n_times, figsize=(4 * n_times, 8))

    for j, t_us in enumerate(time_points_us):
        t_target = t_us * 1e-6

        # Find closest frame for healthy
        idx_h = np.argmin(np.abs(t_h - t_target))
        idx_d = np.argmin(np.abs(t_d - t_target))

        # Healthy (top row)
        ax = axes[0, j]
        ax.tricontourf(tri_h, u3_h[idx_h], levels=50,
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_title('t = %.0f us' % (t_h[idx_h] * 1e6), fontsize=10)
        if j == 0:
            ax.set_ylabel('Healthy\nY [mm]', fontsize=11, fontweight='bold')
        ax.set_xlim(x_h.min() - 5, x_h.max() + 5)
        ax.set_ylim(y_h.min() - 5, y_h.max() + 5)

        # Defect (bottom row)
        ax = axes[1, j]
        ax.tricontourf(tri_d, u3_d[idx_d], levels=50,
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_title('t = %.0f us' % (t_d[idx_d] * 1e6), fontsize=10)
        if j == 0:
            ax.set_ylabel('Debonding\nY [mm]', fontsize=11, fontweight='bold')
        ax.set_xlim(x_d.min() - 5, x_d.max() + 5)
        ax.set_ylim(y_d.min() - 5, y_d.max() + 5)

        # Defect circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(80 + 25 * np.cos(theta), 0 + 25 * np.sin(theta),
                'g--', linewidth=1.5, alpha=0.7)

    fig.suptitle('Guided Wave Propagation Snapshots: Healthy vs Debonding\n'
                 '(50 kHz A0 mode, CFRP/Al-HC panel)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Snapshot grid saved: %s" % output_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_gw_animation.py <prefix1> [prefix2]")
        print("  prefix = path without _coords.csv / _frames.csv")
        sys.exit(1)

    prefix1 = sys.argv[1]

    if len(sys.argv) >= 3:
        prefix2 = sys.argv[2]
        out_dir = os.path.dirname(prefix1) or '.'

        # Individual animations
        create_single_animation(prefix1,
                                os.path.join(out_dir, 'gw_wave_healthy.gif'))
        create_single_animation(prefix2,
                                os.path.join(out_dir, 'gw_wave_debond.gif'))

        # Side-by-side comparison
        create_comparison_animation(
            prefix1, prefix2,
            os.path.join(out_dir, 'gw_wave_comparison.gif'))

        # Snapshot grid
        create_snapshot_grid(
            prefix1, prefix2,
            os.path.join(out_dir, 'gw_snapshots.png'))
    else:
        create_single_animation(prefix1)
