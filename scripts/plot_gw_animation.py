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
                                output_path='gw_wave_comparison.gif', fps=20,
                                defect_x=80, defect_y=0, defect_r=25,
                                defect_label=None, dpi=120, levels=50):
    """Create side-by-side Healthy vs Defect animation.

    defect_x, defect_y, defect_r: defect zone circle (mm)
    defect_label: optional title override (e.g. 'D2 r=15mm')
    """
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

    def_title = defect_label or 'Debonding (x=%d, r=%dmm)' % (defect_x, defect_r)

    def animate(i):
        ax1.clear()
        ax2.clear()

        ax1.tricontourf(tri_h, u3_h[i], levels=levels,
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax1.set_xlim(x_h.min() - 5, x_h.max() + 5)
        ax1.set_ylim(y_h.min() - 5, y_h.max() + 5)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X [mm]')
        ax1.set_ylabel('Y [mm]')
        ax1.set_title('Healthy\nt = %.1f us' % (t_h[i] * 1e6),
                       fontsize=12, fontweight='bold')

        ax2.tricontourf(tri_d, u3_d[i], levels=levels,
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax2.set_xlim(x_d.min() - 5, x_d.max() + 5)
        ax2.set_ylim(y_d.min() - 5, y_d.max() + 5)
        ax2.set_aspect('equal')
        ax2.set_xlabel('X [mm]')
        ax2.set_ylabel('Y [mm]')
        ax2.set_title('%s\nt = %.1f us' % (def_title, t_d[i] * 1e6),
                       fontsize=12, fontweight='bold')

        # Draw defect circle on defect plot
        theta = np.linspace(0, 2 * np.pi, 100)
        ax2.plot(defect_x + defect_r * np.cos(theta), defect_y + defect_r * np.sin(theta),
                 'g--', linewidth=2.5, alpha=0.8, label='Defect zone')
        ax2.legend(loc='upper left', fontsize=9)

        fig.suptitle('Guided Wave Propagation: A0 Mode (50 kHz)\n'
                     'CFRP/Al-HC Sandwich Panel (300x300 mm)',
                     fontsize=14, fontweight='bold')
        return []

    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   interval=1000 // fps, blit=False)
    anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    plt.close()
    print("Comparison animation saved: %s (%d frames)" % (output_path, n_frames))


def create_snapshot_grid(prefix_healthy, prefix_defect,
                         output_path='gw_snapshots.png',
                         time_points_us=None, defect_x=80, defect_y=0, defect_r=25):
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
        ax.plot(defect_x + defect_r * np.cos(theta), defect_y + defect_r * np.sin(theta),
                'g--', linewidth=1.5, alpha=0.7)

    fig.suptitle('Guided Wave Propagation Snapshots: Healthy vs Debonding\n'
                 '(50 kHz A0 mode, CFRP/Al-HC panel)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Snapshot grid saved: %s" % output_path)


# DOE: (prefix_defect, x, y, r, label) for flat panel sweep
DEFECT_CASES = [
    ('Job-GW-Debond-v2', 80, 0, 25, 'D1 r=25mm (基準)'),
    ('Job-GW-D2-r15', 80, 0, 15, 'D2 r=15mm (小)'),
    ('Job-GW-D3-r40', 80, 0, 40, 'D3 r=40mm (大)'),
    ('Job-GW-D4-near', 40, 0, 25, 'D4 x=40mm (近)'),
    ('Job-GW-D5-edge', 120, 0, 25, 'D5 x=120mm (端)'),
]


def run_batch_comparisons(prefix_healthy, work_dir, output_dir=None,
                          fps=20, dpi=120):
    """Generate comparison GIFs for all defect cases."""
    if output_dir is None:
        output_dir = work_dir
    os.makedirs(output_dir, exist_ok=True)

    for job_defect, dx, dy, dr, label in DEFECT_CASES:
        prefix_d = os.path.join(work_dir, job_defect)
        coords = os.path.join(work_dir, job_defect + '_coords.csv')
        if not os.path.exists(coords):
            print("  SKIP %s (no _coords.csv)" % job_defect)
            continue
        out_name = 'gw_wave_comparison_%s.gif' % job_defect.replace('Job-GW-', '').replace('-', '_').lower()
        out_path = os.path.join(output_dir, out_name)
        print("\n--- %s ---" % label)
        create_comparison_animation(
            prefix_healthy, prefix_d,
            output_path=out_path,
            fps=fps, dpi=dpi,
            defect_x=dx, defect_y=dy, defect_r=dr,
            defect_label=label)
    print("\nBatch complete. Outputs in %s" % output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GW wave propagation animation')
    parser.add_argument('prefix1', help='Healthy prefix (path without _coords.csv)')
    parser.add_argument('prefix2', nargs='?', help='Defect prefix (optional)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for GIF')
    parser.add_argument('--fps', type=int, default=20, help='FPS (default 20)')
    parser.add_argument('--dpi', type=int, default=120, help='DPI (default 120)')
    parser.add_argument('--defect', type=str, default=None,
                        help='Defect params: x,y,r e.g. 80,0,25')
    parser.add_argument('--batch', action='store_true',
                        help='Generate GIFs for all defect cases (D1-D5)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output dir for batch (default: wiki_repo/images/guided_wave)')
    args = parser.parse_args()

    prefix1 = args.prefix1
    work_dir = os.path.dirname(os.path.abspath(prefix1)) or '.'
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'wiki_repo', 'images', 'guided_wave')

    if args.batch:
        run_batch_comparisons(prefix1, work_dir, output_dir=out_dir,
                              fps=args.fps, dpi=args.dpi)
    elif args.prefix2:
        dx, dy, dr = 80, 0, 25
        if args.defect:
            parts = [int(x.strip()) for x in args.defect.split(',')]
            if len(parts) >= 3:
                dx, dy, dr = parts[0], parts[1], parts[2]
        out_path = args.output or os.path.join(work_dir, 'gw_wave_comparison.gif')
        create_comparison_animation(
            prefix1, args.prefix2, output_path=out_path,
            fps=args.fps, dpi=args.dpi,
            defect_x=dx, defect_y=dy, defect_r=dr)
        create_snapshot_grid(prefix1, args.prefix2,
                            os.path.join(os.path.dirname(out_path), 'gw_snapshots.png'),
                            defect_x=dx, defect_y=dy, defect_r=dr)
    else:
        create_single_animation(prefix1, args.output)
