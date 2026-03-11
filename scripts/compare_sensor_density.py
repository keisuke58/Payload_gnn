#!/usr/bin/env python3
"""Compare defect detectability across sensor densities (36, 64, 104).

Subsamples the existing 104-sensor (8x13) grid to approximate sparser
configurations, then computes Damage Index for each.

Usage:
  python scripts/compare_sensor_density.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors


def load_sensor_csv(csv_path):
    """Load sensor CSV → (time_s, Ur, arc_mm, n_theta=8, n_z=13)."""
    with open(csv_path, 'r') as f:
        f.readline()  # header
        pos_line = f.readline().strip()
    arc = np.array([float(p) for p in pos_line.split(',')[1:]])
    data = np.loadtxt(csv_path, delimiter=',', skiprows=2)
    time_s, Ur = data[:, 0], data[:, 1:]
    n_theta = 1
    for i in range(1, len(arc)):
        if arc[i] <= arc[i - 1] + 1e-3:
            break
        n_theta += 1
    n_z = len(arc) // n_theta
    return time_s, Ur.reshape(len(time_s), n_z, n_theta), arc[:n_theta], n_theta, n_z


def subsample_grid(Ur_grid, arc_1d, z_1d, idx_theta, idx_z):
    """Subsample (N_time, n_z, n_theta) grid."""
    sub = Ur_grid[:, idx_z, :][:, :, idx_theta]
    arc_sub = arc_1d[idx_theta]
    z_sub = z_1d[idx_z]
    return sub, arc_sub, z_sub


def compute_damage_indices(Ur_h, Ur_d):
    """Compute energy-based and correlation-based DI per sensor.

    Ur_h, Ur_d: (N_time, n_z_sub, n_theta_sub)
    Returns: DI_energy, DI_corr — (n_z_sub, n_theta_sub)
    """
    n_t = min(Ur_h.shape[0], Ur_d.shape[0])
    t_start = n_t // 10

    residual = Ur_d[t_start:n_t] - Ur_h[t_start:n_t]
    E_h = np.sum(Ur_h[t_start:n_t] ** 2, axis=0)
    E_res = np.sum(residual ** 2, axis=0)
    DI_energy = np.where(E_h > 1e-30, E_res / E_h, 0.0)

    n_z_sub, n_theta_sub = Ur_h.shape[1], Ur_h.shape[2]
    DI_corr = np.zeros((n_z_sub, n_theta_sub))
    for iz in range(n_z_sub):
        for it in range(n_theta_sub):
            h = Ur_h[t_start:n_t, iz, it]
            d = Ur_d[t_start:n_t, iz, it]
            if np.std(h) > 1e-20 and np.std(d) > 1e-20:
                DI_corr[iz, it] = 1.0 - np.corrcoef(h, d)[0, 1]
    return DI_energy, DI_corr


def main():
    csv_dir = os.path.expanduser('~/Payload2026/abaqus_work/gw_fairing_dataset')
    out_dir = os.path.expanduser('~/Payload2026/results/gw_validation')
    os.makedirs(out_dir, exist_ok=True)

    csv_h = os.path.join(csv_dir, 'Job-GW-Valid-Healthy-v3_sensors.csv')
    csv_d = os.path.join(csv_dir, 'Job-GW-Valid-Debond30-v3_sensors.csv')

    print("Loading data...")
    time_h, Ur_h_full, arc_1d, n_theta, n_z = load_sensor_csv(csv_h)
    time_d, Ur_d_full, _, _, _ = load_sensor_csv(csv_d)

    z_1d = np.linspace(550, 2450, n_z)

    print("  Full grid: %d x %d = %d sensors" % (n_theta, n_z, n_theta * n_z))

    # Define subsampling configs
    configs = [
        ('36 (6x6)', 6, 6),
        ('64 (8x8)', 8, 8),
        ('104 (8x13)', 8, 13),
    ]

    results = []
    for label, nt_target, nz_target in configs:
        idx_t = np.round(np.linspace(0, n_theta - 1, min(nt_target, n_theta))).astype(int)
        idx_z = np.round(np.linspace(0, n_z - 1, min(nz_target, n_z))).astype(int)
        # Remove duplicates
        idx_t = np.unique(idx_t)
        idx_z = np.unique(idx_z)

        Ur_h_sub, arc_sub, z_sub = subsample_grid(Ur_h_full, arc_1d, z_1d, idx_t, idx_z)
        Ur_d_sub, _, _ = subsample_grid(Ur_d_full, arc_1d, z_1d, idx_t, idx_z)

        actual_n = len(idx_t) * len(idx_z)
        arc_spacing = np.mean(np.diff(arc_sub)) if len(arc_sub) > 1 else 0
        z_spacing = np.mean(np.diff(z_sub)) if len(z_sub) > 1 else 0

        DI_e, DI_c = compute_damage_indices(Ur_h_sub, Ur_d_sub)

        # Find defect location in subsampled grid
        defect_arc_idx = np.argmin(np.abs(arc_sub - arc_1d[n_theta // 2]))
        defect_z_idx = np.argmin(np.abs(z_sub - 1500.0))

        # SNR: DI at defect vs mean DI
        di_at_defect = DI_e[defect_z_idx, defect_arc_idx]
        di_background = np.mean(DI_e)
        snr_energy = di_at_defect / di_background if di_background > 0 else 0

        di_corr_defect = DI_c[defect_z_idx, defect_arc_idx]
        di_corr_bg = np.mean(DI_c)
        snr_corr = di_corr_defect / di_corr_bg if di_corr_bg > 0 else 0

        # Peak DI and its location
        peak_iz, peak_it = np.unravel_index(np.argmax(DI_e), DI_e.shape)
        peak_z = z_sub[peak_iz]
        peak_arc = arc_sub[peak_it]
        dist_to_defect = np.sqrt((peak_arc - arc_1d[n_theta // 2]) ** 2 +
                                  (peak_z - 1500.0) ** 2)

        print("\n%s: %dx%d=%d sensors (arc=%.0fmm, z=%.0fmm spacing)" % (
            label, len(idx_t), len(idx_z), actual_n, arc_spacing, z_spacing))
        print("  Energy DI: defect=%.4f, bg=%.4f, SNR=%.1f" % (
            di_at_defect, di_background, snr_energy))
        print("  Corr DI:   defect=%.4f, bg=%.4f, SNR=%.1f" % (
            di_corr_defect, di_corr_bg, snr_corr))
        print("  Peak DI at (arc=%.0f, z=%.0f), dist to defect=%.0fmm" % (
            peak_arc, peak_z, dist_to_defect))

        ARC_sub, Z_sub = np.meshgrid(arc_sub, z_sub)
        results.append({
            'label': label,
            'n_actual': actual_n,
            'nt': len(idx_t), 'nz': len(idx_z),
            'arc_spacing': arc_spacing, 'z_spacing': z_spacing,
            'DI_e': DI_e, 'DI_c': DI_c,
            'ARC': ARC_sub, 'Z': Z_sub,
            'arc_sub': arc_sub, 'z_sub': z_sub,
            'snr_energy': snr_energy, 'snr_corr': snr_corr,
            'peak_dist': dist_to_defect,
        })

    # --- Plot comparison figure ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    defect_arc = arc_1d[n_theta // 2]
    defect_z = 1500.0

    for col, res in enumerate(results):
        # Energy DI (top row)
        ax = axes[0, col]
        DI_log = res['DI_e'].copy()
        DI_log[DI_log < 1e-6] = 1e-6
        vmin = 1e-5
        vmax = max(r['DI_e'].max() for r in results)
        im = ax.pcolormesh(res['ARC'], res['Z'], DI_log,
                            cmap='inferno',
                            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                            shading='gouraud')
        ax.set_title('%s\nSNR=%.1f, peak dist=%.0fmm' % (
            res['label'], res['snr_energy'], res['peak_dist']))
        ax.set_xlabel('Arc distance [mm]')
        if col == 0:
            ax.set_ylabel('Axial position [mm]')
        ax.set_aspect('equal')
        circle = plt.Circle((defect_arc, defect_z), 30, fill=False,
                             edgecolor='cyan', linewidth=2, linestyle='--')
        ax.add_patch(circle)
        # Mark sensor positions
        for iz in range(res['nz']):
            for it in range(res['nt']):
                ax.plot(res['arc_sub'][it], res['z_sub'][iz],
                        '.', color='white', markersize=2, alpha=0.5)
        plt.colorbar(im, ax=ax, label='Energy DI', shrink=0.8)

        # Correlation DI (bottom row)
        ax2 = axes[1, col]
        DI_c_log = res['DI_c'].copy()
        DI_c_log[DI_c_log < 1e-6] = 1e-6
        vmax_c = max(r['DI_c'].max() for r in results)
        im2 = ax2.pcolormesh(res['ARC'], res['Z'], DI_c_log,
                              cmap='inferno',
                              norm=colors.LogNorm(vmin=1e-6, vmax=vmax_c),
                              shading='gouraud')
        ax2.set_title('SNR=%.1f' % res['snr_corr'])
        ax2.set_xlabel('Arc distance [mm]')
        if col == 0:
            ax2.set_ylabel('Axial position [mm]')
        ax2.set_aspect('equal')
        circle2 = plt.Circle((defect_arc, defect_z), 30, fill=False,
                              edgecolor='cyan', linewidth=2, linestyle='--')
        ax2.add_patch(circle2)
        for iz in range(res['nz']):
            for it in range(res['nt']):
                ax2.plot(res['arc_sub'][it], res['z_sub'][iz],
                         '.', color='white', markersize=2, alpha=0.5)
        plt.colorbar(im2, ax=ax2, label='Correlation DI', shrink=0.8)

    axes[0, 0].set_ylabel('Energy-based DI\nAxial position [mm]')
    axes[1, 0].set_ylabel('Correlation-based DI\nAxial position [mm]')

    fig.suptitle('Sensor Density Comparison — Debond R=30mm Detection',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fname = os.path.join(out_dir, 'sensor_density_comparison.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\nSaved: %s" % fname)

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SENSOR DENSITY COMPARISON SUMMARY")
    print("=" * 70)
    print("%-15s %6s %10s %10s %8s %8s %10s" % (
        'Config', 'Count', 'Arc[mm]', 'Z[mm]', 'SNR_E', 'SNR_C', 'PeakDist'))
    print("-" * 70)
    for r in results:
        print("%-15s %6d %10.0f %10.0f %8.1f %8.1f %10.0f" % (
            r['label'], r['n_actual'],
            r['arc_spacing'], r['z_spacing'],
            r['snr_energy'], r['snr_corr'],
            r['peak_dist']))
    print("=" * 70)


if __name__ == '__main__':
    main()
