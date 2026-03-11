#!/usr/bin/env python3
"""Visualize GW validation results: wavefield, B-scan, C-scan, damage index.

Generates 4 types of figures from sensor CSV data:
  1. Wavefield snapshots (Healthy vs Debond) at multiple time steps
  2. B-scan (time vs arc distance) waterfall plots
  3. C-scan (arc × axial) RMS amplitude maps
  4. Damage Index map (residual energy between Healthy and Debond)

Usage:
  python scripts/visualize_gw_validation.py
  python scripts/visualize_gw_validation.py --csv_dir path/to/csvs --out_dir results/gw_valid
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_sensor_csv(csv_path):
    """Load sensor CSV and reconstruct 2D grid coordinates.

    Returns:
        time_s: (N_time,) array
        arc_mm: (n_sensors,) arc distance from sensor 0
        z_mm:   (n_sensors,) axial position (reconstructed from grid pattern)
        Ur:     (N_time, n_sensors) radial displacement
        n_theta, n_z: grid dimensions
    """
    with open(csv_path, 'r') as f:
        header = f.readline().strip()
        pos_line = f.readline().strip()

    # Parse arc distances from comment line
    parts = pos_line.split(',')
    arc_vals = []
    for p in parts[1:]:  # skip "# x_mm"
        try:
            arc_vals.append(float(p))
        except ValueError:
            arc_vals.append(0.0)
    arc_mm = np.array(arc_vals)

    # Detect grid: find repeating pattern in arc values
    n_sensors = len(arc_mm)
    # Find n_theta = length of first non-repeating block
    n_theta = 1
    for i in range(1, n_sensors):
        if arc_mm[i] <= arc_mm[i - 1] + 1e-3:
            break
        n_theta += 1
    n_z = n_sensors // n_theta

    # Load data (skip header + comment line)
    data = np.loadtxt(csv_path, delimiter=',', skiprows=2)
    time_s = data[:, 0]
    Ur = data[:, 1:]

    # Reconstruct z positions from generation parameters
    # z_min + 50 to z_max - 50, n_z points
    # From generate_gw_fairing.py: z_min=500, z_max=2500 (default)
    z_lo = 550.0   # 500 + 50 margin
    z_hi = 2450.0  # 2500 - 50 margin
    z_1d = np.linspace(z_lo, z_hi, n_z)

    # Build per-sensor z array (sensor order: iz outer, it inner)
    z_mm = np.zeros(n_sensors)
    for iz in range(n_z):
        for it in range(n_theta):
            idx = iz * n_theta + it
            if idx < n_sensors:
                z_mm[idx] = z_1d[iz]

    print("  Loaded: %s" % os.path.basename(csv_path))
    print("    %d time steps, %d sensors (%d theta x %d z)" % (
        len(time_s), n_sensors, n_theta, n_z))
    print("    Arc range: %.0f - %.0f mm" % (arc_mm.min(), arc_mm.max()))
    print("    Z range:   %.0f - %.0f mm" % (z_mm.min(), z_mm.max()))

    return time_s, arc_mm, z_mm, Ur, n_theta, n_z


def reshape_to_grid(Ur, n_theta, n_z):
    """Reshape (N_time, n_sensors) -> (N_time, n_z, n_theta)."""
    N_time = Ur.shape[0]
    return Ur[:, :n_z * n_theta].reshape(N_time, n_z, n_theta)


# ---------------------------------------------------------------------------
# 1. Wavefield snapshots (3D surface + colormap comparison)
# ---------------------------------------------------------------------------

def plot_wavefield_snapshots(time_h, arc_h, z_h, Ur_h, n_theta, n_z,
                             time_d, arc_d, z_d, Ur_d,
                             out_dir, times_us=None):
    """3D surface plots of wavefield at selected time snapshots."""
    if times_us is None:
        # Select meaningful time points covering wave propagation
        t_max_us = time_h[-1] * 1e6
        times_us = [50, 100, 200, 500, 1000, 2000, 3000]
        times_us = [t for t in times_us if t < t_max_us * 0.7]
        if not times_us:
            times_us = np.linspace(t_max_us * 0.05, t_max_us * 0.6, 5).tolist()

    Ur_h_grid = reshape_to_grid(Ur_h, n_theta, n_z)
    Ur_d_grid = reshape_to_grid(Ur_d, n_theta, n_z)

    arc_1d = arc_h[:n_theta]
    z_1d = np.array([z_h[i * n_theta] for i in range(n_z)])
    ARC, Z = np.meshgrid(arc_1d, z_1d)

    for t_us in times_us:
        t_s = t_us * 1e-6
        idx_h = np.argmin(np.abs(time_h - t_s))
        idx_d = np.argmin(np.abs(time_d - t_s))

        fig = plt.figure(figsize=(16, 6))

        # Common colorscale
        vmax = max(np.abs(Ur_h_grid[idx_h]).max(), np.abs(Ur_d_grid[idx_d]).max())
        if vmax < 1e-15:
            vmax = 1e-10

        # Healthy
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(ARC, Z, Ur_h_grid[idx_h] * 1e3,
                                  cmap='RdBu_r', vmin=-vmax * 1e3, vmax=vmax * 1e3,
                                  alpha=0.9, linewidth=0.1, edgecolor='gray')
        ax1.set_xlabel('Arc distance [mm]')
        ax1.set_ylabel('Axial position [mm]')
        ax1.set_zlabel('Ur [x10$^{-3}$ mm]')
        ax1.set_title('Healthy (t = %.0f us)' % t_us)
        ax1.view_init(elev=25, azim=-60)

        # Debond
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(ARC, Z, Ur_d_grid[idx_d] * 1e3,
                                  cmap='RdBu_r', vmin=-vmax * 1e3, vmax=vmax * 1e3,
                                  alpha=0.9, linewidth=0.1, edgecolor='gray')
        ax2.set_xlabel('Arc distance [mm]')
        ax2.set_ylabel('Axial position [mm]')
        ax2.set_zlabel('Ur [x10$^{-3}$ mm]')
        ax2.set_title('Debond R=30mm (t = %.0f us)' % t_us)
        ax2.view_init(elev=25, azim=-60)

        fig.suptitle('Guided Wave Wavefield — t = %.0f $\\mu$s' % t_us,
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        fname = os.path.join(out_dir, 'wavefield_3d_t%04d.png' % int(t_us))
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved: %s" % fname)

    # 2D colormap version with residual panel (3-column: Healthy, Debond, Diff)
    for t_us in times_us:
        t_s = t_us * 1e-6
        idx_h = np.argmin(np.abs(time_h - t_s))
        idx_d = np.argmin(np.abs(time_d - t_s))

        residual = Ur_d_grid[idx_d] - Ur_h_grid[idx_h]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        vmax = max(np.abs(Ur_h_grid[idx_h]).max(), np.abs(Ur_d_grid[idx_d]).max())
        if vmax < 1e-15:
            vmax = 1e-10

        im1 = ax1.pcolormesh(ARC, Z, Ur_h_grid[idx_h],
                              cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='gouraud')
        ax1.set_xlabel('Arc distance [mm]')
        ax1.set_ylabel('Axial position [mm]')
        ax1.set_title('Healthy')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label='Ur [mm]', shrink=0.8)

        im2 = ax2.pcolormesh(ARC, Z, Ur_d_grid[idx_d],
                              cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='gouraud')
        ax2.set_xlabel('Arc distance [mm]')
        ax2.set_ylabel('Axial position [mm]')
        ax2.set_title('Debond R=30mm')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='Ur [mm]', shrink=0.8)

        # Residual (Debond - Healthy) with independent scale
        vmax_r = np.abs(residual).max()
        if vmax_r < 1e-15:
            vmax_r = vmax * 0.1
        im3 = ax3.pcolormesh(ARC, Z, residual,
                              cmap='RdBu_r', vmin=-vmax_r, vmax=vmax_r, shading='gouraud')
        ax3.set_xlabel('Arc distance [mm]')
        ax3.set_ylabel('Axial position [mm]')
        ax3.set_title('Residual (D $-$ H)')
        ax3.set_aspect('equal')
        plt.colorbar(im3, ax=ax3, label='$\\Delta$Ur [mm]', shrink=0.8)
        # Defect marker
        circle = plt.Circle((arc_1d[n_theta // 2], 1500), 30, fill=False,
                             edgecolor='lime', linewidth=1.5, linestyle='--')
        ax3.add_patch(circle)

        fig.suptitle('Wavefield Snapshot — t = %.0f $\\mu$s' % t_us,
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        fname = os.path.join(out_dir, 'wavefield_2d_t%04d.png' % int(t_us))
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved: %s" % fname)


# ---------------------------------------------------------------------------
# 2. B-scan (waterfall: time vs arc distance at fixed z)
# ---------------------------------------------------------------------------

def plot_bscan(time_h, arc_h, z_h, Ur_h, n_theta, n_z,
               time_d, arc_d, z_d, Ur_d, out_dir):
    """B-scan: time-distance waterfall at sensor row nearest to defect."""
    Ur_h_grid = reshape_to_grid(Ur_h, n_theta, n_z)
    Ur_d_grid = reshape_to_grid(Ur_d, n_theta, n_z)

    arc_1d = arc_h[:n_theta]
    z_1d = np.array([z_h[i * n_theta] for i in range(n_z)])

    # Defect at z=1500mm → find nearest z index
    defect_z = 1500.0
    iz_defect = np.argmin(np.abs(z_1d - defect_z))

    # Also plot at excitation z (z=1500 typically, but also try mid-point)
    iz_mid = n_z // 2

    for iz, label in [(iz_defect, 'defect_z'), (iz_mid, 'mid_z')]:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        time_us = time_h * 1e6

        vmax = max(np.abs(Ur_h_grid[:, iz, :]).max(),
                   np.abs(Ur_d_grid[:, iz, :]).max())
        if vmax < 1e-15:
            vmax = 1e-10

        T, A = np.meshgrid(arc_1d, time_us)
        im1 = ax1.pcolormesh(T, A, Ur_h_grid[:, iz, :],
                              cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='gouraud')
        ax1.set_xlabel('Arc distance [mm]')
        ax1.set_ylabel('Time [$\\mu$s]')
        ax1.set_title('Healthy')
        ax1.invert_yaxis()
        plt.colorbar(im1, ax=ax1, label='Ur [mm]')

        im2 = ax2.pcolormesh(T, A, Ur_d_grid[:, iz, :],
                              cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='gouraud')
        ax2.set_xlabel('Arc distance [mm]')
        ax2.set_ylabel('Time [$\\mu$s]')
        ax2.set_title('Debond R=30mm')
        ax2.invert_yaxis()
        plt.colorbar(im2, ax=ax2, label='Ur [mm]')

        # Residual B-scan
        residual_b = Ur_d_grid[:, iz, :] - Ur_h_grid[:, iz, :]
        vmax_r = np.abs(residual_b).max()
        if vmax_r < 1e-15:
            vmax_r = vmax * 0.1
        im3 = ax3.pcolormesh(T, A, residual_b,
                              cmap='RdBu_r', vmin=-vmax_r, vmax=vmax_r, shading='gouraud')
        ax3.set_xlabel('Arc distance [mm]')
        ax3.set_ylabel('Time [$\\mu$s]')
        ax3.set_title('Residual (D $-$ H)')
        ax3.invert_yaxis()
        plt.colorbar(im3, ax=ax3, label='$\\Delta$Ur [mm]')

        fig.suptitle('B-scan at z = %.0f mm (row %d/%d)' % (z_1d[iz], iz, n_z),
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        fname = os.path.join(out_dir, 'bscan_%s.png' % label)
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved: %s" % fname)


# ---------------------------------------------------------------------------
# 3. C-scan (RMS amplitude map over arc x axial)
# ---------------------------------------------------------------------------

def plot_cscan(time_h, arc_h, z_h, Ur_h, n_theta, n_z,
               time_d, arc_d, z_d, Ur_d, out_dir):
    """C-scan: RMS amplitude map on sensor grid."""
    Ur_h_grid = reshape_to_grid(Ur_h, n_theta, n_z)
    Ur_d_grid = reshape_to_grid(Ur_d, n_theta, n_z)

    arc_1d = arc_h[:n_theta]
    z_1d = np.array([z_h[i * n_theta] for i in range(n_z)])
    ARC, Z = np.meshgrid(arc_1d, z_1d)

    # Compute RMS over time (skip first 10% = excitation transient)
    t_start_idx = len(time_h) // 10
    rms_h = np.sqrt(np.mean(Ur_h_grid[t_start_idx:] ** 2, axis=0))
    rms_d = np.sqrt(np.mean(Ur_d_grid[t_start_idx:] ** 2, axis=0))

    # RMS difference map
    rms_diff = rms_d - rms_h

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    vmax = max(rms_h.max(), rms_d.max())
    if vmax < 1e-15:
        vmax = 1e-10

    im1 = ax1.pcolormesh(ARC, Z, rms_h, cmap='hot_r', vmin=0, vmax=vmax,
                          shading='gouraud')
    ax1.set_xlabel('Arc distance [mm]')
    ax1.set_ylabel('Axial position [mm]')
    ax1.set_title('Healthy')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='RMS Ur [mm]')

    im2 = ax2.pcolormesh(ARC, Z, rms_d, cmap='hot_r', vmin=0, vmax=vmax,
                          shading='gouraud')
    ax2.set_xlabel('Arc distance [mm]')
    ax2.set_ylabel('Axial position [mm]')
    ax2.set_title('Debond R=30mm')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='RMS Ur [mm]')

    # Difference (Debond - Healthy) with symmetric scale
    vmax_d = np.abs(rms_diff).max()
    if vmax_d < 1e-15:
        vmax_d = vmax * 0.1
    im3 = ax3.pcolormesh(ARC, Z, rms_diff, cmap='RdBu_r',
                          vmin=-vmax_d, vmax=vmax_d, shading='gouraud')
    ax3.set_xlabel('Arc distance [mm]')
    ax3.set_ylabel('Axial position [mm]')
    ax3.set_title('$\\Delta$RMS (D $-$ H)')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, label='$\\Delta$RMS Ur [mm]')

    # Mark defect location on debond and diff panels
    defect_arc = arc_1d[n_theta // 2]
    defect_z = 1500.0
    for ax in [ax2, ax3]:
        circle = plt.Circle((defect_arc, defect_z), 30, fill=False,
                             edgecolor='lime', linewidth=2, linestyle='--')
        ax.add_patch(circle)

    fig.suptitle('C-scan: RMS Amplitude Map', fontsize=13, fontweight='bold')
    plt.tight_layout()

    fname = os.path.join(out_dir, 'cscan_rms.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: %s" % fname)


# ---------------------------------------------------------------------------
# 4. Damage Index map (residual energy)
# ---------------------------------------------------------------------------

def plot_damage_index(time_h, arc_h, z_h, Ur_h, n_theta, n_z,
                      time_d, arc_d, z_d, Ur_d, out_dir):
    """Damage Index = normalized residual energy between Healthy and Debond."""
    Ur_h_grid = reshape_to_grid(Ur_h, n_theta, n_z)
    Ur_d_grid = reshape_to_grid(Ur_d, n_theta, n_z)

    arc_1d = arc_h[:n_theta]
    z_1d = np.array([z_h[i * n_theta] for i in range(n_z)])
    ARC, Z = np.meshgrid(arc_1d, z_1d)

    # Ensure same time length
    n_t = min(Ur_h_grid.shape[0], Ur_d_grid.shape[0])
    t_start = n_t // 10  # skip excitation transient

    # Residual signal
    residual = Ur_d_grid[t_start:n_t] - Ur_h_grid[t_start:n_t]

    # Energy of healthy signal per sensor
    E_h = np.sum(Ur_h_grid[t_start:n_t] ** 2, axis=0)
    E_res = np.sum(residual ** 2, axis=0)

    # Damage Index: normalized residual energy
    DI = np.where(E_h > 1e-30, E_res / E_h, 0.0)

    # Also compute correlation-based DI
    # DI_corr = 1 - correlation coefficient per sensor
    DI_corr = np.zeros((n_z, n_theta))
    for iz in range(n_z):
        for it in range(n_theta):
            h_sig = Ur_h_grid[t_start:n_t, iz, it]
            d_sig = Ur_d_grid[t_start:n_t, iz, it]
            if np.std(h_sig) > 1e-20 and np.std(d_sig) > 1e-20:
                corr = np.corrcoef(h_sig, d_sig)[0, 1]
                DI_corr[iz, it] = 1.0 - corr
            else:
                DI_corr[iz, it] = 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Energy-based DI (log scale for better contrast)
    DI_log = DI.copy()
    DI_log[DI_log < 1e-6] = 1e-6  # floor for log
    im1 = ax1.pcolormesh(ARC, Z, DI_log, cmap='inferno',
                          norm=colors.LogNorm(vmin=DI_log[DI_log > 0].min(),
                                              vmax=DI_log.max()),
                          shading='gouraud')
    ax1.set_xlabel('Arc distance [mm]')
    ax1.set_ylabel('Axial position [mm]')
    ax1.set_title('Energy-based DI (log scale)')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='DI = E_res / E_healthy')

    # Correlation-based DI (log scale)
    DI_corr_log = DI_corr.copy()
    DI_corr_log[DI_corr_log < 1e-6] = 1e-6
    im2 = ax2.pcolormesh(ARC, Z, DI_corr_log, cmap='inferno',
                          norm=colors.LogNorm(vmin=DI_corr_log[DI_corr_log > 0].min(),
                                              vmax=DI_corr_log.max()),
                          shading='gouraud')
    ax2.set_xlabel('Arc distance [mm]')
    ax2.set_ylabel('Axial position [mm]')
    ax2.set_title('Correlation-based DI (log scale)')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='DI = 1 - corr(H, D)')

    # Mark defect location on both
    defect_arc = arc_1d[n_theta // 2]
    defect_z = 1500.0
    for ax in [ax1, ax2]:
        circle = plt.Circle((defect_arc, defect_z), 30, fill=False,
                             edgecolor='cyan', linewidth=2, linestyle='--')
        ax.add_patch(circle)

    fig.suptitle('Damage Index Map — Debond R=30mm',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    fname = os.path.join(out_dir, 'damage_index.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: %s" % fname)


# ---------------------------------------------------------------------------
# 5. Wavefield animation (GIF)
# ---------------------------------------------------------------------------

def create_wavefield_animation(time_h, arc_h, z_h, Ur_h, n_theta, n_z,
                                time_d, arc_d, z_d, Ur_d, out_dir,
                                fps=20, t_start_us=5, t_end_us=None):
    """Create animated GIF of wave propagation: Healthy vs Debond side by side."""
    Ur_h_grid = reshape_to_grid(Ur_h, n_theta, n_z)
    Ur_d_grid = reshape_to_grid(Ur_d, n_theta, n_z)

    arc_1d = arc_h[:n_theta]
    z_1d = np.array([z_h[i * n_theta] for i in range(n_z)])
    ARC, Z = np.meshgrid(arc_1d, z_1d)

    time_us = time_h * 1e6
    if t_end_us is None:
        t_end_us = time_us[-1] * 0.8

    mask = (time_us >= t_start_us) & (time_us <= t_end_us)
    indices = np.where(mask)[0]

    # Subsample to ~120 frames for smaller file
    max_frames = 120
    if len(indices) > max_frames:
        indices = indices[::len(indices) // max_frames]

    # Global color scale
    vmax = max(np.abs(Ur_h_grid[indices]).max(),
               np.abs(Ur_d_grid[indices]).max())
    if vmax < 1e-15:
        vmax = 1e-10

    # Residual scale (typically much smaller)
    residual_all = Ur_d_grid[indices] - Ur_h_grid[indices]
    vmax_r = np.abs(residual_all).max()
    if vmax_r < 1e-15:
        vmax_r = vmax * 0.1
    del residual_all  # free memory

    defect_arc = arc_1d[n_theta // 2]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    def animate(frame_idx):
        idx = indices[frame_idx]
        t_us_val = time_us[idx]

        ax1.clear()
        ax2.clear()
        ax3.clear()

        ax1.pcolormesh(ARC, Z, Ur_h_grid[idx],
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='gouraud')
        ax1.set_xlabel('Arc distance [mm]')
        ax1.set_ylabel('Axial position [mm]')
        ax1.set_title('Healthy')
        ax1.set_aspect('equal')

        ax2.pcolormesh(ARC, Z, Ur_d_grid[idx],
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='gouraud')
        ax2.set_xlabel('Arc distance [mm]')
        ax2.set_ylabel('Axial position [mm]')
        ax2.set_title('Debond R=30mm')
        ax2.set_aspect('equal')

        residual = Ur_d_grid[idx] - Ur_h_grid[idx]
        ax3.pcolormesh(ARC, Z, residual,
                        cmap='RdBu_r', vmin=-vmax_r, vmax=vmax_r, shading='gouraud')
        ax3.set_xlabel('Arc distance [mm]')
        ax3.set_ylabel('Axial position [mm]')
        ax3.set_title('Residual (D $-$ H)')
        ax3.set_aspect('equal')

        # Defect circle on debond + residual panels
        for ax in [ax2, ax3]:
            circle = plt.Circle((defect_arc, 1500), 30, fill=False,
                                 edgecolor='lime', linewidth=1.5, linestyle='--')
            ax.add_patch(circle)

        fig.suptitle('GW Propagation — t = %.1f $\\mu$s' % t_us_val,
                     fontsize=13, fontweight='bold')
        return []

    print("  Creating animation (%d frames)..." % len(indices))
    anim = animation.FuncAnimation(fig, animate, frames=len(indices), blit=False)

    gif_path = os.path.join(out_dir, 'wavefield_animation.gif')
    fig.set_dpi(100)
    anim.save(gif_path, writer='pillow', fps=fps)
    plt.close(fig)
    print("  Saved: %s" % gif_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='GW Validation Visualization')
    parser.add_argument('--csv_dir', default=os.path.expanduser(
        '~/Payload2026/abaqus_work/gw_fairing_dataset'))
    parser.add_argument('--healthy', default='Job-GW-Valid-Healthy-v3_sensors.csv')
    parser.add_argument('--debond', default='Job-GW-Valid-Debond30-v3_sensors.csv')
    parser.add_argument('--out_dir', default=os.path.expanduser(
        '~/Payload2026/results/gw_validation'))
    parser.add_argument('--no_anim', action='store_true',
                        help='Skip animation (slow)')
    parser.add_argument('--times_us', type=float, nargs='+',
                        default=None,
                        help='Snapshot times in microseconds')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csv_h = os.path.join(args.csv_dir, args.healthy)
    csv_d = os.path.join(args.csv_dir, args.debond)

    if not os.path.exists(csv_h):
        print("ERROR: Healthy CSV not found: %s" % csv_h)
        sys.exit(1)
    if not os.path.exists(csv_d):
        print("ERROR: Debond CSV not found: %s" % csv_d)
        sys.exit(1)

    print("Loading data...")
    time_h, arc_h, z_h, Ur_h, n_theta_h, n_z_h = load_sensor_csv(csv_h)
    time_d, arc_d, z_d, Ur_d, n_theta_d, n_z_d = load_sensor_csv(csv_d)

    if n_theta_h != n_theta_d or n_z_h != n_z_d:
        print("WARNING: Grid dimensions differ! H=%dx%d, D=%dx%d" % (
            n_theta_h, n_z_h, n_theta_d, n_z_d))
    n_theta, n_z = n_theta_h, n_z_h

    print("\n--- 1. Wavefield Snapshots ---")
    plot_wavefield_snapshots(time_h, arc_h, z_h, Ur_h, n_theta, n_z,
                             time_d, arc_d, z_d, Ur_d,
                             args.out_dir, args.times_us)

    print("\n--- 2. B-scan ---")
    plot_bscan(time_h, arc_h, z_h, Ur_h, n_theta, n_z,
               time_d, arc_d, z_d, Ur_d, args.out_dir)

    print("\n--- 3. C-scan ---")
    plot_cscan(time_h, arc_h, z_h, Ur_h, n_theta, n_z,
               time_d, arc_d, z_d, Ur_d, args.out_dir)

    print("\n--- 4. Damage Index ---")
    plot_damage_index(time_h, arc_h, z_h, Ur_h, n_theta, n_z,
                      time_d, arc_d, z_d, Ur_d, args.out_dir)

    if not args.no_anim:
        print("\n--- 5. Animation ---")
        create_wavefield_animation(time_h, arc_h, z_h, Ur_h, n_theta, n_z,
                                    time_d, arc_d, z_d, Ur_d, args.out_dir)

    print("\nAll done! Output: %s" % args.out_dir)


if __name__ == '__main__':
    main()
