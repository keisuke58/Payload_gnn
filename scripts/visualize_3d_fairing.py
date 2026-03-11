#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D visualization of H3 fairing GW model — full 360° from 1/12 sector.
Publication-quality figure.

Usage:
  python scripts/visualize_3d_fairing.py
  python scripts/visualize_3d_fairing.py --defect_z 1500 --defect_theta 15 --defect_r 50
"""

import argparse
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe

# ---------- Geometry constants ----------
RADIUS_INNER = 2600.0      # mm
CORE_T = 38.0              # mm
FACE_T = 1.0               # mm
R_OUTER = RADIUS_INNER + CORE_T  # 2638 mm

# FEM analysis domain (barrel section)
Z_MIN = 500.0              # mm
Z_MAX = 2500.0             # mm
SECTOR_ANGLE = 30.0        # degrees (1/12)
ABL_WIDTH_DEG = 2.0
N_SENSORS_DEFAULT = 104    # 8 x 13 grid

# Full fairing visual geometry (H3: diameter ~5.2m, total ~12m)
BARREL_VIS_BOTTOM = 0.0
BARREL_VIS_TOP = 4000.0
NOSE_TIP_Z = 12000.0

EXCITE_Z_DEFAULT = (Z_MIN + Z_MAX) / 2.0
EXCITE_THETA_DEFAULT = SECTOR_ANGLE / 2.0

# ---------- Publication style ----------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def cylinder_surface(r, theta_range, z_range, n_theta=40, n_z=20):
    theta = np.linspace(np.radians(theta_range[0]),
                        np.radians(theta_range[1]), n_theta)
    z = np.linspace(z_range[0], z_range[1], n_z)
    Theta, Z = np.meshgrid(theta, z)
    X = r * np.cos(Theta)
    Y = r * np.sin(Theta)
    return X, Y, Z


def ogive_surface(r_base, z_start, z_tip, theta_range, n_theta=40, n_z=40):
    L = z_tip - z_start
    rho = (r_base**2 + L**2) / (2 * r_base)
    theta = np.linspace(np.radians(theta_range[0]),
                        np.radians(theta_range[1]), n_theta)
    z_arr = np.linspace(z_start, z_tip, n_z)
    X = np.zeros((n_z, n_theta))
    Y = np.zeros((n_z, n_theta))
    Z = np.zeros((n_z, n_theta))
    for i, z_val in enumerate(z_arr):
        x_from_tip = z_tip - z_val
        if x_from_tip <= 0:
            r_local = 0.0
        else:
            inner = rho**2 - (L - x_from_tip)**2
            r_local = max(0.0, np.sqrt(max(0, inner)) + r_base - rho)
        for j, th in enumerate(theta):
            X[i, j] = r_local * np.cos(th)
            Y[i, j] = r_local * np.sin(th)
            Z[i, j] = z_val
    return X, Y, Z


def sensor_positions(n_sensors=N_SENSORS_DEFAULT, sector_angle=SECTOR_ANGLE,
                     z_min=Z_MIN, z_max=Z_MAX, r=R_OUTER,
                     abl_width_deg=ABL_WIDTH_DEG):
    theta_lo = math.radians(abl_width_deg + 1.0)
    theta_hi = math.radians(sector_angle - abl_width_deg - 1.0)
    z_lo = z_min + 50.0
    z_hi = z_max - 50.0
    arc_span = r * (theta_hi - theta_lo)
    z_span = z_hi - z_lo
    aspect = arc_span / z_span
    n_theta = max(2, int(math.sqrt(n_sensors * aspect) + 0.5))
    n_z = max(2, int(n_sensors / n_theta + 0.5))
    d_theta = (theta_hi - theta_lo) / max(n_theta - 1, 1)
    d_z = z_span / max(n_z - 1, 1)
    positions = []
    for iz in range(n_z):
        z_s = z_lo + iz * d_z
        for it in range(n_theta):
            theta_s = theta_lo + it * d_theta
            positions.append((r * math.cos(theta_s),
                              r * math.sin(theta_s), z_s))
    return np.array(positions), n_theta, n_z


def defect_patch(z_center, theta_deg, radius_mm, r=R_OUTER, n_pts=80):
    theta_c = math.radians(theta_deg)
    angles = np.linspace(0, 2 * np.pi, n_pts)
    arc_r = radius_mm / r
    xs, ys, zs = [], [], []
    for a in angles:
        theta = theta_c + arc_r * math.cos(a)
        xs.append(r * math.cos(theta))
        ys.append(r * math.sin(theta))
        zs.append(z_center + radius_mm * math.sin(a))
    return np.array(xs), np.array(ys), np.array(zs)


def main():
    parser = argparse.ArgumentParser(description='3D Fairing Visualization')
    parser.add_argument('--defect_z', type=float, default=None)
    parser.add_argument('--defect_theta', type=float, default=None)
    parser.add_argument('--defect_r', type=float, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--elev', type=float, default=None)
    parser.add_argument('--azim', type=float, default=None)
    args = parser.parse_args()

    has_defect = all(v is not None for v in
                     [args.defect_z, args.defect_theta, args.defect_r])

    face_theta = args.defect_theta if has_defect else SECTOR_ANGLE / 2.0
    cam_azim = args.azim if args.azim is not None else -face_theta
    cam_elev = args.elev if args.elev is not None else 15

    # --- Figure: main + inset ---
    fig = plt.figure(figsize=(9, 13))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('persp', focal_length=0.3)

    # ==================== Full fairing ====================
    n_th = 18
    n_z_barrel = 12
    n_z_nose = 30

    # Colors
    COL_SECTOR = '#4FC3F7'
    COL_SECTOR_EDGE = '#0277BD'
    COL_PASSIVE = '#CFD8DC'
    COL_PASSIVE_EDGE = '#B0BEC5'
    COL_FEM_BAND = '#E1F5FE'

    for i in range(12):
        th0 = i * SECTOR_ANGLE
        th1 = (i + 1) * SECTOR_ANGLE
        is_active = (i == 0)

        # --- Barrel (full visual extent) ---
        X, Y, Z = cylinder_surface(R_OUTER, (th0, th1),
                                   (BARREL_VIS_BOTTOM, BARREL_VIS_TOP),
                                   n_theta=n_th, n_z=n_z_barrel)
        if is_active:
            ax.plot_surface(X, Y, Z, alpha=0.40, color=COL_SECTOR,
                            edgecolor=COL_SECTOR_EDGE, linewidth=0.3)
        else:
            ax.plot_surface(X, Y, Z, alpha=0.06, color=COL_PASSIVE,
                            edgecolor=COL_PASSIVE_EDGE, linewidth=0.03)

        # --- Nose cone ---
        Xn, Yn, Zn = ogive_surface(R_OUTER, BARREL_VIS_TOP, NOSE_TIP_Z,
                                    (th0, th1), n_theta=n_th, n_z=n_z_nose)
        if is_active:
            ax.plot_surface(Xn, Yn, Zn, alpha=0.30, color='#81D4FA',
                            edgecolor=COL_SECTOR_EDGE, linewidth=0.2)
        else:
            ax.plot_surface(Xn, Yn, Zn, alpha=0.04, color=COL_PASSIVE,
                            edgecolor=COL_PASSIVE_EDGE, linewidth=0.02)

    # --- FEM analysis domain highlight band (on sector 0) ---
    Xf, Yf, Zf = cylinder_surface(R_OUTER + 5, (0, SECTOR_ANGLE),
                                   (Z_MIN, Z_MAX), n_theta=n_th, n_z=8)
    ax.plot_surface(Xf, Yf, Zf, alpha=0.15, color='#FFEB3B',
                    edgecolor='#F9A825', linewidth=0.4)

    # --- Rings ---
    ring_theta = np.linspace(0, 2 * np.pi, 200)
    rx = R_OUTER * np.cos(ring_theta)
    ry = R_OUTER * np.sin(ring_theta)
    for z_val, lw, alpha in [(BARREL_VIS_BOTTOM, 1.2, 0.5),
                              (BARREL_VIS_TOP, 0.8, 0.35),
                              (Z_MIN, 0.6, 0.4),
                              (Z_MAX, 0.6, 0.4)]:
        ax.plot(rx, ry, np.full_like(ring_theta, z_val),
                color='#37474F', linewidth=lw, alpha=alpha)

    # --- Sector boundary lines ---
    for i in range(12):
        th = math.radians(i * SECTOR_ANGLE)
        xl = [R_OUTER * math.cos(th)] * 2
        yl = [R_OUTER * math.sin(th)] * 2
        if i == 0:
            ax.plot(xl, yl, [BARREL_VIS_BOTTOM, BARREL_VIS_TOP],
                    color=COL_SECTOR_EDGE, linewidth=1.2, alpha=0.6)
        else:
            ax.plot(xl, yl, [BARREL_VIS_BOTTOM, BARREL_VIS_TOP],
                    color='#90A4AE', linewidth=0.2, alpha=0.2)

    # ==================== Sensors ====================
    sens_pos, nt, nz = sensor_positions()
    ax.scatter(sens_pos[:, 0], sens_pos[:, 1], sens_pos[:, 2],
               c='#FF6F00', s=8, marker='o', edgecolors='#BF360C',
               linewidths=0.3, zorder=10, depthshade=False,
               label='Sensor grid (%d$\\times$%d = %d)' % (nt, nz, len(sens_pos)))

    # ==================== Excitation ====================
    exc_theta = math.radians(EXCITE_THETA_DEFAULT)
    ax.scatter([R_OUTER * math.cos(exc_theta)],
               [R_OUTER * math.sin(exc_theta)],
               [EXCITE_Z_DEFAULT],
               c='#D32F2F', s=80, marker='*', zorder=11, depthshade=False,
               label='PZT excitation (50 kHz)')

    # ==================== Defect ====================
    if has_defect:
        dx, dy, dz = defect_patch(args.defect_z, args.defect_theta,
                                  args.defect_r)
        verts = [list(zip(dx, dy, dz))]
        poly = Poly3DCollection(verts, alpha=0.65, facecolor='#EF5350',
                                edgecolor='#B71C1C', linewidth=1.5)
        ax.add_collection3d(poly)
        dc_th = math.radians(args.defect_theta)
        ax.scatter([R_OUTER * math.cos(dc_th)],
                   [R_OUTER * math.sin(dc_th)],
                   [args.defect_z],
                   c='#B71C1C', s=50, marker='x', zorder=11,
                   depthshade=False,
                   label='Debonding ($r$ = %g mm)' % args.defect_r)

    # ==================== Dimension annotations ====================
    # FEM domain bracket (right side)
    bk_th = math.radians(SECTOR_ANGLE + 3)
    bk_r = R_OUTER * 1.08
    bk_x, bk_y = bk_r * math.cos(bk_th), bk_r * math.sin(bk_th)
    ax.plot([bk_x, bk_x], [bk_y, bk_y], [Z_MIN, Z_MAX],
            color='#333', linewidth=1.5, alpha=0.8)
    # Small ticks
    for z_tick in [Z_MIN, Z_MAX]:
        ax.plot([bk_x * 0.97, bk_x * 1.03],
                [bk_y * 0.97, bk_y * 1.03],
                [z_tick, z_tick],
                color='#333', linewidth=1.2, alpha=0.8)

    # ==================== Styling ====================
    # Remove axis labels (cleaner for publication)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Set equal aspect ratio matching real proportions
    # X,Y range: -R to R = ±2638; Z range: 0 to 12000
    max_range = max(2 * R_OUTER, NOSE_TIP_Z - BARREL_VIS_BOTTOM)
    ax.set_xlim(-R_OUTER * 1.15, R_OUTER * 1.15)
    ax.set_ylim(-R_OUTER * 1.15, R_OUTER * 1.15)
    ax.set_zlim(BARREL_VIS_BOTTOM - 200, NOSE_TIP_Z + 200)

    # Proper box aspect for real proportions
    x_span = 2 * R_OUTER * 1.15
    y_span = 2 * R_OUTER * 1.15
    z_span = (NOSE_TIP_Z + 200) - (BARREL_VIS_BOTTOM - 200)
    ax.set_box_aspect([x_span, y_span, z_span])

    ax.view_init(elev=cam_elev, azim=cam_azim)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Title — single line to avoid inset overlap
    ax.text2D(0.02, 0.98,
              'H3 Fairing GW-SHM (30° sector)',
              transform=ax.transAxes, fontsize=11, fontweight='bold',
              verticalalignment='top')
    ax.set_title('')

    # Legend — place at top-left, away from the fairing body
    leg = ax.legend(loc='upper left', fontsize=7.5, framealpha=0.95,
                    edgecolor='#999', borderpad=0.6, labelspacing=0.5,
                    handletextpad=0.4, bbox_to_anchor=(0.0, 0.95))
    leg.get_frame().set_linewidth(0.5)

    # Dimension info box — bottom-left
    dim_lines = [
        '$R_{\\mathrm{inner}}$ = %.1f m' % (RADIUS_INNER / 1000),
        'Barrel: $z$ = %.1f–%.1f m' % (Z_MIN / 1000, Z_MAX / 1000),
        'Core: %.0f mm Al-honeycomb' % CORE_T,
        'Skin: %.1f mm CFRP (×2)' % FACE_T,
        'Sector: %d° (1/12)' % int(SECTOR_ANGLE),
    ]
    if has_defect:
        dim_lines.append('Defect: $r$ = %g mm at $z$ = %g m, '
                         '$\\theta$ = %g°' % (
                             args.defect_r, args.defect_z / 1000,
                             args.defect_theta))

    ax.text2D(0.02, 0.01, '\n'.join(dim_lines), transform=ax.transAxes,
              fontsize=7, verticalalignment='bottom',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor='#999', alpha=0.95, linewidth=0.5),
              family='serif')

    # FEM domain label
    ax.text2D(0.88, 0.32, 'FEM\ndomain',
              transform=ax.transAxes, fontsize=7.5, ha='center',
              color='#F57F17', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7',
                        edgecolor='#F9A825', alpha=0.9, linewidth=0.5))

    # ==================== INSET: Barrel close-up ====================
    ax_inset = fig.add_axes([0.60, 0.48, 0.38, 0.35], projection='3d')
    ax_inset.set_proj_type('persp', focal_length=0.25)

    # Draw only the analysis sector barrel
    Xi, Yi, Zi = cylinder_surface(R_OUTER, (0, SECTOR_ANGLE),
                                  (Z_MIN, Z_MAX), n_theta=20, n_z=15)
    ax_inset.plot_surface(Xi, Yi, Zi, alpha=0.30, color=COL_SECTOR,
                          edgecolor=COL_SECTOR_EDGE, linewidth=0.2)

    # Draw adjacent sectors (partial, for context)
    for offset in [-1, 1]:
        th0_adj = offset * SECTOR_ANGLE
        th1_adj = (offset + 1) * SECTOR_ANGLE
        if th0_adj < 0:
            th0_adj += 360
            th1_adj += 360
        Xa, Ya, Za = cylinder_surface(R_OUTER, (th0_adj, th1_adj),
                                      (Z_MIN, Z_MAX), n_theta=10, n_z=8)
        ax_inset.plot_surface(Xa, Ya, Za, alpha=0.06, color=COL_PASSIVE,
                              edgecolor=COL_PASSIVE_EDGE, linewidth=0.02)

    # Sensors in inset (larger markers)
    ax_inset.scatter(sens_pos[:, 0], sens_pos[:, 1], sens_pos[:, 2],
                     c='#FF6F00', s=12, marker='o', edgecolors='#BF360C',
                     linewidths=0.3, zorder=10, depthshade=False)

    # Excitation in inset
    ax_inset.scatter([R_OUTER * math.cos(exc_theta)],
                     [R_OUTER * math.sin(exc_theta)],
                     [EXCITE_Z_DEFAULT],
                     c='#D32F2F', s=60, marker='*', zorder=11,
                     depthshade=False)

    # Defect in inset
    if has_defect:
        dx2, dy2, dz2 = defect_patch(args.defect_z, args.defect_theta,
                                      args.defect_r)
        verts2 = [list(zip(dx2, dy2, dz2))]
        poly2 = Poly3DCollection(verts2, alpha=0.7, facecolor='#EF5350',
                                 edgecolor='#B71C1C', linewidth=1.2)
        ax_inset.add_collection3d(poly2)

    # Inset view: zoom into barrel, face sector center
    # Compute tight limits around the sector
    margin = 400
    theta_mid = math.radians(SECTOR_ANGLE / 2.0)
    x_center = R_OUTER * math.cos(theta_mid)
    y_center = R_OUTER * math.sin(theta_mid)
    ax_inset.set_xlim(x_center - R_OUTER * 0.25, x_center + R_OUTER * 0.25)
    ax_inset.set_ylim(y_center - R_OUTER * 0.25, y_center + R_OUTER * 0.25)
    ax_inset.set_zlim(Z_MIN - margin, Z_MAX + margin)

    x_s = 2 * R_OUTER * 0.25
    y_s = 2 * R_OUTER * 0.25
    z_s = (Z_MAX + margin) - (Z_MIN - margin)
    ax_inset.set_box_aspect([x_s, y_s, z_s])

    ax_inset.view_init(elev=12, azim=-face_theta)
    ax_inset.grid(False)
    ax_inset.set_xticklabels([])
    ax_inset.set_yticklabels([])
    ax_inset.set_zticklabels([])
    ax_inset.xaxis.pane.fill = False
    ax_inset.yaxis.pane.fill = False
    ax_inset.zaxis.pane.fill = False
    ax_inset.xaxis.pane.set_edgecolor('#ddd')
    ax_inset.yaxis.pane.set_edgecolor('#ddd')
    ax_inset.zaxis.pane.set_edgecolor('#ddd')

    # Inset title
    ax_inset.set_title('FEM domain close-up\n($z$ = 0.5–2.5 m)',
                       fontsize=8, pad=2)

    # Inset border
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('#666')
        spine.set_linewidth(0.8)

    out_path = args.output or 'fairing_3d.png'
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: %s (dpi=%d)" % (out_path, args.dpi))


if __name__ == '__main__':
    main()
