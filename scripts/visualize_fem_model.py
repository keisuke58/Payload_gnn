#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-quality FEM model overview figure.

4 panels:
  (a) Cross-section: sandwich layup (inner CFRP / Al-honeycomb / outer CFRP)
  (b) Sector plan view: openings, ring frames, sensors, ABL, defect
  (c) Material & CZM properties table
  (d) Defect types schematic

Usage:
  python scripts/visualize_fem_model.py
  python scripts/visualize_fem_model.py --output fem_model_overview.png
"""

import argparse
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import matplotlib.gridspec as gridspec

# ---------- Publication style ----------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'figure.dpi': 200,
    'savefig.dpi': 300,
})

# ---------- Geometry ----------
R_INNER = 2600.0
CORE_T = 38.0
FACE_T = 1.0
R_OUTER = R_INNER + CORE_T
Z_MIN, Z_MAX = 500.0, 2500.0
SECTOR_ANGLE = 30.0
ABL_WIDTH = 2.0
RING_Z = [500, 1000, 1500, 2000, 2500]


def draw_cross_section(ax):
    """(a) Sandwich cross-section with layup detail."""
    # Scale: show ~80mm radial range centered on sandwich
    r_base = 0  # relative coordinates
    y_lo, y_hi = -5, 80  # mm range to show

    # Inner skin (CFRP)
    inner_skin = Rectangle((0, 0), 100, FACE_T, facecolor='#1565C0',
                           edgecolor='k', linewidth=0.8, alpha=0.9)
    ax.add_patch(inner_skin)

    # Honeycomb core
    core = Rectangle((0, FACE_T), 100, CORE_T, facecolor='#FFF9C4',
                     edgecolor='k', linewidth=0.8, alpha=0.9)
    ax.add_patch(core)

    # Draw honeycomb pattern in core
    cell_h = 6.0  # cell height
    cell_w = 5.0  # cell width
    for row in range(int(CORE_T / cell_h)):
        y_c = FACE_T + row * cell_h + cell_h / 2
        offset = cell_w / 2 if row % 2 else 0
        for col in range(int(100 / cell_w) + 1):
            x_c = col * cell_w + offset
            if 0 < x_c < 100:
                hex_y = [y_c - cell_h * 0.4, y_c - cell_h * 0.2,
                         y_c + cell_h * 0.2, y_c + cell_h * 0.4,
                         y_c + cell_h * 0.2, y_c - cell_h * 0.2]
                hex_x = [x_c, x_c + cell_w * 0.35, x_c + cell_w * 0.35,
                         x_c, x_c - cell_w * 0.35, x_c - cell_w * 0.35]
                ax.plot(hex_x + [hex_x[0]], hex_y + [hex_y[0]],
                        color='#C0A000', linewidth=0.3, alpha=0.5)

    # Outer skin (CFRP)
    outer_skin = Rectangle((0, FACE_T + CORE_T), 100, FACE_T,
                           facecolor='#1565C0', edgecolor='k',
                           linewidth=0.8, alpha=0.9)
    ax.add_patch(outer_skin)

    # CZM interface lines
    for y_czm in [FACE_T, FACE_T + CORE_T]:
        ax.plot([0, 100], [y_czm, y_czm], color='#F44336',
                linewidth=1.5, linestyle='--', alpha=0.8)

    # Dimension annotations
    def dim_arrow(x, y1, y2, label, side='right'):
        xoff = 105 if side == 'right' else -5
        ax.annotate('', xy=(xoff, y2), xytext=(xoff, y1),
                    arrowprops=dict(arrowstyle='<->', color='#333',
                                   lw=1.0, shrinkA=0, shrinkB=0))
        ax.text(xoff + (4 if side == 'right' else -4),
                (y1 + y2) / 2, label,
                fontsize=7, ha='left' if side == 'right' else 'right',
                va='center', color='#333')

    dim_arrow(105, 0, FACE_T, '%.1f mm\nCFRP' % FACE_T)
    dim_arrow(105, FACE_T, FACE_T + CORE_T, '%.0f mm\nAl-HC' % CORE_T)
    dim_arrow(105, FACE_T + CORE_T, FACE_T + CORE_T + FACE_T,
              '%.1f mm\nCFRP' % FACE_T)

    # Layup detail (left side)
    ply_angles = [45, 0, -45, 90, 90, -45, 0, 45]
    ply_t = FACE_T / 8
    ply_colors = {45: '#42A5F5', 0: '#1565C0', -45: '#7E57C2', 90: '#26A69A'}

    # Zoom box for inner skin layup
    zoom_x, zoom_y = -45, -2
    zoom_w, zoom_h = 40, 12
    ax.add_patch(Rectangle((zoom_x, zoom_y), zoom_w, zoom_h,
                           facecolor='white', edgecolor='#666',
                           linewidth=0.5, zorder=5))
    for i, angle in enumerate(ply_angles):
        y_ply = zoom_y + 1 + i * (zoom_h - 2) / 8
        ax.add_patch(Rectangle((zoom_x + 1, y_ply),
                               zoom_w - 2, (zoom_h - 2) / 8 - 0.1,
                               facecolor=ply_colors[angle], alpha=0.7,
                               edgecolor='none', zorder=6))
        ax.text(zoom_x + zoom_w / 2, y_ply + (zoom_h - 2) / 16,
                '%+d°' % angle, fontsize=5, ha='center', va='center',
                color='white', fontweight='bold', zorder=7)

    ax.text(zoom_x + zoom_w / 2, zoom_y + zoom_h + 1,
            '[45/0/$-$45/90]$_s$', fontsize=7.5, ha='center',
            fontweight='bold')
    # Connect zoom to inner skin
    ax.plot([zoom_x + zoom_w, 0], [zoom_y + zoom_h / 2, FACE_T / 2],
            color='#666', linewidth=0.5, linestyle=':', zorder=4)

    # CZM label
    ax.text(50, FACE_T + 1.5, 'CZM interface', fontsize=6.5,
            ha='center', color='#F44336', fontstyle='italic')
    ax.text(50, FACE_T + CORE_T - 1.5, 'CZM interface', fontsize=6.5,
            ha='center', color='#F44336', fontstyle='italic', va='top')

    # Labels
    ax.text(50, FACE_T / 2, 'Inner CFRP skin', fontsize=7.5,
            ha='center', va='center', color='white', fontweight='bold')
    ax.text(50, FACE_T + CORE_T / 2, 'Al-5052 Honeycomb Core',
            fontsize=8, ha='center', va='center', fontweight='bold',
            color='#5D4037')
    ax.text(50, FACE_T + CORE_T + FACE_T / 2, 'Outer CFRP skin',
            fontsize=7.5, ha='center', va='center', color='white',
            fontweight='bold')

    # Radial direction arrow
    ax.annotate('', xy=(95, FACE_T + CORE_T + FACE_T + 4),
                xytext=(95, -3),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))
    ax.text(97, FACE_T + CORE_T / 2, '$r$', fontsize=10, ha='left',
            va='center', fontstyle='italic')

    ax.set_xlim(-50, 130)
    ax.set_ylim(-5, FACE_T + CORE_T + FACE_T + 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Sandwich cross-section', fontweight='bold', pad=8)


def draw_sector_plan(ax):
    """(b) Sector plan view (theta vs z) with structural features."""
    # Sector domain
    ax.add_patch(Rectangle((0, Z_MIN), SECTOR_ANGLE, Z_MAX - Z_MIN,
                           facecolor='#E3F2FD', edgecolor='#0277BD',
                           linewidth=1.5))

    # ABL zones
    for th_lo, th_hi in [(0, ABL_WIDTH), (SECTOR_ANGLE - ABL_WIDTH, SECTOR_ANGLE)]:
        ax.add_patch(Rectangle((th_lo, Z_MIN), th_hi - th_lo, Z_MAX - Z_MIN,
                               facecolor='#BDBDBD', alpha=0.4, edgecolor='none'))
    ax.text(ABL_WIDTH / 2, Z_MIN + 50, 'ABL', fontsize=6, ha='center',
            rotation=90, color='#616161')
    ax.text(SECTOR_ANGLE - ABL_WIDTH / 2, Z_MIN + 50, 'ABL', fontsize=6,
            ha='center', rotation=90, color='#616161')

    # Ring frames
    for z_f in RING_Z:
        ax.axhline(z_f, color='#795548', linewidth=2.0, alpha=0.6)
        ax.text(SECTOR_ANGLE + 0.5, z_f, 'Frame', fontsize=5.5,
                va='center', color='#795548')

    # HVAC door opening
    hvac_th, hvac_z, hvac_d = 20.0, 2500.0, 400.0
    arc_r_deg = math.degrees(hvac_d / 2 / R_OUTER)
    opening = mpatches.Ellipse((hvac_th, hvac_z), arc_r_deg * 2, hvac_d,
                               facecolor='white', edgecolor='#333',
                               linewidth=1.5, zorder=5)
    ax.add_patch(opening)
    ax.text(hvac_th, hvac_z, 'HVAC\nDoor\n$\\phi$400', fontsize=5.5,
            ha='center', va='center', fontweight='bold', zorder=6)

    # Vent hole (Phase 2)
    vent_th, vent_z, vent_d = 15.0, 600.0, 100.0
    vent_r_deg = math.degrees(vent_d / 2 / R_OUTER)
    vent = mpatches.Ellipse((vent_th, vent_z), vent_r_deg * 2, vent_d,
                            facecolor='white', edgecolor='#666',
                            linewidth=1.0, linestyle='--', zorder=5)
    ax.add_patch(vent)
    ax.text(vent_th, vent_z, 'Vent\n$\\phi$100', fontsize=5,
            ha='center', va='center', zorder=6, color='#666')

    # Sensor grid (8x13)
    theta_lo = ABL_WIDTH + 1.0
    theta_hi = SECTOR_ANGLE - ABL_WIDTH - 1.0
    z_lo, z_hi = Z_MIN + 50, Z_MAX - 50
    for iz in range(13):
        z_s = z_lo + iz * (z_hi - z_lo) / 12
        for it in range(8):
            th_s = theta_lo + it * (theta_hi - theta_lo) / 7
            ax.plot(th_s, z_s, 'o', color='#FF6F00', ms=2.5,
                    mec='#BF360C', mew=0.3, zorder=7)

    # Excitation point
    ax.plot(15, 1500, '*', color='#D32F2F', ms=10, zorder=8)
    ax.text(15, 1540, 'PZT', fontsize=6, ha='center', color='#D32F2F',
            fontweight='bold')

    # Example defect
    defect_th, defect_z, defect_r = 10.0, 1200.0, 50.0
    d_arc = math.degrees(defect_r / R_OUTER)
    defect = mpatches.Ellipse((defect_th, defect_z), d_arc * 2, defect_r * 2,
                              facecolor='#EF5350', alpha=0.5,
                              edgecolor='#B71C1C', linewidth=1.2, zorder=6)
    ax.add_patch(defect)
    ax.text(defect_th, defect_z, 'Defect\n$r$=50', fontsize=5.5,
            ha='center', va='center', color='#B71C1C', fontweight='bold',
            zorder=7)

    ax.set_xlabel('$\\theta$ (deg)')
    ax.set_ylabel('$z$ (mm)')
    ax.set_xlim(-1, SECTOR_ANGLE + 4)
    ax.set_ylim(Z_MIN - 100, Z_MAX + 200)
    ax.set_title('(b) 30° sector plan view', fontweight='bold', pad=8)

    # Custom legend
    legend_items = [
        mpatches.Patch(facecolor='#E3F2FD', edgecolor='#0277BD', label='FEM domain'),
        mpatches.Patch(facecolor='#BDBDBD', alpha=0.5, label='ABL (2°)'),
        plt.Line2D([0], [0], color='#795548', lw=2, label='Ring frame (Al-7075)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6F00',
                   ms=5, label='Sensor (104)'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#D32F2F',
                   ms=8, label='PZT excitation'),
        mpatches.Patch(facecolor='#EF5350', alpha=0.5, edgecolor='#B71C1C',
                       label='Defect zone'),
    ]
    ax.legend(handles=legend_items, fontsize=6.5, loc='upper left',
              framealpha=0.95)


def draw_properties_table(ax):
    """(c) Material properties summary table."""
    ax.axis('off')
    ax.set_title('(c) Material properties', fontweight='bold', pad=8)

    table_data = [
        ['Component', 'Material', 'Key Properties'],
        ['Face sheets', 'T1000G/Epoxy\nCFRP [45/0/-45/90]$_s$',
         '$E_1$ = 160 GPa, $E_2$ = 10 GPa\n'
         '$G_{12}$ = 5 GPa, $\\nu_{12}$ = 0.3\n'
         '$\\rho$ = 1600 kg/m³\n'
         '$\\alpha_1$ = $-$0.3×10$^{-6}$ /°C'],
        ['Core', 'Al-5052\nHoneycomb',
         '$E_R$ = 1000 MPa, $E_{\\theta}$ = 10 MPa\n'
         '$G_{R\\theta}$ = 400, $G_{Rz}$ = 240 MPa\n'
         '$\\rho$ = 50 kg/m³\n'
         '$\\alpha$ = 23×10$^{-6}$ /°C'],
        ['Frames', 'Al-7075',
         '$E$ = 71.7 GPa, $\\nu$ = 0.33\n'
         '$\\rho$ = 2810 kg/m³'],
        ['CZM\n(skin-core)', 'Surface-based\ncohesive',
         '$K_n$ = 10$^5$ N/mm³, $K_s$ = 5×10$^4$\n'
         '$t_n$ = 50 MPa, $t_s$ = 40 MPa\n'
         '$G_{Ic}$ = 0.3, $G_{IIc}$ = 1.0 N/mm\n'
         'BK law ($\\eta$ = 2.284)'],
    ]

    col_widths = [0.18, 0.22, 0.60]
    row_heights = [0.06, 0.20, 0.20, 0.12, 0.22]

    y_pos = 0.95
    for i, row in enumerate(table_data):
        x_pos = 0.02
        h = row_heights[i]
        for j, cell in enumerate(row):
            w = col_widths[j]
            if i == 0:  # header
                ax.add_patch(FancyBboxPatch((x_pos, y_pos - h), w - 0.01, h,
                             boxstyle='round,pad=0.005',
                             facecolor='#37474F', edgecolor='#263238'))
                ax.text(x_pos + w / 2, y_pos - h / 2, cell,
                        fontsize=7.5, ha='center', va='center',
                        color='white', fontweight='bold',
                        transform=ax.transAxes)
            else:
                fc = '#FAFAFA' if i % 2 == 0 else '#F5F5F5'
                ax.add_patch(FancyBboxPatch((x_pos, y_pos - h), w - 0.01, h,
                             boxstyle='round,pad=0.005',
                             facecolor=fc, edgecolor='#BDBDBD',
                             linewidth=0.5))
                ax.text(x_pos + w / 2, y_pos - h / 2, cell,
                        fontsize=6.5, ha='center', va='center',
                        transform=ax.transAxes, linespacing=1.4)
            x_pos += w
        y_pos -= h

    # Thermal field note
    ax.text(0.02, y_pos - 0.05,
            'Thermal field: $T_{\\mathrm{ref}}$ = 180°C (cure), '
            '$T_{\\mathrm{outer}}$ = 40°C, '
            '$T_{\\mathrm{inner}}$ = 25°C\n'
            'Moisture: 2% absorption → $E_2$, $G_{ij}$ reduced by 8%\n'
            'Excitation: 50 kHz, 5-cycle Hanning-windowed toneburst, '
            '$v_g$ ≈ 1100 m/s (A$_0$)',
            fontsize=6.5, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1',
                      edgecolor='#FFC107', linewidth=0.5))


def draw_defect_types(ax):
    """(d) 7 defect types schematic."""
    ax.axis('off')
    ax.set_title('(d) Defect types (7 categories)', fontweight='bold', pad=8)

    defects = [
        ('Debonding', 'Skin-core CZM\ndegraded 90%',
         '#E53935', 'Most common;\nadhesive failure'),
        ('FOD', 'Core density ×3\nstiffness ×5',
         '#FB8C00', 'Foreign object\ndebris inclusion'),
        ('Impact', 'Skin: $E$ ×0.3\nCore: crush zone',
         '#7B1FA2', 'BVID from\nhandling/hail'),
        ('Delamination', 'Inter-ply CZM\nin CFRP skin',
         '#1E88E5', 'Between ply\ninterfaces'),
        ('Inner debond', 'Inner skin CZM\nonly degraded',
         '#00897B', 'Single-side\ndebonding'),
        ('Thermal prog.', 'CZM progressive\nweakening',
         '#D81B60', 'Cyclic thermal\nfatigue damage'),
        ('Acoustic fatigue', 'Core shear $G$\nreduced 50%',
         '#6D4C41', 'Vibration-induced\ncore degradation'),
    ]

    y_start = 0.92
    dy = 0.125

    for i, (name, mechanism, color, note) in enumerate(defects):
        y = y_start - i * dy

        # Color dot
        ax.add_patch(plt.Circle((0.04, y), 0.018, facecolor=color,
                                edgecolor='k', linewidth=0.5,
                                transform=ax.transAxes))

        # Name
        ax.text(0.10, y, name, fontsize=7.5, fontweight='bold',
                va='center', transform=ax.transAxes)

        # Mechanism
        ax.text(0.42, y, mechanism, fontsize=6.5, va='center',
                transform=ax.transAxes, color='#333',
                family='monospace')

        # Note
        ax.text(0.78, y, note, fontsize=6, va='center',
                transform=ax.transAxes, color='#666',
                fontstyle='italic')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(2, 2, hspace=0.25, wspace=0.25,
                           height_ratios=[1, 1.1])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_cross_section(ax_a)
    draw_sector_plan(ax_b)
    draw_properties_table(ax_c)
    draw_defect_types(ax_d)

    fig.suptitle('FEM Model Configuration: H3 Fairing GW-SHM',
                 fontsize=14, fontweight='bold', y=0.98)

    out_path = args.output or 'fem_model_overview.png'
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: %s (dpi=%d)" % (out_path, args.dpi))


if __name__ == '__main__':
    main()
