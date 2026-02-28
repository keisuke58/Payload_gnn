#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize H3 Fairing Sandwich Panel Layup Structure

Structure (from generate_fairing_dataset.py, generate_realistic_fairing.py):
  - Outer Skin: 8 plies CFRP [45/0/-45/90]s, 1.0 mm total (0.125 mm/ply)
  - Core: Al honeycomb 38 mm
  - Inner Skin: 8 plies CFRP [45/0/-45/90]s, 1.0 mm total (0.125 mm/ply)

Total: 40 mm (1 + 38 + 1)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# From generate_fairing_dataset.py
FACE_T = 1.0   # mm (CFRP Face Sheet Thickness)
CORE_T = 38.0  # mm (Honeycomb Core Thickness)
LAYUP_ANGLES = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]  # [45/0/-45/90]s
PLY_T = FACE_T / 8.0  # 0.125 mm per ply

# Colors
COLOR_45 = '#4A90D9'   # blue
COLOR_0 = '#50C878'    # green
COLOR_M45 = '#E8A317'  # amber
COLOR_90 = '#CD5C5C'   # red
COLOR_CORE = '#C0C0C0'  # silver

ANGLE_COLORS = {
    45.0: COLOR_45,
    0.0: COLOR_0,
    -45.0: COLOR_M45,
    90.0: COLOR_90,
}


def draw_ply_stack(ax, y_base, plies, height_scale=1.0, label_side='left'):
    """Draw a stack of plies (8 plies for one face sheet)."""
    for i, ang in enumerate(plies):
        color = ANGLE_COLORS.get(ang, '#888888')
        rect = mpatches.Rectangle((0, y_base + i * PLY_T * height_scale),
                                  1.0, PLY_T * height_scale,
                                  facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        # Label angle
        tx = -0.08 if label_side == 'left' else 1.08
        ax.text(tx, y_base + (i + 0.5) * PLY_T * height_scale, f'{int(ang)}°',
                va='center', ha='right' if label_side == 'left' else 'left',
                fontsize=8, fontweight='bold')
    return y_base + 8 * PLY_T * height_scale


def main():
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-1, 42)

    # Scale: core is 38mm, each skin is 1mm. For visual balance, scale skins up.
    SKIN_VIS_SCALE = 8.0  # 1mm * 8 = 8 units visually
    CORE_VIS = 24.0      # 38mm -> 24 units (so total ~40)

    y = 0

    # Inner Skin (bottom, 8 plies)
    ax.text(-0.25, y + 4 * PLY_T * SKIN_VIS_SCALE, 'Inner\nSkin', fontsize=9,
            va='center', ha='right', fontweight='bold')
    y = draw_ply_stack(ax, y, LAYUP_ANGLES, height_scale=SKIN_VIS_SCALE, label_side='left')
    ax.axhline(y, color='black', linewidth=1, linestyle='-')

    # Core (Al honeycomb)
    core_bottom = y
    rect = mpatches.Rectangle((0, y), 1.0, CORE_VIS,
                               facecolor=COLOR_CORE, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(0.5, y + CORE_VIS / 2, 'Al Honeycomb\nCore\n38 mm',
            va='center', ha='center', fontsize=10, fontweight='bold')
    ax.text(-0.08, y + CORE_VIS / 2, '38 mm', va='center', ha='right', fontsize=9)
    y += CORE_VIS
    ax.axhline(y, color='black', linewidth=1, linestyle='-')

    # Outer Skin (top, 8 plies)
    ax.text(-0.25, y + 4 * PLY_T * SKIN_VIS_SCALE, 'Outer\nSkin', fontsize=9,
            va='center', ha='right', fontweight='bold')
    y = draw_ply_stack(ax, y, LAYUP_ANGLES, height_scale=SKIN_VIS_SCALE, label_side='left')

    ax.set_ylim(-1, y + 2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_45, edgecolor='black', label='45°'),
        mpatches.Patch(facecolor=COLOR_0, edgecolor='black', label='0°'),
        mpatches.Patch(facecolor=COLOR_M45, edgecolor='black', label='-45°'),
        mpatches.Patch(facecolor=COLOR_90, edgecolor='black', label='90°'),
        mpatches.Patch(facecolor=COLOR_CORE, edgecolor='black', label='Al Core'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.set_title('H3 Fairing Sandwich Panel Layup Structure\n[45/0/-45/90]s × 2 (Inner + Outer) | Total 40 mm',
                 fontsize=12, fontweight='bold')

    # Summary text
    summary = (
        'Structure: 17 layers total\n'
        '  • Inner Skin: 8 plies CFRP, 1.0 mm (0.125 mm/ply)\n'
        '  • Core: Al-5052 honeycomb, 38 mm\n'
        '  • Outer Skin: 8 plies CFRP, 1.0 mm (0.125 mm/ply)\n'
        'Layup: [45/0/-45/90]s (quasi-isotropic)'
    )
    ax.text(1.15, 20, summary, fontsize=8, va='center', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'wiki_repo', 'images')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'layup_structure.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
