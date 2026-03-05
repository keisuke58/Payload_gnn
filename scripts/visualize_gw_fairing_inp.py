#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GW フェアリング INP 3次元可視化

Abaqus INP をパースし、外皮メッシュ・センサ位置・欠陥ゾーンを 3D 表示。
Healthy と Defect モデルの比較用。

Usage:
  python scripts/visualize_gw_fairing_inp.py abaqus_work/Job-GW-Fair-Healthy.inp
  python scripts/visualize_gw_fairing_inp.py abaqus_work/Job-GW-Fair-0000.inp --defect '{"z_center":818.7,"theta_deg":21.33,"radius":59}'
"""

import argparse
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

RADIUS_INNER = 2600.0
CORE_T = 38.0
R_OUTER = RADIUS_INNER + CORE_T


def parse_inp_nodes(inp_path, instance_name='Part-OuterSkin-1'):
    """Parse nodes from INP for given instance. Returns {label: (x,y,z)}."""
    nodes = {}
    in_target = False
    in_node_block = False

    with open(inp_path) as f:
        for line in f:
            raw = line
            line = line.strip()
            if not line or line.startswith('**'):
                continue
            if line.upper().startswith('*INSTANCE'):
                m = re.search(r'name=([^,\s]+)', line, re.I)
                current = m.group(1).strip() if m else None
                in_target = (current == instance_name)
                in_node_block = False
                continue
            if line.upper().startswith('*END INSTANCE'):
                in_target = False
                in_node_block = False
                continue
            if in_target and line.upper().startswith('*NODE'):
                in_node_block = True
                continue
            if in_target and in_node_block:
                if line.startswith('*'):
                    in_node_block = False
                    continue
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        label = int(parts[0].strip())
                        x = float(parts[1].strip())
                        y = float(parts[2].strip())
                        z = float(parts[3].strip())
                        nodes[label] = (x, y, z)
                    except (ValueError, IndexError):
                        pass
    return nodes


def parse_sensor_nodes(inp_path):
    """Parse Set-Sensor-N nodes for Part-OuterSkin-1. Returns {sensor_id: node_label}."""
    sensors = {}
    with open(inp_path) as f:
        content = f.read()
    for i in range(10):
        pattern = r'\*Nset,\s*nset=Set-Sensor-%d[^,]*,\s*instance=Part-OuterSkin-1\s*\n\s*(\d+)' % i
        m = re.search(pattern, content, re.I)
        if m:
            sensors[i] = int(m.group(1).strip())
    return sensors


def parse_defect_surface_elements(inp_path):
    """Parse defect surface element IDs from INP."""
    elem_ids = set()
    in_defect = False
    with open(inp_path) as f:
        for line in f:
            if 'Surf-Core-Outer-Defect' in line and 'Elset' in line:
                in_defect = True
                continue
            if in_defect:
                if line.strip().startswith('*'):
                    in_defect = False
                    continue
                for part in line.replace(',', ' ').split():
                    try:
                        elem_ids.add(int(part))
                    except ValueError:
                        pass
    return elem_ids


def get_defect_zone_circle(defect_params, n_pts=32):
    """Return (x,y,z) circle points for defect zone on cylindrical surface."""
    z_c = defect_params['z_center']
    theta_c = math.radians(defect_params['theta_deg'])
    r_def = defect_params['radius']
    pts = []
    for i in range(n_pts + 1):
        t = theta_c + 2 * math.pi * i / n_pts
        x = R_OUTER * math.cos(t)
        z = R_OUTER * math.sin(t)
        pts.append((x, z_c, z))
    return np.array(pts)


def main():
    parser = argparse.ArgumentParser(description='GW Fairing INP 3D visualization')
    parser.add_argument('inp', type=str, help='INP file path')
    parser.add_argument('--defect', type=str, default=None,
                        help='Defect JSON for defect zone circle')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG path')
    parser.add_argument('--sample', type=int, default=3000,
                        help='Max nodes to plot (for performance)')
    args = parser.parse_args()

    inp_path = args.inp
    if not os.path.exists(inp_path):
        print("ERROR: INP not found: %s" % inp_path)
        sys.exit(1)

    print("Parsing: %s" % inp_path)
    nodes = parse_inp_nodes(inp_path)
    sensors = parse_sensor_nodes(inp_path)
    print("  Nodes (OuterSkin): %d" % len(nodes))
    print("  Sensors: %d" % len(sensors))

    # Build point cloud
    pts = np.array(list(nodes.values()))
    if len(pts) == 0:
        print("ERROR: No nodes found")
        sys.exit(1)

    # Subsample for visualization (reproducible)
    n_plot = min(args.sample, len(pts))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pts), n_plot, replace=False) if len(pts) > n_plot else np.arange(len(pts))
    pts_plot = pts[idx]

    # Sensor positions
    sensor_pts = []
    sensor_labels = []
    for sid, nid in sorted(sensors.items()):
        if nid in nodes:
            sensor_pts.append(nodes[nid])
            sensor_labels.append('S%d' % sid)
    sensor_pts = np.array(sensor_pts) if sensor_pts else np.zeros((0, 3))

    # Defect zone
    defect_circle = None
    defect_params = None
    if args.defect:
        try:
            defect_params = json.loads(args.defect)
            defect_circle = get_defect_zone_circle(defect_params)
        except (json.JSONDecodeError, KeyError) as e:
            print("WARNING: Invalid defect JSON: %s" % e)

    # Plot
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')

    # Outer skin points (scatter) — 密に表示
    ax.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2],
               c='#87CEEB', s=1.2, alpha=0.6, label='OuterSkin mesh')

    # Sensors
    if len(sensor_pts) > 0:
        ax.scatter(sensor_pts[:, 0], sensor_pts[:, 1], sensor_pts[:, 2],
                   c='red', s=80, marker='o', edgecolors='black', linewidths=1.5,
                   label='Sensors', zorder=10)
        for i, lbl in enumerate(sensor_labels):
            ax.text(sensor_pts[i, 0], sensor_pts[i, 1], sensor_pts[i, 2], lbl,
                    fontsize=8, fontweight='bold')

    # Defect zone circle
    if defect_circle is not None:
        ax.plot(defect_circle[:, 0], defect_circle[:, 1], defect_circle[:, 2],
                'g-', linewidth=3, alpha=0.9, label='Defect zone (r=%.0f mm)' % defect_params['radius'])

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y (axial) [mm]')
    ax.set_zlabel('Z [mm]')
    job_name = os.path.splitext(os.path.basename(inp_path))[0]
    title = '%s\n30° sector, R=%.0f mm' % (job_name, R_OUTER)
    if defect_params:
        title += '\nDefect: z=%.0f, θ=%.1f°, r=%.0f mm' % (
            defect_params['z_center'], defect_params['theta_deg'], defect_params['radius'])
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper left', fontsize=9)

    # Equal aspect
    max_range = max(pts_plot.max(axis=0) - pts_plot.min(axis=0)) / 2.0
    mid = (pts_plot.max(axis=0) + pts_plot.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # 視点調整（欠陥・センサがよく見える角度）
    ax.view_init(elev=15, azim=-60)
    out_path = args.output or inp_path.replace('.inp', '_3d.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print("Saved: %s" % out_path)
    plt.close()


if __name__ == '__main__':
    main()
