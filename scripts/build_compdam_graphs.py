#!/usr/bin/env python3
"""Build PyG graphs from CompDam parametric CSV data.

Converts all case CSVs to PyG Data objects with 34-dim features
aligned with the fairing dataset for combined pre-training.

Usage:
    python scripts/build_compdam_graphs.py \
        --data_dir abaqus_work/compdam_parametric \
        --output data/compdam_graphs.pt
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Layup angle mappings from DOE
LAYUP_ANGLES = {
    'QI_45':   [45, 0, -45, 90],
    'CP_0_90': [0, 90, 0, 90],
    'QI_std':  [0, 45, -45, 90],
    'QI_60':   [60, 0, -60, 90],
}


def compute_flat_plate_normals(coords, df_elem):
    """Compute surface normals from element connectivity."""
    n_nodes = len(coords)
    normals = np.zeros((n_nodes, 3), dtype=np.float32)

    node_cols = [c for c in df_elem.columns if c.startswith('n')]
    if len(node_cols) < 4:
        # Fallback: z-direction normals for flat plate
        normals[:, 2] = 1.0
        return normals

    elem_nodes = df_elem[node_cols].values.astype(int)

    for elem_row in elem_nodes:
        valid_nodes = [n for n in elem_row if 0 < n <= n_nodes]
        if len(valid_nodes) < 3:
            continue
        p0 = coords[valid_nodes[0] - 1]
        p1 = coords[valid_nodes[1] - 1]
        p2 = coords[valid_nodes[2] - 1]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal /= norm
        for nid in valid_nodes:
            normals[nid - 1] += normal

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    normals /= norms

    return normals


def build_case_graph(node_csv, elem_csv, layup_name, pressure, impact_r,
                     x_offset, y_offset, k_neighbors=12):
    """Convert a single CompDam case to PyG Data with 34-dim features."""
    df = pd.read_csv(node_csv)
    df_elem = pd.read_csv(elem_csv)

    n_nodes = len(df)
    coords = df[['x', 'y', 'z']].values.astype(np.float32)
    pos = torch.tensor(coords, dtype=torch.float32)

    # Normal
    normals = compute_flat_plate_normals(coords, df_elem)
    normal_tensor = torch.tensor(normals, dtype=torch.float32)

    # Curvature: 0 for flat plate
    curvature = torch.zeros(n_nodes, 4, dtype=torch.float32)

    # Displacement
    disp = torch.tensor(df[['U1', 'U2', 'U3']].values, dtype=torch.float32)
    u_mag = torch.tensor(df['Umag'].values, dtype=torch.float32).unsqueeze(1)

    # Temperature: reference (no thermal in impact)
    temp = torch.full((n_nodes, 1), 150.0, dtype=torch.float32)

    # Stress
    s11 = torch.tensor(df['S11'].values, dtype=torch.float32).unsqueeze(1)
    s22 = torch.tensor(df['S22'].values, dtype=torch.float32).unsqueeze(1)
    s12 = torch.tensor(df['S12'].values, dtype=torch.float32).unsqueeze(1)
    mises = torch.tensor(df['Mises'].values, dtype=torch.float32).unsqueeze(1)
    stress = torch.cat([s11, s22, s12, mises], dim=1)

    # Principal stress sum
    principal_sum = s11 + s22

    # Thermal mises: 0 (no thermal)
    thermal_mises = torch.zeros(n_nodes, 1, dtype=torch.float32)

    # Strain from stress (IM7-8552)
    E1, E2, nu12, G12 = 171420.0, 9080.0, 0.32, 5290.0
    le11 = (s11 / E1 - nu12 * s22 / E1).squeeze()
    le22 = (-nu12 * s11 / E1 + s22 / E2).squeeze()
    le12 = (s12 / (2 * G12)).squeeze()
    strain = torch.stack([le11, le22, le12], dim=1)

    # Fiber orientation: use dominant angle from layup
    layup_deg = LAYUP_ANGLES.get(layup_name, [0, 45, -45, 90])
    # Average fiber direction over all plies
    fx = np.mean([math.cos(math.radians(a)) for a in layup_deg])
    fy = np.mean([math.sin(math.radians(a)) for a in layup_deg])
    fiber = torch.zeros(n_nodes, 3, dtype=torch.float32)
    fiber[:, 0] = fx
    fiber[:, 1] = fy

    # Layup angles (standardize to 4 values)
    layup_rad = torch.tensor([math.radians(a) for a in layup_deg[:4]], dtype=torch.float32)
    layup = layup_rad.unsqueeze(0).expand(n_nodes, 4)

    # Circumferential angle: 0
    circum_angle = torch.zeros(n_nodes, 1, dtype=torch.float32)

    # Node type
    x_arr = coords[:, 0]
    y_arr = coords[:, 1]
    plate_size = 100.0
    tol = 2.0
    is_boundary = ((x_arr < tol) | (x_arr > plate_size - tol) |
                   (y_arr < tol) | (y_arr > plate_size - tol)).astype(np.float32)

    cx = plate_size / 2 + x_offset
    cy = plate_size / 2 + y_offset
    dist_from_center = np.sqrt((x_arr - cx)**2 + (y_arr - cy)**2)
    is_loaded = (dist_from_center < impact_r * 1.5).astype(np.float32)
    node_type = torch.tensor(np.stack([is_boundary, is_loaded], axis=1), dtype=torch.float32)

    # Assemble 34-dim features
    x_features = torch.cat([
        pos,             # 3
        normal_tensor,   # 3
        curvature,       # 4
        disp,            # 3
        u_mag,           # 1
        temp,            # 1
        stress,          # 4
        principal_sum,   # 1
        thermal_mises,   # 1
        strain,          # 3
        fiber,           # 3
        layup,           # 4
        circum_angle,    # 1
        node_type,       # 2
    ], dim=1)
    assert x_features.shape[1] == 34

    # Damage labels
    y = torch.tensor(df['damage_label'].values, dtype=torch.long)

    # Build k-NN graph
    tree = KDTree(coords)
    _, indices = tree.query(coords, k=k_neighbors + 1)
    src_list, dst_list = [], []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            src_list.append(i)
            dst_list.append(j)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Edge features
    row, col = edge_index
    dx = pos[col, 0] - pos[row, 0]
    dy = pos[col, 1] - pos[row, 1]
    dz = pos[col, 2] - pos[row, 2]
    dist = torch.sqrt(dx**2 + dy**2 + dz**2)
    curv_diff = torch.zeros_like(dist)
    edge_attr = torch.stack([dx, dy, dz, dist, curv_diff], dim=1)

    data = Data(x=x_features, edge_index=edge_index, edge_attr=edge_attr,
                y=y, pos=pos)
    return data


def main():
    parser = argparse.ArgumentParser(description="Build PyG graphs from CompDam CSVs")
    parser.add_argument('--data_dir', default='abaqus_work/compdam_parametric')
    parser.add_argument('--output', default='data/compdam_graphs.pt')
    parser.add_argument('--k_neighbors', type=int, default=12)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    doe_path = data_dir / 'doe_summary.json'

    if not doe_path.exists():
        print(f"ERROR: DOE summary not found: {doe_path}")
        sys.exit(1)

    with open(doe_path) as f:
        doe = json.load(f)

    print(f"{'=' * 60}")
    print(f"Building CompDam PyG Graphs")
    print(f"{'=' * 60}")
    print(f"  DOE cases:    {len(doe)}")
    print(f"  k_neighbors:  {args.k_neighbors}")

    graphs = []
    meta = []

    for case in doe:
        cid = case['case_id']
        node_csv = data_dir / f"compdam_case_{cid:03d}_nodes.csv"
        elem_csv = data_dir / f"compdam_case_{cid:03d}_elements.csv"

        if not node_csv.exists():
            print(f"  Case {cid:03d}: SKIP (CSV not found)")
            continue

        print(f"  Case {cid:03d}: {case['layup_name']}, P={case['pressure']}MPa...",
              end='', flush=True)

        try:
            data = build_case_graph(
                node_csv, elem_csv,
                case['layup_name'], case['pressure'],
                case['impact_r'], case['x_offset'], case['y_offset'],
                k_neighbors=args.k_neighbors,
            )
            n_damaged = (data.y > 0).sum().item()
            n_total = data.x.size(0)
            print(f" nodes={n_total}, damaged={n_damaged} "
                  f"({100*n_damaged/n_total:.1f}%)")
            graphs.append(data)
            meta.append(case)
        except Exception as e:
            print(f" ERROR: {e}")

    if not graphs:
        print("No graphs built!")
        sys.exit(1)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'graphs': graphs, 'meta': meta}, output_path)

    print(f"\n{'=' * 60}")
    print(f"Saved {len(graphs)} graphs to {output_path}")
    total_nodes = sum(g.x.size(0) for g in graphs)
    total_damaged = sum((g.y > 0).sum().item() for g in graphs)
    print(f"  Total nodes:   {total_nodes:,}")
    print(f"  Total damaged: {total_damaged:,} ({100*total_damaged/total_nodes:.2f}%)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
