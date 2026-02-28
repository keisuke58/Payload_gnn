# -*- coding: utf-8 -*-
"""
Curvature-Aware Graph Construction from FEM Mesh

Converts FEM mesh (nodes + elements CSV) into a PyTorch Geometric graph
that preserves geometric information crucial for curved sandwich structures:

Node features:
  - Position (x, y, z)
  - Surface normal (nx, ny, nz)
  - Principal curvatures (kappa1, kappa2)
  - Mean curvature H = (k1+k2)/2
  - Gaussian curvature K = k1*k2
  - Stress components (s11, s22, s12, dspss)
  - Node type one-hot (internal / boundary / loaded)

Edge features:
  - Relative position vector (dx, dy, dz)
  - Euclidean distance
  - Geodesic distance (approximated along mesh edges)
  - Relative normal angle (captures curvature change across edge)

Usage:
  python src/build_graph.py --nodes_csv data/nodes.csv --elems_csv data/elements.csv
  python src/build_graph.py --sample_dir dataset_output/sample_0000 --output graph.pt
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import lil_matrix


# =========================================================================
# Mesh parsing
# =========================================================================
def parse_mesh(df_nodes, df_elems):
    """
    Parse node and element DataFrames into arrays.

    Returns:
        coords: (N, 3) node positions
        node_ids: (N,) original node IDs
        elements: list of tuples (element connectivity as 0-based indices)
        node_id_map: dict {original_id -> 0-based index}
    """
    node_ids = df_nodes['node_id'].values.astype(int)
    node_id_map = {int(nid): idx for idx, nid in enumerate(node_ids)}
    coords = df_nodes[['x', 'y', 'z']].values.astype(np.float64)

    elements = []
    has_elem_type = 'elem_type' in df_elems.columns
    for _, row in df_elems.iterrows():
        etype = row['elem_type'] if has_elem_type else 'S4R'
        n4_val = int(row.get('n4', -1)) if 'n4' in df_elems.columns else -1
        n1 = node_id_map[int(row['n1'])]
        n2 = node_id_map[int(row['n2'])]
        n3 = node_id_map[int(row['n3'])]
        if etype in ('S4R', 'S4RT') and n4_val > 0 and n4_val in node_id_map:
            n4 = node_id_map[n4_val]
            elements.append((n1, n2, n3, n4))
        else:
            elements.append((n1, n2, n3))

    return coords, node_ids, elements, node_id_map


# =========================================================================
# Surface normals
# =========================================================================
def compute_face_normals(coords, elements):
    """Compute per-face normals using cross product of edge vectors."""
    face_normals = []
    for elem in elements:
        p0 = coords[elem[0]]
        p1 = coords[elem[1]]
        p2 = coords[elem[2]]
        e1 = p1 - p0
        e2 = p2 - p0
        n = np.cross(e1, e2)
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            n /= norm
        face_normals.append(n)
    return np.array(face_normals)


def compute_node_normals(coords, elements):
    """
    Compute per-node normals by area-weighted averaging of adjacent face normals.
    """
    n_nodes = len(coords)
    normals = np.zeros((n_nodes, 3))
    weights = np.zeros(n_nodes)

    for elem in elements:
        p0 = coords[elem[0]]
        p1 = coords[elem[1]]
        p2 = coords[elem[2]]
        e1 = p1 - p0
        e2 = p2 - p0
        cross = np.cross(e1, e2)
        area = np.linalg.norm(cross) / 2.0

        if area > 1e-12:
            n = cross / (2.0 * area)
        else:
            n = np.zeros(3)

        for idx in elem:
            normals[idx] += n * area
            weights[idx] += area

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals /= norms

    return normals


# =========================================================================
# Curvature estimation (discrete differential geometry)
# =========================================================================
def build_adjacency(n_nodes, elements):
    """Build node adjacency list and one-ring neighborhoods."""
    adj = defaultdict(set)
    node_to_elems = defaultdict(list)

    for eidx, elem in enumerate(elements):
        node_to_elems[elem[0]].append(eidx)
        node_to_elems[elem[1]].append(eidx)
        node_to_elems[elem[2]].append(eidx)
        if len(elem) == 4:
            node_to_elems[elem[3]].append(eidx)

        # Edges within element
        ne = len(elem)
        for i in range(ne):
            for j in range(i + 1, ne):
                adj[elem[i]].add(elem[j])
                adj[elem[j]].add(elem[i])

    return adj, node_to_elems


def estimate_curvature(coords, normals, adj):
    """
    Estimate principal curvatures per node using the shape operator
    (discrete Weingarten map) on the one-ring neighborhood.

    For each node i with neighbors j:
      - Project (p_j - p_i) onto the tangent plane at i
      - Compute normal curvature kn = (n_j - n_i) . (p_j - p_i) / |p_j - p_i|^2
      - Fit a 2x2 shape matrix and extract eigenvalues (principal curvatures)

    Returns:
        kappa1, kappa2: (N,) principal curvatures (kappa1 >= kappa2)
        H: (N,) mean curvature
        K: (N,) Gaussian curvature
    """
    n_nodes = len(coords)
    kappa1 = np.zeros(n_nodes)
    kappa2 = np.zeros(n_nodes)

    for i in range(n_nodes):
        ni = normals[i]
        pi = coords[i]
        neighbors = list(adj[i])
        if len(neighbors) < 3:
            continue

        # Build local tangent frame
        # e1: projection of first neighbor direction onto tangent plane
        diff0 = coords[neighbors[0]] - pi
        e1 = diff0 - np.dot(diff0, ni) * ni
        e1_norm = np.linalg.norm(e1)
        if e1_norm < 1e-12:
            continue
        e1 /= e1_norm
        e2 = np.cross(ni, e1)
        e2_norm = np.linalg.norm(e2)
        if e2_norm < 1e-12:
            continue
        e2 /= e2_norm

        # Accumulate shape operator (least squares)
        # For each neighbor j:
        #   d = p_j - p_i
        #   dn = n_j - n_i
        #   In tangent coords: u = (d.e1, d.e2), kn = -dn.d / |d|^2
        #   Shape operator: S * u_hat = kn * u_hat (normal curvature in direction u)
        # Fit S = [[a, b], [b, c]] minimizing sum |S*u_hat - kn*u_hat|^2
        A = []
        b_vec = []
        for j in neighbors:
            d = coords[j] - pi
            d_len = np.linalg.norm(d)
            if d_len < 1e-12:
                continue
            dn = normals[j] - ni
            # Normal curvature
            kn = -np.dot(dn, d) / (d_len * d_len)
            # Tangent coordinates
            u1 = np.dot(d, e1) / d_len
            u2 = np.dot(d, e2) / d_len
            # S * [u1, u2]^T . [u1, u2] = kn
            # => a*u1^2 + 2*b*u1*u2 + c*u2^2 = kn
            A.append([u1 * u1, 2.0 * u1 * u2, u2 * u2])
            b_vec.append(kn)

        if len(A) < 3:
            continue

        A = np.array(A)
        b_vec = np.array(b_vec)

        # Least squares: [a, b, c]
        result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        a, b, c = result

        # Eigenvalues of [[a, b], [b, c]]
        trace = a + c
        det = a * c - b * b
        disc = trace * trace - 4.0 * det
        if disc < 0:
            disc = 0.0
        sqrt_disc = np.sqrt(disc)
        k1 = (trace + sqrt_disc) / 2.0
        k2 = (trace - sqrt_disc) / 2.0

        kappa1[i] = k1
        kappa2[i] = k2

    H = (kappa1 + kappa2) / 2.0
    K = kappa1 * kappa2

    return kappa1, kappa2, H, K


# =========================================================================
# Edge construction
# =========================================================================
def build_edges(coords, adj, normals):
    """
    Build edge_index and edge attributes.

    Edge attributes:
      - Relative position (dx, dy, dz): 3
      - Euclidean distance: 1
      - Relative normal angle (arccos of dot product): 1
    """
    edges_src, edges_dst = [], []
    edge_attrs = []

    for i, neighbors in adj.items():
        for j in neighbors:
            edges_src.append(i)
            edges_dst.append(j)

            d = coords[j] - coords[i]
            dist = np.linalg.norm(d)

            # Angle between normals
            cos_angle = np.clip(np.dot(normals[i], normals[j]), -1.0, 1.0)
            angle = np.arccos(cos_angle)

            edge_attrs.append(np.concatenate([d, [dist, angle]]))

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float)

    return edge_index, edge_attr


def compute_geodesic_distances(coords, adj, edge_index):
    """
    Approximate geodesic distances using shortest path on the mesh graph.
    Returns per-edge geodesic distance.
    """
    n_nodes = len(coords)
    graph = lil_matrix((n_nodes, n_nodes))

    for i, neighbors in adj.items():
        for j in neighbors:
            d = np.linalg.norm(coords[j] - coords[i])
            graph[i, j] = d

    graph = graph.tocsr()

    # For large meshes, only compute geodesic for actual edges (not all-pairs)
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    # Use Dijkstra from unique source nodes
    unique_src = np.unique(src)
    geo_map = {}

    # For efficiency, compute row-by-row only for needed sources
    for s in unique_src:
        dists = shortest_path(graph, method='D', directed=False, indices=s)
        for j in adj[s]:
            geo_map[(s, j)] = dists[j]

    geo_dists = []
    for s, d in zip(src, dst):
        geo_dists.append(geo_map.get((s, d), np.linalg.norm(coords[d] - coords[s])))

    return torch.tensor(geo_dists, dtype=torch.float).unsqueeze(1)


# =========================================================================
# Full graph construction
# =========================================================================
def build_curvature_graph(df_nodes, df_elems, compute_geodesic=True, verbose=True):
    """
    Build a PyG Data object with curvature-aware features.

    Node features (dim=34):
      [x, y, z, nx, ny, nz, k1, k2, H, K,
       ux, uy, uz, u_mag, temp,
       s11, s22, s12, smises, principal_stress_sum, thermal_smises,
       le11, le22, le12, fiber_circum_x, fiber_circum_y, fiber_circum_z,
       layup_0, layup_45, layup_minus45, layup_90, circum_angle,
       node_type_boundary, node_type_loaded]

    Edge features (dim=5 or 6):
      [dx, dy, dz, euclidean_dist, normal_angle, (geodesic_dist)]
    """
    def log(msg):
        if verbose:
            print(msg)

    coords, node_ids, elements, node_id_map = parse_mesh(df_nodes, df_elems)
    n_nodes = len(coords)

    log("  Nodes: %d | Elements: %d" % (n_nodes, len(elements)))

    # Surface normals
    normals = compute_node_normals(coords, elements)
    log("  Computed node normals")

    # Adjacency
    adj, node_to_elems = build_adjacency(n_nodes, elements)
    log("  Built adjacency (avg degree: %.1f)" %
          (sum(len(v) for v in adj.values()) / max(n_nodes, 1)))

    # Curvature
    kappa1, kappa2, H, K = estimate_curvature(coords, normals, adj)
    print("  Curvature — H: [%.4f, %.4f]  K: [%.6f, %.6f]" %
          (H.min(), H.max(), K.min(), K.max()))

    # Edges
    edge_index, edge_attr = build_edges(coords, adj, normals)
    print("  Edges: %d" % edge_index.shape[1])

    # Geodesic distances (optional, can be slow for large meshes)
    if compute_geodesic and n_nodes < 100000:
        geo_dists = compute_geodesic_distances(coords, adj, edge_index)
        edge_attr = torch.cat([edge_attr, geo_dists], dim=1)
        print("  Computed geodesic distances")
    else:
        print("  Skipping geodesic distances (mesh too large or disabled)")

    # Displacement features (ux, uy, uz) — key physical signal for defect detection
    disp_features = []
    for col in ['ux', 'uy', 'uz']:
        if col in df_nodes.columns:
            disp_features.append(
                torch.tensor(df_nodes[col].values, dtype=torch.float).unsqueeze(1))
    if disp_features:
        disp_tensor = torch.cat(disp_features, dim=1)
    else:
        disp_tensor = torch.zeros(n_nodes, 3)

    # u_mag = |u| — displacement magnitude (1 dim)
    if 'u_mag' in df_nodes.columns:
        u_mag_tensor = torch.tensor(df_nodes['u_mag'].values, dtype=torch.float).unsqueeze(1)
    else:
        u_mag_tensor = torch.norm(disp_tensor, dim=1, keepdim=True)

    # Temperature feature
    temp_tensor = None
    for col in ['temp', 'temperature']:
        if col in df_nodes.columns:
            temp_tensor = torch.tensor(
                df_nodes[col].values, dtype=torch.float).unsqueeze(1)
            break
    if temp_tensor is None:
        temp_tensor = torch.zeros(n_nodes, 1)

    # Stress features (s11, s22, s12, smises/dspss)
    stress_features = []
    for col in ['s11', 's22', 's12']:
        if col in df_nodes.columns:
            stress_features.append(
                torch.tensor(df_nodes[col].values, dtype=torch.float).unsqueeze(1))
    # Accept either 'smises' or 'dspss' for von Mises
    smises_col = None
    for col in ['smises', 'dspss']:
        if col in df_nodes.columns:
            smises_col = col
            break
    if smises_col:
        stress_features.append(
            torch.tensor(df_nodes[smises_col].values, dtype=torch.float).unsqueeze(1))
    if stress_features:
        stress_tensor = torch.cat(stress_features, dim=1)
        # Pad to 4 if fewer columns
        if stress_tensor.shape[1] < 4:
            pad = torch.zeros(n_nodes, 4 - stress_tensor.shape[1])
            stress_tensor = torch.cat([stress_tensor, pad], dim=1)
    else:
        stress_tensor = torch.zeros(n_nodes, 4)

    # principal_stress_sum = σ₁+σ₂ ≈ s11+s22 (2D plane stress trace)
    principal_stress_sum = (stress_tensor[:, 0] + stress_tensor[:, 1]).unsqueeze(1)

    # thermal_smises — thermal stress von Mises (1 dim)
    if 'thermal_smises' in df_nodes.columns:
        thermal_smises_tensor = torch.tensor(
            df_nodes['thermal_smises'].values, dtype=torch.float).unsqueeze(1)
    else:
        thermal_smises_tensor = torch.zeros(n_nodes, 1)

    # Strain (le11, le22, le12) — 3 dims
    strain_cols = ['le11', 'le22', 'le12']
    strain_tensor = torch.zeros(n_nodes, 3)
    for i, col in enumerate(strain_cols):
        if col in df_nodes.columns:
            strain_tensor[:, i] = torch.tensor(df_nodes[col].values, dtype=torch.float)

    # Fiber orientation — circumferential direction (3 dims) for CFRP anisotropy
    # Abaqus Revolve: Y=axial, XZ=radial. Circumferential = (-z/r, 0, x/r)
    x_arr = coords[:, 0]
    y_arr = coords[:, 1]
    z_arr = coords[:, 2]
    r_xy = np.sqrt(x_arr**2 + z_arr**2)
    r_safe = np.where(r_xy > 1.0, r_xy, 1.0)
    fiber_circum_x = -z_arr / r_safe
    fiber_circum_y = np.zeros_like(x_arr)
    fiber_circum_z = x_arr / r_safe
    fiber_tensor = torch.tensor(
        np.stack([fiber_circum_x, fiber_circum_y, fiber_circum_z], axis=1),
        dtype=torch.float)

    # Layup angles [45/0/-45/90]s — 4 unique ply angles (radians). Same for all nodes.
    LAYUP_ANGLES_DEG = [0.0, 45.0, -45.0, 90.0]
    layup_rad = np.array([np.radians(a) for a in LAYUP_ANGLES_DEG], dtype=np.float32)
    layup_tensor = torch.tensor(layup_rad, dtype=torch.float).unsqueeze(0).expand(n_nodes, 4)

    # Circumferential angle θ = atan2(x, -z) — angle of 0° ply direction in XZ plane (radians)
    circum_angle = np.arctan2(x_arr, np.where(np.abs(z_arr) < 1e-12, 1e-12, -z_arr))
    circum_tensor = torch.tensor(circum_angle, dtype=torch.float).unsqueeze(1)

    # Node type (from position)
    z_coords = coords[:, 2]
    mesh_size = 50.0  # default
    height = z_coords.max()
    tol = mesh_size * 1.5
    is_boundary = ((z_coords < tol) | (z_coords > height - tol)).astype(float)
    is_loaded = (z_coords > height - tol).astype(float)
    node_type = torch.tensor(
        np.stack([is_boundary, is_loaded], axis=1), dtype=torch.float)

    # Assemble node features
    pos = torch.tensor(coords, dtype=torch.float)
    normal_tensor = torch.tensor(normals, dtype=torch.float)
    curvature_tensor = torch.tensor(
        np.stack([kappa1, kappa2, H, K], axis=1), dtype=torch.float)

    feature_list = [
        pos,                # 3: x, y, z
        normal_tensor,      # 3: nx, ny, nz
        curvature_tensor,   # 4: k1, k2, H, K
        disp_tensor,        # 3: ux, uy, uz
        u_mag_tensor,       # 1: |u| displacement magnitude
        temp_tensor,        # 1: temp
        stress_tensor,      # 4: s11, s22, s12, smises
        principal_stress_sum,   # 1: σ₁+σ₂ (主応力和)
        thermal_smises_tensor,  # 1: thermal stress von Mises
        strain_tensor,      # 3: le11, le22, le12
        fiber_tensor,       # 3: fiber circumferential direction (CFRP anisotropy)
        layup_tensor,       # 4: layup angles [0, 45, -45, 90] deg in rad
        circum_tensor,      # 1: circumferential angle θ (0° ply direction)
        node_type,          # 2: boundary, loaded
    ]
    x = torch.cat(feature_list, dim=1)

    # Defect labels
    if 'defect_label' in df_nodes.columns:
        y = torch.tensor(df_nodes['defect_label'].values, dtype=torch.long)
    else:
        y = torch.zeros(n_nodes, dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        pos=pos,
    )

    # Store geometric features separately for visualization
    data.normals = normal_tensor
    data.kappa1 = torch.tensor(kappa1, dtype=torch.float)
    data.kappa2 = torch.tensor(kappa2, dtype=torch.float)
    data.mean_curvature = torch.tensor(H, dtype=torch.float)
    data.gaussian_curvature = torch.tensor(K, dtype=torch.float)

    log("  Node features: %d dims | Edge features: %d dims" %
          (x.shape[1], edge_attr.shape[1]))

    return data


# =========================================================================
# CLI
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Build curvature-aware graph from FEM mesh')
    parser.add_argument('--sample_dir', type=str, default=None,
                        help='Sample directory containing nodes.csv, elements.csv')
    parser.add_argument('--nodes_csv', type=str, default=None)
    parser.add_argument('--elems_csv', type=str, default=None)
    parser.add_argument('--output', type=str, default='graph.pt',
                        help='Output path for PyG Data object')
    parser.add_argument('--no_geodesic', action='store_true',
                        help='Skip geodesic distance computation')
    args = parser.parse_args()

    if args.sample_dir:
        nodes_csv = os.path.join(args.sample_dir, 'nodes.csv')
        elems_csv = os.path.join(args.sample_dir, 'elements.csv')
    elif args.nodes_csv and args.elems_csv:
        nodes_csv = args.nodes_csv
        elems_csv = args.elems_csv
    else:
        parser.error('Provide --sample_dir or both --nodes_csv and --elems_csv')

    print("Loading mesh...")
    df_nodes = pd.read_csv(nodes_csv)
    df_elems = pd.read_csv(elems_csv)

    print("Building curvature-aware graph...")
    data = build_curvature_graph(df_nodes, df_elems,
                                 compute_geodesic=not args.no_geodesic)

    torch.save(data, args.output)
    print("\nSaved graph to %s" % args.output)
    print("  Nodes: %d | Edges: %d" % (data.num_nodes, data.num_edges))
    print("  x shape: %s | edge_attr shape: %s" %
          (list(data.x.shape), list(data.edge_attr.shape)))


if __name__ == '__main__':
    main()
