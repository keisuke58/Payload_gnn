# -*- coding: utf-8 -*-
"""
FEM Result -> PyTorch Geometric Graph Conversion Pipeline

Reads CSV exports from Abaqus (nodes.csv, elements.csv, metadata.csv)
and converts them into PyG Data objects for GNN training.

Supports:
- S4R (quad) and S3 (tri) shell elements
- DSPSS subtraction against healthy baseline
- z-score normalization
- Train / Val / Test / OOD split
"""

import os
import glob
import json
import enum
import argparse

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


class NodeType(enum.IntEnum):
    INTERNAL = 0
    BOUNDARY = 1
    LOADED = 2
    SIZE = 3


# =========================================================================
# Element -> Edge conversion
# =========================================================================
def quads_to_edges(quads):
    """Convert S4R quad elements (N x 4) to edge pairs."""
    edges = []
    for q in quads:
        n0, n1, n2, n3 = q
        edges.extend([(n0, n1), (n1, n2), (n2, n3), (n3, n0)])
    return edges


def tris_to_edges(tris):
    """Convert S3 triangular elements (N x 3) to edge pairs."""
    edges = []
    for t in tris:
        n0, n1, n2 = t
        edges.extend([(n0, n1), (n1, n2), (n2, n0)])
    return edges


def elements_to_edge_index(df_elems, node_id_map):
    """
    Convert element connectivity to PyG edge_index (2 x num_edges).
    Handles mixed S4R / S3 meshes.
    """
    all_edges = []

    for _, row in df_elems.iterrows():
        etype = row['elem_type']
        n1 = node_id_map[int(row['n1'])]
        n2 = node_id_map[int(row['n2'])]
        n3 = node_id_map[int(row['n3'])]

        if etype == 'S4R' and int(row['n4']) >= 0:
            n4 = node_id_map[int(row['n4'])]
            all_edges.extend([(n1, n2), (n2, n3), (n3, n4), (n4, n1)])
        else:
            all_edges.extend([(n1, n2), (n2, n3), (n3, n1)])

    if not all_edges:
        return torch.zeros((2, 0), dtype=torch.long)

    edges = torch.tensor(all_edges, dtype=torch.long).t()  # (2, E)

    # Make undirected: add reverse edges
    edge_index = torch.cat([edges, edges.flip(0)], dim=1)

    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)

    return edge_index


# =========================================================================
# Feature engineering
# =========================================================================
def compute_edge_attr(pos, edge_index):
    """Compute edge attributes: relative position + distance."""
    src, dst = edge_index[0], edge_index[1]
    rel_pos = pos[src] - pos[dst]  # (E, 3)
    dist = torch.norm(rel_pos, p=2, dim=1, keepdim=True)  # (E, 1)
    return torch.cat([rel_pos, dist], dim=-1).float()  # (E, 4)


def classify_nodes(pos, mesh_size, height):
    """Assign node types based on position."""
    z = pos[:, 2]
    node_type = torch.zeros(len(pos), dtype=torch.long)

    tol = mesh_size * 1.5
    boundary_mask = (z < tol) | (z > height - tol)
    loaded_mask = z > height - tol

    node_type[boundary_mask] = NodeType.BOUNDARY
    node_type[loaded_mask] = NodeType.LOADED
    return node_type


# =========================================================================
# Single sample processing
# =========================================================================
def process_single_sample(sample_dir, baseline_dspss=None, mesh_size=50.0,
                          height=5000.0):
    """
    Process a single sample directory containing nodes.csv, elements.csv, metadata.csv.
    Returns a PyG Data object.
    """
    nodes_path = os.path.join(sample_dir, 'nodes.csv')
    elems_path = os.path.join(sample_dir, 'elements.csv')
    meta_path = os.path.join(sample_dir, 'metadata.csv')

    if not os.path.exists(nodes_path) or not os.path.exists(elems_path):
        return None

    df_nodes = pd.read_csv(nodes_path)
    df_elems = pd.read_csv(elems_path)

    # Build node ID mapping (Abaqus labels -> contiguous 0-based indices)
    node_ids = df_nodes['node_id'].values
    node_id_map = {int(nid): idx for idx, nid in enumerate(node_ids)}

    # Positions
    pos = torch.tensor(df_nodes[['x', 'y', 'z']].values, dtype=torch.float)

    # Stress features
    s11 = torch.tensor(df_nodes['s11'].values, dtype=torch.float).unsqueeze(1)
    s22 = torch.tensor(df_nodes['s22'].values, dtype=torch.float).unsqueeze(1)
    s12 = torch.tensor(df_nodes['s12'].values, dtype=torch.float).unsqueeze(1)
    dspss = torch.tensor(df_nodes['dspss'].values, dtype=torch.float).unsqueeze(1)

    # DSPSS subtraction (healthy baseline)
    if baseline_dspss is not None:
        n = min(len(dspss), len(baseline_dspss))
        delta_dspss = dspss[:n] - baseline_dspss[:n]
        if n < len(dspss):
            delta_dspss = torch.cat([delta_dspss,
                                     dspss[n:]], dim=0)
    else:
        delta_dspss = dspss

    # Node types
    node_type = classify_nodes(pos, mesh_size, height)
    node_type_onehot = torch.nn.functional.one_hot(
        node_type, num_classes=int(NodeType.SIZE)).float()

    # Assemble node features: [x, y, z, s11, s22, s12, delta_dspss, node_type(3)]
    x = torch.cat([pos, s11, s22, s12, delta_dspss, node_type_onehot], dim=-1)

    # Defect labels (binary: 0=healthy, 1=defect)
    y = torch.tensor(df_nodes['defect_label'].values, dtype=torch.long)

    # Edges
    edge_index = elements_to_edge_index(df_elems, node_id_map)
    edge_attr = compute_edge_attr(pos, edge_index)

    # Metadata
    meta = {}
    if os.path.exists(meta_path):
        df_meta = pd.read_csv(meta_path)
        for _, row in df_meta.iterrows():
            meta[row['key']] = row['value']

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        pos=pos,
    )
    data.sample_dir = sample_dir
    data.defect_type = meta.get('defect_type', 'unknown')
    data.defect_radius = float(meta.get('defect_radius', 0))

    return data


# =========================================================================
# Dataset processing
# =========================================================================
def load_baseline_dspss(baseline_dir):
    """Load DSPSS from healthy baseline for subtraction."""
    nodes_path = os.path.join(baseline_dir, 'nodes.csv')
    if not os.path.exists(nodes_path):
        return None
    df = pd.read_csv(nodes_path)
    return torch.tensor(df['dspss'].values, dtype=torch.float).unsqueeze(1)


def normalize_dataset(data_list):
    """Apply z-score normalization to node features across the dataset."""
    all_x = torch.cat([d.x for d in data_list], dim=0)
    mean = all_x.mean(dim=0)
    std = all_x.std(dim=0)
    std[std < 1e-8] = 1.0  # avoid division by zero

    for d in data_list:
        d.x = (d.x - mean) / std

    stats = {'mean': mean, 'std': std}
    return data_list, stats


def split_dataset(data_list, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split dataset into train / val / test.
    Also extracts OOD samples (largest defect radius quartile).
    """
    rng = np.random.RandomState(seed)

    # Separate OOD: top 10% by defect radius
    radii = np.array([d.defect_radius for d in data_list])
    ood_threshold = np.percentile(radii, 90)
    ood_data = [d for d in data_list if d.defect_radius >= ood_threshold]
    in_dist = [d for d in data_list if d.defect_radius < ood_threshold]

    # Shuffle in-distribution
    indices = list(range(len(in_dist)))
    rng.shuffle(indices)

    n_train = int(len(in_dist) * train_ratio)
    n_val = int(len(in_dist) * val_ratio)

    train_data = [in_dist[i] for i in indices[:n_train]]
    val_data = [in_dist[i] for i in indices[n_train:n_train + n_val]]
    test_data = [in_dist[i] for i in indices[n_train + n_val:]]

    return train_data, val_data, test_data, ood_data


def process_dataset(raw_dir, output_dir, mesh_size=50.0, height=5000.0):
    """Process all samples and save as PyG dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Load baseline
    baseline_dir = os.path.join(raw_dir, 'healthy_baseline')
    baseline_dspss = load_baseline_dspss(baseline_dir)
    if baseline_dspss is not None:
        print("Loaded healthy baseline DSPSS (%d nodes)" % len(baseline_dspss))
    else:
        print("Warning: No healthy baseline found, skipping DSPSS subtraction")

    # Process all sample directories
    sample_dirs = sorted(glob.glob(os.path.join(raw_dir, 'sample_*')))
    print("Found %d sample directories" % len(sample_dirs))

    data_list = []
    for i, sdir in enumerate(sample_dirs):
        data = process_single_sample(sdir, baseline_dspss, mesh_size, height)
        if data is not None:
            data_list.append(data)
            if (i + 1) % 10 == 0:
                print("  Processed %d / %d" % (i + 1, len(sample_dirs)))

    if not data_list:
        print("Error: No samples processed.")
        return

    print("Total samples: %d" % len(data_list))

    # Normalize
    data_list, stats = normalize_dataset(data_list)
    torch.save(stats, os.path.join(output_dir, 'norm_stats.pt'))

    # Split
    train, val, test, ood = split_dataset(data_list)
    print("Split: train=%d, val=%d, test=%d, ood=%d" %
          (len(train), len(val), len(test), len(ood)))

    # Save
    torch.save(train, os.path.join(output_dir, 'train.pt'))
    torch.save(val, os.path.join(output_dir, 'val.pt'))
    torch.save(test, os.path.join(output_dir, 'test.pt'))
    torch.save(ood, os.path.join(output_dir, 'ood.pt'))

    # Save split info
    split_info = {
        'n_train': len(train), 'n_val': len(val),
        'n_test': len(test), 'n_ood': len(ood),
        'feature_dim': int(data_list[0].x.shape[1]),
        'edge_attr_dim': int(data_list[0].edge_attr.shape[1]),
    }
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    print("Dataset saved to %s" % output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess FEM data for GNN')
    parser.add_argument('--raw_dir', type=str, default='dataset_output',
                        help='Directory containing raw CSV samples')
    parser.add_argument('--output_dir', type=str, default='dataset/processed',
                        help='Output directory for processed PyG data')
    parser.add_argument('--mesh_size', type=float, default=50.0)
    parser.add_argument('--height', type=float, default=5000.0)
    args = parser.parse_args()

    process_dataset(args.raw_dir, args.output_dir, args.mesh_size, args.height)
