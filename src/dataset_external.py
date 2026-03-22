#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External Dataset Loaders for Cross-Dataset Validation

Loads public SHM datasets and converts to PyG graph format
compatible with our GNN pipeline (train.py).

Supported datasets:
  1. OGW #3 — CFRP stringer wavefield (intact vs impact damage)
  2. NASA CFRP Composites — Lamb wave fatigue (16 PZT sensors)
  3. OGW #4 — Omega stringer SHM (bonded vs disbonded)

Usage:
  python src/dataset_external.py --dataset ogw3 --output data/processed_ogw3
  python src/dataset_external.py --dataset nasa --output data/processed_nasa
  python src/dataset_external.py --dataset all --output data/processed_external
"""

import os
import argparse
import numpy as np
import torch
from torch_geometric.data import Data


# =============================================================================
# OGW #3: CFRP Stringer Wavefield
# =============================================================================

def load_ogw3(data_dir, freq='100kHz', max_points=5000):
    """Load OGW #3 wavefield data and build spatial graph.

    The wavefield is a 2D grid of out-of-plane velocity measurements.
    We subsample to max_points and build a kNN graph.

    Args:
        data_dir: path to data/external/ directory
        freq: excitation frequency (e.g., '100kHz')
        max_points: max spatial points to use

    Returns:
        list of PyG Data objects [intact, damaged]
    """
    import h5py

    intact_dir = os.path.join(data_dir,
        'OGW_CFRP_Stringer_Wavefield_Intact')
    damaged_dir = os.path.join(data_dir,
        'OGW_CFRP_Stringer_Wavefield_Damaged')
    # Fallback: SecondImpact is the damaged version
    if not os.path.isdir(damaged_dir):
        alt = os.path.join(data_dir, 'OGW_CFRP_Stringer_Wavefield_SecondImpact')
        if os.path.isdir(alt):
            damaged_dir = alt

    # Find frequency directory
    intact_freq = None
    if os.path.isdir(intact_dir):
        for d in os.listdir(intact_dir):
            if freq.replace('kHz', '') in d:
                intact_freq = os.path.join(intact_dir, d)
                break

    if intact_freq is None:
        # Try to extract from zip
        import zipfile
        intact_zip = os.path.join(data_dir,
            'OGW_CFRP_Stringer_Wavefield_Intact.zip')
        damaged_zip = os.path.join(data_dir,
            'OGW3_Wavefield_Damaged.zip')

        if os.path.exists(intact_zip) and not os.path.isdir(intact_dir):
            print("  Extracting OGW3 intact...")
            with zipfile.ZipFile(intact_zip, 'r') as zf:
                # Extract only the target frequency
                members = [m for m in zf.namelist()
                          if freq.replace('kHz', '') in m]
                zf.extractall(data_dir, members)
            # Re-find
            for d in os.listdir(intact_dir):
                if freq.replace('kHz', '') in d:
                    intact_freq = os.path.join(intact_dir, d)
                    break

        if os.path.exists(damaged_zip) and not os.path.isdir(damaged_dir):
            print("  Extracting OGW3 damaged...")
            with zipfile.ZipFile(damaged_zip, 'r') as zf:
                members = [m for m in zf.namelist()
                          if freq.replace('kHz', '') in m]
                # Determine the top-level directory name
                top = zf.namelist()[0].split('/')[0]
                zf.extractall(data_dir, members)
                # Rename if needed
                extracted = os.path.join(data_dir, top)
                if extracted != damaged_dir and os.path.isdir(extracted):
                    os.rename(extracted, damaged_dir)

    datasets = []
    for label, base_dir, name in [
        (0, intact_dir, 'Intact'),
        (1, damaged_dir, 'Damaged')
    ]:
        freq_dir = None
        if os.path.isdir(base_dir):
            for d in os.listdir(base_dir):
                if freq.replace('kHz', '') in d:
                    freq_dir = os.path.join(base_dir, d)
                    break

        if freq_dir is None:
            print("  WARNING: %s %s directory not found" % (name, freq))
            continue

        # Load H5 files
        coords_path = os.path.join(freq_dir, 'coordinates.h5')
        data_path = os.path.join(freq_dir, 'data_z.h5')
        time_path = os.path.join(freq_dir, 'time.h5')

        if not os.path.exists(coords_path):
            print("  WARNING: %s coordinates.h5 not found" % name)
            continue

        print("  Loading %s %s..." % (name, freq))
        with h5py.File(coords_path, 'r') as f:
            for key in f.keys():
                coords = f[key][:].T  # (2, N) → (N, 2)
                break

        with h5py.File(data_path, 'r') as f:
            for key in f.keys():
                wavefield = f[key][:].T  # (T, N) → (N, T)
                break

        with h5py.File(time_path, 'r') as f:
            for key in f.keys():
                time_vec = f[key][:].flatten()
                break

        n_points = coords.shape[0]
        n_timesteps = wavefield.shape[1] if len(wavefield.shape) > 1 else 1
        print("    Points: %d, Timesteps: %d" % (n_points, n_timesteps))

        # Subsample if too large
        if n_points > max_points:
            idx = np.random.RandomState(42).choice(n_points, max_points, replace=False)
            idx.sort()
            coords = coords[idx]
            wavefield = wavefield[idx] if len(wavefield.shape) > 1 else wavefield
            n_points = max_points
            print("    Subsampled to %d points" % n_points)

        # Build node features: position (2-3) + wavefield statistics
        if coords.shape[1] == 2:
            pos = np.column_stack([coords, np.zeros(n_points)])
        else:
            pos = coords[:, :3]

        # Wavefield features: max, min, rms, peak-to-peak, energy
        if len(wavefield.shape) > 1:
            w_max = wavefield.max(axis=1)
            w_min = wavefield.min(axis=1)
            w_rms = np.sqrt(np.mean(wavefield**2, axis=1))
            w_p2p = w_max - w_min
            w_energy = np.sum(wavefield**2, axis=1)
        else:
            w_max = w_min = w_rms = w_p2p = w_energy = wavefield

        features = np.column_stack([
            pos,                         # 0-2: position
            w_max, w_min, w_rms,        # 3-5: wavefield stats
            w_p2p, w_energy,            # 6-7: peak-to-peak, energy
        ]).astype(np.float32)

        x = torch.tensor(features, dtype=torch.float32)

        # Build kNN graph (without torch-cluster dependency)
        from scipy.spatial import cKDTree
        pos_tensor = torch.tensor(pos, dtype=torch.float32)
        tree = cKDTree(pos)
        _, indices = tree.query(pos, k=9)  # k+1 (includes self)
        src_list, dst_list = [], []
        for i in range(n_points):
            for j in indices[i, 1:]:  # skip self
                src_list.append(i)
                dst_list.append(j)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # Edge features
        src, dst = edge_index[0], edge_index[1]
        diff = pos_tensor[dst] - pos_tensor[src]
        dist = torch.norm(diff, dim=1, keepdim=True)
        edge_attr = torch.cat([diff, dist], dim=1)

        # Labels: all nodes same label (graph-level classification)
        y = torch.full((n_points,), label, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=y, pos=pos_tensor)
        data.dataset_name = 'OGW3'
        data.case_name = name
        data.graph_label = label
        datasets.append(data)

        print("    Graph: %d nodes, %d edges, label=%d (%s)" % (
            n_points, edge_index.shape[1], label, name))

    return datasets


# =============================================================================
# NASA CFRP Composites
# =============================================================================

def load_nasa_composites(data_dir, layup=1, max_samples=50):
    """Load NASA CFRP Composites Lamb wave data.

    Each sample is a fatigue state with 16 PZT sensor signals.
    We build a fully-connected sensor graph.

    Returns:
        list of PyG Data objects
    """
    import scipy.io as sio
    import zipfile

    base = os.path.join(data_dir, 'NASA_Composites', '2. Composites')
    zip_path = os.path.join(base, 'Layup%d.zip' % layup)

    if not os.path.exists(zip_path):
        print("  WARNING: Layup%d.zip not found" % layup)
        return []

    extract_dir = os.path.join(base, 'Layup%d' % layup)
    if not os.path.isdir(extract_dir):
        print("  Extracting Layup%d..." % layup)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(base)

    # Find sample directories
    sample_dirs = []
    for root, dirs, files in os.walk(base):
        if 'PZT-data' in dirs:
            sample_dirs.append(os.path.join(root, 'PZT-data'))
    sample_dirs.sort()

    if not sample_dirs:
        print("  WARNING: No PZT-data directories found")
        return []

    print("  Found %d sample directories" % len(sample_dirs))

    # PZT sensor positions (approximate grid layout, 4x4)
    # NASA documentation: 16 PZT on rectangular grid
    sensor_pos = np.array([
        [i * 50, j * 50, 0] for i in range(4) for j in range(4)
    ], dtype=np.float32)
    n_sensors = 16

    datasets = []
    for i, pzt_dir in enumerate(sample_dirs[:max_samples]):
        mat_files = sorted([f for f in os.listdir(pzt_dir)
                           if f.endswith('.mat')])
        if not mat_files:
            continue

        # Load first .mat to get signal shape
        try:
            mat = sio.loadmat(os.path.join(pzt_dir, mat_files[0]))
        except Exception:
            continue

        # Extract signal features per sensor
        features_list = []
        for mf in mat_files[:16]:  # up to 16 actuator-receiver pairs
            try:
                mat = sio.loadmat(os.path.join(pzt_dir, mf))
                # Find the signal array
                for key in mat:
                    if not key.startswith('_'):
                        sig = mat[key]
                        if hasattr(sig, 'shape') and len(sig.shape) >= 1:
                            sig = sig.flatten().astype(np.float32)
                            features_list.append([
                                sig.max(), sig.min(),
                                np.sqrt(np.mean(sig**2)),
                                sig.max() - sig.min(),
                                np.sum(sig**2),
                            ])
                            break
            except Exception:
                continue

        if len(features_list) < n_sensors:
            # Pad with zeros
            while len(features_list) < n_sensors:
                features_list.append([0, 0, 0, 0, 0])

        features = np.array(features_list[:n_sensors], dtype=np.float32)
        # Node features: position (3) + signal stats (5) = 8 dim
        x = torch.tensor(np.column_stack([sensor_pos, features]),
                         dtype=torch.float32)

        # Fully connected graph
        src_list, dst_list = [], []
        for s in range(n_sensors):
            for d in range(n_sensors):
                if s != d:
                    src_list.append(s)
                    dst_list.append(d)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # Edge features
        pos_tensor = torch.tensor(sensor_pos, dtype=torch.float32)
        diff = pos_tensor[edge_index[1]] - pos_tensor[edge_index[0]]
        dist = torch.norm(diff, dim=1, keepdim=True)
        edge_attr = torch.cat([diff, dist], dim=1)

        # Label from directory name (damage progression)
        dir_name = os.path.basename(os.path.dirname(pzt_dir))
        if '_F' in dir_name:
            label = 1  # fatigue damage
        else:
            label = 0  # healthy/baseline

        y = torch.full((n_sensors,), label, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=y, pos=pos_tensor)
        data.dataset_name = 'NASA_CFRP'
        data.case_name = dir_name
        data.graph_label = label
        datasets.append(data)

    print("  Loaded %d graphs from NASA Layup%d" % (len(datasets), layup))
    return datasets


# =============================================================================
# Main: Build and Save
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Load external SHM datasets for cross-validation')
    parser.add_argument('--dataset', default='all',
                        choices=['ogw3', 'nasa', 'all'],
                        help='Dataset to load')
    parser.add_argument('--data_dir', default='data/external',
                        help='Path to external data')
    parser.add_argument('--output', default='data/processed_external',
                        help='Output directory')
    parser.add_argument('--max_points', type=int, default=5000,
                        help='Max spatial points for wavefield (OGW3)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_graphs = []

    if args.dataset in ['ogw3', 'all']:
        print("\n=== OGW #3: CFRP Stringer Wavefield ===")
        ogw3 = load_ogw3(args.data_dir, freq='100kHz',
                          max_points=args.max_points)
        all_graphs.extend(ogw3)
        if ogw3:
            torch.save(ogw3, os.path.join(args.output, 'ogw3.pt'))
            print("  Saved: ogw3.pt (%d graphs)" % len(ogw3))

    if args.dataset in ['nasa', 'all']:
        print("\n=== NASA CFRP Composites ===")
        nasa = load_nasa_composites(args.data_dir, layup=1)
        all_graphs.extend(nasa)
        if nasa:
            torch.save(nasa, os.path.join(args.output, 'nasa_cfrp.pt'))
            print("  Saved: nasa_cfrp.pt (%d graphs)" % len(nasa))

    if all_graphs:
        # Summary
        print("\n" + "=" * 60)
        print("External dataset summary:")
        for g in all_graphs:
            print("  %s/%s: %d nodes, %d edges, label=%d" % (
                g.dataset_name, g.case_name,
                g.x.shape[0], g.edge_index.shape[1], g.graph_label))
        print("Total: %d graphs" % len(all_graphs))

        torch.save(all_graphs, os.path.join(args.output, 'all_external.pt'))
        print("Saved: all_external.pt")


if __name__ == '__main__':
    main()
