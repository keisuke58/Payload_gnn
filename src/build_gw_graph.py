# -*- coding: utf-8 -*-
"""
GW Sensor Graph Construction from CSV

Converts sensor time history CSV into PyTorch Geometric graph for GNN.
Nodes = sensors, edges = spatial adjacency, node features = time-series derived.

Usage:
  python src/build_gw_graph.py --csv abaqus_work/Job-GW-Fair-Healthy_sensors.csv --label 0
  python src/build_gw_graph.py --sample_dir abaqus_work/gw_fairing_dataset --output data/gw_test.pt
"""

import argparse
import csv
import os
import numpy as np
import torch
from torch_geometric.data import Data


def load_sensor_csv(csv_path):
    """Load sensor CSV, return (times, sensor_data, positions).

    Returns:
        times: (n_steps,) array
        sensor_data: dict {sensor_id: (n_steps,) array}
        positions: list of (arc_mm or x_mm) per sensor
    """
    times = []
    sensor_data = {}
    positions = []

    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        return None, {}, []

    header = rows[0]
    pos_row = rows[1]
    data_rows = [r for r in rows[2:] if r and not r[0].startswith('#')]

    # Parse header: time_s, sensor_0_Ur, sensor_1_Ur, ...
    col_to_sensor = {}  # col_index -> sensor_id
    for i, col in enumerate(header):
        if i == 0:
            continue
        if col.startswith('sensor_'):
            try:
                sid = int(col.replace('sensor_', '').split('_')[0])
                sensor_data[sid] = []
                col_to_sensor[i] = sid
            except ValueError:
                pass

    sensor_ids = sorted(sensor_data.keys())

    # Parse position row: col index -> position, then order by sensor_id
    pos_by_col = {}
    for i in range(1, min(len(pos_row), len(header))):
        try:
            pos_by_col[i] = float(pos_row[i])
        except (ValueError, TypeError):
            pos_by_col[i] = 0.0
    sensor_to_col = {s: c for c, s in col_to_sensor.items()}
    positions = [pos_by_col.get(sensor_to_col.get(sid, -1), 0.0) for sid in sensor_ids]

    # Parse data: column i -> sensor col_to_sensor[i]
    for row in data_rows:
        if len(row) < len(header):
            continue
        try:
            t = float(row[0])
            times.append(t)
            for col_idx, sid in col_to_sensor.items():
                if col_idx < len(row) and row[col_idx]:
                    sensor_data[sid].append(float(row[col_idx]))
                else:
                    sensor_data[sid].append(0.0)
        except (ValueError, IndexError):
            continue

    times = np.array(times) if times else np.array([])
    for sid in sensor_data:
        sensor_data[sid] = np.array(sensor_data[sid]) if sensor_data[sid] else np.array([])

    return times, sensor_data, positions[:len(sensor_ids)]


def extract_time_features(times, sensor_data):
    """Extract per-sensor features from time series.

    Returns (n_sensors, n_features) array.
    Features: max_abs, rms, peak_time_norm
    """
    sensor_ids = sorted(sensor_data.keys())
    features = []
    t_max = times[-1] if len(times) > 0 else 1.0

    for sid in sensor_ids:
        sig = sensor_data[sid]
        if len(sig) == 0:
            features.append([0.0, 0.0, 0.0])
            continue
        max_abs = np.max(np.abs(sig))
        rms = np.sqrt(np.mean(sig ** 2))
        peak_idx = np.argmax(np.abs(sig))
        peak_time = times[peak_idx] if peak_idx < len(times) else 0.0
        peak_time_norm = peak_time / t_max if t_max > 0 else 0.0
        features.append([max_abs, rms, peak_time_norm])

    return np.array(features, dtype=np.float32)


def build_edge_index(n_sensors, positions, connectivity='full'):
    """Build edge_index for sensor graph.

    connectivity: 'full' | 'knn' (k=2)
    """
    if connectivity == 'full':
        edges = []
        for i in range(n_sensors):
            for j in range(n_sensors):
                if i != j:
                    edges.append([i, j])
        if edges:
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
    # TODO: k-NN by position
    return torch.zeros((2, 0), dtype=torch.long)


def build_gw_graph(csv_path, label, positions=None, connectivity='full'):
    """Build PyG Data from sensor CSV.

    Args:
        csv_path: path to _sensors.csv
        label: 0=healthy, 1=defect
        positions: optional override (arc_mm or x_mm)
        connectivity: 'full' or 'knn'

    Returns:
        Data(x, edge_index, y, pos)
    """
    times, sensor_data, pos_from_csv = load_sensor_csv(csv_path)
    if not sensor_data:
        return None

    pos_use = positions if positions is not None else pos_from_csv
    features = extract_time_features(times, sensor_data)
    n_sensors = features.shape[0]

    # Normalize positions for pos attribute
    pos_arr = np.array(pos_use, dtype=np.float32).reshape(-1, 1)
    if pos_arr.shape[0] < n_sensors:
        pos_arr = np.pad(pos_arr, ((0, n_sensors - pos_arr.shape[0]), (0, 0)), constant_values=0)
    # Add dummy z for 2D (arc, z) if we only have arc
    if pos_arr.shape[1] == 1:
        pos_arr = np.hstack([pos_arr, np.zeros((pos_arr.shape[0], 1))])

    edge_index = build_edge_index(n_sensors, pos_use, connectivity)

    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        pos=torch.tensor(pos_arr[:n_sensors], dtype=torch.float),
    )
    return data


def main():
    parser = argparse.ArgumentParser(description='Build GW sensor graph from CSV')
    parser.add_argument('--csv', type=str, help='Path to _sensors.csv')
    parser.add_argument('--label', type=int, default=0, help='0=healthy, 1=defect')
    parser.add_argument('--output', type=str, default='gw_graph.pt')
    parser.add_argument('--connectivity', type=str, default='full',
                        choices=['full', 'knn'])
    args = parser.parse_args()

    if not args.csv or not os.path.exists(args.csv):
        print("ERROR: CSV not found: %s" % args.csv)
        return 1

    data = build_gw_graph(args.csv, args.label, connectivity=args.connectivity)
    if data is None:
        print("ERROR: Failed to build graph")
        return 1

    torch.save(data, args.output)
    print("Saved: %s" % args.output)
    print("  Nodes: %d | Edges: %d | x: %s | y: %d" % (
        data.num_nodes, data.num_edges, list(data.x.shape), data.y.item()))
    return 0


if __name__ == '__main__':
    exit(main())
