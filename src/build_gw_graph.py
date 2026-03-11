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


def _safe_fft_features(sig, dt):
    """Spectral features. Returns (dom_freq, centroid, bandwidth, rolloff) all norm by Nyquist."""
    if len(sig) < 4 or dt <= 0:
        return 0.0, 0.0, 0.0, 0.0
    n = len(sig)
    fft = np.fft.rfft(sig)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(n, dt)
    total = np.sum(mag)
    if total < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    dominant_idx = np.argmax(mag[1:]) + 1
    dominant_freq = freqs[dominant_idx]
    centroid = np.sum(freqs * mag) / total
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / (total + 1e-12))
    # Rolloff: freq below which 85% of energy
    cum = np.cumsum(mag)
    thresh = 0.85 * cum[-1]
    rolloff_idx = np.searchsorted(cum, thresh)
    rolloff_idx = min(rolloff_idx, len(freqs) - 1)
    rolloff = freqs[rolloff_idx]
    nyq = 0.5 / dt if dt > 0 else 1.0
    return (float(dominant_freq / nyq) if nyq > 0 else 0.0,
            float(centroid / nyq) if nyq > 0 else 0.0,
            float(bandwidth / nyq) if nyq > 0 else 0.0,
            float(rolloff / nyq) if nyq > 0 else 0.0)


def _hilbert_envelope(sig):
    """Hilbert envelope. Returns (envelope_max, decay_rate)."""
    if len(sig) < 4:
        return 0.0, 0.0
    from scipy.signal import hilbert
    try:
        analytic = hilbert(sig)
        envelope = np.abs(analytic)
        env_max = float(np.max(envelope))
        # Decay rate: fit log(envelope) after peak
        peak_idx = np.argmax(envelope)
        tail = envelope[peak_idx:]
        if len(tail) > 10 and env_max > 1e-20:
            log_env = np.log(np.clip(tail / env_max, 1e-12, None))
            x = np.arange(len(tail), dtype=float)
            # Simple linear fit: log_env ~ slope * x
            slope = np.polyfit(x, log_env, 1)[0]
            decay_rate = float(-slope)  # positive = decaying
        else:
            decay_rate = 0.0
        return env_max, decay_rate
    except Exception:
        return 0.0, 0.0


def _tof_threshold(sig, times, threshold_frac=0.1):
    """Time of flight: first time abs(sig) exceeds threshold_frac * max."""
    if len(sig) < 4:
        return 0.0
    max_abs = np.max(np.abs(sig))
    if max_abs < 1e-20:
        return 0.0
    thresh = threshold_frac * max_abs
    idx = np.argmax(np.abs(sig) > thresh)
    return float(times[idx]) if idx < len(times) else 0.0


def _velocity_features(sig, dt):
    """Velocity (dU/dt) peak and RMS."""
    if len(sig) < 4 or dt <= 0:
        return 0.0, 0.0
    vel = np.diff(sig) / dt
    vel_peak = float(np.max(np.abs(vel)))
    vel_rms = float(np.sqrt(np.mean(vel ** 2)))
    return vel_peak, vel_rms


def _ricker_wavelet(points, a):
    """Mexican hat (Ricker) wavelet."""
    A = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    t = np.arange(points) - (points - 1) / 2.0
    mod = 1.0 - (t ** 2) / wsq
    gauss = np.exp(-(t ** 2) / (2.0 * wsq))
    return A * mod * gauss


def _wavelet_features(sig, n_scales=4):
    """Multi-scale energy using Ricker wavelet convolution."""
    if len(sig) < 16:
        return [0.0] * n_scales
    try:
        widths = np.geomspace(2, len(sig) // 4, n_scales).astype(int)
        widths = np.clip(widths, 2, len(sig) // 2)
        energies = []
        for w in widths:
            wavelet = _ricker_wavelet(min(10 * w, len(sig)), w)
            conv = np.convolve(sig, wavelet, mode='same')
            energies.append(float(np.sum(conv ** 2)))
        total = sum(energies) + 1e-30
        return [e / total for e in energies]
    except Exception:
        return [0.0] * n_scales


def _safe_skew_kurt(sig):
    """Skewness and excess kurtosis. Returns (0,0) if std too small."""
    if len(sig) < 4:
        return 0.0, 0.0
    std = np.std(sig)
    if std < 1e-12:
        return 0.0, 0.0
    mean = np.mean(sig)
    z = (sig - mean) / std
    skew = np.mean(z ** 3)
    kurt = np.mean(z ** 4) - 3.0  # excess
    return float(np.clip(skew, -10, 10)), float(np.clip(kurt, -10, 10))


def extract_time_features(times, sensor_data, feature_set='baseline'):
    """Extract per-sensor features from time series.

    Args:
        times: (n_steps,) array
        sensor_data: dict {sensor_id: (n_steps,) array}
        feature_set: 'baseline' (3), 'extended' (10), 'full' (15),
                     or 'comprehensive' (24)

    Returns (n_sensors, n_features) array.

    Baseline (3):  max_abs, rms, peak_time_norm
    Extended (10): + mean, std, peak_to_peak, energy, zcr, dom_freq, spec_centroid
    Full (15):     + envelope_max, skewness, kurtosis, spec_bandwidth, spec_rolloff
    Comprehensive (24): + tof, vel_peak, vel_rms, crest_factor,
                          envelope_decay, wavelet_e(x4)
    """
    FEAT_COUNTS = {'baseline': 3, 'extended': 10, 'full': 15, 'comprehensive': 24}
    n_feat = FEAT_COUNTS.get(feature_set, 3)

    sensor_ids = sorted(sensor_data.keys())
    features = []
    t_max = times[-1] if len(times) > 0 else 1.0
    dt = (times[1] - times[0]) if len(times) > 1 else 1e-6

    for sid in sensor_ids:
        sig = sensor_data[sid]
        if len(sig) == 0:
            features.append([0.0] * n_feat)
            continue

        # Baseline (3)
        max_abs = np.max(np.abs(sig))
        rms = np.sqrt(np.mean(sig ** 2))
        peak_idx = np.argmax(np.abs(sig))
        peak_time = times[peak_idx] if peak_idx < len(times) else 0.0
        peak_time_norm = peak_time / t_max if t_max > 0 else 0.0
        row = [max_abs, rms, peak_time_norm]

        if feature_set in ('extended', 'full', 'comprehensive'):
            mean = np.mean(sig)
            std = np.std(sig)
            std = std if std > 1e-12 else 1e-12
            peak_to_peak = np.ptp(sig)
            energy_raw = np.sum(sig ** 2) * dt if dt > 0 else np.sum(sig ** 2)
            energy = np.log1p(energy_raw)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(sig)))) / 2
            zcr = zero_crossings / (len(sig) - 1) if len(sig) > 1 else 0.0
            dom_freq, spec_cent, spec_bw, spec_rolloff = _safe_fft_features(sig, dt)
            row.extend([mean, std, peak_to_peak, energy, zcr, dom_freq, spec_cent])

        if feature_set in ('full', 'comprehensive'):
            envelope_max, envelope_decay = _hilbert_envelope(sig)
            skew, kurt = _safe_skew_kurt(sig)
            row.extend([envelope_max, skew, kurt, spec_bw, spec_rolloff])

        if feature_set == 'comprehensive':
            # ToF (normalized)
            tof = _tof_threshold(sig, times)
            tof_norm = tof / t_max if t_max > 0 else 0.0
            # Velocity features
            vel_peak, vel_rms = _velocity_features(sig, dt)
            # Crest factor
            crest = max_abs / rms if rms > 1e-20 else 0.0
            # Wavelet energies (4 scales)
            wav = _wavelet_features(sig, n_scales=4)
            row.extend([tof_norm, vel_peak, vel_rms, crest, envelope_decay] + wav)

        features.append(row)

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


def build_gw_graph(csv_path, label, positions=None, connectivity='full', feature_set='baseline'):
    """Build PyG Data from sensor CSV.

    Args:
        csv_path: path to _sensors.csv
        label: 0=healthy, 1=defect
        positions: optional override (arc_mm or x_mm)
        connectivity: 'full' or 'knn'
        feature_set: 'baseline' (3), 'extended' (10), or 'full' (15)

    Returns:
        Data(x, edge_index, y, pos)
    """
    times, sensor_data, pos_from_csv = load_sensor_csv(csv_path)
    if not sensor_data:
        return None

    pos_use = positions if positions is not None else pos_from_csv
    features = extract_time_features(times, sensor_data, feature_set=feature_set)
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
    parser.add_argument('--feature_set', type=str, default='baseline',
                        choices=['baseline', 'extended', 'full', 'comprehensive'])
    args = parser.parse_args()

    if not args.csv or not os.path.exists(args.csv):
        print("ERROR: CSV not found: %s" % args.csv)
        return 1

    data = build_gw_graph(args.csv, args.label, connectivity=args.connectivity,
                          feature_set=args.feature_set)
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
