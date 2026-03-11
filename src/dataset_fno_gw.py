# -*- coding: utf-8 -*-
"""
GW Sensor CSV → FNO/DeepONet Training Dataset

Loads healthy reference + defect CSVs, pairs with DOE defect parameters.
Supports residual learning (predict scattering signal = defect - healthy).

Usage:
  from dataset_fno_gw import GWOperatorDataset
  ds = GWOperatorDataset('abaqus_work/gw_fairing_dataset', 'doe_gw_fairing.json')
"""

import json
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def load_sensor_csv(csv_path):
    """Load sensor CSV → (times, waveforms, positions).

    Returns:
        times: (T,) array
        waveforms: (n_sensors, T) array
        positions: (n_sensors,) array — x_mm or arc_mm
    """
    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) < 3:
        return None, None, None

    header = lines[0].strip().split(',')
    pos_row = lines[1].strip().split(',')

    # Parse sensor columns
    sensor_cols = []
    for i, col in enumerate(header):
        if col.startswith('sensor_') and '_Ur' in col:
            sensor_cols.append(i)

    n_sensors = len(sensor_cols)
    positions = np.zeros(n_sensors, dtype=np.float32)
    for j, ci in enumerate(sensor_cols):
        try:
            positions[j] = float(pos_row[ci])
        except (ValueError, IndexError):
            pass

    # Parse data rows
    data_rows = []
    for line in lines[2:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        vals = line.split(',')
        if len(vals) >= len(header):
            try:
                data_rows.append([float(v) for v in vals])
            except ValueError:
                continue

    if not data_rows:
        return None, None, None

    arr = np.array(data_rows, dtype=np.float32)
    times = arr[:, 0]
    waveforms = np.zeros((n_sensors, len(times)), dtype=np.float32)
    for j, ci in enumerate(sensor_cols):
        waveforms[j] = arr[:, ci]

    return times, waveforms, positions


class GWOperatorDataset(Dataset):
    """Dataset for FNO/DeepONet GW surrogate learning.

    Each sample: (defect_params, healthy_waveform, defect_waveform)
    where:
        defect_params: (n_params,) normalized — z_center, theta_deg, radius[, defect_type]
        healthy_waveform: (n_sensors, T)
        defect_waveform: (n_sensors, T)

    For residual mode: target = defect - healthy (scattering signal).

    Supports multiple DOE formats:
      - v1 (fairing): Job-GW-Fair-XXXX, 3 params (z, theta, radius)
      - v3 (barrel):  Job-GW-v3-XXXX, 4+ params (z, theta, radius, defect_type)
      - junction:     Job-Junc-XXXX, 4+ params
    """

    # Defect type encoding (one-hot index)
    DEFECT_TYPES = [
        'debonding', 'fod', 'impact', 'delamination',
        'inner_debond', 'thermal_progression', 'acoustic_fatigue',
    ]

    def __init__(self, data_dir, doe_path, residual=True,
                 max_timesteps=None, downsample=1,
                 healthy_pattern=None, encode_defect_type=True):
        self.data_dir = data_dir
        self.residual = residual
        self.downsample = downsample
        self.encode_defect_type = encode_defect_type

        # Load DOE
        with open(doe_path) as f:
            doe = json.load(f)
        self.n_samples_doe = doe.get('n_samples', len(doe.get('samples', [])))
        self.bounds = doe.get('bounds', {})

        # Normalization bounds for defect params
        if 'z_center' in self.bounds:
            self.z_min, self.z_max = self.bounds['z_center']
        else:
            self.z_min, self.z_max = 500, 2500  # fallback
        if 'theta_deg' in self.bounds:
            self.theta_min, self.theta_max = self.bounds['theta_deg']
        else:
            self.theta_min, self.theta_max = 2, 28
        # radius: use tier extremes or flat range
        if 'radius_tiers' in self.bounds:
            r_min = min(t['min'] for t in self.bounds['radius_tiers'])
            r_max = max(t['max'] for t in self.bounds['radius_tiers'])
        elif 'radius' in self.bounds:
            r_min, r_max = self.bounds['radius']
        else:
            r_min, r_max = 20, 80
        self.r_min, self.r_max = r_min, r_max

        # Auto-detect healthy CSV pattern from DOE
        if healthy_pattern is None:
            healthy_pattern = self._detect_healthy_pattern(doe, data_dir)
        healthy_path = os.path.join(data_dir, healthy_pattern)
        if not os.path.exists(healthy_path):
            # Try augmented A000 variants
            for suffix in ['A000', 'A0', 'A00']:
                base = healthy_pattern.replace('_sensors.csv', '')
                alt = os.path.join(data_dir, f'{base}-{suffix}_sensors.csv')
                if os.path.exists(alt):
                    healthy_path = alt
                    break
        self.times, self.healthy_waveform, self.positions = \
            load_sensor_csv(healthy_path)
        if self.times is None:
            raise FileNotFoundError(f"Healthy CSV not found: {healthy_path}")

        self.n_sensors = self.healthy_waveform.shape[0]
        self.T_full = len(self.times)

        # Number of params: 3 (z, theta, r) + 7 (defect_type one-hot) if enabled
        self.n_params = 3 + (len(self.DEFECT_TYPES) if encode_defect_type else 0)

        # Downsample
        if downsample > 1:
            self.times = self.times[::downsample]
            self.healthy_waveform = self.healthy_waveform[:, ::downsample]
        if max_timesteps and len(self.times) > max_timesteps:
            self.times = self.times[:max_timesteps]
            self.healthy_waveform = self.healthy_waveform[:, :max_timesteps]

        self.T = len(self.times)
        self.dt = self.times[1] - self.times[0] if self.T > 1 else 1e-6

        # Collect defect samples (auto-detect job name from DOE)
        self.samples = []  # list of (defect_params_norm, csv_path)
        for sample in doe.get('samples', []):
            job = sample.get('job_name', None)
            if job is None:
                # Legacy format: index-based
                continue
            csv_path = os.path.join(data_dir, f"{job}_sensors.csv")
            if not os.path.exists(csv_path):
                continue
            p = sample.get('defect_params', sample.get('defect', {}))
            params_norm = self._normalize_params(
                p.get('z_center', 0), p.get('theta_deg', 0),
                p.get('radius', 0), p.get('defect_type', 'debonding'))
            self.samples.append((params_norm, csv_path))

        # Also add healthy as "no defect" samples (params = 0)
        healthy_glob = os.path.join(data_dir, '*Healthy*_sensors.csv')
        self.healthy_csvs = sorted(glob.glob(healthy_glob))

        # Normalization stats (computed from healthy reference)
        self.waveform_scale = np.max(np.abs(self.healthy_waveform)) + 1e-12

    def _detect_healthy_pattern(self, doe, data_dir):
        """Auto-detect healthy CSV filename from DOE or data_dir."""
        # Check healthy_samples in DOE
        for hs in doe.get('healthy_samples', []):
            job = hs.get('job_name', '')
            csv = os.path.join(data_dir, f"{job}_sensors.csv")
            if os.path.exists(csv):
                return f"{job}_sensors.csv"
        # Fallback: glob for any healthy CSV
        for pattern in ['*-H0_sensors.csv', '*-H0000_sensors.csv',
                        '*Healthy_sensors.csv', '*-H00_sensors.csv']:
            matches = glob.glob(os.path.join(data_dir, pattern))
            if matches:
                return os.path.basename(matches[0])
        return 'Job-GW-Fair-Healthy_sensors.csv'

    def _normalize_params(self, z, theta, radius, defect_type='debonding'):
        """Normalize defect params to [0, 1], optionally with defect type one-hot."""
        z_n = (z - self.z_min) / (self.z_max - self.z_min + 1e-8)
        t_n = (theta - self.theta_min) / (self.theta_max - self.theta_min + 1e-8)
        r_n = (radius - self.r_min) / (self.r_max - self.r_min + 1e-8)
        base = [z_n, t_n, r_n]
        if self.encode_defect_type:
            one_hot = [0.0] * len(self.DEFECT_TYPES)
            if defect_type in self.DEFECT_TYPES:
                one_hot[self.DEFECT_TYPES.index(defect_type)] = 1.0
            base.extend(one_hot)
        return np.array(base, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        params_norm, csv_path = self.samples[idx]

        _, defect_waveform, _ = load_sensor_csv(csv_path)
        if defect_waveform is None:
            # Return zeros as fallback
            defect_waveform = np.zeros_like(self.healthy_waveform)

        # Downsample and truncate
        if self.downsample > 1:
            defect_waveform = defect_waveform[:, ::self.downsample]
        defect_waveform = defect_waveform[:, :self.T]

        # Normalize
        healthy = self.healthy_waveform / self.waveform_scale
        defect = defect_waveform / self.waveform_scale

        if self.residual:
            target = defect - healthy  # scattering signal
        else:
            target = defect

        return {
            'params': torch.tensor(params_norm, dtype=torch.float32),
            'healthy': torch.tensor(healthy, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'defect_full': torch.tensor(defect, dtype=torch.float32),
        }

    def get_metadata(self):
        return {
            'n_sensors': self.n_sensors,
            'n_params': self.n_params,
            'T': self.T,
            'dt': float(self.dt),
            'n_defect_samples': len(self.samples),
            'n_healthy_augmented': len(self.healthy_csvs),
            'waveform_scale': float(self.waveform_scale),
            'residual': self.residual,
            'downsample': self.downsample,
            'positions': self.positions.tolist(),
        }


class GWDeepONetDataset(Dataset):
    """Dataset for DeepONet: branch input = defect params, trunk input = (sensor, time).

    Each sample: (branch_input, trunk_coords, target_values)
    where:
        branch_input: (n_params,) — normalized defect params
        trunk_coords: (N_query, 2) — (sensor_id_norm, time_norm)
        target_values: (N_query,) — waveform amplitude at query points
    """

    def __init__(self, data_dir, doe_path, residual=True,
                 n_query=256, downsample=1, max_timesteps=None):
        self.base = GWOperatorDataset(
            data_dir, doe_path, residual=residual,
            downsample=downsample, max_timesteps=max_timesteps)
        self.n_query = n_query

        # Pre-compute trunk coordinate grid (sensor_norm, time_norm)
        s_coords = np.linspace(0, 1, self.base.n_sensors)
        t_coords = np.linspace(0, 1, self.base.T)
        ss, tt = np.meshgrid(s_coords, t_coords, indexing='ij')
        self.all_coords = np.stack([ss.ravel(), tt.ravel()], axis=-1).astype(np.float32)
        # total query points = n_sensors * T

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        target_flat = sample['target'].numpy().ravel()  # (n_sensors * T,)

        # Random subsample of query points
        total = len(target_flat)
        if self.n_query < total:
            indices = np.random.choice(total, self.n_query, replace=False)
        else:
            indices = np.arange(total)

        coords = self.all_coords[indices]
        values = target_flat[indices]

        return {
            'params': sample['params'],
            'coords': torch.tensor(coords, dtype=torch.float32),
            'values': torch.tensor(values, dtype=torch.float32),
        }
