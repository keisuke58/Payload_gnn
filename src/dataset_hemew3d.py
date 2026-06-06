# -*- coding: utf-8 -*-
"""
HEMEW3D Dataset Loader for FNO Pre-training

Loads the HEMEW3D 3D elastic wave simulation dataset (Lehmann et al. 2024)
for pre-training FNO before fine-tuning on fairing GW data.

HEMEW3D: 30,000 seismic wave simulations with heterogeneous geology.
  - Input: 3D material field (32³) + source params (position + orientation)
  - Output: Surface waveforms at 32×32 grid, 3 components, 320 timesteps

For our purpose (1D FNO pre-training):
  - Extract 1D waveform traces (single component, row of sensors)
  - Map geology/source params → simplified defect-like parameters
  - Pre-train spectral convolution layers on wave propagation patterns

Data source: doi:10.57745/LAI6YU (Recherche Data Gouv)
GitHub: https://github.com/lehmannfa/MIFNO

Usage:
  from dataset_hemew3d import HEMEW3DDataset
  ds = HEMEW3DDataset('data/hemew3d/', n_sensors=9, component='uZ')
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class HEMEW3DDataset(Dataset):
    """HEMEW3D dataset adapted for 1D FNO GW pre-training.

    Extracts 1D waveform rows from 32×32 surface grid to match
    our fairing sensor array format (n_sensors × T).

    Each sample:
      params: (n_params,) — normalized source/material parameters
      healthy: (n_sensors, T) — baseline waveform (homogeneous reference)
      target: (n_sensors, T) — residual (heterogeneous - homogeneous)
      defect_full: (n_sensors, T) — full heterogeneous waveform
    """

    def __init__(self, data_dir, n_sensors=9, component='uZ',
                 max_samples=None, downsample_t=1, max_timesteps=None,
                 n_params=3, sensor_row=16, residual=True):
        """
        Args:
            data_dir: path to HEMEW3D data directory
            n_sensors: number of sensors to extract per row (resampled from 32)
            component: velocity component ('uE', 'uN', 'uZ')
            max_samples: limit number of samples (None = all)
            downsample_t: temporal downsampling factor
            sensor_row: which row (0-31) of the 32×32 grid to use
            residual: if True, target = heterogeneous - homogeneous
        """
        self.n_sensors = n_sensors
        self.component = component
        self.residual = residual
        self.downsample_t = downsample_t
        self.n_params = n_params

        # Find HDF5 sample files
        h5_pattern = os.path.join(data_dir, 'sample_*.h5')
        self.h5_files = sorted(glob.glob(h5_pattern))

        # Also check for velocity field files (v2 format)
        if not self.h5_files:
            h5_pattern = os.path.join(data_dir, 'velocities', '*.h5')
            self.h5_files = sorted(glob.glob(h5_pattern))

        if max_samples:
            self.h5_files = self.h5_files[:max_samples]

        if not self.h5_files:
            raise FileNotFoundError(
                f"No HDF5 files found in {data_dir}. "
                "Download from: doi:10.57745/LAI6YU")

        # Sensor indices: resample 32 positions to n_sensors
        indices_32 = np.linspace(0, 31, n_sensors, dtype=int)
        self.sensor_indices = indices_32
        self.sensor_row = sensor_row

        # Load first sample to get dimensions
        sample0 = self._load_h5(self.h5_files[0])
        self.T_full = sample0['waveform'].shape[-1]
        self.T = self.T_full // downsample_t
        if max_timesteps and self.T > max_timesteps:
            self.T = max_timesteps

        # Compute homogeneous reference (average of first few samples)
        self._compute_reference(min(10, len(self.h5_files)))

        # Normalization
        self.waveform_scale = np.max(np.abs(self.reference)) + 1e-12

    def _load_h5(self, path):
        """Load a single HEMEW3D HDF5 file."""
        import h5py
        with h5py.File(path, 'r') as f:
            # Try preprocessed format (sample_i.h5)
            if self.component in f:
                waveform = f[self.component][:]  # (32, 32, T)
            elif 'velocity' in f:
                waveform = f['velocity'][self.component][:]
            else:
                # Try flat array format
                keys = list(f.keys())
                waveform = f[keys[0]][:]

            # Extract source parameters if available
            if 'source' in f:
                source = f['source'][:]  # position + orientation
            elif 'params' in f:
                source = f['params'][:]
            else:
                source = np.zeros(self.n_params, dtype=np.float32)

            # Extract material if available
            if 'material' in f:
                material = f['material'][:]  # (32, 32, 32)
            else:
                material = None

        # Extract sensor row: (32, T) → select n_sensors
        if waveform.ndim == 3:
            # (32, 32, T) → select one row
            row_wf = waveform[self.sensor_row, :, :]  # (32, T)
            sensor_wf = row_wf[self.sensor_indices, :]  # (n_sensors, T)
        elif waveform.ndim == 2:
            # Already (n_points, T)
            sensor_wf = waveform[self.sensor_indices[:min(self.n_sensors, len(waveform))], :]
        else:
            raise ValueError(f"Unexpected waveform shape: {waveform.shape}")

        # Normalize source params to [0, 1]
        if len(source) > self.n_params:
            source = source[:self.n_params]
        params = np.zeros(self.n_params, dtype=np.float32)
        params[:len(source)] = source.astype(np.float32)

        return {
            'waveform': sensor_wf.astype(np.float32),
            'params': params,
            'material': material,
        }

    def _compute_reference(self, n_ref):
        """Compute reference waveform (mean of first n_ref samples)."""
        wfs = []
        for i in range(min(n_ref, len(self.h5_files))):
            s = self._load_h5(self.h5_files[i])
            wfs.append(s['waveform'])
        self.reference = np.mean(wfs, axis=0)  # (n_sensors, T_full)

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        sample = self._load_h5(self.h5_files[idx])
        wf = sample['waveform']
        ref = self.reference.copy()

        # Downsample
        if self.downsample_t > 1:
            wf = wf[:, ::self.downsample_t]
            ref = ref[:, ::self.downsample_t]
        wf = wf[:, :self.T]
        ref = ref[:, :self.T]

        # Normalize
        wf_norm = wf / self.waveform_scale
        ref_norm = ref / self.waveform_scale

        if self.residual:
            target = wf_norm - ref_norm
        else:
            target = wf_norm

        # Normalize params to [0, 1]
        params = sample['params']
        p_min = params.min()
        p_max = params.max()
        if p_max - p_min > 1e-8:
            params = (params - p_min) / (p_max - p_min)

        return {
            'params': torch.tensor(params, dtype=torch.float32),
            'healthy': torch.tensor(ref_norm, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'defect_full': torch.tensor(wf_norm, dtype=torch.float32),
        }

    def get_metadata(self):
        return {
            'n_sensors': self.n_sensors,
            'n_params': self.n_params,
            'T': self.T,
            'T_full': self.T_full,
            'n_samples': len(self.h5_files),
            'component': self.component,
            'waveform_scale': float(self.waveform_scale),
            'residual': self.residual,
            'dataset': 'HEMEW3D',
        }
