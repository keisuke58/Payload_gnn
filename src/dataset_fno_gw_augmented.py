# -*- coding: utf-8 -*-
"""
Augmented GW Dataset: Mixup + Noise Injection + Amplitude Scaling

Wraps GWOperatorDataset to generate additional training samples
without running more FEM simulations.

Physics justification:
  - Mixup: Linear superposition valid for linear elastodynamics (Born approx.)
  - Noise: Models sensor noise / environmental variation (SNR 20-40 dB)
  - Amplitude scaling: Linear system → excitation amplitude is scalable

Usage:
  from dataset_fno_gw_augmented import AugmentedGWDataset
  ds = AugmentedGWDataset('data_dir', 'doe.json', augment_factor=10)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from dataset_fno_gw import GWOperatorDataset, load_sensor_csv


class AugmentedGWDataset(Dataset):
    """Augmented GW dataset with Mixup, noise injection, and scaling.

    Given N original FEM samples, generates N * augment_factor samples:
      - Original samples (unchanged)
      - Mixup pairs (linear interpolation of waveforms + params)
      - Noise-injected variants
      - Amplitude-scaled variants
    """

    def __init__(self, data_dir, doe_path, residual=True,
                 downsample=1, max_timesteps=None,
                 augment_factor=10, noise_snr_db=30.0,
                 amplitude_range=(0.8, 1.2), mixup_alpha=0.4,
                 time_shift_max=0.05, seed=42, **kwargs):
        self.base = GWOperatorDataset(
            data_dir, doe_path, residual=residual,
            downsample=downsample, max_timesteps=max_timesteps, **kwargs)
        self.augment_factor = augment_factor
        self.noise_snr_db = noise_snr_db
        self.amplitude_range = amplitude_range
        self.mixup_alpha = mixup_alpha
        self.time_shift_max = time_shift_max  # max fraction of T to shift
        self.rng = np.random.RandomState(seed)

        n_base = len(self.base)
        if n_base == 0:
            raise ValueError("Base dataset is empty")

        # Pre-load all base samples (small dataset, OK to cache)
        self._cache = []
        for i in range(n_base):
            self._cache.append(self.base[i])

        # Pre-generate augmentation plan
        self._plan = self._build_augmentation_plan(n_base)

    def _build_augmentation_plan(self, n_base):
        """Build list of (aug_type, indices, aug_params) for all samples."""
        plan = []

        # Original samples first
        for i in range(n_base):
            plan.append(('original', i, None))

        n_aug = n_base * self.augment_factor - n_base
        if n_aug <= 0:
            return plan

        # Distribute augmentation types: 40% mixup, 25% noise, 15% scale, 20% time_shift
        n_mixup = int(n_aug * 0.40)
        n_noise = int(n_aug * 0.25)
        n_scale = int(n_aug * 0.15)
        n_tshift = n_aug - n_mixup - n_noise - n_scale

        # Mixup pairs
        for _ in range(n_mixup):
            i, j = self.rng.choice(n_base, 2, replace=False)
            alpha = self.rng.beta(self.mixup_alpha, self.mixup_alpha)
            plan.append(('mixup', (i, j), alpha))

        # Noise injection
        for _ in range(n_noise):
            i = self.rng.randint(n_base)
            snr_var = self.rng.uniform(-5, 5)  # vary SNR ±5 dB
            plan.append(('noise', i, self.noise_snr_db + snr_var))

        # Amplitude scaling
        for _ in range(n_scale):
            i = self.rng.randint(n_base)
            scale = self.rng.uniform(*self.amplitude_range)
            plan.append(('scale', i, scale))

        # Time shift (circular shift — models excitation timing variation)
        for _ in range(n_tshift):
            i = self.rng.randint(n_base)
            shift_frac = self.rng.uniform(-self.time_shift_max,
                                           self.time_shift_max)
            plan.append(('time_shift', i, shift_frac))

        return plan

    def __len__(self):
        return len(self._plan)

    def __getitem__(self, idx):
        aug_type, indices, aug_param = self._plan[idx]

        if aug_type == 'original':
            return self._cache[indices]

        elif aug_type == 'mixup':
            return self._apply_mixup(indices[0], indices[1], aug_param)

        elif aug_type == 'noise':
            return self._apply_noise(indices, aug_param)

        elif aug_type == 'scale':
            return self._apply_scale(indices, aug_param)

        elif aug_type == 'time_shift':
            return self._apply_time_shift(indices, aug_param)

    def _apply_mixup(self, i, j, alpha):
        """Linear interpolation of two samples (physics: Born approximation)."""
        s_i = self._cache[i]
        s_j = self._cache[j]

        params = alpha * s_i['params'] + (1 - alpha) * s_j['params']
        healthy = s_i['healthy']  # healthy is shared
        target = alpha * s_i['target'] + (1 - alpha) * s_j['target']
        defect_full = alpha * s_i['defect_full'] + (1 - alpha) * s_j['defect_full']

        return {
            'params': params,
            'healthy': healthy,
            'target': target,
            'defect_full': defect_full,
        }

    def _apply_noise(self, i, snr_db):
        """Add Gaussian noise at specified SNR."""
        s = self._cache[i]
        target = s['target'].clone()
        signal_power = torch.mean(target ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10) + 1e-12)
        noise = torch.randn_like(target) * torch.sqrt(noise_power + 1e-12)

        noisy_target = target + noise
        noisy_defect = s['defect_full'] + noise

        return {
            'params': s['params'],
            'healthy': s['healthy'],
            'target': noisy_target,
            'defect_full': noisy_defect,
        }

    def _apply_scale(self, i, scale):
        """Scale waveform amplitude (physics: linear elasticity)."""
        s = self._cache[i]
        return {
            'params': s['params'],
            'healthy': s['healthy'],
            'target': s['target'] * scale,
            'defect_full': s['healthy'] + s['target'] * scale,
        }

    def _apply_time_shift(self, i, shift_frac):
        """Circular time shift (physics: excitation timing variation).

        Shifts both target and healthy by the same amount to preserve
        the relative scattering pattern.
        """
        s = self._cache[i]
        T = s['target'].shape[-1]
        shift = int(shift_frac * T)
        if shift == 0:
            return s

        target = torch.roll(s['target'], shifts=shift, dims=-1)
        healthy = torch.roll(s['healthy'], shifts=shift, dims=-1)
        defect_full = torch.roll(s['defect_full'], shifts=shift, dims=-1)

        # Zero-pad the wrapped region (not circular — causal signal)
        if shift > 0:
            target[..., :shift] = 0.0
            healthy[..., :shift] = 0.0
            defect_full[..., :shift] = 0.0
        elif shift < 0:
            target[..., shift:] = 0.0
            healthy[..., shift:] = 0.0
            defect_full[..., shift:] = 0.0

        return {
            'params': s['params'],
            'healthy': healthy,
            'target': target,
            'defect_full': defect_full,
        }

    def get_metadata(self):
        meta = self.base.get_metadata()
        meta['augmented'] = True
        meta['augment_factor'] = self.augment_factor
        meta['n_total_samples'] = len(self)
        meta['n_original'] = len(self.base)
        meta['noise_snr_db'] = self.noise_snr_db
        meta['mixup_alpha'] = self.mixup_alpha
        return meta

    # Delegate attributes to base
    @property
    def n_sensors(self):
        return self.base.n_sensors

    @property
    def n_params(self):
        return self.base.n_params

    @property
    def T(self):
        return self.base.T

    @property
    def waveform_scale(self):
        return self.base.waveform_scale

    @property
    def DEFECT_TYPES(self):
        return self.base.DEFECT_TYPES

    @property
    def encode_defect_type(self):
        return self.base.encode_defect_type


class MultiDefectLocalDataset(Dataset):
    """Dataset for multi-defect FEM: extracts local sensor patches per defect.

    Given a FEM model with N defects and a grid of sensors:
    - For each defect, find k nearest sensors
    - Extract local waveforms at those sensors
    - Subtract healthy reference → local scattering signal
    - Each (defect_params, local_waveforms) becomes one training sample

    This allows 1 FEM run → N training samples.
    """

    # Same 7-type encoding as GWOperatorDataset
    DEFECT_TYPES = [
        'debonding', 'fod', 'impact', 'delamination',
        'inner_debond', 'thermal_progression', 'acoustic_fatigue',
    ]

    def __init__(self, multi_defect_csv, healthy_csv, defect_list,
                 sensor_positions_2d, k_nearest=9, residual=True,
                 downsample=1, max_timesteps=None,
                 z_bounds=(500, 2500), theta_bounds=(2, 28),
                 r_bounds=(20, 80), encode_defect_type=True):
        """
        Args:
            multi_defect_csv: path to CSV with all sensors from multi-defect FEM
            healthy_csv: path to healthy reference CSV (same sensor layout)
            defect_list: list of dicts with keys:
                z_center, theta_deg, radius, defect_type
            sensor_positions_2d: (n_sensors, 2) array of (z_mm, arc_mm)
            k_nearest: number of nearest sensors per defect
            encode_defect_type: if True, append 7-dim one-hot for defect type
        """
        self.residual = residual
        self.k_nearest = k_nearest
        self.encode_defect_type = encode_defect_type

        # Load waveforms
        times_d, wf_defect, _ = load_sensor_csv(multi_defect_csv)
        times_h, wf_healthy, _ = load_sensor_csv(healthy_csv)

        if wf_defect is None or wf_healthy is None:
            raise FileNotFoundError("CSV load failed")

        # Downsample
        if downsample > 1:
            times_d = times_d[::downsample]
            wf_defect = wf_defect[:, ::downsample]
            wf_healthy = wf_healthy[:, ::downsample]
        if max_timesteps and len(times_d) > max_timesteps:
            times_d = times_d[:max_timesteps]
            wf_defect = wf_defect[:, :max_timesteps]
            wf_healthy = wf_healthy[:, :max_timesteps]

        self.times = times_d
        self.T = len(times_d)
        self.n_sensors_total = wf_defect.shape[0]

        # Normalization
        self.waveform_scale = np.max(np.abs(wf_healthy)) + 1e-12
        self.wf_defect = wf_defect / self.waveform_scale
        self.wf_healthy = wf_healthy / self.waveform_scale

        # Scattering field
        self.scattering = self.wf_defect - self.wf_healthy

        # Sensor positions
        self.sensor_pos = np.array(sensor_positions_2d, dtype=np.float32)

        # Normalization bounds
        self.z_min, self.z_max = z_bounds
        self.theta_min, self.theta_max = theta_bounds
        self.r_min, self.r_max = r_bounds

        # Build samples: for each defect, find k nearest sensors
        self.samples = []
        for defect in defect_list:
            z = defect['z_center']
            theta = defect['theta_deg']
            r = defect['radius']
            dtype = defect.get('defect_type', 'debonding')

            # Defect position in (z_mm, arc_mm) space
            R_outer = 2638.0  # mm (outer skin radius)
            defect_arc = R_outer * np.radians(theta)
            defect_pos = np.array([z, defect_arc])

            # Find k nearest sensors
            dists = np.linalg.norm(self.sensor_pos - defect_pos, axis=1)
            nearest_idx = np.argsort(dists)[:k_nearest]
            nearest_idx = np.sort(nearest_idx)  # keep spatial order

            # Normalize params
            z_n = (z - self.z_min) / (self.z_max - self.z_min + 1e-8)
            t_n = (theta - self.theta_min) / (self.theta_max - self.theta_min + 1e-8)
            r_n = (r - self.r_min) / (self.r_max - self.r_min + 1e-8)
            params_list = [z_n, t_n, r_n]

            # Defect type one-hot encoding (7 types)
            if self.encode_defect_type:
                one_hot = [0.0] * len(self.DEFECT_TYPES)
                if dtype in self.DEFECT_TYPES:
                    one_hot[self.DEFECT_TYPES.index(dtype)] = 1.0
                params_list.extend(one_hot)

            params = np.array(params_list, dtype=np.float32)

            self.samples.append({
                'params': params,
                'sensor_idx': nearest_idx,
                'defect_info': defect,
            })

        # 3 spatial + 7 type one-hot (if enabled)
        self.n_params = 3 + (len(self.DEFECT_TYPES) if self.encode_defect_type else 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        si = s['sensor_idx']

        healthy_local = self.wf_healthy[si]       # (k, T)
        defect_local = self.wf_defect[si]         # (k, T)
        scatter_local = self.scattering[si]       # (k, T)

        if self.residual:
            target = scatter_local
        else:
            target = defect_local

        return {
            'params': torch.tensor(s['params'], dtype=torch.float32),
            'healthy': torch.tensor(healthy_local, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'defect_full': torch.tensor(defect_local, dtype=torch.float32),
        }

    def get_metadata(self):
        return {
            'n_sensors': self.k_nearest,
            'n_sensors_total': self.n_sensors_total,
            'n_params': self.n_params,
            'T': self.T,
            'n_defect_samples': len(self.samples),
            'waveform_scale': float(self.waveform_scale),
            'residual': self.residual,
            'multi_defect': True,
            'encode_defect_type': self.encode_defect_type,
            'defect_types': self.DEFECT_TYPES,
        }
