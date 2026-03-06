# -*- coding: utf-8 -*-
"""
FNO1d / DeepONet for Guided Wave Surrogate

FNO1d: Spectral convolution in temporal dimension.
  Input: (batch, in_ch, T) — defect-encoded channels + healthy reference
  Output: (batch, n_sensors, T) — predicted waveform or residual

DeepONet: Branch-Trunk architecture for operator learning.
  Branch: defect_params → R^p
  Trunk: (sensor_norm, time_norm) → R^p
  Output: dot product → waveform amplitude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# 1D Spectral Convolution (FNO1d core)
# =============================================================================
class SpectralConv1d(nn.Module):
    """1D Fourier layer: FFT → multiply modes → iFFT."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes,
                               dtype=torch.cfloat))

    def forward(self, x):
        # x: (batch, in_ch, T)
        batch = x.shape[0]
        x_ft = torch.fft.rfft(x)  # (batch, in_ch, T//2+1)

        out_ft = torch.zeros(batch, self.out_channels, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        m = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :m] = torch.einsum(
            'bix,iox->box', x_ft[:, :, :m], self.weights[:, :, :m])

        return torch.fft.irfft(out_ft, n=x.size(-1))  # (batch, out_ch, T)


class FNO1d(nn.Module):
    """1D Fourier Neural Operator for GW waveform prediction.

    Architecture:
      1. Lift: (in_ch + 1_grid) → width via Linear
      2. N_layers × (SpectralConv1d + Conv1d + GELU)
      3. Project: width → out_ch via Linear

    Input: (batch, in_channels, T)
      in_channels = n_sensors (healthy ref) + n_params_broadcast
    Output: (batch, n_sensors, T) — predicted residual or waveform
    """

    def __init__(self, in_channels, out_channels, modes=64, width=64,
                 n_layers=4):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers

        # +1 for positional grid
        self.fc_lift = nn.Linear(in_channels + 1, width)

        self.spectral_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_layers.append(
                SpectralConv1d(width, width, modes))
            self.conv_layers.append(nn.Conv1d(width, width, 1))

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: (batch, in_ch, T)
        batch, _, T = x.shape

        # Grid encoding
        grid = torch.linspace(0, 1, T, device=x.device).reshape(1, 1, T)
        grid = grid.expand(batch, -1, -1)  # (batch, 1, T)
        x = torch.cat([x, grid], dim=1)    # (batch, in_ch+1, T)

        # Lift
        x = x.permute(0, 2, 1)   # (batch, T, in_ch+1)
        x = self.fc_lift(x)       # (batch, T, width)
        x = x.permute(0, 2, 1)   # (batch, width, T)

        # Spectral layers
        for s_layer, c_layer in zip(self.spectral_layers, self.conv_layers):
            x1 = s_layer(x)
            x2 = c_layer(x)
            x = F.gelu(x1 + x2)

        # Project
        x = x.permute(0, 2, 1)   # (batch, T, width)
        x = F.gelu(self.fc1(x))  # (batch, T, 128)
        x = self.fc2(x)          # (batch, T, out_ch)
        x = x.permute(0, 2, 1)   # (batch, out_ch, T)

        return x


class FNOGWSurrogate(nn.Module):
    """Full FNO surrogate: defect_params + healthy_ref → predicted waveform.

    Encodes defect parameters as spatially-broadcast channels,
    concatenates with healthy reference, feeds through FNO1d.
    """

    def __init__(self, n_sensors=9, n_params=3, modes=64, width=64,
                 n_layers=4, residual=True):
        super().__init__()
        self.n_sensors = n_sensors
        self.n_params = n_params
        self.residual = residual

        # Param encoder: 3 → n_sensors channels (broadcast over time)
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, 32),
            nn.GELU(),
            nn.Linear(32, n_sensors),
        )

        # FNO: in_ch = n_sensors (healthy) + n_sensors (param_encoded)
        self.fno = FNO1d(
            in_channels=n_sensors * 2,
            out_channels=n_sensors,
            modes=modes, width=width, n_layers=n_layers,
        )

    def forward(self, params, healthy):
        """
        Args:
            params: (batch, n_params)
            healthy: (batch, n_sensors, T)
        Returns:
            pred: (batch, n_sensors, T)
        """
        batch, _, T = healthy.shape

        # Encode params → (batch, n_sensors) → broadcast to (batch, n_sensors, T)
        p_enc = self.param_encoder(params)  # (batch, n_sensors)
        p_broadcast = p_enc.unsqueeze(-1).expand(-1, -1, T)

        # Concatenate: healthy + param_encoded
        x = torch.cat([healthy, p_broadcast], dim=1)  # (batch, 2*n_sensors, T)

        residual_pred = self.fno(x)  # (batch, n_sensors, T)

        if self.residual:
            return healthy + residual_pred
        return residual_pred


# =============================================================================
# DeepONet for GW
# =============================================================================
class DeepONetGW(nn.Module):
    """DeepONet for GW waveform prediction.

    Branch: defect_params → R^p
    Trunk: (sensor_norm, time_norm) → R^p
    Output: branch ⊙ trunk + bias → waveform amplitude
    """

    def __init__(self, n_params=3, coord_dim=2,
                 hidden_dim=128, basis_dim=64, n_branch_layers=3,
                 n_trunk_layers=3):
        super().__init__()
        self.basis_dim = basis_dim

        # Branch: params → basis coefficients
        branch_layers = [nn.Linear(n_params, hidden_dim), nn.GELU()]
        for _ in range(n_branch_layers - 2):
            branch_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        branch_layers.append(nn.Linear(hidden_dim, basis_dim))
        self.branch = nn.Sequential(*branch_layers)

        # Trunk: (sensor, time) → basis functions
        trunk_layers = [nn.Linear(coord_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_trunk_layers - 2):
            trunk_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        trunk_layers.append(nn.Linear(hidden_dim, basis_dim))
        self.trunk = nn.Sequential(*trunk_layers)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, params, coords):
        """
        Args:
            params: (batch, n_params)
            coords: (N, 2) or (batch, N, 2) — query (sensor_norm, time_norm)
        Returns:
            output: (batch, N) — predicted amplitude
        """
        b_out = self.branch(params)  # (batch, basis_dim)

        if coords.dim() == 2:
            t_out = self.trunk(coords)  # (N, basis_dim)
            output = torch.einsum('bp,np->bn', b_out, t_out) + self.bias
        else:
            batch, N, _ = coords.shape
            t_out = self.trunk(coords.reshape(-1, coords.size(-1)))
            t_out = t_out.view(batch, N, -1)
            output = torch.einsum('bp,bnp->bn', b_out, t_out) + self.bias

        return output


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    # FNO test
    batch, n_sensors, T = 4, 9, 512
    params = torch.randn(batch, 3)
    healthy = torch.randn(batch, n_sensors, T)
    model_fno = FNOGWSurrogate(n_sensors=n_sensors, modes=32, width=32)
    out_fno = model_fno(params, healthy)
    print(f"FNO: params {params.shape} + healthy {healthy.shape} → {out_fno.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_fno.parameters()):,}")

    # DeepONet test
    coords = torch.rand(100, 2)
    model_don = DeepONetGW(n_params=3, basis_dim=64)
    out_don = model_don(params, coords)
    print(f"DeepONet: params {params.shape} + coords {coords.shape} → {out_don.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_don.parameters()):,}")
