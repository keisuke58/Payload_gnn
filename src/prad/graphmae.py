# -*- coding: utf-8 -*-
"""PI-GraphMAE — Physics-Informed Graph Masked Autoencoder.

Self-supervised pre-training on FEM mesh graphs. Masks random node
features, reconstructs them via GNN encoder + MLP decoder, and
enforces physics constraints (stress equilibrium, strain compatibility).

Reference: GraphMAE (KDD 2022) adapted with physics-informed loss.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import build_model
from prad import IN_CHANNELS, EDGE_ATTR_DIM
from prad.physics_loss import combined_physics_loss


class MaskToken(nn.Module):
    """Learnable mask token replacing masked node features."""

    def __init__(self, dim):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, dim))
        nn.init.xavier_uniform_(self.token)

    def forward(self):
        return self.token


class PIGraphMAE(nn.Module):
    """Physics-Informed Graph Masked Autoencoder.

    Architecture:
        1. Randomly mask node features (replace with learnable mask token)
        2. GNN encoder → latent representation (N, hidden)
        3. Re-mask: replace masked positions with another learnable token
        4. MLP decoder → reconstructed features (N, in_channels)
        5. Loss = reconstruction error (masked nodes) + physics constraint

    Args:
        encoder_arch: 'gat', 'sage', 'gcn', or 'gin'.
        in_channels: input feature dimension (34).
        hidden_channels: encoder hidden dimension.
        num_layers: number of GNN layers.
        dropout: dropout rate in encoder.
        mask_ratio: fraction of nodes to mask (0.0–1.0).
        decoder_layers: number of MLP decoder layers.
        lambda_physics: weight for physics constraint loss.
    """

    def __init__(self, encoder_arch='sage', in_channels=IN_CHANNELS,
                 hidden_channels=128, num_layers=4, dropout=0.1,
                 mask_ratio=0.5, decoder_layers=2, lambda_physics=0.1,
                 decoder_type='mlp'):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.mask_ratio = mask_ratio
        self.lambda_physics = lambda_physics

        # Encoder: reuse existing GNN architecture (encoder part only)
        edge_attr_dim = EDGE_ATTR_DIM if encoder_arch == 'gat' else 0
        self.encoder = build_model(
            encoder_arch, in_channels=in_channels,
            edge_attr_dim=edge_attr_dim,
            hidden_channels=hidden_channels,
            num_classes=2,  # unused, only .encode() is called
            num_layers=num_layers,
            dropout=dropout,
        )

        # Mask tokens
        self.enc_mask_token = MaskToken(in_channels)
        self.dec_mask_token = MaskToken(hidden_channels)

        # Decoder
        if decoder_type == 'bottleneck':
            # Bottleneck MLP: forces compressed representation,
            # preventing trivial pass-through.
            bottleneck = hidden_channels // 2
            self.decoder = nn.Sequential(
                nn.Linear(hidden_channels, bottleneck),
                nn.LayerNorm(bottleneck),
                nn.PReLU(),
                nn.Linear(bottleneck, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.PReLU(),
                nn.Linear(hidden_channels, in_channels),
            )
        else:
            # Default MLP decoder (compatible with v3 checkpoints)
            dec_layers = []
            for i in range(decoder_layers):
                d_in = hidden_channels
                d_out = in_channels if i == decoder_layers - 1 \
                    else hidden_channels
                dec_layers.append(nn.Linear(d_in, d_out))
                if i < decoder_layers - 1:
                    dec_layers.append(nn.PReLU())
            self.decoder = nn.Sequential(*dec_layers)

    def mask_nodes(self, x, mask_ratio=None):
        """Randomly select nodes to mask.

        Args:
            x: (N, D) node features.
            mask_ratio: override default mask ratio.

        Returns:
            masked_x: (N, D) with masked nodes replaced by mask token.
            mask: (N,) boolean mask (True = masked).
        """
        ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        N = x.size(0)
        n_mask = max(1, int(N * ratio))

        perm = torch.randperm(N, device=x.device)
        mask_indices = perm[:n_mask]

        mask = torch.zeros(N, dtype=torch.bool, device=x.device)
        mask[mask_indices] = True

        masked_x = x.clone()
        masked_x[mask] = self.enc_mask_token()

        return masked_x, mask

    def forward(self, data, return_recon=False):
        """Forward pass: mask → encode → decode → loss.

        Loss is computed only on HEALTHY masked nodes (data.y == 0) so the
        model learns "what healthy physics looks like". Defect nodes still
        participate in message passing but don't contribute to the loss.

        Args:
            data: PyG Data with .x, .edge_index, .edge_attr, .y.
            return_recon: if True, also return reconstructed features.

        Returns:
            loss: scalar (reconstruction + physics).
            x_recon: (N, D) reconstructed features (if return_recon=True).
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)

        # 1. Mask input features
        masked_x, mask = self.mask_nodes(x)

        # 2. Encode (full graph with masked features)
        h = self.encoder.encode(masked_x, edge_index, edge_attr)  # (N, hidden)

        # 3. Decode to reconstruct features
        x_recon = self.decoder(h)  # (N, in_channels)

        # 4. Loss: only on healthy masked nodes
        # Healthy mask: nodes with label 0 AND selected for feature masking
        healthy = (data.y == 0)
        loss_mask = mask & healthy  # only healthy masked nodes

        loss_recon = self._reconstruction_loss(x_recon, x, loss_mask)

        loss_physics = torch.tensor(0.0, device=x.device)
        if self.lambda_physics > 0:
            pos = x[:, :3]
            # Physics loss on healthy nodes only
            healthy_idx = healthy.nonzero(as_tuple=True)[0]
            if len(healthy_idx) > 0:
                loss_physics = combined_physics_loss(
                    x_recon, edge_index, pos)

        loss = loss_recon + self.lambda_physics * loss_physics

        if return_recon:
            return loss, x_recon
        return loss

    def _reconstruction_loss(self, x_recon, x_target, mask):
        """Scaled cosine error on selected nodes (GraphMAE style).

        sce = 1 - cos(x_recon, x_target), averaged over selected nodes.
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x_recon.device)

        x_r = x_recon[mask]   # (M, D)
        x_t = x_target[mask]  # (M, D)

        # Cosine similarity per node
        cos_sim = F.cosine_similarity(x_r, x_t, dim=1)  # (M,)

        # Scaled cosine error: (1 - cos_sim) in [0, 2]
        loss = (1.0 - cos_sim).mean()
        return loss

    @torch.no_grad()
    def get_embeddings(self, data):
        """Get encoder embeddings without masking (for downstream tasks).

        Args:
            data: PyG Data.

        Returns:
            h: (N, hidden_channels) node embeddings.
        """
        edge_attr = getattr(data, 'edge_attr', None)
        return self.encoder.encode(data.x, data.edge_index, edge_attr)

    @torch.no_grad()
    def reconstruct(self, data):
        """Full reconstruction without masking (for anomaly scoring).

        Args:
            data: PyG Data.

        Returns:
            x_recon: (N, in_channels) reconstructed features.
        """
        edge_attr = getattr(data, 'edge_attr', None)
        h = self.encoder.encode(data.x, data.edge_index, edge_attr)
        return self.decoder(h)
