# -*- coding: utf-8 -*-
"""
GNN Model Architectures for Fairing Defect Localization

Implements node-level binary classification (defect / healthy) using:
- GCN  (Graph Convolutional Network)
- GAT  (Graph Attention Network)
- GIN  (Graph Isomorphism Network)
- SAGE (GraphSAGE)

All models share the same interface:
    model = build_model(arch, in_channels, edge_attr_dim, **kwargs)
    out = model(x, edge_index, edge_attr)  # (N, 2) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, SAGEConv,
    BatchNorm, global_mean_pool,
)


# =========================================================================
# Base class
# =========================================================================
class BaseGNN(nn.Module):
    """Shared skeleton for all GNN variants."""

    def __init__(self, in_channels, hidden_channels, num_classes,
                 num_layers, dropout, use_residual=False):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()  # residual projections

        # Head (node classifier)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def encode(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs):
            residual = x
            x = self._conv_forward(conv, x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + self.skip_projs[i](residual)
        return x

    def _conv_forward(self, conv, x, edge_index, edge_attr):
        """Override in subclass if conv signature differs."""
        return conv(x, edge_index)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.encode(x, edge_index, edge_attr)
        return self.head(h)  # (N, num_classes)


# =========================================================================
# GCN
# =========================================================================
class GCNModel(BaseGNN):
    def __init__(self, in_channels, hidden_channels=128, num_classes=2,
                 num_layers=4, dropout=0.1, use_residual=False):
        super().__init__(in_channels, hidden_channels, num_classes,
                         num_layers, dropout, use_residual)
        for i in range(num_layers):
            inc = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(inc, hidden_channels))
            self.norms.append(BatchNorm(hidden_channels))
            if use_residual:
                if inc != hidden_channels:
                    self.skip_projs.append(nn.Linear(inc, hidden_channels))
                else:
                    self.skip_projs.append(nn.Identity())


# =========================================================================
# GAT
# =========================================================================
class GATModel(BaseGNN):
    def __init__(self, in_channels, hidden_channels=128, num_classes=2,
                 num_layers=4, dropout=0.1, heads=4, edge_attr_dim=0,
                 use_residual=False):
        super().__init__(in_channels, hidden_channels, num_classes,
                         num_layers, dropout, use_residual)
        self.edge_attr_dim = edge_attr_dim

        for i in range(num_layers):
            if i == 0:
                inc = in_channels
            else:
                inc = hidden_channels * heads  # concat heads

            is_last = (i == num_layers - 1)
            out_c = hidden_channels
            h = 1 if is_last else heads
            concat = not is_last

            self.convs.append(GATConv(
                inc, out_c, heads=h, concat=concat, dropout=dropout,
                edge_dim=edge_attr_dim if edge_attr_dim > 0 else None,
            ))
            out_dim = out_c * h if concat else out_c
            self.norms.append(BatchNorm(out_dim))
            if use_residual:
                if inc != out_dim:
                    self.skip_projs.append(nn.Linear(inc, out_dim))
                else:
                    self.skip_projs.append(nn.Identity())

    def _conv_forward(self, conv, x, edge_index, edge_attr):
        if self.edge_attr_dim > 0 and edge_attr is not None:
            return conv(x, edge_index, edge_attr=edge_attr)
        return conv(x, edge_index)


# =========================================================================
# GIN
# =========================================================================
class GINModel(BaseGNN):
    def __init__(self, in_channels, hidden_channels=128, num_classes=2,
                 num_layers=4, dropout=0.1, use_residual=False):
        super().__init__(in_channels, hidden_channels, num_classes,
                         num_layers, dropout, use_residual)
        for i in range(num_layers):
            inc = in_channels if i == 0 else hidden_channels
            mlp = nn.Sequential(
                nn.Linear(inc, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(BatchNorm(hidden_channels))
            if use_residual:
                if inc != hidden_channels:
                    self.skip_projs.append(nn.Linear(inc, hidden_channels))
                else:
                    self.skip_projs.append(nn.Identity())


# =========================================================================
# GraphSAGE
# =========================================================================
class SAGEModel(BaseGNN):
    def __init__(self, in_channels, hidden_channels=128, num_classes=2,
                 num_layers=4, dropout=0.1, use_residual=False):
        super().__init__(in_channels, hidden_channels, num_classes,
                         num_layers, dropout, use_residual)
        for i in range(num_layers):
            inc = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(inc, hidden_channels))
            self.norms.append(BatchNorm(hidden_channels))
            if use_residual:
                if inc != hidden_channels:
                    self.skip_projs.append(nn.Linear(inc, hidden_channels))
                else:
                    self.skip_projs.append(nn.Identity())


# =========================================================================
# LGSTA — Local-Global Spatiotemporal Attention (LGSTA-GNN inspired)
# =========================================================================
class LocalGlobalAttentionGNN(nn.Module):
    """Local-Global Dual Attention GNN for node-level classification.

    Inspired by LGSTA-GNN (Buildings, 2025):
    - Local branch: GAT layers capture neighborhood structure
    - Global branch: Multi-head self-attention over all nodes for long-range dependencies
    - Fusion: Gated combination of local and global representations

    Designed for fairing defect detection where both local anomalies
    (debonding region) and global mode shapes matter.
    """

    def __init__(self, in_channels, hidden_channels=128, num_classes=2,
                 num_layers=4, dropout=0.1, heads=4, edge_attr_dim=0,
                 global_heads=4, use_residual=True):
        super().__init__()
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Local branch: GAT layers
        self.local_convs = nn.ModuleList()
        self.local_norms = nn.ModuleList()
        self.local_skip_projs = nn.ModuleList()
        self.edge_attr_dim = edge_attr_dim

        in_dim = hidden_channels
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            h = 1 if is_last else heads
            concat = not is_last
            out_c = hidden_channels
            self.local_convs.append(GATConv(
                in_dim, out_c, heads=h, concat=concat, dropout=dropout,
                edge_dim=edge_attr_dim if edge_attr_dim > 0 else None,
            ))
            out_dim = out_c * h if concat else out_c
            self.local_norms.append(BatchNorm(out_dim))
            if in_dim != out_dim:
                self.local_skip_projs.append(nn.Linear(in_dim, out_dim))
            else:
                self.local_skip_projs.append(nn.Identity())
            in_dim = out_dim
        self._local_final_dim = in_dim

        # Global branch: Transformer-style self-attention (lightweight)
        self.global_attn = nn.MultiheadAttention(
            hidden_channels, global_heads, dropout=dropout, batch_first=True)
        self.global_norm = nn.LayerNorm(hidden_channels)
        self.global_ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
        )
        self.global_ffn_norm = nn.LayerNorm(hidden_channels)

        # Fusion gate: learnable combination of local + global
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid(),
        )

        # Head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.input_proj(x)  # (N, hidden)

        # === Local branch (GAT) ===
        h_local = h
        for conv, norm, skip in zip(self.local_convs, self.local_norms, self.local_skip_projs):
            residual = h_local
            if self.edge_attr_dim > 0 and edge_attr is not None:
                h_local = conv(h_local, edge_index, edge_attr=edge_attr)
            else:
                h_local = conv(h_local, edge_index)
            h_local = norm(h_local)
            h_local = F.relu(h_local)
            h_local = F.dropout(h_local, p=self.dropout, training=self.training)
            h_local = h_local + skip(residual)
        # Project back to hidden_channels for fusion
        if self._local_final_dim != h.shape[-1]:
            if not hasattr(self, '_local_out_proj'):
                self._local_out_proj = nn.Linear(
                    self._local_final_dim, h.shape[-1])
            h_local = self._local_out_proj(h_local)

        # === Global branch (Self-Attention) ===
        # Process per-graph for batched data
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Attention over all nodes (per graph in batch)
        unique_graphs = batch.unique()
        h_global = torch.zeros_like(h)
        for g_id in unique_graphs:
            mask = (batch == g_id)
            h_g = h[mask].unsqueeze(0)  # (1, N_g, hidden)
            attn_out, _ = self.global_attn(h_g, h_g, h_g)
            h_g = self.global_norm(h_g + attn_out)
            ffn_out = self.global_ffn(h_g)
            h_g = self.global_ffn_norm(h_g + ffn_out)
            h_global[mask] = h_g.squeeze(0)

        # === Gated Fusion ===
        gate = self.gate(torch.cat([h_local, h_global], dim=-1))
        h_fused = gate * h_local + (1.0 - gate) * h_global

        return self.head(h_fused)


# =========================================================================
# Factory
# =========================================================================
# =========================================================================
# GPS Graph Transformer
# =========================================================================
try:
    from torch_geometric.nn import GPSConv

    class GPSTransformerModel(nn.Module):
        """GPS (General Powerful Scalable) Graph Transformer.

        Each layer combines:
        - Local MPNN (GINConv) for structural message passing
        - Global Multi-Head Self-Attention for long-range dependencies
        - Feed-Forward Network (inside GPSConv)

        Compatible with BaseGNN interface (encode/forward).
        """

        def __init__(self, in_channels, hidden_channels=128, num_classes=2,
                     num_layers=4, dropout=0.1, heads=4,
                     attn_type='multihead', use_residual=False):
            super().__init__()
            self.dropout = dropout

            # Project input features to hidden dimension
            self.input_proj = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )

            # GPS layers: local GIN + global attention
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                local_nn = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                gps_layer = GPSConv(
                    hidden_channels,
                    GINConv(local_nn),
                    heads=heads,
                    dropout=dropout,
                    attn_type=attn_type,
                )
                self.convs.append(gps_layer)

            # Classification head
            self.head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, num_classes),
            )

        def encode(self, x, edge_index, edge_attr=None, batch=None):
            """Encode graph to node embeddings.

            If batch is None, assumes single graph (all nodes batch=0).
            """
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long,
                                    device=x.device)
            x = self.input_proj(x)
            for conv in self.convs:
                x = conv(x, edge_index, batch)
            return x

        def forward(self, x, edge_index, edge_attr=None, batch=None):
            h = self.encode(x, edge_index, edge_attr, batch)
            return self.head(h)

    _GPS_AVAILABLE = True
except ImportError:
    _GPS_AVAILABLE = False


MODEL_REGISTRY = {
    'gcn': GCNModel,
    'gat': GATModel,
    'gin': GINModel,
    'sage': SAGEModel,
    'lgsta': LocalGlobalAttentionGNN,
}

if _GPS_AVAILABLE:
    MODEL_REGISTRY['gps'] = GPSTransformerModel


def build_model(arch, in_channels, edge_attr_dim=0, **kwargs):
    """
    Build a GNN model by name.

    Args:
        arch: one of 'gcn', 'gat', 'gin', 'sage', 'gps',
              'quantum', 'classical_graph'
        in_channels: number of node input features
        edge_attr_dim: number of edge attributes (used by GAT)
        **kwargs: forwarded to model constructor
                  (hidden_channels, num_layers, dropout, num_classes, use_residual)
    """
    if arch in ('quantum', 'classical_graph'):
        from models_quantum import build_quantum_model
        return build_quantum_model(arch, in_channels, edge_attr_dim, **kwargs)
    if arch == 'gps' and not _GPS_AVAILABLE:
        raise ImportError("GPSConv not available. Requires PyG >= 2.3.0")
    if arch not in MODEL_REGISTRY:
        raise ValueError("Unknown architecture '%s'. Choose from: %s" %
                         (arch, list(MODEL_REGISTRY.keys())))
    cls = MODEL_REGISTRY[arch]
    if arch in ('gat', 'lgsta'):
        return cls(in_channels, edge_attr_dim=edge_attr_dim, **kwargs)
    # GPS: filter out edge_attr_dim from kwargs (handled internally)
    if arch == 'gps':
        kwargs.pop('use_residual', None)  # GPS handles residuals internally
        return cls(in_channels, **kwargs)
    return cls(in_channels, **kwargs)
