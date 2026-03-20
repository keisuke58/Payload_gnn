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
# MeshGNN — Encode-Process-Decode (MeshGraphNets-inspired, PyG)
# =========================================================================
class _EdgeUpdateMLP(nn.Module):
    """Edge update: concat(sender, receiver, edge_feat) -> MLP -> new edge_feat."""

    def __init__(self, node_dim, edge_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        in_dim = node_dim * 2 + edge_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else out_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, node_features, edge_index, edge_features):
        src, dst = edge_index
        x_src = node_features[src]
        x_dst = node_features[dst]
        inp = torch.cat([x_src, x_dst, edge_features], dim=-1)
        return self.mlp(inp)


class _NodeUpdateMLP(nn.Module):
    """Node update: concat(node_feat, aggregated_edge_feat) -> MLP -> new node_feat."""

    def __init__(self, node_dim, edge_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        in_dim = node_dim + edge_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else out_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, node_features, edge_features, edge_index, num_nodes):
        # Aggregate incoming edge features (sum)
        dst = edge_index[1]
        agg = torch.zeros(num_nodes, edge_features.size(-1),
                          device=edge_features.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(edge_features),
                         edge_features)
        inp = torch.cat([node_features, agg], dim=-1)
        return self.mlp(inp)


class _GraphNetBlock(nn.Module):
    """Single message-passing block with edge + node update and residual."""

    def __init__(self, latent_size, edge_latent_size, mlp_layers=2):
        super().__init__()
        self.edge_fn = _EdgeUpdateMLP(latent_size, edge_latent_size,
                                       edge_latent_size, mlp_layers)
        self.node_fn = _NodeUpdateMLP(latent_size, edge_latent_size,
                                       latent_size, mlp_layers)

    def forward(self, node_features, edge_features, edge_index):
        # Edge update
        new_edge = self.edge_fn(node_features, edge_index, edge_features)
        # Node update
        new_node = self.node_fn(node_features, new_edge, edge_index,
                                node_features.size(0))
        # Residual connections on both nodes and edges
        new_node = new_node + node_features
        new_edge = new_edge + edge_features
        return new_node, new_edge


class MeshGNNModel(nn.Module):
    """Encode-Process-Decode GNN inspired by MeshGraphNets (Pfaff et al. 2021).

    - Encoder: independent MLPs project node/edge features to latent space
    - Processor: M rounds of GraphNetBlock (edge update + node update + residual)
    - Decoder: MLP from latent node features to output

    Key differences from BaseGNN:
    - Explicit edge feature updates at every step
    - Residual connections on both nodes AND edges
    - LayerNorm instead of BatchNorm (stable for small batches)
    - Same block repeated M times (can optionally share weights)
    """

    def __init__(self, in_channels, hidden_channels=128, num_classes=2,
                 num_layers=6, dropout=0.1, edge_attr_dim=0,
                 mlp_layers=2, share_weights=False, use_residual=False):
        super().__init__()
        self.dropout = dropout
        self._latent_size = hidden_channels
        self._edge_latent_size = hidden_channels
        self._num_steps = num_layers
        self._edge_attr_dim = edge_attr_dim

        # Encoder: node features -> latent
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )

        # Encoder: edge features -> latent (or create from scratch if no edge_attr)
        edge_in = edge_attr_dim if edge_attr_dim > 0 else 1
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )

        # Processor: M GraphNetBlocks
        if share_weights:
            block = _GraphNetBlock(hidden_channels, hidden_channels, mlp_layers)
            self.processors = nn.ModuleList([block] * num_layers)
        else:
            self.processors = nn.ModuleList([
                _GraphNetBlock(hidden_channels, hidden_channels, mlp_layers)
                for _ in range(num_layers)
            ])

        # Decoder: latent node features -> output
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def _encode_edges(self, edge_index, edge_attr):
        """Encode edge features; create dummy if none provided."""
        if edge_attr is not None and self._edge_attr_dim > 0:
            return self.edge_encoder(edge_attr)
        # Default: constant 1.0 per edge
        num_edges = edge_index.size(1)
        dummy = torch.ones(num_edges, 1, device=edge_index.device)
        return self.edge_encoder(dummy)

    def encode(self, x, edge_index, edge_attr=None):
        """Encode + Process, return node embeddings."""
        node_latent = self.node_encoder(x)
        edge_latent = self._encode_edges(edge_index, edge_attr)

        for block in self.processors:
            node_latent, edge_latent = block(node_latent, edge_latent, edge_index)
            node_latent = F.dropout(node_latent, p=self.dropout,
                                    training=self.training)

        return node_latent

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.encode(x, edge_index, edge_attr)
        return self.head(h)


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


# =========================================================================
# MultiScale GNN — Hierarchical graph pooling + unpooling
# =========================================================================
try:
    from torch_geometric.nn import TopKPooling
    from torch_geometric.utils import to_dense_adj, to_dense_batch

    class MultiScaleGNN(nn.Module):
        """Multi-scale GNN with hierarchical pooling and unpooling.

        Architecture:
          Level 0 (full resolution): GATConv layers → node embeddings
          Level 1 (coarsened 50%):   TopKPooling → GATConv → coarse embeddings
          Level 2 (coarsened 25%):   TopKPooling → GATConv → global context
          Unpool: interpolate coarse embeddings back to full resolution
          Fuse: concatenate local (L0) + upsampled (L1, L2) → classifier

        This captures both local defect patterns and global structural modes,
        which is critical for detecting debonding that affects mode shapes.
        """

        def __init__(self, in_channels, hidden_channels=128, num_classes=2,
                     num_layers=3, dropout=0.1, edge_attr_dim=0,
                     pool_ratio=0.5, use_residual=False):
            super().__init__()
            self.dropout = dropout
            self.pool_ratio = pool_ratio

            # Level 0: Full resolution encoder
            self.encoder0 = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs0 = nn.ModuleList()
            self.norms0 = nn.ModuleList()
            for _ in range(num_layers):
                self.convs0.append(GATConv(hidden_channels, hidden_channels // 4,
                                           heads=4, concat=True, dropout=dropout))
                self.norms0.append(nn.LayerNorm(hidden_channels))

            # Level 1: Coarsened graph
            self.pool1 = TopKPooling(hidden_channels, ratio=pool_ratio)
            self.convs1 = nn.ModuleList()
            self.norms1 = nn.ModuleList()
            for _ in range(num_layers):
                self.convs1.append(GATConv(hidden_channels, hidden_channels // 4,
                                           heads=4, concat=True, dropout=dropout))
                self.norms1.append(nn.LayerNorm(hidden_channels))

            # Level 2: Further coarsened
            self.pool2 = TopKPooling(hidden_channels, ratio=pool_ratio)
            self.convs2 = nn.ModuleList()
            self.norms2 = nn.ModuleList()
            for _ in range(max(1, num_layers - 1)):
                self.convs2.append(GATConv(hidden_channels, hidden_channels // 4,
                                           heads=4, concat=True, dropout=dropout))
                self.norms2.append(nn.LayerNorm(hidden_channels))

            # Fusion: local + coarse1 + coarse2 → classifier
            self.fusion = nn.Sequential(
                nn.Linear(hidden_channels * 3, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, num_classes),
            )

        def _apply_convs(self, x, edge_index, convs, norms):
            for conv, norm in zip(convs, norms):
                x_res = x
                x = conv(x, edge_index)
                x = norm(x)
                x = F.relu(x) + x_res
                x = F.dropout(x, p=self.dropout, training=self.training)
            return x

        def forward(self, x, edge_index, edge_attr=None, batch=None):
            n_nodes = x.size(0)
            if batch is None:
                batch = torch.zeros(n_nodes, dtype=torch.long, device=x.device)

            # Level 0: full resolution
            h0 = self.encoder0(x)
            h0 = self._apply_convs(h0, edge_index, self.convs0, self.norms0)

            # Level 1: pool → process
            h1, edge1, _, batch1, perm1, score1 = self.pool1(
                h0, edge_index, batch=batch)
            h1 = self._apply_convs(h1, edge1, self.convs1, self.norms1)

            # Level 2: pool → process
            h2, edge2, _, batch2, perm2, score2 = self.pool2(
                h1, edge1, batch=batch1)
            h2 = self._apply_convs(h2, edge2, self.convs2, self.norms2)

            # Unpool level 2 → level 1 resolution
            h2_up = torch.zeros_like(h1)
            h2_up[perm2] = h2

            # Unpool level 1 → level 0 resolution
            h1_up = torch.zeros(n_nodes, h1.size(1), device=x.device)
            h1_up[perm1] = h1

            h2_full = torch.zeros(n_nodes, h2.size(1), device=x.device)
            # Map L2 back through both unpooling steps
            h2_temp = torch.zeros_like(h1)
            h2_temp[perm2] = h2
            h2_full[perm1] = h2_temp

            # Fuse: concat local + coarse1 + coarse2
            h_fused = torch.cat([h0, h1_up, h2_full], dim=1)
            h_fused = self.fusion(h_fused)
            return self.head(h_fused)

    _MULTISCALE_AVAILABLE = True
except ImportError:
    _MULTISCALE_AVAILABLE = False


MODEL_REGISTRY = {
    'gcn': GCNModel,
    'gat': GATModel,
    'gin': GINModel,
    'sage': SAGEModel,
    'lgsta': LocalGlobalAttentionGNN,
    'meshgnn': MeshGNNModel,
}

if _MULTISCALE_AVAILABLE:
    MODEL_REGISTRY['multiscale'] = MultiScaleGNN

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
    if arch in ('gat', 'lgsta', 'meshgnn', 'multiscale'):
        return cls(in_channels, edge_attr_dim=edge_attr_dim, **kwargs)
    # GPS: filter out edge_attr_dim from kwargs (handled internally)
    if arch == 'gps':
        kwargs.pop('use_residual', None)  # GPS handles residuals internally
        return cls(in_channels, **kwargs)
    return cls(in_channels, **kwargs)
