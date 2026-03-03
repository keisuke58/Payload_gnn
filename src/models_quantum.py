# -*- coding: utf-8 -*-
"""
Hybrid Quantum-Classical GNN for Graph-Level Defect Detection

Architecture:
    GNN Encoder (reuse BaseGNN.encode)
    → GlobalAttention Pooling (node → graph embedding)
    → Classical Dim Reduction (hidden → n_qubits)
    → VQC (Variational Quantum Circuit via Qiskit)
    → 2-class graph-level logits

Also provides ClassicalGNNGraphLevel as a fair baseline (MLP replaces VQC).
"""

import warnings
import math

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import GlobalAttention, global_mean_pool, global_max_pool

from models import build_model as build_gnn_model

# Suppress Qiskit deprecation warnings (ZZFeatureMap/RealAmplitudes class API)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")


# =========================================================================
# VQC Layer (Qiskit + TorchConnector)
# =========================================================================
class VQCLayer(nn.Module):
    """Variational Quantum Circuit wrapped as a PyTorch module.

    Uses Qiskit EstimatorQNN + TorchConnector for autograd-compatible
    forward/backward through a parameterized quantum circuit.

    Args:
        n_qubits: Number of qubits (= input dimension).
        reps: Number of ansatz repetitions.
    """

    def __init__(self, n_qubits=4, reps=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.reps = reps
        self._torch_layer = self._build(n_qubits, reps)

    # ---- construction --------------------------------------------------
    @staticmethod
    def _build(n_qubits, reps):
        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.connectors import TorchConnector

        feature_map = ZZFeatureMap(n_qubits, reps=1)
        ansatz = RealAmplitudes(n_qubits, reps=reps)

        qc = QuantumCircuit(n_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        # Two observables → 2 output logits
        pauli_0 = "Z" + "I" * (n_qubits - 1)
        pauli_1 = "I" * (n_qubits - 1) + "Z"
        obs = [
            SparsePauliOp.from_list([(pauli_0, 1.0)]),
            SparsePauliOp.from_list([(pauli_1, 1.0)]),
        ]

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            observables=obs,
            input_gradients=True,  # end-to-end gradient flow
        )
        return TorchConnector(qnn)

    # ---- forward --------------------------------------------------------
    def forward(self, x):
        """x: (B, n_qubits) → (B, 2)"""
        return self._torch_layer(x)


# =========================================================================
# Graph Pooling + Dim Reduction
# =========================================================================
class GraphPooling(nn.Module):
    """Multi-mode global pooling + dimension reduction to n_qubits.

    Uses mean + max + std pooling to capture both average structure
    and anomalous outlier signals (critical for rare defect detection).
    Maps node embeddings [N, hidden] → graph embedding [B, n_qubits]
    with values in [-pi, pi] (suitable for quantum rotation angles).
    """

    def __init__(self, hidden_channels, n_qubits):
        super().__init__()
        # mean + max + std → 3 * hidden
        pool_dim = hidden_channels * 3
        self.reduce = nn.Sequential(
            nn.Linear(pool_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_qubits),
        )

    def _std_pool(self, h, batch):
        """Per-graph standard deviation pooling."""
        mean = global_mean_pool(h, batch)  # (B, hidden)
        # Expand mean back to node level
        mean_expanded = mean[batch]  # (N, hidden)
        diff_sq = (h - mean_expanded) ** 2
        var = global_mean_pool(diff_sq, batch)  # (B, hidden)
        return torch.sqrt(var + 1e-8)

    def forward(self, h, batch):
        """h: (N, hidden), batch: (N,) → (B, n_qubits)"""
        g_mean = global_mean_pool(h, batch)   # (B, hidden)
        g_max = global_max_pool(h, batch)     # (B, hidden)
        g_std = self._std_pool(h, batch)      # (B, hidden)
        g = torch.cat([g_mean, g_max, g_std], dim=1)  # (B, 3*hidden)
        z = torch.tanh(self.reduce(g)) * math.pi  # (B, n_qubits) ∈ [-π, π]
        return z


# =========================================================================
# Hybrid Quantum GNN
# =========================================================================
class QuantumGNN(nn.Module):
    """GNN encoder + attention pooling + VQC for graph-level classification.

    Args:
        gnn_encoder: A BaseGNN instance (provides .encode()).
        hidden_channels: GNN output dim (must match encoder).
        n_qubits: Number of qubits for VQC.
        vqc_reps: Ansatz repetition count.
        freeze_gnn: If True, freeze GNN encoder weights.
    """

    def __init__(self, gnn_encoder, hidden_channels=128, n_qubits=4,
                 vqc_reps=2, freeze_gnn=False):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        if freeze_gnn:
            for p in self.gnn_encoder.parameters():
                p.requires_grad = False

        self.pool = GraphPooling(hidden_channels, n_qubits)
        self.vqc = VQCLayer(n_qubits, vqc_reps)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.gnn_encoder.encode(x, edge_index, edge_attr)  # (N, hidden)
        z = self.pool(h, batch)                                  # (B, n_qubits)
        logits = self.vqc(z)                                     # (B, 2)
        return logits


# =========================================================================
# Classical Graph-Level Baseline (fair comparison)
# =========================================================================
class ClassicalGNNGraphLevel(nn.Module):
    """Same as QuantumGNN but with a classical MLP replacing VQC.

    Uses the same bottleneck dimension (n_qubits) for fair comparison.
    """

    def __init__(self, gnn_encoder, hidden_channels=128, n_qubits=4,
                 freeze_gnn=False, dropout=0.1):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        if freeze_gnn:
            for p in self.gnn_encoder.parameters():
                p.requires_grad = False

        self.pool = GraphPooling(hidden_channels, n_qubits)
        self.head = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.gnn_encoder.encode(x, edge_index, edge_attr)
        z = self.pool(h, batch)
        logits = self.head(z)
        return logits


# =========================================================================
# GNN-free: Feature Statistics → VQC / MLP
# =========================================================================
class FeatureStatsPooling(nn.Module):
    """Compute graph-level statistics directly from raw node features.

    For each graph: mean, max, std over all nodes → 3 * in_channels dims.
    Then reduce to n_qubits via MLP.
    No GNN required — tests quantum classifier in isolation.
    """

    def __init__(self, in_channels, n_qubits):
        super().__init__()
        pool_dim = in_channels * 3  # mean + max + std
        self.reduce = nn.Sequential(
            nn.Linear(pool_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_qubits),
        )

    def forward(self, x, batch):
        """x: (N, in_channels), batch: (N,) → (B, n_qubits)"""
        g_mean = global_mean_pool(x, batch)
        g_max = global_max_pool(x, batch)
        # std pooling
        mean_exp = g_mean[batch]
        var = global_mean_pool((x - mean_exp) ** 2, batch)
        g_std = torch.sqrt(var + 1e-8)
        g = torch.cat([g_mean, g_max, g_std], dim=1)
        z = torch.tanh(self.reduce(g)) * math.pi
        return z


class QuantumStats(nn.Module):
    """Feature stats → VQC. No GNN, pure quantum classification test."""

    def __init__(self, in_channels, n_qubits=4, vqc_reps=2):
        super().__init__()
        self.pool = FeatureStatsPooling(in_channels, n_qubits)
        self.vqc = VQCLayer(n_qubits, vqc_reps)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        z = self.pool(x, batch)
        return self.vqc(z)


class ClassicalStats(nn.Module):
    """Feature stats → MLP. Classical baseline for QuantumStats."""

    def __init__(self, in_channels, n_qubits=4, dropout=0.1):
        super().__init__()
        self.pool = FeatureStatsPooling(in_channels, n_qubits)
        self.head = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2),
        )

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        z = self.pool(x, batch)
        return self.head(z)


# =========================================================================
# Factory
# =========================================================================
def build_quantum_model(arch, in_channels, edge_attr_dim=0,
                        hidden_channels=128, num_layers=4, dropout=0.1,
                        use_residual=False, n_qubits=4, vqc_reps=2,
                        freeze_gnn=False, gnn_checkpoint=None):
    """Build a graph-level quantum or classical model.

    Args:
        arch: 'quantum' or 'classical_graph'
        in_channels: Node feature dimension.
        edge_attr_dim: Edge attribute dimension.
        hidden_channels, num_layers, dropout, use_residual: GNN params.
        n_qubits, vqc_reps: Quantum circuit params.
        freeze_gnn: Freeze GNN encoder weights.
        gnn_checkpoint: Path to pretrained GNN checkpoint.
    """
    # Build GNN encoder (always GAT for best edge_attr support)
    gnn = build_gnn_model(
        "gat", in_channels, edge_attr_dim,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        use_residual=use_residual,
    )

    # Load pretrained weights if provided
    if gnn_checkpoint:
        ckpt = torch.load(gnn_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        # Load only encoder weights (skip head)
        encoder_state = {k: v for k, v in state.items()
                         if not k.startswith("head.")}
        gnn.load_state_dict(encoder_state, strict=False)

    if arch == "quantum_stats":
        return QuantumStats(in_channels, n_qubits=n_qubits, vqc_reps=vqc_reps)
    elif arch == "classical_stats":
        return ClassicalStats(in_channels, n_qubits=n_qubits, dropout=dropout)
    elif arch == "quantum":
        return QuantumGNN(
            gnn, hidden_channels=hidden_channels,
            n_qubits=n_qubits, vqc_reps=vqc_reps,
            freeze_gnn=freeze_gnn,
        )
    elif arch == "classical_graph":
        return ClassicalGNNGraphLevel(
            gnn, hidden_channels=hidden_channels,
            n_qubits=n_qubits, freeze_gnn=freeze_gnn,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown quantum arch: {arch}")
