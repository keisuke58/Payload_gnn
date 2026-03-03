# -*- coding: utf-8 -*-
"""Physics-informed losses for PI-GraphMAE.

Discrete approximations of continuum mechanics constraints
computed on the FEM mesh graph.
"""

import torch

try:
    from torch_scatter import scatter_mean
except ImportError:
    from torch_geometric.utils import scatter
    def scatter_mean(src, index, dim=0, dim_size=None):
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')

from . import STRESS_DIMS, STRAIN_DIMS, POSITION_DIMS


def stress_equilibrium_loss(x, edge_index, pos=None):
    """Discrete divergence of stress ≈ 0 (static equilibrium).

    For each node i, computes:
        div_i = mean_{j in N(i)} (sigma_j - sigma_i) / |r_j - r_i|

    In a healthy structure under static loads, ∇·σ ≈ 0 everywhere.
    Defect regions break this because of discontinuous stiffness changes.

    Args:
        x: (N, D) node features (reconstructed or actual).
        edge_index: (2, E) graph edges.
        pos: (N, 3) node positions. If None, uses x[:, :3].

    Returns:
        Scalar loss: mean |div_i| across all nodes.
    """
    if pos is None:
        pos = x[:, POSITION_DIMS]

    src, dst = edge_index[0], edge_index[1]

    # Stress at source and destination nodes (s11, s22, s12)
    sigma_src = x[src][:, STRESS_DIMS]  # (E, 3)
    sigma_dst = x[dst][:, STRESS_DIMS]  # (E, 3)

    # Distance between connected nodes
    dr = pos[dst] - pos[src]            # (E, 3)
    dist = dr.norm(dim=1, keepdim=True).clamp(min=1e-8)  # (E, 1)

    # Stress gradient along each edge
    grad_sigma = (sigma_dst - sigma_src) / dist  # (E, 3)

    # Aggregate: mean gradient per node (discrete divergence)
    div_per_node = scatter_mean(grad_sigma, src, dim=0,
                                dim_size=x.size(0))  # (N, 3)

    return div_per_node.abs().mean()


def strain_compatibility_loss(x, edge_index):
    """Strain field smoothness constraint.

    Strain should vary smoothly in healthy regions. Large jumps indicate
    either damage or reconstruction errors.

    Args:
        x: (N, D) node features.
        edge_index: (2, E) graph edges.

    Returns:
        Scalar loss: mean |strain_j - strain_i| across all edges.
    """
    src, dst = edge_index[0], edge_index[1]

    strain_src = x[src][:, STRAIN_DIMS]  # (E, 3)
    strain_dst = x[dst][:, STRAIN_DIMS]  # (E, 3)

    diff = (strain_dst - strain_src).abs()
    return diff.mean()


def displacement_continuity_loss(x, edge_index):
    """Displacement field continuity.

    Displacement should be continuous across the mesh. Jumps indicate
    either crack-like damage or bad reconstruction.

    Args:
        x: (N, D) node features.
        edge_index: (2, E) graph edges.

    Returns:
        Scalar loss.
    """
    from . import DISPLACEMENT_DIMS

    src, dst = edge_index[0], edge_index[1]

    u_src = x[src][:, DISPLACEMENT_DIMS]  # (E, 3)
    u_dst = x[dst][:, DISPLACEMENT_DIMS]  # (E, 3)

    diff = (u_dst - u_src).abs()
    return diff.mean()


def combined_physics_loss(x, edge_index, pos=None,
                          w_equil=1.0, w_strain=0.5, w_disp=0.5):
    """Combined physics loss with configurable weights.

    Args:
        x: (N, D) reconstructed node features.
        edge_index: (2, E) graph edges.
        pos: (N, 3) node positions (optional).
        w_equil: weight for stress equilibrium.
        w_strain: weight for strain compatibility.
        w_disp: weight for displacement continuity.

    Returns:
        Scalar loss.
    """
    loss = torch.tensor(0.0, device=x.device)

    if w_equil > 0:
        loss = loss + w_equil * stress_equilibrium_loss(x, edge_index, pos)
    if w_strain > 0:
        loss = loss + w_strain * strain_compatibility_loss(x, edge_index)
    if w_disp > 0:
        loss = loss + w_disp * displacement_continuity_loss(x, edge_index)

    return loss
