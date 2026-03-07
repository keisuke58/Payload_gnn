#!/usr/bin/env python3
"""Graph Posterior Network (GPN) for SHM Node-Level Defect Detection.

Bayesian node classification with uncertainty quantification using:
  1. MLP Encoder → latent space
  2. Normalizing Flow (Radial) → class-conditional densities
  3. Evidence layer → Dirichlet pseudo-counts
  4. APPNP propagation → posterior aggregation

Based on: Stadler et al., "Graph Posterior Network: Bayesian Predictive
Uncertainty for Node Classification", NeurIPS 2021.

Usage:
    python src/gpn_shm.py --mode train --data_dir data/processed_s12_thermal_500
    python src/gpn_shm.py --mode eval  --checkpoint runs/gpn/best_gpn.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

# ── Radial Flow ──────────────────────────────────────────────────────


class RadialTransform(nn.Module):
    """Single radial normalizing flow transform.

    f(z) = z + beta * (z - x0) / (alpha + ||z - x0||)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.x0 = nn.Parameter(torch.randn(dim) * 0.1)
        self.alpha_prime = nn.Parameter(torch.zeros(1))
        self.beta_prime = nn.Parameter(torch.zeros(1))

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = F.softplus(self.alpha_prime)
        beta = -alpha + F.softplus(self.beta_prime)
        diff = z - self.x0
        r = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        h = 1.0 / (alpha + r)
        z_new = z + beta * h * diff

        # Log determinant of Jacobian
        d = z.shape[-1]
        h_prime = -1.0 / (alpha + r) ** 2
        log_det = (d - 1) * torch.log1p(beta * h).squeeze(-1) + torch.log1p(
            beta * h + beta * h_prime * r
        ).squeeze(-1)
        return z_new, log_det


class RadialFlow(nn.Module):
    """Stack of radial transforms with learnable base distribution."""

    def __init__(self, dim: int, n_layers: int = 10):
        super().__init__()
        self.transforms = nn.ModuleList(
            [RadialTransform(dim) for _ in range(n_layers)]
        )
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_var = nn.Parameter(torch.zeros(dim))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z) under the flow distribution."""
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        for transform in reversed(self.transforms):
            z, log_det = transform(z)
            log_det_sum += log_det

        # Base distribution: diagonal Gaussian
        var = self.log_var.exp()
        log_p_base = (
            -0.5 * z.shape[-1] * math.log(2 * math.pi)
            - 0.5 * self.log_var.sum()
            - 0.5 * ((z - self.mu) ** 2 / var).sum(dim=-1)
        )
        return log_p_base + log_det_sum


class BatchedRadialFlow(nn.Module):
    """C independent radial flows for C classes."""

    def __init__(self, dim: int, n_classes: int, n_layers: int = 10):
        super().__init__()
        self.flows = nn.ModuleList(
            [RadialFlow(dim, n_layers) for _ in range(n_classes)]
        )

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Returns log p(z|c) for each class c. Shape: [N, C]."""
        return torch.stack([flow.log_prob(z) for flow in self.flows], dim=-1)


# ── APPNP Propagation ────────────────────────────────────────────────


class APPNPPropagation(nn.Module):
    """Approximate Personalized PageRank propagation.

    H^(k) = (1 - alpha) * A_hat * H^(k-1) + alpha * H^(0)
    """

    def __init__(self, K: int = 10, alpha: float = 0.1):
        super().__init__()
        self.K = K
        self.alpha = alpha

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes: int | None = None,
    ) -> torch.Tensor:
        if n_nodes is None:
            n_nodes = x.shape[0]

        row, col = edge_index
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        deg = torch.zeros(n_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.shape[0], device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        h = x
        for _ in range(self.K):
            # Sparse message passing
            msg = x.new_zeros(n_nodes, x.shape[-1])
            msg.scatter_add_(
                0,
                row.unsqueeze(-1).expand_as(h[col]),
                norm.unsqueeze(-1) * h[col],
            )
            h = (1 - self.alpha) * msg + self.alpha * x
        return h


# ── Physics Edge Weight ──────────────────────────────────────────────


class PhysicsEdgeMLP(nn.Module):
    """Learnable physics-informed edge weights from edge features.

    Maps edge attributes (dx, dy, dz, distance, curvature_diff)
    to scalar weights representing equivalent stiffness/damping.
    """

    def __init__(self, edge_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Returns positive scalar weight per edge. Shape: [E]."""
        return F.softplus(self.mlp(edge_attr).squeeze(-1))


class PhysicsAPPNPPropagation(nn.Module):
    """APPNP with physics-informed edge weights.

    Instead of symmetric normalization D^{-1/2}AD^{-1/2},
    uses learned edge weights: w_ij / sum_j(w_ij).
    """

    def __init__(self, K: int = 10, alpha: float = 0.1):
        super().__init__()
        self.K = K
        self.alpha = alpha

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        n_nodes: int | None = None,
    ) -> torch.Tensor:
        if n_nodes is None:
            n_nodes = x.shape[0]

        row, col = edge_index
        # Row-normalize: w_ij / sum_j(w_ij)
        deg_w = torch.zeros(n_nodes, device=x.device)
        deg_w.scatter_add_(0, row, edge_weight)
        norm = edge_weight / (deg_w[row] + 1e-10)

        h = x
        for _ in range(self.K):
            msg = x.new_zeros(n_nodes, x.shape[-1])
            msg.scatter_add_(
                0,
                row.unsqueeze(-1).expand_as(h[col]),
                norm.unsqueeze(-1) * h[col],
            )
            h = (1 - self.alpha) * msg + self.alpha * x
        return h


# ── GPN Model ────────────────────────────────────────────────────────


class GPNModel(nn.Module):
    """Graph Posterior Network for node classification.

    Architecture:
        MLP Encoder → Radial Flow → Evidence → APPNP → Dirichlet posterior
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        dim_hidden: int = 64,
        dim_latent: int = 16,
        radial_layers: int = 10,
        appnp_K: int = 10,
        appnp_alpha: float = 0.1,
        beta_prior: float = 1e-3,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.dim_latent = dim_latent
        self.beta_prior = beta_prior

        # MLP Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_hidden, dim_latent),
        )

        # Normalizing Flow (per-class density)
        self.flow = BatchedRadialFlow(dim_latent, n_classes, radial_layers)

        # APPNP propagation
        self.propagation = APPNPPropagation(K=appnp_K, alpha=appnp_alpha)

        # Evidence scaling constant
        self.log_scale = 0.5 * dim_latent * math.log(4 * math.pi)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        N = x.shape[0]

        # 1. Encode
        z = self.encoder(x)  # [N, dim_latent]

        # 2. Flow → class-conditional log densities
        log_q = self.flow.log_prob(z)  # [N, C]

        # 3. Evidence: convert to Dirichlet pseudo-counts
        scaled_log_q = log_q + self.log_scale
        beta_ft = scaled_log_q.clamp(-30.0, 30.0).exp()  # [N, C]

        # 4. APPNP propagation of pseudo-counts
        # Add self-loops
        self_loops = torch.arange(N, device=x.device).unsqueeze(0).repeat(2, 1)
        edge_index_sl = torch.cat([edge_index, self_loops], dim=1)
        beta_agg = self.propagation(beta_ft, edge_index_sl, N)  # [N, C]

        # 5. Dirichlet posterior
        alpha = self.beta_prior + beta_agg.clamp(min=0)  # [N, C]

        # Predictions and uncertainty
        alpha_sum = alpha.sum(dim=-1)  # [N]
        probs = alpha / alpha_sum.unsqueeze(-1)  # [N, C]

        return {
            "alpha": alpha,
            "probs": probs,
            "z": z,
            "beta_ft": beta_ft,
            "log_q": log_q,
        }

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass + uncertainty decomposition."""
        out = self.forward(x, edge_index)
        alpha = out["alpha"]
        alpha_sum = alpha.sum(dim=-1)
        probs = out["probs"]

        # Epistemic uncertainty: inverse of total evidence
        epistemic = self.n_classes / alpha_sum

        # Aleatoric uncertainty: expected entropy
        aleatoric = -(probs * (probs + 1e-10).log()).sum(dim=-1)

        # Total uncertainty
        total = epistemic + aleatoric

        out["epistemic_uncertainty"] = epistemic
        out["aleatoric_uncertainty"] = aleatoric
        out["total_uncertainty"] = total
        return out


class PhysicsGPNModel(nn.Module):
    """Physics-Informed GPN with learnable edge weights.

    Replaces standard APPNP normalization with physics-informed edge
    weights derived from edge attributes (stiffness/damping analogy).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        dim_hidden: int = 64,
        dim_latent: int = 16,
        radial_layers: int = 10,
        appnp_K: int = 10,
        appnp_alpha: float = 0.1,
        beta_prior: float = 1e-3,
        edge_dim: int = 5,
        edge_hidden: int = 32,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.dim_latent = dim_latent
        self.beta_prior = beta_prior

        # MLP Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_hidden, dim_latent),
        )

        # Normalizing Flow
        self.flow = BatchedRadialFlow(dim_latent, n_classes, radial_layers)

        # Physics edge weight
        self.edge_mlp = PhysicsEdgeMLP(edge_dim, edge_hidden)

        # Physics-aware APPNP
        self.propagation = PhysicsAPPNPPropagation(K=appnp_K, alpha=appnp_alpha)

        # Evidence scaling
        self.log_scale = 0.5 * dim_latent * math.log(4 * math.pi)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        N = x.shape[0]

        # 1. Encode
        z = self.encoder(x)

        # 2. Flow
        log_q = self.flow.log_prob(z)

        # 3. Evidence
        scaled_log_q = log_q + self.log_scale
        beta_ft = scaled_log_q.clamp(-30.0, 30.0).exp()

        # 4. Physics-informed APPNP
        # Add self-loops with unit edge features
        self_loops = torch.arange(N, device=x.device).unsqueeze(0).repeat(2, 1)
        edge_index_sl = torch.cat([edge_index, self_loops], dim=1)

        if edge_attr is not None:
            # Self-loop edge features: zero displacement, small distance
            self_loop_attr = torch.zeros(N, edge_attr.shape[1], device=x.device)
            edge_attr_sl = torch.cat([edge_attr, self_loop_attr], dim=0)
            edge_weight = self.edge_mlp(edge_attr_sl)
        else:
            # Fallback: uniform weights
            edge_weight = torch.ones(edge_index_sl.shape[1], device=x.device)

        beta_agg = self.propagation(beta_ft, edge_index_sl, edge_weight, N)

        # 5. Dirichlet posterior
        alpha = self.beta_prior + beta_agg.clamp(min=0)
        alpha_sum = alpha.sum(dim=-1)
        probs = alpha / alpha_sum.unsqueeze(-1)

        return {
            "alpha": alpha,
            "probs": probs,
            "z": z,
            "beta_ft": beta_ft,
            "log_q": log_q,
            "edge_weight": edge_weight,
        }

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(x, edge_index, edge_attr)
        alpha = out["alpha"]
        alpha_sum = alpha.sum(dim=-1)
        probs = out["probs"]

        epistemic = self.n_classes / alpha_sum
        aleatoric = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        total = epistemic + aleatoric

        out["epistemic_uncertainty"] = epistemic
        out["aleatoric_uncertainty"] = aleatoric
        out["total_uncertainty"] = total
        return out


# ── Loss Functions ────────────────────────────────────────────────────


def uce_loss(
    alpha: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Uncertainty Cross-Entropy: E_{p~Dir(alpha)}[-log p_y].

    = digamma(alpha_0) - digamma(alpha_y)
    """
    alpha_sum = alpha.sum(dim=-1)
    alpha_y = alpha.gather(-1, y.view(-1, 1)).squeeze(-1)
    loss = torch.digamma(alpha_sum) - torch.digamma(alpha_y)
    if reduction == "mean":
        return loss.mean()
    return loss


def entropy_reg(
    alpha: torch.Tensor,
    beta_reg: float = 1e-4,
    reduction: str = "mean",
) -> torch.Tensor:
    """Entropy regularization to prevent overconfident predictions."""
    dirichlet = Dirichlet(alpha)
    reg = -beta_reg * dirichlet.entropy()
    if reduction == "mean":
        return reg.mean()
    return reg


def gpn_loss(
    alpha: torch.Tensor,
    y: torch.Tensor,
    beta_reg: float = 1e-4,
) -> torch.Tensor:
    """Combined GPN loss: UCE + entropy regularization."""
    return uce_loss(alpha, y) + entropy_reg(alpha, beta_reg)


# ── Data Utilities ────────────────────────────────────────────────────


def subsample_graph(
    data,
    max_nodes: int = 4000,
    seed: int = 0,
):
    """Subsample nodes keeping ALL defect nodes (vectorized)."""
    import torch_geometric.data as gd

    N = data.x.shape[0]
    has_edge_attr = hasattr(data, "edge_attr") and data.edge_attr is not None

    if N <= max_nodes:
        # Still binarize labels
        new_data = gd.Data(
            x=data.x,
            edge_index=data.edge_index,
            y=(data.y > 0).long(),
            pos=data.pos if data.pos is not None else None,
        )
        if has_edge_attr:
            new_data.edge_attr = data.edge_attr
        return new_data

    labels = (data.y > 0).long()
    defect_mask = labels > 0
    defect_idx = defect_mask.nonzero(as_tuple=True)[0]
    normal_idx = (~defect_mask).nonzero(as_tuple=True)[0]
    n_normal = max_nodes - len(defect_idx)

    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(normal_idx), size=min(n_normal, len(normal_idx)), replace=False)
    sampled_normal = normal_idx[chosen]
    keep_idx = torch.cat([defect_idx, sampled_normal]).sort()[0]

    # Vectorized edge filtering: build boolean membership mask
    keep_mask = torch.zeros(N, dtype=torch.bool)
    keep_mask[keep_idx] = True

    ei = data.edge_index
    edge_mask = keep_mask[ei[0]] & keep_mask[ei[1]]
    filtered_ei = ei[:, edge_mask]

    # Vectorized remapping: old index → new index
    remap = torch.full((N,), -1, dtype=torch.long)
    remap[keep_idx] = torch.arange(len(keep_idx))
    new_ei = remap[filtered_ei]

    new_data = gd.Data(
        x=data.x[keep_idx],
        edge_index=new_ei,
        y=labels[keep_idx],
        pos=data.pos[keep_idx] if data.pos is not None else None,
    )
    if has_edge_attr:
        new_data.edge_attr = data.edge_attr[edge_mask]
    return new_data


# ── Training ──────────────────────────────────────────────────────────


def flush_print(*args, **kwargs):
    """Print with immediate flush for log visibility."""
    print(*args, **kwargs, flush=True)


def train_gpn(
    data_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    dim_hidden: int = 64,
    dim_latent: int = 16,
    radial_layers: int = 10,
    appnp_K: int = 10,
    appnp_alpha: float = 0.1,
    epochs: int = 100,
    warmup_epochs: int = 10,
    lr: float = 1e-3,
    lr_flow: float = 1e-4,
    max_nodes: int = 4000,
    max_train: int = 50,
    max_val: int = 20,
    beta_reg: float = 1e-4,
    physics: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    flush_print("Loading data...")
    train_data = torch.load(data_dir / "train.pt", weights_only=False, map_location="cpu")
    val_data = torch.load(data_dir / "val.pt", weights_only=False, map_location="cpu")

    # Filter to defect-containing samples only
    train_data = [g for g in train_data if g.y.sum() > 0][:max_train]
    val_data = [g for g in val_data if g.y.sum() > 0][:max_val]
    flush_print(f"Train: {len(train_data)} graphs, Val: {len(val_data)} graphs")

    n_features = train_data[0].x.shape[0] if len(train_data) == 0 else train_data[0].x.shape[1]
    edge_dim = train_data[0].edge_attr.shape[1] if hasattr(train_data[0], "edge_attr") and train_data[0].edge_attr is not None else 5

    if physics:
        flush_print("Using PhysicsGPN (learnable edge weights)")
        model = PhysicsGPNModel(
            n_features=n_features,
            n_classes=2,
            dim_hidden=dim_hidden,
            dim_latent=dim_latent,
            radial_layers=radial_layers,
            appnp_K=appnp_K,
            appnp_alpha=appnp_alpha,
            edge_dim=edge_dim,
        ).to(device)
    else:
        model = GPNModel(
            n_features=n_features,
            n_classes=2,
            dim_hidden=dim_hidden,
            dim_latent=dim_latent,
            radial_layers=radial_layers,
            appnp_K=appnp_K,
            appnp_alpha=appnp_alpha,
        ).to(device)

    # Separate optimizers for encoder and flow (different learning rates)
    encoder_params = list(model.encoder.parameters()) + list(model.propagation.parameters())
    if physics:
        encoder_params += list(model.edge_mlp.parameters())
    flow_params = list(model.flow.parameters())

    opt_encoder = torch.optim.Adam(encoder_params, lr=lr, weight_decay=1e-4)
    opt_flow = torch.optim.Adam(flow_params, lr=lr_flow, weight_decay=1e-4)

    best_val_auroc = 0.0
    history = []
    model_tag = "PhysicsGPN" if physics else "GPN"
    ckpt_name = "best_physics_gpn.pt" if physics else "best_gpn.pt"

    flush_print(f"Model [{model_tag}]: {sum(p.numel() for p in model.parameters()):,} params")
    flush_print(f"Training for {epochs} epochs (warmup={warmup_epochs})...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_graphs = 0
        t0 = time.time()

        for i, data in enumerate(train_data):
            sub = subsample_graph(data, max_nodes=max_nodes, seed=epoch * 1000 + i)
            x = sub.x.to(device)
            ei = sub.edge_index.to(device)
            y = sub.y.to(device)
            ea = sub.edge_attr.to(device) if hasattr(sub, "edge_attr") and sub.edge_attr is not None else None

            if physics:
                out = model(x, ei, ea)
            else:
                out = model(x, ei)

            if epoch <= warmup_epochs:
                # Warmup: train flow only with NLL
                loss = -out["log_q"].gather(-1, y.view(-1, 1)).squeeze(-1).mean()
                opt_flow.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow_params, 1.0)
                opt_flow.step()
            else:
                # Full training: UCE + entropy reg
                loss = gpn_loss(out["alpha"], y, beta_reg=beta_reg)
                opt_encoder.zero_grad()
                opt_flow.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_encoder.step()
                opt_flow.step()

            epoch_loss += loss.item()
            n_graphs += 1

        avg_loss = epoch_loss / max(n_graphs, 1)

        # Validation
        if epoch % 5 == 0 or epoch <= 3:
            val_auroc, val_auprc = evaluate_gpn(model, val_data, device, max_nodes, physics=physics)
            dt = time.time() - t0
            phase = "warmup" if epoch <= warmup_epochs else "full"
            flush_print(
                f"Epoch {epoch:3d} [{phase}] loss={avg_loss:.4f} "
                f"val_AUROC={val_auroc:.4f} val_AUPRC={val_auprc:.4f} "
                f"({dt:.1f}s)"
            )
            history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
            })

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_auroc": val_auroc,
                        "val_auprc": val_auprc,
                        "physics": physics,
                        "config": {
                            "n_features": n_features,
                            "dim_hidden": dim_hidden,
                            "dim_latent": dim_latent,
                            "radial_layers": radial_layers,
                            "appnp_K": appnp_K,
                            "appnp_alpha": appnp_alpha,
                            "edge_dim": edge_dim if physics else 0,
                        },
                    },
                    output_dir / ckpt_name,
                )
                flush_print(f"  → Best model saved (AUROC={val_auroc:.4f})")

    # Save history
    with open(output_dir / "gpn_history.json", "w") as f:
        json.dump(history, f, indent=2)
    flush_print(f"\nTraining complete. Best val AUROC: {best_val_auroc:.4f}")
    return model


def evaluate_gpn(
    model: nn.Module,
    data_list: list,
    device: str,
    max_nodes: int = 4000,
    physics: bool = False,
) -> tuple[float, float]:
    """Evaluate GPN on a list of graphs. Returns (AUROC, AUPRC)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for i, data in enumerate(data_list):
            sub = subsample_graph(data, max_nodes=max_nodes, seed=i)
            x = sub.x.to(device)
            ei = sub.edge_index.to(device)
            y = sub.y
            ea = sub.edge_attr.to(device) if physics and hasattr(sub, "edge_attr") and sub.edge_attr is not None else None

            if physics:
                out = model.predict_with_uncertainty(x, ei, ea)
            else:
                out = model.predict_with_uncertainty(x, ei)
            probs = out["probs"][:, 1].cpu().numpy()
            labels = y.numpy()

            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    if len(np.unique(all_labels)) < 2:
        return 0.5, 0.0

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    return auroc, auprc


# ── Inference with Uncertainty ────────────────────────────────────────


def infer_with_uncertainty(
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    max_nodes: int = 6000,
):
    """Run inference and produce uncertainty-aware predictions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, roc_auc_score

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = ckpt["config"]
    is_physics = ckpt.get("physics", False)

    if is_physics:
        model = PhysicsGPNModel(
            n_features=config["n_features"],
            n_classes=2,
            dim_hidden=config["dim_hidden"],
            dim_latent=config["dim_latent"],
            radial_layers=config["radial_layers"],
            appnp_K=config["appnp_K"],
            appnp_alpha=config["appnp_alpha"],
            edge_dim=config.get("edge_dim", 5),
        ).to(device)
    else:
        model = GPNModel(
            n_features=config["n_features"],
            n_classes=2,
            dim_hidden=config["dim_hidden"],
            dim_latent=config["dim_latent"],
            radial_layers=config["radial_layers"],
            appnp_K=config["appnp_K"],
            appnp_alpha=config["appnp_alpha"],
        ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tag = "PhysicsGPN" if is_physics else "GPN"
    print(f"Loaded {tag} from {checkpoint_path} (epoch {ckpt['epoch']})")

    # Load validation data
    val_data = torch.load(data_dir / "val.pt", weights_only=False, map_location="cpu")
    val_data = [g for g in val_data if g.y.sum() > 0]

    # Run inference
    results = []
    all_probs, all_labels = [], []
    all_epistemic, all_aleatoric = [], []

    with torch.no_grad():
        for i, data in enumerate(val_data):
            sub = subsample_graph(data, max_nodes=max_nodes, seed=i)
            x = sub.x.to(device)
            ei = sub.edge_index.to(device)
            y = sub.y.numpy()
            ea = sub.edge_attr.to(device) if is_physics and hasattr(sub, "edge_attr") and sub.edge_attr is not None else None

            if is_physics:
                out = model.predict_with_uncertainty(x, ei, ea)
            else:
                out = model.predict_with_uncertainty(x, ei)

            probs = out["probs"][:, 1].cpu().numpy()
            epistemic = out["epistemic_uncertainty"].cpu().numpy()
            aleatoric = out["aleatoric_uncertainty"].cpu().numpy()

            all_probs.append(probs)
            all_labels.append(y)
            all_epistemic.append(epistemic)
            all_aleatoric.append(aleatoric)

            results.append({
                "sample_idx": i,
                "n_nodes": len(y),
                "n_defect": int(y.sum()),
                "mean_defect_prob": float(probs[y > 0].mean()) if y.sum() > 0 else 0,
                "mean_epistemic": float(epistemic.mean()),
                "mean_aleatoric": float(aleatoric.mean()),
            })

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_epistemic = np.concatenate(all_epistemic)
    all_aleatoric = np.concatenate(all_aleatoric)

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    print(f"Overall AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

    # Visualization: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Defect probability histogram
    ax = axes[0]
    ax.hist(
        all_probs[all_labels == 0], bins=50, alpha=0.7, label="Normal",
        color="#2196F3", density=True,
    )
    ax.hist(
        all_probs[all_labels == 1], bins=50, alpha=0.7, label="Defect",
        color="#F44336", density=True,
    )
    ax.set_xlabel("Defect Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"{tag} Predictions (AUROC={auroc:.3f})")
    ax.legend()

    # Panel 2: Epistemic uncertainty by class
    ax = axes[1]
    ax.hist(
        all_epistemic[all_labels == 0], bins=50, alpha=0.7, label="Normal",
        color="#2196F3", density=True,
    )
    ax.hist(
        all_epistemic[all_labels == 1], bins=50, alpha=0.7, label="Defect",
        color="#F44336", density=True,
    )
    ax.set_xlabel("Epistemic Uncertainty")
    ax.set_ylabel("Density")
    ax.set_title("Epistemic Uncertainty Distribution")
    ax.legend()

    # Panel 3: Spatial plot (first sample)
    ax = axes[2]
    data0 = val_data[0]
    sub0 = subsample_graph(data0, max_nodes=max_nodes, seed=0)
    with torch.no_grad():
        if is_physics:
            ea0 = sub0.edge_attr.to(device) if hasattr(sub0, "edge_attr") and sub0.edge_attr is not None else None
            out0 = model.predict_with_uncertainty(
                sub0.x.to(device), sub0.edge_index.to(device), ea0
            )
        else:
            out0 = model.predict_with_uncertainty(
                sub0.x.to(device), sub0.edge_index.to(device)
            )
    pos = sub0.pos.numpy() if sub0.pos is not None else sub0.x[:, :3].numpy()
    probs0 = out0["probs"][:, 1].cpu().numpy()
    sc = ax.scatter(
        pos[:, 0], pos[:, 1], c=probs0, cmap="RdYlBu_r", s=2, alpha=0.8,
    )
    plt.colorbar(sc, ax=ax, label="P(defect)")
    y0 = sub0.y.numpy()
    if y0.sum() > 0:
        ax.scatter(
            pos[y0 > 0, 0], pos[y0 > 0, 1],
            facecolors="none", edgecolors="red", s=30, linewidths=1.5,
            label="True defect",
        )
        ax.legend()
    ax.set_title("Spatial Defect Probability (Sample 0)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    plt.tight_layout()
    fig_name = "physics_gpn_results.png" if is_physics else "gpn_results.png"
    fig_path = output_dir / fig_name
    fig.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")
    plt.close()

    # Save metrics
    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "n_nodes_total": int(len(all_labels)),
        "n_defect_total": int(all_labels.sum()),
        "per_sample": results,
    }
    metrics_name = "physics_gpn_metrics.json" if is_physics else "gpn_metrics.json"
    with open(output_dir / metrics_name, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {output_dir / metrics_name}")


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Graph Posterior Network for SHM Defect Detection"
    )
    parser.add_argument(
        "--mode", choices=["train", "eval"], default="train",
    )
    parser.add_argument(
        "--data_dir", type=Path,
        default=Path("data/processed_s12_thermal_500"),
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("runs/gpn"),
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--physics", action="store_true",
                        help="Use Physics-Informed GPN with learnable edge weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--dim_hidden", type=int, default=64)
    parser.add_argument("--dim_latent", type=int, default=16)
    parser.add_argument("--radial_layers", type=int, default=10)
    parser.add_argument("--appnp_K", type=int, default=10)
    parser.add_argument("--appnp_alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_nodes", type=int, default=4000)
    parser.add_argument("--max_train", type=int, default=50)
    parser.add_argument("--max_val", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "train":
        train_gpn(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            dim_hidden=args.dim_hidden,
            dim_latent=args.dim_latent,
            radial_layers=args.radial_layers,
            appnp_K=args.appnp_K,
            appnp_alpha=args.appnp_alpha,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            lr=args.lr,
            max_nodes=args.max_nodes,
            max_train=args.max_train,
            max_val=args.max_val,
            physics=args.physics,
        )
    elif args.mode == "eval":
        default_ckpt = "best_physics_gpn.pt" if args.physics else "best_gpn.pt"
        ckpt_path = args.checkpoint or (args.output_dir / default_ckpt)
        infer_with_uncertainty(
            checkpoint_path=ckpt_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            max_nodes=args.max_nodes,
        )


if __name__ == "__main__":
    main()
