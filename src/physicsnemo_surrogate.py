#!/usr/bin/env python3
"""PhysicsNeMo MeshGraphNet: FEM Surrogate for GW Data Generation.

Trains a MeshGraphNet to learn the mapping:
    (mesh geometry, material, defect params) → (stress, displacement, GW response)

This enables rapid generation of synthetic FEA data without running Abaqus,
solving the data scarcity problem (currently only 100 FEM samples).

Two modes:
    train:  Train MeshGraphNet on existing FEM data
    infer:  Generate new samples with trained model (~1000x faster than FEM)

Usage:
    python src/physicsnemo_surrogate.py --mode train --data_dir data/processed_s12_thermal_500
    python src/physicsnemo_surrogate.py --mode infer --checkpoint runs/surrogate/best.pt --n_samples 500
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── MeshGraphNet Architecture ────────────────────────────────────────


class EdgeModel(nn.Module):
    """Edge update: combines sender, receiver, and edge features."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, src, dst, edge_attr):
        return self.mlp(torch.cat([src, dst, edge_attr], dim=-1))


class NodeModel(nn.Module):
    """Node update: aggregates edge messages and updates node features."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        # Aggregate incoming edge features
        row, col = edge_index
        agg = scatter(edge_attr, col, dim=0, dim_size=x.size(0), reduce="mean")
        return self.mlp(torch.cat([x, agg], dim=-1))


class MeshGraphNetBlock(nn.Module):
    """One message-passing block of MeshGraphNet."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_model = EdgeModel(node_dim, edge_dim, hidden_dim)
        self.node_model = NodeModel(node_dim, hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        # Edge update
        edge_attr_new = self.edge_model(x[row], x[col], edge_attr)
        # Node update
        x_new = self.node_model(x, edge_index, edge_attr_new)
        return x_new, edge_attr_new


class MeshGraphNet(nn.Module):
    """MeshGraphNet for FEM surrogate modeling.

    Encodes mesh nodes and edges, applies N message-passing steps,
    then decodes to predict target fields (stress, displacement, etc.).

    Architecture:
        Node Encoder → Edge Encoder → N × MGN Blocks → Decoder
    """

    def __init__(
        self,
        node_in: int,
        edge_in: int,
        node_out: int,
        hidden_dim: int = 128,
        n_blocks: int = 6,
    ):
        super().__init__()

        # Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing blocks with residual connections
        self.blocks = nn.ModuleList([
            MeshGraphNetBlock(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(n_blocks)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_out),
        )

    def forward(self, x, edge_index, edge_attr):
        # Encode
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        # Message passing with residual
        for block in self.blocks:
            h_new, e_new = block(h, edge_index, e)
            h = h + h_new  # residual
            e = e + e_new

        # Decode
        return self.decoder(h)

    def encode(self, x, edge_index, edge_attr):
        """Extract node embeddings (for use as features in other models)."""
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for block in self.blocks:
            h_new, e_new = block(h, edge_index, e)
            h = h + h_new
            e = e + e_new
        return h


# ── Training ─────────────────────────────────────────────────────────


def split_features_targets(data, target_dims: list[int] | None = None):
    """Split node features into input (geometry+boundary) and target (FEA results).

    Default split for 34-dim features:
        Input  (10 dims): position(3), normal(3), curvature(4)
        Target (24 dims): displacement(4), temp(1), stress(5), strain(3),
                         fiber(3), layup(5), boundary(2)

    For surrogate: predict FEA outputs from geometry alone.
    """
    if target_dims is None:
        # Geometry + material as input (known before FEA)
        # Position(3) + Normal(3) + Curvature(4) + Fiber(3) + Layup(5) + Boundary(2) = 20
        input_idx = list(range(0, 10)) + list(range(27, 37))  # geom + material + boundary
        # FEA results as target (what we want to predict)
        # Displacement(4) + Temp(1) + Stress(5) + Strain(3) = 13
        target_idx = list(range(10, 23))

        # Clamp to actual feature dimensions
        D = data.x.shape[1]
        input_idx = [i for i in input_idx if i < D]
        target_idx = [i for i in target_idx if i < D]
    else:
        D = data.x.shape[1]
        target_idx = target_dims
        input_idx = [i for i in range(D) if i not in target_idx]

    return input_idx, target_idx


def train_surrogate(
    data_dir: Path,
    output_dir: Path,
    hidden_dim: int = 128,
    n_blocks: int = 6,
    epochs: int = 200,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """Train MeshGraphNet surrogate model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data = torch.load(data_dir / "train.pt", weights_only=False)
    val_data = torch.load(data_dir / "val.pt", weights_only=False)
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # Determine input/target split
    sample = train_data[0]
    input_idx, target_idx = split_features_targets(sample)
    node_in = len(input_idx)
    node_out = len(target_idx)
    edge_in = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
    print(f"Input dims: {node_in} | Target dims: {node_out} | Edge dims: {edge_in}")

    # Prepare data: split features
    for d in train_data + val_data:
        d.x_input = d.x[:, input_idx]
        d.x_target = d.x[:, target_idx]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Model
    model = MeshGraphNet(
        node_in=node_in, edge_in=edge_in, node_out=node_out,
        hidden_dim=hidden_dim, n_blocks=n_blocks,
    ).to(device)
    print(f"MeshGraphNet: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x_input, batch.edge_index, batch.edge_attr)
            loss = F.mse_loss(pred, batch.x_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_data)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x_input, batch.edge_index, batch.edge_attr)
                loss = F.mse_loss(pred, batch.x_target)
                val_loss += loss.item() * batch.num_graphs
        val_loss /= len(val_data)

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train MSE={train_loss:.6f} | Val MSE={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "node_in": node_in,
                "node_out": node_out,
                "edge_in": edge_in,
                "hidden_dim": hidden_dim,
                "n_blocks": n_blocks,
                "input_idx": input_idx,
                "target_idx": target_idx,
            }, output_dir / "best_surrogate.pt")
        else:
            patience_counter += 1
            if patience_counter >= 30:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nBest Val MSE: {best_val_loss:.6f}")
    print(f"Saved: {output_dir / 'best_surrogate.pt'}")


@torch.no_grad()
def generate_samples(
    checkpoint_path: Path,
    template_data_dir: Path,
    n_samples: int = 100,
    device: str = "cuda",
) -> list:
    """Generate synthetic FEM samples using trained surrogate."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = MeshGraphNet(
        node_in=ckpt["node_in"], edge_in=ckpt["edge_in"],
        node_out=ckpt["node_out"],
        hidden_dim=ckpt["hidden_dim"], n_blocks=ckpt["n_blocks"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load template graphs (use real mesh, predict new FEA fields)
    templates = torch.load(template_data_dir / "train.pt", weights_only=False)
    input_idx = ckpt["input_idx"]
    target_idx = ckpt["target_idx"]

    generated = []
    for i in range(n_samples):
        # Pick a random template and add perturbation
        template = templates[i % len(templates)].clone()
        x_input = template.x[:, input_idx].to(device)

        # Add small noise to geometry (data augmentation)
        noise = torch.randn_like(x_input) * 0.01
        x_input_perturbed = x_input + noise

        # Predict FEA fields
        pred = model(
            x_input_perturbed,
            template.edge_index.to(device),
            template.edge_attr.to(device) if template.edge_attr is not None else None,
        )

        # Reconstruct full feature vector
        x_full = template.x.clone()
        x_full[:, input_idx] = x_input_perturbed.cpu()
        x_full[:, target_idx] = pred.cpu()
        template.x = x_full

        generated.append(template)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_samples}")

    return generated


def main():
    parser = argparse.ArgumentParser(
        description="PhysicsNeMo MeshGraphNet FEM Surrogate"
    )
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--data_dir", type=str,
                        default="data/processed_s12_thermal_500")
    parser.add_argument("--output_dir", type=str, default="runs/surrogate")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of synthetic samples to generate (infer mode)")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir

    if args.mode == "train":
        train_surrogate(
            data_dir, output_dir,
            hidden_dim=args.hidden_dim, n_blocks=args.n_blocks,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, device=args.device,
        )
    else:
        ckpt = args.checkpoint or str(output_dir / "best_surrogate.pt")
        generated = generate_samples(
            Path(ckpt), data_dir,
            n_samples=args.n_samples, device=args.device,
        )
        out_path = output_dir / "synthetic_data.pt"
        torch.save(generated, out_path)
        print(f"Saved {len(generated)} synthetic samples: {out_path}")


if __name__ == "__main__":
    main()
