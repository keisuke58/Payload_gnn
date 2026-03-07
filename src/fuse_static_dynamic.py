#!/usr/bin/env python3
"""Fuse static FEA graph features with dynamic GW Chronos embeddings.

Takes existing PyG graph data (34-dim node features) and augments each
node with projected Chronos-Bolt time-series embeddings via sensor-to-node
interpolation (Inverse Distance Weighting on fairing geometry).

Pipeline:
    1. Load processed graphs (train.pt, val.pt)
    2. Load pre-computed Chronos embeddings (.pt per sample)
    3. Fit PCA on all sensor embeddings → project 768 → proj_dim
    4. For each graph node, interpolate sensor embeddings via IDW
    5. Concatenate: x_fused = [x_static (34) | x_dynamic (proj_dim)]
    6. Save augmented dataset

Sensor-to-node mapping modes:
    uniform:  All nodes receive the mean of all sensor embeddings (simplest)
    idw:      Inverse Distance Weighting based on 3D coordinates (recommended)
    nearest:  Each node receives the nearest sensor's embedding

Usage:
    python src/fuse_static_dynamic.py \\
        --graph-dir data/processed_s12_thermal_500 \\
        --emb-dir data/chronos_embeddings \\
        --out-dir data/processed_hybrid_500

    # With more sensors in future:
    python src/fuse_static_dynamic.py \\
        --emb-dir data/chronos_embeddings_v2 \\
        --proj-dim 64 --interpolation idw
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fairing geometry constants (from generate_gw_fairing.py)
FAIRING_RADIUS = 2600.0    # mm, inner skin radius
FAIRING_CORE_T = 38.0      # mm, honeycomb core thickness
FAIRING_R_OUTER = FAIRING_RADIUS + FAIRING_CORE_T  # 2638 mm


def load_chronos_embeddings(emb_dir: Path) -> dict[str, dict]:
    """Load all pre-computed Chronos embeddings from directory.

    Returns dict keyed by sample_id:
        {sample_id: {embeddings, sensor_positions, n_sensors, ...}}
    """
    emb_files = sorted(emb_dir.glob("*.pt"))
    if not emb_files:
        raise FileNotFoundError(f"No .pt files in {emb_dir}")

    all_emb = {}
    for f in emb_files:
        data = torch.load(f, weights_only=False)
        sid = data.get("sample_id", f.stem)
        all_emb[sid] = data

    print(f"Loaded {len(all_emb)} Chronos embeddings from {emb_dir}")
    return all_emb


def fit_pca_projection(
    all_emb: dict[str, dict], proj_dim: int
) -> tuple[PCA, np.ndarray, np.ndarray]:
    """Fit PCA on all sensor embeddings to project 768 → proj_dim.

    Returns:
        pca: fitted PCA model
        emb_mean: (768,) mean for centering
        emb_std:  (768,) std for scaling
    """
    # Collect all sensor embeddings
    all_vectors = []
    for data in all_emb.values():
        emb = data["embeddings"]  # (n_sensors, 768)
        if isinstance(emb, torch.Tensor):
            emb = emb.numpy()
        all_vectors.append(emb)

    all_vectors = np.concatenate(all_vectors, axis=0)  # (total_sensors, 768)
    print(f"PCA: fitting on {all_vectors.shape[0]} sensor embeddings "
          f"({all_vectors.shape[1]} → {proj_dim})")

    # Standardize before PCA
    emb_mean = all_vectors.mean(axis=0)
    emb_std = all_vectors.std(axis=0)
    emb_std[emb_std < 1e-8] = 1.0
    all_vectors_norm = (all_vectors - emb_mean) / emb_std

    pca = PCA(n_components=proj_dim, random_state=42)
    pca.fit(all_vectors_norm)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {proj_dim} components explain {explained:.1%} of variance")

    return pca, emb_mean, emb_std


def project_embeddings(
    emb: np.ndarray, pca: PCA, emb_mean: np.ndarray, emb_std: np.ndarray
) -> np.ndarray:
    """Project sensor embeddings from 768-dim to proj_dim.

    Args:
        emb: (n_sensors, 768)
    Returns:
        projected: (n_sensors, proj_dim)
    """
    emb_norm = (emb - emb_mean) / emb_std
    return pca.transform(emb_norm).astype(np.float32)


def estimate_sensor_3d_coords(
    sensor_positions: np.ndarray,
    excitation_theta: float = 0.0,
    excitation_z: float = 1500.0,
    direction: str = "circumferential",
) -> np.ndarray:
    """Estimate sensor 3D coordinates from arc distances on fairing.

    For circumferential sensors:
        theta_sensor = excitation_theta + arc_dist / R_outer
        x = R * cos(theta), y = z_axial, z = R * sin(theta)

    Args:
        sensor_positions: (n_sensors,) arc distances in mm
        excitation_theta: excitation point angle (rad)
        excitation_z: excitation point axial position (mm)
        direction: "circumferential" or "axial"

    Returns:
        coords_3d: (n_sensors, 3) estimated [x, y, z]
    """
    n = len(sensor_positions)
    coords = np.zeros((n, 3), dtype=np.float32)
    R = FAIRING_R_OUTER

    if direction == "circumferential":
        for i, arc in enumerate(sensor_positions):
            d_theta = arc / R
            theta = excitation_theta + d_theta
            coords[i] = [R * np.cos(theta), excitation_z, R * np.sin(theta)]
    else:  # axial
        for i, arc in enumerate(sensor_positions):
            coords[i] = [
                R * np.cos(excitation_theta),
                excitation_z + arc,
                R * np.sin(excitation_theta),
            ]

    return coords


def interpolate_to_nodes(
    node_coords: np.ndarray,
    sensor_coords: np.ndarray,
    sensor_features: np.ndarray,
    method: str = "idw",
    idw_power: float = 2.0,
    idw_eps: float = 1.0,
) -> np.ndarray:
    """Interpolate sensor features to graph nodes.

    Args:
        node_coords:    (N, 3) graph node positions
        sensor_coords:  (n_sensors, 3) sensor positions
        sensor_features: (n_sensors, D) projected embeddings
        method:         "uniform", "idw", or "nearest"
        idw_power:      power for inverse distance weighting
        idw_eps:        small epsilon to avoid division by zero (mm)

    Returns:
        node_features: (N, D) interpolated features per node
    """
    N = node_coords.shape[0]
    n_sensors, D = sensor_features.shape

    if method == "uniform":
        # Every node gets the same mean embedding
        mean_emb = sensor_features.mean(axis=0, keepdims=True)  # (1, D)
        return np.broadcast_to(mean_emb, (N, D)).copy()

    # Compute distances: (N, n_sensors)
    # Use broadcasting for efficiency
    diffs = node_coords[:, None, :] - sensor_coords[None, :, :]  # (N, S, 3)
    dists = np.linalg.norm(diffs, axis=2)  # (N, S)

    if method == "nearest":
        nearest_idx = dists.argmin(axis=1)  # (N,)
        return sensor_features[nearest_idx]  # (N, D)

    # IDW
    weights = 1.0 / (dists + idw_eps) ** idw_power  # (N, S)
    weights /= weights.sum(axis=1, keepdims=True)    # normalize
    return (weights @ sensor_features).astype(np.float32)  # (N, D)


def augment_graph(
    graph_data: torch.Tensor,  # PyG Data object
    sensor_emb_projected: np.ndarray,  # (n_sensors, proj_dim)
    sensor_coords_3d: np.ndarray,      # (n_sensors, 3)
    method: str = "idw",
) -> torch.Tensor:
    """Augment a single PyG graph with interpolated sensor embeddings.

    x_fused = [x_static (34-dim) | x_dynamic (proj_dim)]
    """
    node_coords = graph_data.pos.numpy()  # (N, 3)
    node_dynamic = interpolate_to_nodes(
        node_coords, sensor_coords_3d, sensor_emb_projected, method=method
    )
    x_dynamic = torch.tensor(node_dynamic, dtype=torch.float32)

    # Concatenate: (N, 34) + (N, proj_dim) = (N, 34+proj_dim)
    graph_data.x = torch.cat([graph_data.x, x_dynamic], dim=1)

    # Store metadata
    graph_data.n_static_features = 34
    graph_data.n_dynamic_features = sensor_emb_projected.shape[1]

    return graph_data


def create_sample_mapping(
    graph_list: list,
    emb_dict: dict[str, dict],
    mapping_mode: str = "auto",
) -> list[str | None]:
    """Create mapping from graph index to Chronos embedding sample_id.

    Args:
        graph_list: list of PyG Data objects
        emb_dict: dict of {sample_id: embedding_data}
        mapping_mode: "auto" (try index-based), "index" (strict ordering)

    Returns:
        List of sample_ids (or None if no match) for each graph
    """
    n_graphs = len(graph_list)
    available_ids = sorted(emb_dict.keys())

    # Strategy: if the number of healthy embeddings matches the number of
    # graphs, assume they correspond by index. Otherwise, try name matching.
    healthy_ids = [sid for sid in available_ids if sid.startswith("healthy_")]
    defect_ids = [sid for sid in available_ids if sid.startswith("defect_")]

    # For now, use a simple approach:
    # - If graph has a sample_id attribute, use it
    # - Otherwise, assign healthy embeddings in order
    mapping = []
    healthy_idx = 0
    for i, g in enumerate(graph_list):
        sid = getattr(g, "sample_id", None)
        if sid and sid in emb_dict:
            mapping.append(sid)
        elif healthy_idx < len(healthy_ids):
            mapping.append(healthy_ids[healthy_idx])
            healthy_idx += 1
        else:
            mapping.append(None)

    n_matched = sum(1 for m in mapping if m is not None)
    print(f"Mapped {n_matched}/{n_graphs} graphs to Chronos embeddings")
    return mapping


def fuse_dataset(
    graph_dir: Path,
    emb_dir: Path,
    out_dir: Path,
    proj_dim: int = 32,
    interpolation: str = "idw",
) -> None:
    """Main fusion pipeline."""
    graph_dir = PROJECT_ROOT / graph_dir if not graph_dir.is_absolute() else graph_dir
    emb_dir = PROJECT_ROOT / emb_dir if not emb_dir.is_absolute() else emb_dir
    out_dir = PROJECT_ROOT / out_dir if not out_dir.is_absolute() else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Chronos embeddings
    all_emb = load_chronos_embeddings(emb_dir)

    # 2. Fit PCA projection
    pca, emb_mean, emb_std = fit_pca_projection(all_emb, proj_dim)

    # 3. Pre-compute projected embeddings and sensor 3D coords
    projected_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid, data in all_emb.items():
        emb = data["embeddings"]
        if isinstance(emb, torch.Tensor):
            emb = emb.numpy()
        proj = project_embeddings(emb, pca, emb_mean, emb_std)

        sensor_pos = data["sensor_positions"]
        if isinstance(sensor_pos, torch.Tensor):
            sensor_pos = sensor_pos.numpy()
        coords_3d = estimate_sensor_3d_coords(sensor_pos)

        projected_cache[sid] = (proj, coords_3d)

    # 4. Compute a "default" embedding for graphs without GW data
    # (mean of all projected embeddings, zero-padded)
    all_proj = np.concatenate([p for p, _ in projected_cache.values()], axis=0)
    default_proj = all_proj.mean(axis=0)  # (proj_dim,)

    # 5. Process train and val splits
    for split in ["train", "val", "test"]:
        split_path = graph_dir / f"{split}.pt"
        if not split_path.exists():
            continue

        print(f"\nProcessing {split}...")
        graph_list = torch.load(split_path, weights_only=False)
        mapping = create_sample_mapping(graph_list, all_emb)

        for i, (g, sid) in enumerate(zip(graph_list, mapping)):
            if sid and sid in projected_cache:
                proj, sensor_3d = projected_cache[sid]
            else:
                # No GW data — use default (uniform broadcast)
                proj = default_proj[None, :]  # (1, proj_dim)
                sensor_3d = np.zeros((1, 3), dtype=np.float32)
                method = "uniform"
            method_use = interpolation if sid else "uniform"

            augment_graph(g, proj, sensor_3d, method=method_use)

        # Recompute normalization stats (extended features)
        if split == "train":
            x_cat = torch.cat([g.x for g in graph_list], dim=0)
            mean = x_cat.mean(dim=0)
            std = x_cat.std(dim=0)
            std[std < 1e-8] = 1.0
            norm_stats = {"mean": mean, "std": std}
            torch.save(norm_stats, out_dir / "norm_stats.pt")
            print(f"Norm stats: {mean.shape[0]} dims")

        torch.save(graph_list, out_dir / f"{split}.pt")
        print(f"Saved {split}.pt: {len(graph_list)} graphs, "
              f"features={graph_list[0].x.shape[1]} dims")

    # Save fusion metadata
    meta = {
        "proj_dim": proj_dim,
        "interpolation": interpolation,
        "n_static": 34,
        "n_dynamic": proj_dim,
        "n_total": 34 + proj_dim,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "pca_components": pca.components_.shape,
        "n_embeddings": len(all_emb),
        "sensor_counts": {sid: d["n_sensors"] for sid, d in all_emb.items()},
    }
    torch.save(meta, out_dir / "fusion_meta.pt")
    torch.save(
        {"pca": pca, "emb_mean": emb_mean, "emb_std": emb_std},
        out_dir / "pca_model.pt",
    )
    print(f"\nFusion complete: {out_dir}")
    print(f"  Static features:  34 dims")
    print(f"  Dynamic features: {proj_dim} dims (PCA from 768)")
    print(f"  Total features:   {34 + proj_dim} dims")


def main():
    parser = argparse.ArgumentParser(
        description="Fuse static FEA graphs with dynamic GW Chronos embeddings"
    )
    parser.add_argument(
        "--graph-dir", type=Path,
        default=Path("data/processed_s12_thermal_500"),
        help="Directory with train.pt, val.pt (existing PyG graphs)",
    )
    parser.add_argument(
        "--emb-dir", type=Path,
        default=Path("data/chronos_embeddings"),
        help="Directory with Chronos embedding .pt files",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/processed_hybrid"),
        help="Output directory for augmented dataset",
    )
    parser.add_argument(
        "--proj-dim", type=int, default=32,
        help="PCA projection dimension for Chronos embeddings (default: 32)",
    )
    parser.add_argument(
        "--interpolation", choices=["uniform", "idw", "nearest"],
        default="idw",
        help="Sensor-to-node interpolation method (default: idw)",
    )
    args = parser.parse_args()

    fuse_dataset(
        args.graph_dir, args.emb_dir, args.out_dir,
        args.proj_dim, args.interpolation,
    )


if __name__ == "__main__":
    main()
