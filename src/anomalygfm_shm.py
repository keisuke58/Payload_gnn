#!/usr/bin/env python3
"""AnomalyGFM: Graph Foundation Model for SHM Node-level Anomaly Detection.

Zero-shot / few-shot graph anomaly detection using the KDD2025 AnomalyGFM
pre-trained model. Converts PyG graphs from our pipeline to the expected
format, runs inference, and outputs per-node anomaly scores.

Modes:
    zero-shot:  Random init → inference (baseline)
    finetune:   Train on labeled train.pt → inference on val.pt
    infer:      Load fine-tuned checkpoint → inference

For large graphs (N > 5000 nodes), uses node subsampling during training.

Usage:
    python src/anomalygfm_shm.py --data_dir data/processed_s12_thermal_500
    python src/anomalygfm_shm.py --mode finetune --data_dir data/processed_s12_thermal_500
    python src/anomalygfm_shm.py --mode infer --checkpoint results/anomalygfm_shm/best_model.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── AnomalyGFM Model (reimplemented for portability) ─────────────────


class GCNLayer(nn.Module):
    """Graph Convolutional layer (dense adjacency)."""

    def __init__(self, in_ft: int, out_ft: int, act: str = "prelu"):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == "prelu" else nn.ReLU()
        self.bias = nn.Parameter(torch.zeros(out_ft))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D_in), adj: (B, N, N)
        h = self.fc(x)            # (B, N, D_out)
        h = torch.bmm(adj, h)     # graph convolution
        return self.act(h + self.bias)


class AnomalyGFMModel(nn.Module):
    """AnomalyGFM: Graph Foundation Model for anomaly detection.

    Architecture:
        Input (D_in) → fc_map → (D_internal) → GCN1 → GCN2 → (D_emb)
        + Normal/Abnormal prompt prototypes for scoring
    """

    def __init__(
        self,
        n_in: int,
        n_internal: int = 8,
        n_emb: int = 400,
        act: str = "prelu",
    ):
        super().__init__()
        self.fc_map = nn.Linear(n_in, n_internal)
        self.gcn1 = GCNLayer(n_internal, n_emb, act)
        self.gcn2 = GCNLayer(n_emb, n_emb, act)
        self.fc_normal_prompt = nn.Linear(n_emb, n_emb)
        self.fc_abnormal_prompt = nn.Linear(n_emb, n_emb)
        self.fc_score = nn.Linear(n_emb, 1)
        self.fc_residual_score = nn.Linear(n_emb, 1)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        raw_adj: torch.Tensor,
        normal_prompt: torch.Tensor,
        abnormal_prompt: torch.Tensor,
    ):
        """
        Args:
            x:       (1, N, D_in) node features
            adj:     (1, N, N) normalized adjacency (with self-loops)
            raw_adj: (N, N) raw adjacency
            normal_prompt:   (D_emb,) random prompt
            abnormal_prompt: (D_emb,) random prompt

        Returns:
            ano_scores: (N,) per-node anomaly scores
            emb:        (N, D_emb) node embeddings
        """
        h = self.fc_map(x)        # (1, N, D_internal)
        h = self.gcn1(h, adj)      # (1, N, D_emb)
        emb = self.gcn2(h, adj)    # (1, N, D_emb)

        # Prompts
        np_out = F.relu(self.fc_normal_prompt(normal_prompt))
        ap_out = F.relu(self.fc_abnormal_prompt(abnormal_prompt))

        # Residual features: node embedding - neighbor mean
        degree = raw_adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = raw_adj / degree
        emb_neighbors = torch.bmm(adj_norm.unsqueeze(0), emb)
        emb_residual = emb - emb_neighbors  # (1, N, D_emb)

        # Anomaly scoring via prompt similarity
        emb_r = F.normalize(emb_residual.squeeze(0), p=2, dim=1)  # (N, D_emb)
        np_norm = F.normalize(np_out, p=2, dim=0)
        ap_norm = F.normalize(ap_out, p=2, dim=0)

        score_normal = emb_r @ np_norm   # (N,)
        score_abnormal = emb_r @ ap_norm  # (N,)

        ano_scores = torch.exp(score_abnormal) + 6.0 * torch.exp(-score_normal)

        return ano_scores, emb.squeeze(0)


# ── Data conversion utilities ────────────────────────────────────────


def pyg_to_dense(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert PyG Data to dense feature matrix and adjacency matrix.

    Returns:
        features: (N, D) numpy array
        adj:      (N, N) scipy sparse matrix
        labels:   (N,) numpy array (0/1)
    """
    features = data.x.numpy()
    N = features.shape[0]

    # Build adjacency from edge_index
    edge_index = data.edge_index.numpy()
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(N, N),
    )
    adj = adj.tocsr()

    labels = data.y.numpy() if hasattr(data, "y") else np.zeros(N)
    # Binarize labels (any defect class → 1)
    labels = (labels > 0).astype(np.float32)

    return features, adj, labels


def normalize_adj(adj: sp.spmatrix) -> np.ndarray:
    """Symmetric normalization: D^{-1/2} A D^{-1/2} + I."""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)
    return (D @ adj @ D).toarray().astype(np.float32)


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Row-normalize feature matrix."""
    rowsum = np.abs(features).sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    return (features / rowsum).astype(np.float32)


# ── Node subsampling for large graphs ─────────────────────────────────


def subsample_graph(data, max_nodes: int = 3000, seed: int = 0):
    """Subsample nodes from a large graph, keeping ALL defect nodes.

    Returns a new PyG-like data object with subsampled nodes and
    re-indexed edges.
    """
    N = data.x.shape[0]
    if N <= max_nodes:
        return data

    labels = data.y.numpy() if hasattr(data, "y") else np.zeros(N)
    defect_idx = np.where(labels > 0)[0]
    normal_idx = np.where(labels == 0)[0]

    n_normal = max_nodes - len(defect_idx)
    if n_normal < 0:
        n_normal = max_nodes

    rng = np.random.RandomState(seed)
    sampled_normal = rng.choice(normal_idx, size=min(n_normal, len(normal_idx)),
                                replace=False)
    keep_idx = np.sort(np.concatenate([defect_idx, sampled_normal]))

    # Remap
    old_to_new = -np.ones(N, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(len(keep_idx))

    # Filter edges
    edge_index = data.edge_index.numpy()
    src, dst = edge_index[0], edge_index[1]
    mask = (old_to_new[src] >= 0) & (old_to_new[dst] >= 0)
    new_src = old_to_new[src[mask]]
    new_dst = old_to_new[dst[mask]]

    import copy
    sub = copy.copy(data)
    sub.x = data.x[keep_idx]
    sub.y = data.y[keep_idx] if hasattr(data, "y") else torch.zeros(len(keep_idx))
    sub.edge_index = torch.tensor(np.stack([new_src, new_dst]), dtype=torch.long)
    if hasattr(data, "pos") and data.pos is not None:
        sub.pos = data.pos[keep_idx]
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        sub.edge_attr = data.edge_attr[mask]
    return sub


# ── Fine-tuning ──────────────────────────────────────────────────────


def finetune_anomalygfm(
    train_data: list,
    val_data: list,
    output_dir: Path,
    device: str = "cuda",
    n_emb: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    max_nodes: int = 3000,
    max_train_samples: int = 30,
    max_val_samples: int = 10,
):
    """Fine-tune AnomalyGFM on labeled graph data.

    Loss: BCE on per-node anomaly scores vs binary defect labels.
    Node subsampling to handle 15K-node graphs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sub = train_data[:max_train_samples]
    val_sub = val_data[:max_val_samples]

    D = train_sub[0].x.shape[1]
    model = AnomalyGFMModel(n_in=D, n_internal=16, n_emb=n_emb).to(device)

    # Learnable prompts instead of random
    normal_prompt = nn.Parameter(torch.randn(n_emb, device=device) * 0.1)
    abnormal_prompt = nn.Parameter(torch.randn(n_emb, device=device) * 0.1)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + [normal_prompt, abnormal_prompt],
        lr=lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auroc = 0.0
    patience_counter = 0

    print(f"Fine-tuning: {len(train_sub)} train, {len(val_sub)} val, "
          f"n_emb={n_emb}, max_nodes={max_nodes}")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        n_graphs = 0

        for i, data in enumerate(train_sub):
            sub = subsample_graph(data, max_nodes=max_nodes, seed=epoch * 100 + i)
            features, adj_sparse, labels = pyg_to_dense(sub)
            feat_norm = normalize_features(features)
            adj_norm = normalize_adj(adj_sparse)

            feat_t = torch.FloatTensor(feat_norm[None]).to(device)
            adj_t = torch.FloatTensor(adj_norm[None]).to(device)
            raw_adj_t = torch.FloatTensor(adj_sparse.toarray()).to(device)
            labels_t = torch.FloatTensor(labels).to(device)

            optimizer.zero_grad()
            scores, _ = model(feat_t, adj_t, raw_adj_t,
                              normal_prompt, abnormal_prompt)

            # Normalize scores to [0, 1] for BCE
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

            # Focal-style weighting for class imbalance
            n_pos = labels_t.sum().clamp(min=1)
            n_neg = (1 - labels_t).sum().clamp(min=1)
            pos_weight = n_neg / n_pos

            loss = F.binary_cross_entropy(
                scores_norm, labels_t,
                weight=labels_t * pos_weight + (1 - labels_t),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_graphs += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_graphs, 1)

        # ── Validate ──
        if epoch % 5 == 0 or epoch == 1:
            val_results = run_anomalygfm_with_model(
                model, val_sub, normal_prompt, abnormal_prompt,
                device=device, max_nodes=20000,
            )
            metrics = evaluate_results(val_results)
            auroc = metrics.get("auroc", 0.0)
            auprc = metrics.get("auprc", 0.0)

            print(f"  Epoch {epoch:3d}/{epochs} | Loss={avg_loss:.4f} | "
                  f"Val AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

            if auroc > best_val_auroc:
                best_val_auroc = auroc
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "normal_prompt": normal_prompt.data,
                    "abnormal_prompt": abnormal_prompt.data,
                    "n_in": D,
                    "n_emb": n_emb,
                    "epoch": epoch,
                    "auroc": auroc,
                    "auprc": auprc,
                }, output_dir / "best_model.pt")
            else:
                patience_counter += 5
                if patience_counter >= 25:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    print(f"\nBest Val AUROC: {best_val_auroc:.4f}")
    print(f"Saved: {output_dir / 'best_model.pt'}")
    return best_val_auroc


@torch.no_grad()
def run_anomalygfm_with_model(
    model: nn.Module,
    data_list: list,
    normal_prompt: torch.Tensor,
    abnormal_prompt: torch.Tensor,
    device: str = "cuda",
    max_nodes: int = 20000,
) -> list[dict]:
    """Run inference with a trained model and learned prompts."""
    model.eval()
    results = []

    for i, data in enumerate(data_list):
        N = data.x.shape[0]
        D = data.x.shape[1]

        if N > max_nodes:
            results.append(None)
            continue

        features, adj_sparse, labels = pyg_to_dense(data)
        feat_norm = normalize_features(features)
        adj_norm = normalize_adj(adj_sparse)

        feat_t = torch.FloatTensor(feat_norm[None]).to(device)
        adj_t = torch.FloatTensor(adj_norm[None]).to(device)
        raw_adj_t = torch.FloatTensor(adj_sparse.toarray()).to(device)

        scores, _ = model(feat_t, adj_t, raw_adj_t,
                          normal_prompt, abnormal_prompt)

        n_defect = int(labels.sum())
        results.append({
            "scores": scores.cpu().numpy(),
            "labels": labels,
            "n_nodes": N,
            "n_defect": n_defect,
        })

    return results


# ── Inference (zero-shot) ────────────────────────────────────────────


@torch.no_grad()
def run_anomalygfm(
    data_list: list,
    device: str = "cuda",
    n_runs: int = 5,
    n_emb: int = 400,
    max_nodes: int = 20000,
) -> list[dict]:
    """Run AnomalyGFM zero-shot on a list of PyG graphs.

    Args:
        data_list: List of PyG Data objects
        device:    Inference device
        n_runs:    Number of random prompt runs (averaged)
        n_emb:     Embedding dimension
        max_nodes: Skip graphs larger than this

    Returns:
        List of dicts with {scores, labels, n_nodes, n_defect} per graph
    """
    results = []

    for i, data in enumerate(data_list):
        N = data.x.shape[0]
        D = data.x.shape[1]

        if N > max_nodes:
            print(f"  [{i}] Skip: {N} nodes > max_nodes={max_nodes}")
            results.append(None)
            continue

        features, adj_sparse, labels = pyg_to_dense(data)
        feat_norm = normalize_features(features)
        adj_norm = normalize_adj(adj_sparse)

        feat_t = torch.FloatTensor(feat_norm[None]).to(device)  # (1, N, D)
        adj_t = torch.FloatTensor(adj_norm[None]).to(device)     # (1, N, N)
        raw_adj_t = torch.FloatTensor(adj_sparse.toarray()).to(device)  # (N, N)

        # Build model for this input dimension
        model = AnomalyGFMModel(n_in=D, n_emb=n_emb).to(device)

        # Pre-trained weights: load if available, else random init
        ckpt_path = Path("/tmp/AnomalyGFM/few_shot/model_weights_abnormal300.pth")
        if ckpt_path.exists() and n_emb == 300:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state, strict=False)

        model.eval()

        # Multiple runs with random prompts → average scores
        all_scores = []
        for run in range(n_runs):
            torch.manual_seed(run * 42)
            normal_prompt = torch.randn(n_emb).to(device)
            abnormal_prompt = torch.randn(n_emb).to(device)

            scores, emb = model(feat_t, adj_t, raw_adj_t,
                                normal_prompt, abnormal_prompt)
            all_scores.append(scores.cpu().numpy())

        avg_scores = np.mean(all_scores, axis=0)  # (N,)

        n_defect = int(labels.sum())
        results.append({
            "scores": avg_scores,
            "labels": labels,
            "n_nodes": N,
            "n_defect": n_defect,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(data_list)}] N={N}, defect={n_defect}, "
                  f"score range=[{avg_scores.min():.3f}, {avg_scores.max():.3f}]")

    return results


def evaluate_results(results: list[dict | None]) -> dict:
    """Evaluate anomaly detection performance across all graphs."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    all_scores, all_labels = [], []
    for r in results:
        if r is None:
            continue
        all_scores.append(r["scores"])
        all_labels.append(r["labels"])

    if not all_scores:
        return {"auroc": 0.0, "auprc": 0.0}

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    if labels.sum() == 0 or labels.sum() == len(labels):
        print("Warning: All labels are same class, cannot compute AUROC")
        return {"auroc": 0.0, "auprc": 0.0, "n_nodes": len(labels)}

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "n_nodes": len(labels),
        "n_defect": int(labels.sum()),
        "n_normal": int((labels == 0).sum()),
    }


def visualize_scores(
    data, scores: np.ndarray, labels: np.ndarray, out_path: Path, title: str = ""
):
    """Visualize anomaly scores on graph nodes."""
    import matplotlib.pyplot as plt

    pos = data.pos.numpy()  # (N, 3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Ground truth
    ax = axes[0]
    colors = ["#2196F3" if l == 0 else "#F44336" for l in labels]
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=1, alpha=0.5)
    ax.set_title("Ground Truth Labels")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # Right: Anomaly scores
    ax = axes[1]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=scores, cmap="YlOrRd",
                    s=1, alpha=0.5)
    ax.set_title(f"AnomalyGFM Anomaly Scores{' — ' + title if title else ''}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(sc, ax=ax, label="Anomaly Score")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="AnomalyGFM SHM anomaly detection (zero-shot / fine-tune)"
    )
    parser.add_argument("--mode", choices=["zero-shot", "finetune", "infer"],
                        default="zero-shot")
    parser.add_argument("--data_dir", type=str,
                        default="data/processed_s12_thermal_500")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_runs", type=int, default=10,
                        help="Number of random prompt runs to average (zero-shot)")
    parser.add_argument("--n_emb", type=int, default=128,
                        help="Embedding dimension (128 for finetune, 400 for zero-shot)")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Max graphs to process (for speed)")
    parser.add_argument("--out_dir", type=str, default="results/anomalygfm_shm")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for infer mode")
    # Fine-tune args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_nodes", type=int, default=3000,
                        help="Max nodes per subsampled graph (training)")
    parser.add_argument("--max_train", type=int, default=30)
    parser.add_argument("--max_val", type=int, default=10)
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading graphs from {data_dir}...")

    if args.mode == "finetune":
        train_data = torch.load(data_dir / "train.pt", weights_only=False)
        val_data = torch.load(data_dir / "val.pt", weights_only=False)
        print(f"Train: {len(train_data)}, Val: {len(val_data)}")

        finetune_anomalygfm(
            train_data, val_data, out_dir,
            device=args.device, n_emb=args.n_emb,
            epochs=args.epochs, lr=args.lr,
            max_nodes=args.max_nodes,
            max_train_samples=args.max_train,
            max_val_samples=args.max_val,
        )

        # Run final evaluation on full val set
        print("\n── Final Evaluation (full val set) ──")
        ckpt = torch.load(out_dir / "best_model.pt", map_location=args.device,
                          weights_only=False)
        model = AnomalyGFMModel(
            n_in=ckpt["n_in"], n_internal=16, n_emb=ckpt["n_emb"]
        ).to(args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        np_t = ckpt["normal_prompt"].to(args.device)
        ap_t = ckpt["abnormal_prompt"].to(args.device)

        val_subset = val_data[:args.max_samples]
        results = run_anomalygfm_with_model(
            model, val_subset, np_t, ap_t, device=args.device,
        )

    elif args.mode == "infer":
        ckpt_path = args.checkpoint or str(out_dir / "best_model.pt")
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model = AnomalyGFMModel(
            n_in=ckpt["n_in"], n_internal=16, n_emb=ckpt["n_emb"]
        ).to(args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        np_t = ckpt["normal_prompt"].to(args.device)
        ap_t = ckpt["abnormal_prompt"].to(args.device)

        val_data = torch.load(data_dir / "val.pt", weights_only=False)
        val_subset = val_data[:args.max_samples]
        print(f"Inference: {len(val_subset)} graphs, checkpoint: {ckpt_path}")

        results = run_anomalygfm_with_model(
            model, val_subset, np_t, ap_t, device=args.device,
        )

    else:  # zero-shot
        val_data = torch.load(data_dir / "val.pt", weights_only=False)

        norm_path = data_dir / "norm_stats.pt"
        if norm_path.exists():
            norm = torch.load(norm_path, weights_only=False)
            x_mean, x_std = norm["mean"], norm["std"]
            for d in val_data:
                d.x = (d.x - x_mean) / x_std

        val_subset = val_data[:args.max_samples]
        print(f"Processing {len(val_subset)} graphs "
              f"(features={val_subset[0].x.shape[1]}d, "
              f"nodes~{val_subset[0].x.shape[0]})")

        results = run_anomalygfm(
            val_subset, device=args.device,
            n_runs=args.n_runs, n_emb=args.n_emb,
        )

    # Evaluate
    metrics = evaluate_results(results)
    mode_label = args.mode.replace("-", " ").title()
    print(f"\n── AnomalyGFM {mode_label} Results ──")
    print(f"  AUROC: {metrics.get('auroc', 0):.4f}")
    print(f"  AUPRC: {metrics.get('auprc', 0):.4f}")
    print(f"  Nodes: {metrics.get('n_nodes', 0)} "
          f"(defect={metrics.get('n_defect', 0)}, "
          f"normal={metrics.get('n_normal', 0)})")

    # Visualize first sample with defects
    suffix = "finetuned" if args.mode != "zero-shot" else "zeroshot"
    for i, r in enumerate(results):
        if r and r["n_defect"] > 0:
            vis_path = out_dir / f"anomalygfm_scores_{suffix}_sample{i}.png"
            visualize_scores(val_subset[i], r["scores"], r["labels"],
                             vis_path, f"{mode_label} Sample {i}")
            print(f"Saved: {vis_path}")
            break

    # Save metrics
    import json
    metrics["mode"] = args.mode
    with open(out_dir / f"metrics_{suffix}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {out_dir / f'metrics_{suffix}.json'}")


if __name__ == "__main__":
    main()
