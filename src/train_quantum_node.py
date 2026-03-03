# -*- coding: utf-8 -*-
"""
Node-Level Quantum Defect Classification

Extracts defect/healthy node features from the 34-dim dataset,
reduces to n_qubits via PCA or MLP, and classifies with VQC vs MLP.

This bypasses the GNN entirely to test quantum ML in isolation on
physically meaningful features (stress, displacement, strain).

Usage:
    python src/train_quantum_node.py --mode stats --epochs 200
    python src/train_quantum_node.py --mode stats --n_qubits 6
"""

import os
import sys
import json
import time
import argparse
import csv as csv_module
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA

from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from models_quantum import VQCLayer


# =========================================================================
# Data preparation
# =========================================================================
def extract_node_dataset(data_list, n_defect_per_graph=None, healthy_ratio=5,
                         seed=42):
    """Extract balanced defect/healthy node features from PyG graph list.

    Args:
        data_list: List of PyG Data objects.
        n_defect_per_graph: Max defect nodes per graph (None = all).
        healthy_ratio: Healthy nodes per defect node for downsampling.
        seed: Random seed.
    Returns:
        X: (N, D) node features, y: (N,) labels.
    """
    rng = np.random.RandomState(seed)
    all_x, all_y = [], []

    for g in data_list:
        defect_mask = g.y > 0
        defect_idx = defect_mask.nonzero(as_tuple=True)[0]
        healthy_idx = (~defect_mask).nonzero(as_tuple=True)[0]

        if len(defect_idx) == 0:
            continue

        # Sample defect nodes
        n_d = len(defect_idx) if n_defect_per_graph is None else min(n_defect_per_graph, len(defect_idx))
        if n_d < len(defect_idx):
            sel = rng.choice(len(defect_idx), n_d, replace=False)
            defect_idx = defect_idx[sel]

        # Sample healthy nodes (balanced)
        n_h = min(n_d * healthy_ratio, len(healthy_idx))
        if n_h > 0:
            sel_h = rng.choice(len(healthy_idx), n_h, replace=False)
            healthy_idx = healthy_idx[sel_h]
        else:
            healthy_idx = healthy_idx[:0]

        idx = torch.cat([defect_idx, healthy_idx])
        all_x.append(g.x[idx])
        all_y.append(torch.cat([
            torch.ones(len(defect_idx), dtype=torch.long),
            torch.zeros(len(healthy_idx), dtype=torch.long),
        ]))

    X = torch.cat(all_x, dim=0)
    y = torch.cat(all_y, dim=0)
    return X, y


def select_features(X, feature_dims=None):
    """Select physically meaningful feature dimensions.

    Default: stress (15-19), displacement (10-13), strain (21-23),
    thermal stress (20), position-derived (0-2).
    """
    if feature_dims is None:
        # Key physics features with clear defect/healthy discrimination
        feature_dims = [
            0, 1, 2,       # position (x, y, z)
            10, 11, 12, 13, # displacement (ux, uy, uz, u_mag)
            15, 16, 17, 18, # stress (s11, s22, s12, smises)
            19,             # principal_stress_sum
            20,             # thermal_smises
            21, 22, 23,     # strain (le11, le22, le12)
        ]
    return X[:, feature_dims], feature_dims


# =========================================================================
# Models
# =========================================================================
class QuantumNodeClassifier(nn.Module):
    """PCA/MLP dim reduction → VQC for node-level defect classification."""

    def __init__(self, in_features, n_qubits=4, vqc_reps=2):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_qubits),
            nn.Tanh(),
        )
        self.scale = nn.Parameter(torch.tensor(np.pi), requires_grad=False)
        self.vqc = VQCLayer(n_qubits, vqc_reps)

    def forward(self, x):
        z = self.pre(x) * self.scale  # (B, n_qubits) ∈ [-π, π]
        return self.vqc(z)            # (B, 2)


class ClassicalNodeClassifier(nn.Module):
    """MLP baseline with same bottleneck for fair comparison."""

    def __init__(self, in_features, n_qubits=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_qubits),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================================
# Training
# =========================================================================
def compute_metrics(logits, targets):
    preds = logits.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    acc = (preds == targets_np).mean()
    f1 = f1_score(targets_np, preds, zero_division=0)
    prec = precision_score(targets_np, preds, zero_division=0)
    rec = recall_score(targets_np, preds, zero_division=0)
    try:
        auc = roc_auc_score(targets_np, probs)
    except ValueError:
        auc = 0.0
    return {"accuracy": float(acc), "f1": float(f1), "precision": float(prec),
            "recall": float(rec), "auc": float(auc)}


def train_one(model, name, train_loader, val_loader, args, run_dir):
    """Train a single model and return best metrics."""
    device = torch.device("cpu")  # Quantum simulator is CPU-only
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[%s] Params: %d" % (name, n_params))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Class-weighted loss
    train_y = torch.cat([batch[1] for batch in train_loader])
    n1 = (train_y == 1).sum().item()
    n0 = (train_y == 0).sum().item()
    w = torch.tensor([len(train_y) / (2 * max(n0, 1)),
                       len(train_y) / (2 * max(n1, 1))], dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=w)
    print("[%s] Samples: defect=%d, healthy=%d | Weights: [%.2f, %.2f]" % (
        name, n1, n0, w[0], w[1]))

    # Logging
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv_module.writer(f).writerow(
            ["epoch", "train_loss", "train_f1", "train_acc",
             "val_loss", "val_f1", "val_acc", "val_auc", "lr"])
    tb_writer = SummaryWriter(log_dir=run_dir)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss = 0
        all_logits, all_targets = [], []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            all_logits.append(out.detach())
            all_targets.append(yb)
        train_m = compute_metrics(torch.cat(all_logits), torch.cat(all_targets))
        train_m["loss"] = total_loss / len(train_loader.dataset)

        # Val
        model.eval()
        total_loss_v = 0
        all_logits_v, all_targets_v = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                total_loss_v += criterion(out, yb).item() * len(yb)
                all_logits_v.append(out)
                all_targets_v.append(yb)
        val_m = compute_metrics(torch.cat(all_logits_v), torch.cat(all_targets_v))
        val_m["loss"] = total_loss_v / len(val_loader.dataset)

        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Log
        with open(log_path, "a", newline="") as f:
            csv_module.writer(f).writerow([
                epoch, "%.6f" % train_m["loss"], "%.4f" % train_m["f1"],
                "%.4f" % train_m["accuracy"],
                "%.6f" % val_m["loss"], "%.4f" % val_m["f1"],
                "%.4f" % val_m["accuracy"], "%.4f" % val_m["auc"],
                "%.2e" % lr])
        tb_writer.add_scalars("loss", {"train": train_m["loss"], "val": val_m["loss"]}, epoch)
        tb_writer.add_scalars("f1", {"train": train_m["f1"], "val": val_m["f1"]}, epoch)
        tb_writer.add_scalar("val/auc", val_m["auc"], epoch)

        if epoch % args.log_every == 0 or epoch == 1:
            print("  [%s] Epoch %3d/%d | Train F1=%.4f | Val F1=%.4f AUC=%.4f | %.1fs" % (
                name, epoch, args.epochs, train_m["f1"], val_m["f1"], val_m["auc"], elapsed))

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "val_metrics": val_m}, os.path.join(run_dir, "best_model.pt"))
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print("  [%s] Early stopping at epoch %d" % (name, epoch))
            break

    tb_writer.close()
    print("  [%s] Best Val F1: %.4f" % (name, best_val_f1))
    return best_val_f1, val_m


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Node-level quantum defect classifier")
    parser.add_argument("--data_dir", type=str, default="data/processed_s12_czm_thermal_200_binary")
    parser.add_argument("--output_dir", type=str, default="runs/quantum_node")
    parser.add_argument("--mode", type=str, default="stats",
                        choices=["quantum", "classical", "stats"])
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--vqc_reps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--healthy_ratio", type=int, default=3,
                        help="Healthy nodes per defect node (downsampling)")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data_dir = (os.path.join(PROJECT_ROOT, args.data_dir)
                if not os.path.isabs(args.data_dir) else args.data_dir)
    train_data = torch.load(os.path.join(data_dir, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(data_dir, "val.pt"), weights_only=False)

    # Extract balanced node-level datasets
    X_train, y_train = extract_node_dataset(train_data, healthy_ratio=args.healthy_ratio,
                                             seed=args.seed)
    X_val, y_val = extract_node_dataset(val_data, healthy_ratio=args.healthy_ratio,
                                         seed=args.seed + 1)

    # Select physics features
    X_train, feat_dims = select_features(X_train)
    X_val, _ = select_features(X_val)
    in_features = X_train.shape[1]

    # Z-score normalize
    mu = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    std[std < 1e-8] = 1.0
    X_train = (X_train - mu) / std
    X_val = (X_val - mu) / std

    print("Train: %d nodes (defect=%d, healthy=%d) | Features: %d dims" % (
        len(y_train), (y_train == 1).sum(), (y_train == 0).sum(), in_features))
    print("Val:   %d nodes (defect=%d, healthy=%d)" % (
        len(y_val), (y_val == 1).sum(), (y_val == 0).sum()))

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=args.batch_size, shuffle=False)

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode in ("classical", "stats"):
        print("\n===== Classical MLP =====")
        torch.manual_seed(args.seed)
        model = ClassicalNodeClassifier(in_features, n_qubits=args.n_qubits)
        run_dir = os.path.join(args.output_dir, "classical_%s" % timestamp)
        f1, metrics = train_one(model, "CLASSICAL", train_loader, val_loader, args, run_dir)
        results["classical"] = {"f1": f1, **metrics}

    if args.mode in ("quantum", "stats"):
        print("\n===== Quantum VQC =====")
        torch.manual_seed(args.seed)
        model = QuantumNodeClassifier(in_features, n_qubits=args.n_qubits,
                                       vqc_reps=args.vqc_reps)
        run_dir = os.path.join(args.output_dir, "quantum_%s" % timestamp)
        f1, metrics = train_one(model, "QUANTUM", train_loader, val_loader, args, run_dir)
        results["quantum"] = {"f1": f1, **metrics}

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print("%-12s  %8s  %8s  %8s  %8s" % ("Model", "F1", "AUC", "Prec", "Recall"))
        print("-" * 60)
        for name, m in results.items():
            print("%-12s  %8.4f  %8.4f  %8.4f  %8.4f" % (
                name, m["f1"], m["auc"], m["precision"], m["recall"]))
        print("=" * 60)

        summary_path = os.path.join(args.output_dir, "comparison_%s.json" % timestamp)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
