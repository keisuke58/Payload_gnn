# -*- coding: utf-8 -*-
"""
Graph-Level Training Script: Quantum GNN vs Classical Baseline

Trains QuantumGNN (GNN + VQC) and/or ClassicalGNNGraphLevel (GNN + MLP)
for binary graph-level defect detection ("does this graph contain a defect?").

Usage:
    # Quantum only
    python src/train_quantum.py --mode quantum --epochs 100

    # Classical baseline only
    python src/train_quantum.py --mode classical --epochs 100

    # Both (side-by-side comparison)
    python src/train_quantum.py --mode both --epochs 100

    # With pretrained GNN encoder
    python src/train_quantum.py --mode both --gnn_checkpoint runs/gat_.../best_model.pt
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
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from torch.utils.tensorboard import SummaryWriter

from models_quantum import build_quantum_model


# =========================================================================
# Graph-level label derivation
# =========================================================================
def add_graph_labels(data_list):
    """Convert node-level labels to graph-level binary labels.

    graph_y = 1 if any node has y > 0 (defect), else 0.
    """
    n_defect = 0
    for data in data_list:
        has_defect = int((data.y > 0).any().item())
        data.graph_y = torch.tensor(has_defect, dtype=torch.long)
        n_defect += has_defect
    return data_list, n_defect


# =========================================================================
# Metrics (graph-level binary)
# =========================================================================
def compute_graph_metrics(logits, targets):
    """Compute graph-level binary classification metrics."""
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

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "auc": float(auc),
    }


# =========================================================================
# Training / Evaluation loops
# =========================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.graph_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(out.detach())
        all_targets.append(batch.graph_y.detach())

    avg_loss = total_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_graph_metrics(logits_cat, targets_cat)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.graph_y)

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(out)
        all_targets.append(batch.graph_y)

    avg_loss = total_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_graph_metrics(logits_cat, targets_cat)
    metrics["loss"] = avg_loss
    return metrics


# =========================================================================
# Single training run
# =========================================================================
def train_single(args, arch, train_data, val_data, tag=""):
    """Train one model (quantum or classical_graph) and return results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Detect dimensions
    sample = train_data[0]
    in_channels = sample.x.shape[1]
    edge_attr_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0

    # Build model
    model = build_quantum_model(
        arch, in_channels, edge_attr_dim,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        use_residual=args.residual,
        n_qubits=args.n_qubits,
        vqc_reps=args.vqc_reps,
        freeze_gnn=args.freeze_gnn,
        gnn_checkpoint=args.gnn_checkpoint,
    ).to(device)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[%s] Params: %d | Device: %s | Qubits: %d" % (
        arch.upper(), total_params, device, args.n_qubits))

    # Separate param groups for different learning rates
    gnn_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and "gnn_encoder" in n]
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and "gnn_encoder" not in n]

    param_groups = []
    if gnn_params:
        param_groups.append({"params": gnn_params, "lr": args.lr * 0.1})
    if other_params:
        param_groups.append({"params": other_params, "lr": args.vqc_lr})

    optimizer = Adam(param_groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Class weights for loss
    all_labels = torch.tensor([d.graph_y.item() for d in train_data])
    n_total = len(all_labels)
    n_defect = (all_labels == 1).sum().item()
    n_healthy = n_total - n_defect
    w0 = n_total / (2 * max(n_healthy, 1))
    w1 = n_total / (2 * max(n_defect, 1))
    weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("[%s] Loss weights: healthy=%.2f, defect=%.2f" % (arch.upper(), w0, w1))

    # Logging
    run_name = "%s_%s_%s" % (arch, tag, datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow(["epoch", "train_loss", "train_f1", "train_acc",
                         "val_loss", "val_f1", "val_acc", "val_auc", "lr"])

    tb_writer = SummaryWriter(log_dir=run_dir)
    tb_writer.add_text("config", json.dumps(vars(args), indent=2))

    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[-1]["lr"]

        # CSV log
        with open(log_path, "a", newline="") as f:
            writer = csv_module.writer(f)
            writer.writerow([
                epoch,
                "%.6f" % train_m["loss"], "%.4f" % train_m["f1"],
                "%.4f" % train_m["accuracy"],
                "%.6f" % val_m["loss"], "%.4f" % val_m["f1"],
                "%.4f" % val_m["accuracy"], "%.4f" % val_m["auc"],
                "%.2e" % lr,
            ])

        # TensorBoard
        tb_writer.add_scalars("loss", {"train": train_m["loss"], "val": val_m["loss"]}, epoch)
        tb_writer.add_scalars("f1", {"train": train_m["f1"], "val": val_m["f1"]}, epoch)
        tb_writer.add_scalars("accuracy", {"train": train_m["accuracy"], "val": val_m["accuracy"]}, epoch)
        tb_writer.add_scalar("val/auc", val_m["auc"], epoch)
        tb_writer.add_scalar("val/precision", val_m["precision"], epoch)
        tb_writer.add_scalar("val/recall", val_m["recall"], epoch)
        tb_writer.add_scalar("lr", lr, epoch)

        if epoch % args.log_every == 0 or epoch == 1:
            print("  [%s] Epoch %3d/%d | Train F1=%.4f Loss=%.4f | "
                  "Val F1=%.4f AUC=%.4f | %.1fs" % (
                      arch.upper(), epoch, args.epochs,
                      train_m["f1"], train_m["loss"],
                      val_m["f1"], val_m["auc"], elapsed))

        # Checkpoint best
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_metrics": val_m,
                "args": vars(args),
                "arch": arch,
            }, os.path.join(run_dir, "best_model.pt"))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("  [%s] Early stopping at epoch %d" % (arch.upper(), epoch))
            break

    tb_writer.close()
    print("  [%s] Best Val F1: %.4f" % (arch.upper(), best_val_f1))
    return run_dir, best_val_f1, val_m


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Graph-level defect detection: Quantum GNN vs Classical")
    # Data
    parser.add_argument("--data_dir", type=str, default="data/processed_50mm_100")
    parser.add_argument("--output_dir", type=str, default="runs/quantum")
    # Mode
    parser.add_argument("--mode", type=str, default="both",
                        choices=["quantum", "classical", "both",
                                 "stats", "quantum_stats", "classical_stats"])
    # GNN encoder
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument("--gnn_checkpoint", type=str, default=None,
                        help="Path to pretrained GNN checkpoint")
    parser.add_argument("--freeze_gnn", action="store_true", default=False)
    # Quantum
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--vqc_reps", type=int, default=2)
    parser.add_argument("--vqc_lr", type=float, default=0.01)
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    # Misc
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_normalize", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data_dir = (os.path.join(PROJECT_ROOT, args.data_dir)
                if not os.path.isabs(args.data_dir) else args.data_dir)
    train_data = torch.load(os.path.join(data_dir, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(data_dir, "val.pt"), weights_only=False)

    # Feature normalization
    if not args.no_normalize:
        norm_path = os.path.join(data_dir, "norm_stats.pt")
        if os.path.exists(norm_path):
            stats = torch.load(norm_path, weights_only=False)
            x_mean, x_std = stats["mean"], stats["std"]
            for d in train_data + val_data:
                d.x = (d.x - x_mean) / x_std
            print("Normalized node features: %d dims" % x_mean.shape[0])

        edge_attrs = [d.edge_attr for d in train_data if d.edge_attr is not None]
        if edge_attrs:
            ea_cat = torch.cat(edge_attrs, dim=0)
            ea_mean = ea_cat.mean(dim=0)
            ea_std = ea_cat.std(dim=0)
            ea_std[ea_std < 1e-8] = 1.0
            for d in train_data + val_data:
                if d.edge_attr is not None:
                    d.edge_attr = (d.edge_attr - ea_mean) / ea_std
            print("Normalized edge features: %d dims" % ea_mean.shape[0])

    # Graph-level labels
    train_data, n_defect_train = add_graph_labels(train_data)
    val_data, n_defect_val = add_graph_labels(val_data)
    print("Train: %d graphs (defect=%d, healthy=%d)" % (
        len(train_data), n_defect_train, len(train_data) - n_defect_train))
    print("Val:   %d graphs (defect=%d, healthy=%d)" % (
        len(val_data), n_defect_val, len(val_data) - n_defect_val))

    # Run training
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode in ("classical_stats", "stats"):
        print("\n===== Classical Stats (Feature Stats + MLP) =====")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        run_dir, f1, metrics = train_single(
            args, "classical_stats", train_data, val_data, tag=timestamp)
        results["classical_stats"] = {"f1": f1, "metrics": metrics, "run_dir": run_dir}

    if args.mode in ("quantum_stats", "stats"):
        print("\n===== Quantum Stats (Feature Stats + VQC) =====")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        run_dir, f1, metrics = train_single(
            args, "quantum_stats", train_data, val_data, tag=timestamp)
        results["quantum_stats"] = {"f1": f1, "metrics": metrics, "run_dir": run_dir}

    if args.mode in ("classical", "both"):
        print("\n===== Classical Baseline (GNN + MLP) =====")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        run_dir, f1, metrics = train_single(
            args, "classical_graph", train_data, val_data, tag=timestamp)
        results["classical"] = {"f1": f1, "metrics": metrics, "run_dir": run_dir}

    if args.mode in ("quantum", "both"):
        print("\n===== Quantum Model (GNN + VQC) =====")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        run_dir, f1, metrics = train_single(
            args, "quantum", train_data, val_data, tag=timestamp)
        results["quantum"] = {"f1": f1, "metrics": metrics, "run_dir": run_dir}

    # Comparison summary
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print("%-15s  %8s  %8s  %8s  %8s" % (
            "Model", "F1", "AUC", "Prec", "Recall"))
        print("-" * 60)
        for name, r in results.items():
            m = r["metrics"]
            print("%-15s  %8.4f  %8.4f  %8.4f  %8.4f" % (
                name, m["f1"], m["auc"], m["precision"], m["recall"]))
        print("=" * 60)

        # Save comparison
        summary_path = os.path.join(args.output_dir, "comparison_%s.json" % timestamp)
        summary = {k: {"f1": v["f1"], "run_dir": v["run_dir"],
                        **v["metrics"]} for k, v in results.items()}
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print("Comparison saved: %s" % summary_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
