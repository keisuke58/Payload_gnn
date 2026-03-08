# -*- coding: utf-8 -*-
"""
Spatio-Temporal GNN for GW-based SHM — train_gw_stgnn.py

Sensor time-series → 1D-CNN temporal encoder → GAT spatial aggregation
→ graph-level binary classification (healthy vs defect).

Key idea: Instead of hand-crafted features, feed raw waveforms through
a 1D-CNN to learn temporal representations, then use GAT over the sensor
graph to capture spatial wave propagation patterns.

Also includes DI (Damage Index) baseline for comparison.

Usage:
  # Prepare data + train ST-GNN
  python src/train_gw_stgnn.py --mode train --csv_dir abaqus_work/gw_fairing_dataset \
      --doe doe_gw_fairing.json --epochs 200

  # DI baseline only
  python src/train_gw_stgnn.py --mode di_baseline --csv_dir abaqus_work/gw_fairing_dataset \
      --doe doe_gw_fairing.json
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
except ImportError:
    print("ERROR: torch_geometric not installed")
    sys.exit(1)


# =========================================================================
# Data loading
# =========================================================================
def load_sensor_csv(csv_path):
    """Load sensor CSV → (times, waveforms, positions).

    Returns:
        times: (T,) ndarray
        waveforms: (n_sensors, T) ndarray — raw radial displacement
        positions: (n_sensors,) ndarray — x_mm or arc positions
    """
    with open(csv_path) as f:
        rows = list(csv.reader(f))

    if len(rows) < 3:
        return None, None, None

    header = rows[0]
    pos_row = rows[1]

    # Parse sensor columns
    sensor_cols = {}  # col_idx -> sensor_id
    for i, col in enumerate(header):
        if i == 0:
            continue
        if col.startswith("sensor_"):
            try:
                sid = int(col.replace("sensor_", "").split("_")[0])
                sensor_cols[i] = sid
            except ValueError:
                pass

    sensor_ids = sorted(set(sensor_cols.values()))
    sid_to_idx = {s: i for i, s in enumerate(sensor_ids)}
    n_sensors = len(sensor_ids)

    # Parse positions
    positions = np.zeros(n_sensors, dtype=np.float32)
    for col_idx, sid in sensor_cols.items():
        if col_idx < len(pos_row):
            try:
                positions[sid_to_idx[sid]] = float(pos_row[col_idx])
            except (ValueError, TypeError):
                pass

    # Parse time series
    times = []
    data_buf = [[] for _ in range(n_sensors)]

    for row in rows[2:]:
        if not row or row[0].startswith("#"):
            continue
        try:
            times.append(float(row[0]))
            for col_idx, sid in sensor_cols.items():
                idx = sid_to_idx[sid]
                val = float(row[col_idx]) if col_idx < len(row) and row[col_idx] else 0.0
                data_buf[idx].append(val)
        except (ValueError, IndexError):
            continue

    times = np.array(times, dtype=np.float64)
    waveforms = np.array(data_buf, dtype=np.float32)  # (n_sensors, T)

    return times, waveforms, positions


def build_sensor_graph_edges(positions, k=8):
    """Build k-NN edges from sensor positions.

    For 1D positions, connects each sensor to its k nearest neighbors.
    Returns edge_index (2, E) and edge_attr (E, 1) = normalized distance.
    """
    n = len(positions)
    pos = positions.reshape(-1, 1) if positions.ndim == 1 else positions

    # Pairwise distance
    dist = np.abs(pos[:, 0:1] - pos[:, 0:1].T)  # (n, n)
    k_eff = min(k, n - 1)

    src, dst, edge_dist = [], [], []
    for i in range(n):
        d = dist[i].copy()
        d[i] = np.inf
        neighbors = np.argsort(d)[:k_eff]
        for j in neighbors:
            src.append(i)
            dst.append(j)
            edge_dist.append(d[j])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    max_d = max(edge_dist) if edge_dist else 1.0
    edge_attr = torch.tensor(
        [[d / max_d] for d in edge_dist], dtype=torch.float
    )
    return edge_index, edge_attr


def prepare_stgnn_data(csv_dir, doe_path=None, max_len=4000, k_neighbors=8,
                       n_sensors_filter=None, min_sensors=2):
    """Build PyG dataset from GW sensor CSVs.

    Each graph: nodes=sensors, node features=raw waveform (T,),
    edges=k-NN by position, graph label=0/1.

    Supports variable sensor counts (10, 20, 99, etc.) — the model
    uses graph-level pooling so node count can differ between samples.

    Args:
        csv_dir: directory containing *_sensors.csv
        doe_path: DOE JSON for defect sample list
        max_len: max waveform length (truncate/pad)
        k_neighbors: k-NN for sensor graph edges
        n_sensors_filter: if set, only include samples with exactly this many sensors.
                          If None, accept all sensor counts >= min_sensors.
        min_sensors: minimum number of sensors to accept (default=2)

    Returns list of Data objects.
    """
    csv_dir = os.path.abspath(csv_dir)
    if not os.path.isdir(csv_dir):
        print("ERROR: csv_dir not found: %s" % csv_dir)
        return []

    # Collect samples: (csv_path, label, name)
    samples = []

    # Healthy: main + augmented
    for f in sorted(os.listdir(csv_dir)):
        if not f.endswith("_sensors.csv"):
            continue
        if "Healthy" in f and "Test" not in f:
            samples.append((os.path.join(csv_dir, f), 0, f))

    # Defect
    if doe_path and os.path.exists(doe_path):
        with open(doe_path) as fh:
            doe = json.load(fh)
        for i in range(doe.get("n_samples", 0)):
            name = "Job-GW-Fair-%04d_sensors.csv" % i
            path = os.path.join(csv_dir, name)
            if os.path.exists(path):
                samples.append((path, 1, name))
    else:
        for f in sorted(os.listdir(csv_dir)):
            if f.endswith("_sensors.csv") and f.startswith("Job-GW-Fair-0"):
                samples.append((os.path.join(csv_dir, f), 1, f))

    if not samples:
        print("No samples found")
        return []

    print("Total samples found: %d (healthy=%d, defect=%d)" % (
        len(samples),
        sum(1 for _, l, _ in samples if l == 0),
        sum(1 for _, l, _ in samples if l == 1),
    ))
    if n_sensors_filter:
        print("Sensor filter: exactly %d sensors" % n_sensors_filter)
    else:
        print("Sensor filter: >= %d sensors (variable count OK)" % min_sensors)

    dataset = []
    skipped = 0
    sensor_counts = {}
    for path, label, name in samples:
        times, waveforms, positions = load_sensor_csv(path)
        if waveforms is None:
            skipped += 1
            continue

        n_sensors, T = waveforms.shape

        # Filter by sensor count
        if n_sensors_filter is not None and n_sensors != n_sensors_filter:
            skipped += 1
            continue
        if n_sensors < min_sensors:
            skipped += 1
            continue

        sensor_counts[n_sensors] = sensor_counts.get(n_sensors, 0) + 1

        # Truncate or pad time series to max_len
        if T > max_len:
            waveforms = waveforms[:, :max_len]
        elif T < max_len:
            pad = np.zeros((n_sensors, max_len - T), dtype=np.float32)
            waveforms = np.concatenate([waveforms, pad], axis=1)

        # Normalize per-sample: zero mean, unit variance
        mean = waveforms.mean()
        std = waveforms.std()
        if std > 1e-10:
            waveforms = (waveforms - mean) / std

        # Build graph (k adjusted for small sensor counts)
        k_eff = min(k_neighbors, n_sensors - 1)
        edge_index, edge_attr = build_sensor_graph_edges(positions, k=k_eff)

        data = Data(
            x=torch.tensor(waveforms, dtype=torch.float),  # (n_sensors, T)
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long),
            pos=torch.tensor(positions.reshape(-1, 1), dtype=torch.float),
            name=name,
        )
        dataset.append(data)

    print("Built %d graphs (skipped %d)" % (len(dataset), skipped))
    for ns, cnt in sorted(sensor_counts.items()):
        print("  %d sensors: %d samples" % (ns, cnt))
    return dataset


# =========================================================================
# Models
# =========================================================================
class TemporalEncoder(nn.Module):
    """1D-CNN to encode raw waveform (T,) → (d_out,) per sensor."""

    def __init__(self, in_len=4000, d_out=64):
        super().__init__()
        # Multi-scale: narrow + wide kernels
        self.branch_narrow = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.branch_wide = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        # After 3x stride-2: T/8 → adaptive pool to fixed size
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc = nn.Sequential(
            nn.Linear(64 * 8, d_out),  # 2 branches × 32 × 8
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        """x: (B * n_sensors, T) → (B * n_sensors, d_out)"""
        x = x.unsqueeze(1)  # (B*S, 1, T)
        h_narrow = self.branch_narrow(x)  # (B*S, 32, T/8)
        h_wide = self.branch_wide(x)      # (B*S, 32, T/8)
        h = torch.cat([h_narrow, h_wide], dim=1)  # (B*S, 64, T/8)
        h = self.pool(h)  # (B*S, 64, 8)
        h = h.flatten(1)  # (B*S, 512)
        return self.fc(h)  # (B*S, d_out)


class SpatioTemporalGNN(nn.Module):
    """ST-GNN: TemporalEncoder → GAT spatial layers → graph pooling → classifier."""

    def __init__(self, in_len=4000, d_temporal=64, d_hidden=64,
                 n_gat_layers=3, n_heads=4, dropout=0.1):
        super().__init__()
        self.temporal = TemporalEncoder(in_len=in_len, d_out=d_temporal)

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        in_dim = d_temporal
        for i in range(n_gat_layers):
            out_dim = d_hidden
            self.gat_layers.append(
                GATConv(in_dim, out_dim // n_heads, heads=n_heads,
                        dropout=dropout, concat=True)
            )
            self.gat_norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.classifier = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),  # mean + max pool
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 2),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        """
        x: (total_nodes, T) raw waveforms
        edge_index: (2, E)
        batch: (total_nodes,) batch assignment
        """
        # Temporal encoding: per-sensor waveform → embedding
        h = self.temporal(x)  # (total_nodes, d_temporal)

        # Spatial GAT layers
        for conv, norm in zip(self.gat_layers, self.gat_norms):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.elu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            # Residual if dimensions match
            if h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new

        # Graph-level readout: mean + max pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        g = torch.cat([h_mean, h_max], dim=1)

        return self.classifier(g)


# =========================================================================
# DI Baseline
# =========================================================================
def compute_damage_index(healthy_waveforms, defect_waveforms):
    """Compute per-sensor Damage Index (correlation-based).

    DI_i = 1 - max(|cross_correlation(healthy_i, defect_i)|)

    Args:
        healthy_waveforms: (n_sensors, T) reference
        defect_waveforms:  (n_sensors, T) test

    Returns: (n_sensors,) DI values
    """
    n_sensors = healthy_waveforms.shape[0]
    di = np.zeros(n_sensors)

    for i in range(n_sensors):
        h = healthy_waveforms[i]
        d = defect_waveforms[i]
        h_norm = np.linalg.norm(h)
        d_norm = np.linalg.norm(d)
        if h_norm < 1e-12 or d_norm < 1e-12:
            di[i] = 0.0
            continue
        # Normalized cross-correlation (at zero lag for speed)
        cc = np.corrcoef(h, d)[0, 1]
        di[i] = 1.0 - abs(cc)

    return di


def di_baseline(csv_dir, doe_path=None):
    """Run DI baseline: compute DI per sample, classify by threshold.

    Uses the first healthy sample as reference.
    """
    print("\n=== DI Baseline ===")
    csv_dir = os.path.abspath(csv_dir)

    # Load reference healthy
    ref_path = None
    for f in sorted(os.listdir(csv_dir)):
        if f.endswith("_sensors.csv") and "Healthy" in f and "Test" not in f:
            candidate = os.path.join(csv_dir, f)
            _, wf, _ = load_sensor_csv(candidate)
            if wf is not None and wf.shape[0] > 20:
                ref_path = candidate
                break

    if ref_path is None:
        print("ERROR: No healthy reference with matching sensors found")
        return

    _, ref_wf, ref_pos = load_sensor_csv(ref_path)
    ref_n = ref_wf.shape[0]
    print("Reference: %s (%d sensors, %d steps)" % (
        os.path.basename(ref_path), ref_wf.shape[0], ref_wf.shape[1]))

    # Collect all samples
    results = []

    # Healthy samples
    for f in sorted(os.listdir(csv_dir)):
        if not f.endswith("_sensors.csv") or "Test" in f:
            continue
        if "Healthy" in f:
            path = os.path.join(csv_dir, f)
            if path == ref_path:
                continue
            _, wf, _ = load_sensor_csv(path)
            if wf is None or wf.shape[0] != ref_n:
                continue
            di = compute_damage_index(ref_wf, wf)
            results.append({"name": f, "label": 0, "di_mean": di.mean(),
                            "di_max": di.max(), "di_std": di.std()})

    # Defect samples
    if doe_path and os.path.exists(doe_path):
        with open(doe_path) as fh:
            doe = json.load(fh)
        for i in range(doe.get("n_samples", 0)):
            name = "Job-GW-Fair-%04d_sensors.csv" % i
            path = os.path.join(csv_dir, name)
            if not os.path.exists(path):
                continue
            _, wf, _ = load_sensor_csv(path)
            if wf is None or wf.shape[0] != ref_n:
                continue
            di = compute_damage_index(ref_wf, wf)
            results.append({"name": name, "label": 1, "di_mean": di.mean(),
                            "di_max": di.max(), "di_std": di.std()})

    if not results:
        print("No samples with matching sensor count found")
        return

    print("\nResults (%d samples):" % len(results))
    print("%-45s  label  DI_mean   DI_max   DI_std" % "name")
    print("-" * 85)
    for r in results:
        print("%-45s  %d      %.4f    %.4f    %.4f" % (
            r["name"], r["label"], r["di_mean"], r["di_max"], r["di_std"]))

    # Threshold sweep for classification
    labels = np.array([r["label"] for r in results])
    di_means = np.array([r["di_mean"] for r in results])

    if len(set(labels)) < 2:
        print("\nOnly one class present — cannot evaluate classifier")
        return

    best_f1, best_thresh = 0.0, 0.0
    for thresh in np.linspace(di_means.min(), di_means.max(), 200):
        preds = (di_means > thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    preds = (di_means > best_thresh).astype(int)
    print("\nBest threshold: %.6f" % best_thresh)
    print("F1: %.4f  Precision: %.4f  Recall: %.4f  Accuracy: %.4f" % (
        f1_score(labels, preds, zero_division=0),
        precision_score(labels, preds, zero_division=0),
        recall_score(labels, preds, zero_division=0),
        accuracy_score(labels, preds),
    ))
    print("Confusion matrix:\n", confusion_matrix(labels, preds))


# =========================================================================
# Training
# =========================================================================
def compute_metrics(logits, targets):
    preds = logits.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    f1 = f1_score(targets_np, preds, zero_division=0)
    prec = precision_score(targets_np, preds, zero_division=0)
    rec = recall_score(targets_np, preds, zero_division=0)
    acc = accuracy_score(targets_np, preds)
    try:
        auc = roc_auc_score(targets_np, probs)
    except ValueError:
        auc = 0.0

    return {"f1": f1, "precision": prec, "recall": rec,
            "accuracy": acc, "auc": auc}


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        targets = batch.y.squeeze()
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        loss = criterion(out, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(out.detach())
        all_targets.append(targets.detach())

    avg_loss = total_loss / len(loader.dataset)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(logits, targets)
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
        targets = batch.y.squeeze()
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        loss = criterion(out, targets)
        total_loss += loss.item() * batch.num_graphs
        all_logits.append(out)
        all_targets.append(targets)

    avg_loss = total_loss / len(loader.dataset)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(logits, targets)
    metrics["loss"] = avg_loss
    return metrics


def train_stgnn(dataset, args):
    """Train Spatio-Temporal GNN."""
    # Split
    np.random.seed(args.seed)
    n = len(dataset)
    indices = np.random.permutation(n)
    n_val = max(1, int(n * args.val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]

    n_healthy_train = sum(1 for d in train_data if d.y.item() == 0)
    n_defect_train = len(train_data) - n_healthy_train
    print("\nTrain: %d (healthy=%d, defect=%d) | Val: %d" % (
        len(train_data), n_healthy_train, n_defect_train, len(val_data)))

    in_len = dataset[0].x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpatioTemporalGNN(
        in_len=in_len,
        d_temporal=args.d_temporal,
        d_hidden=args.d_hidden,
        n_gat_layers=args.n_gat_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("Model: SpatioTemporalGNN (%d params)" % n_params)
    print("  temporal_dim=%d, hidden=%d, gat_layers=%d, heads=%d" % (
        args.d_temporal, args.d_hidden, args.n_gat_layers, args.n_heads))

    # Class weights
    if n_healthy_train > 0 and n_defect_train > 0:
        w = torch.tensor([1.0 / n_healthy_train, 1.0 / n_defect_train])
        w = w / w.sum()
    else:
        w = torch.ones(2)
    criterion = nn.CrossEntropyLoss(weight=w.to(device))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    run_dir = args.run_dir or os.path.join(
        PROJECT_ROOT, "runs/stgnn_%s" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    best_val_f1 = 0.0
    best_val_auc = 0.0
    patience_counter = 0

    print("\nTraining for %d epochs..." % args.epochs)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        improved = ""
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_val_auc = val_m["auc"]
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or improved:
            lr = optimizer.param_groups[0]["lr"]
            print("  Epoch %3d/%d | Train F1=%.4f Loss=%.4f | "
                  "Val F1=%.4f AUC=%.4f Loss=%.4f | LR=%.2e | %.1fs%s" % (
                      epoch, args.epochs,
                      train_m["f1"], train_m["loss"],
                      val_m["f1"], val_m["auc"], val_m["loss"],
                      lr, elapsed, improved))

        if patience_counter >= args.patience:
            print("  Early stopping at epoch %d (patience=%d)" % (epoch, args.patience))
            break

    print("\nBest Val F1: %.4f  AUC: %.4f" % (best_val_f1, best_val_auc))
    print("Checkpoint: %s/best_model.pt" % run_dir)

    # Save config
    config = vars(args).copy()
    config["best_val_f1"] = best_val_f1
    config["best_val_auc"] = best_val_auc
    config["n_params"] = n_params
    config["n_train"] = len(train_data)
    config["n_val"] = len(val_data)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    return best_val_f1, best_val_auc


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="GW ST-GNN training")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "di_baseline", "both"])
    parser.add_argument("--csv_dir", type=str,
                        default="abaqus_work/gw_fairing_dataset")
    parser.add_argument("--doe", type=str, default="doe_gw_fairing.json")

    # Data
    parser.add_argument("--max_len", type=int, default=3920,
                        help="Max waveform length (truncate/pad)")
    parser.add_argument("--k_neighbors", type=int, default=8,
                        help="k-NN for sensor graph")
    parser.add_argument("--n_sensors", type=int, default=None,
                        help="Filter to exactly N sensors (None=accept all)")
    parser.add_argument("--min_sensors", type=int, default=2,
                        help="Minimum sensor count to accept")

    # Model
    parser.add_argument("--d_temporal", type=int, default=64)
    parser.add_argument("--d_hidden", type=int, default=64)
    parser.add_argument("--n_gat_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", type=str, default=None)

    args = parser.parse_args()

    # Resolve paths
    csv_dir = os.path.join(PROJECT_ROOT, args.csv_dir)
    doe_path = os.path.join(PROJECT_ROOT, args.doe) if args.doe else None

    if args.mode in ("di_baseline", "both"):
        di_baseline(csv_dir, doe_path)

    if args.mode in ("train", "both"):
        print("\n=== Preparing ST-GNN dataset ===")
        dataset = prepare_stgnn_data(
            csv_dir, doe_path,
            max_len=args.max_len,
            k_neighbors=args.k_neighbors,
            n_sensors_filter=args.n_sensors,
            min_sensors=args.min_sensors,
        )
        if len(dataset) < 4:
            print("ERROR: Not enough samples (%d). Need at least 4." % len(dataset))
            print("Wait for more Abaqus jobs to complete and extract CSVs.")
            return

        train_stgnn(dataset, args)


if __name__ == "__main__":
    main()
