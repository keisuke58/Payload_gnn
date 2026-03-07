#!/usr/bin/env python3
"""Cross-Domain GNN Benchmark — External SHM Datasets.

Converts external SHM datasets (DINS-SHM, CONCEPT) to graph format
and evaluates GNN models for cross-domain damage detection.

Three evaluation modes:
  1. In-domain: Train & test on each dataset independently
  2. Zero-shot: Train on fairing, test on external (and vice versa)
  3. DANN: Domain-adversarial transfer between domains

Output: figures/cross_domain_benchmark/ with comparison plots

Usage:
    python scripts/benchmark_gnn_cross_domain.py [--device cuda]
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from models import build_model

# ── Paths ──────────────────────────────────────────────────
DINS_DIR = PROJECT_ROOT / "data" / "external" / "dins_shm"
CONCEPT_DIR = (PROJECT_ROOT / "data" / "external" / "concept" /
               "DATASET_PLATEUN01" / "data")
OUT = PROJECT_ROOT / "figures" / "cross_domain_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3,
})


# ====================================================================
# Dataset Converters: Time-Series → PyG Graph
# ====================================================================

def timeseries_to_graph(signal, n_segments=50, k_neighbors=8, label=0):
    """Convert 1D time-series to PyG graph via sliding window features.

    Steps:
        1. Divide signal into n_segments overlapping windows
        2. Extract features per window (RMS, peak, ZCR, energy, spectral)
        3. Build k-NN graph over feature space
        4. Return PyG Data object

    Args:
        signal: (T,) 1D time series
        n_segments: number of graph nodes
        k_neighbors: edges per node
        label: 0=healthy, 1=damaged

    Returns:
        torch_geometric.data.Data with .x, .edge_index, .y
    """
    from torch_geometric.data import Data

    T = len(signal)
    win_size = T // n_segments
    overlap = win_size // 2

    features = []
    positions = []

    for i in range(n_segments):
        start = i * (T - win_size) // max(n_segments - 1, 1)
        end = start + win_size
        window = signal[start:end]

        # Feature extraction (8-dim per node)
        rms = np.sqrt(np.mean(window ** 2))
        peak = np.max(np.abs(window))
        zcr = np.sum(np.diff(np.sign(window)) != 0) / len(window)
        energy = np.sum(window ** 2)
        mean_val = np.mean(window)
        std_val = np.std(window)
        skew = _safe_skewness(window)
        kurt = _safe_kurtosis(window)

        features.append([rms, peak, zcr, energy, mean_val, std_val,
                         skew, kurt])
        positions.append((start + end) / 2.0 / T)  # normalized position

    x = np.array(features, dtype=np.float32)
    pos = np.array(positions, dtype=np.float32).reshape(-1, 1)

    # Normalize features
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    x = (x - x_mean) / x_std

    # Add position as feature
    x = np.concatenate([x, pos], axis=1)  # (N, 9)

    # Build k-NN graph
    from scipy.spatial import KDTree
    tree = KDTree(x)
    edges = []
    k = min(k_neighbors, n_segments - 1)
    for j in range(n_segments):
        _, idx = tree.query(x[j], k=k + 1)
        for nb in idx[1:]:
            edges.append([j, nb])
            edges.append([nb, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.full((n_segments,), label, dtype=torch.long)

    return Data(x=x_tensor, edge_index=edge_index, y=y_tensor)


def _safe_skewness(x):
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0.0
    return np.mean(((x - m) / s) ** 3)


def _safe_kurtosis(x):
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0.0
    return np.mean(((x - m) / s) ** 4) - 3.0


# ====================================================================
# Load and Convert DINS-SHM
# ====================================================================

def load_dins_graphs(n_samples=200, n_segments=50, k_neighbors=8):
    """Load DINS-SHM composite waveguide as graphs.

    Returns list of PyG Data objects (healthy + damaged).
    """
    comp_dir = DINS_DIR / "composite" / "0_DataSet"
    n_per_class = n_samples // 2

    graphs = []

    for fname, lbl in [
        ("Ax2500Comp_UD_07Aug20.txt", 0),  # healthy
        ("Ax2500Comp_D_07Aug20.txt", 1),    # damaged
    ]:
        fpath = comp_dir / fname
        if not fpath.exists():
            print("  WARNING: %s not found, skipping" % fpath)
            continue

        # Load a subset
        all_rows = np.genfromtxt(fpath, delimiter=",",
                                 max_rows=n_per_class)
        n_loaded = min(len(all_rows), n_per_class)

        for i in range(n_loaded):
            g = timeseries_to_graph(all_rows[i], n_segments=n_segments,
                                    k_neighbors=k_neighbors, label=lbl)
            graphs.append(g)

    np.random.shuffle(graphs)
    print("  DINS-SHM: %d graphs (%d features/node, %d nodes/graph)" % (
        len(graphs), graphs[0].x.size(1) if graphs else 0,
        n_segments))
    return graphs


# ====================================================================
# Load and Convert CONCEPT
# ====================================================================

def load_concept_graphs(n_segments=50, k_neighbors=8):
    """Load CONCEPT experimental data as graphs.

    Healthy: all temperatures, multiple measurements
    Damaged: all 11 damage states
    """
    import scipy.io as sio

    graphs = []

    # Healthy states (T=20,30,40°C — middle range, 20 measurements each)
    for temp in [20, 30, 40]:
        fpath = CONCEPT_DIR / ("H_T%d.mat" % temp)
        if not fpath.exists():
            continue
        mat = sio.loadmat(fpath)
        data = mat["data"]  # (1000, 4, 100)
        n_meas = min(20, data.shape[2])
        for m in range(n_meas):
            # Use channel 1 (PZT-2) as main signal
            signal = data[:, 1, m]
            g = timeseries_to_graph(signal, n_segments=n_segments,
                                    k_neighbors=k_neighbors, label=0)
            graphs.append(g)

    # Damaged states (all 11 levels, 20 measurements each)
    for d_level in range(1, 12):
        fpath = CONCEPT_DIR / ("D%d.mat" % d_level)
        if not fpath.exists():
            continue
        mat = sio.loadmat(fpath)
        data = mat["data"]  # (1000, 4, 100)
        n_meas = min(20, data.shape[2])
        for m in range(n_meas):
            signal = data[:, 1, m]
            g = timeseries_to_graph(signal, n_segments=n_segments,
                                    k_neighbors=k_neighbors, label=1)
            graphs.append(g)

    np.random.shuffle(graphs)
    print("  CONCEPT: %d graphs (%d features/node, %d nodes/graph)" % (
        len(graphs), graphs[0].x.size(1) if graphs else 0,
        n_segments))
    return graphs


# ====================================================================
# Load Fairing Data (subset)
# ====================================================================

def load_fairing_graphs(data_dir=None, max_graphs=200):
    """Load fairing PyG graphs (existing processed data)."""
    if data_dir is None:
        # Find best available dataset
        for candidate in [
            'processed_s12_czm_thermal_200_binary',
            'processed_s12_100_binary',
            'processed_25mm_100',
        ]:
            d = PROJECT_ROOT / 'data' / candidate
            if (d / 'train.pt').exists():
                data_dir = d
                break
    if data_dir is None:
        print("  WARNING: No fairing data found")
        return []

    data_dir = Path(data_dir)
    train = torch.load(data_dir / 'train.pt', weights_only=False)
    val = torch.load(data_dir / 'val.pt', weights_only=False)
    graphs = train + val

    # Subsample if too many
    if len(graphs) > max_graphs:
        indices = np.random.choice(len(graphs), max_graphs, replace=False)
        graphs = [graphs[i] for i in indices]

    print("  Fairing: %d graphs (%d features/node, ~%d nodes/graph)" % (
        len(graphs), graphs[0].x.size(1),
        graphs[0].x.size(0)))
    return graphs


# ====================================================================
# Train & Evaluate GNN
# ====================================================================

def train_eval_gnn(train_graphs, val_graphs, in_channels, arch='sage',
                   hidden=64, num_layers=3, epochs=50, lr=1e-3,
                   device='cpu', verbose=True):
    """Train a GNN classifier and return AUROC + F1."""
    device = torch.device(device)

    model = build_model(
        arch, in_channels=in_channels,
        hidden_channels=hidden, num_classes=2,
        num_layers=num_layers, dropout=0.1,
    ).to(device)

    # Move data
    for d in train_graphs + val_graphs:
        d.x = d.x.to(device)
        d.edge_index = d.edge_index.to(device)
        d.y = d.y.to(device)

    # Class weights
    all_y = torch.cat([d.y for d in train_graphs])
    n_pos = (all_y > 0).sum().item()
    n_neg = len(all_y) - n_pos
    pw = min(n_neg / max(n_pos, 1), 10.0)
    weights = torch.tensor([1.0, pw], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_graphs))
        for idx in perm:
            data = train_graphs[idx]
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out, data.y, weight=weights)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for data in val_graphs:
            out = model(data.x, data.edge_index)
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.append(probs.cpu())
            all_labels.append(data.y.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.5

    preds = (all_probs > 0.5).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)

    return auroc, f1, model


# ====================================================================
# Cross-Domain Evaluation
# ====================================================================

def evaluate_cross_domain(model, graphs, device='cpu'):
    """Evaluate a trained model on out-of-domain graphs."""
    device = torch.device(device)
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for data in graphs:
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.y = data.y.to(device)
            out = model(data.x, data.edge_index)
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.append(probs.cpu())
            all_labels.append(data.y.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.5

    preds = (all_probs > 0.5).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)

    return auroc, f1


# ====================================================================
# Visualization
# ====================================================================

def plot_results(results, out_dir):
    """Plot cross-domain benchmark results."""

    # Figure 1: In-domain vs Cross-domain AUROC
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    datasets = list(results['in_domain'].keys())
    in_aurocs = [results['in_domain'][d]['auroc'] for d in datasets]
    in_f1s = [results['in_domain'][d]['f1'] for d in datasets]

    ax = axes[0]
    x_pos = np.arange(len(datasets))
    bars = ax.bar(x_pos, in_aurocs, color=['#4472C4', '#ED7D31', '#70AD47'],
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('AUROC')
    ax.set_title('In-Domain Performance (GNN-SAGE)')
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, in_aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                '%.3f' % val, ha='center', fontsize=10, fontweight='bold')

    # Cross-domain heatmap
    ax = axes[1]
    if 'cross_domain' in results:
        cd = results['cross_domain']
        matrix = np.zeros((len(datasets), len(datasets)))
        for i, src in enumerate(datasets):
            for j, tgt in enumerate(datasets):
                key = '%s→%s' % (src, tgt)
                if key in cd:
                    matrix[i, j] = cd[key]['auroc']
                elif src == tgt:
                    matrix[i, j] = results['in_domain'][src]['auroc']

        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(datasets)))
        ax.set_yticks(range(len(datasets)))
        ax.set_xticklabels(datasets, fontsize=9)
        ax.set_yticklabels(datasets, fontsize=9)
        ax.set_xlabel('Target (Test)')
        ax.set_ylabel('Source (Train)')
        ax.set_title('Cross-Domain Transfer AUROC')

        for i in range(len(datasets)):
            for j in range(len(datasets)):
                ax.text(j, i, '%.3f' % matrix[i, j],
                        ha='center', va='center', fontsize=11,
                        fontweight='bold',
                        color='white' if matrix[i, j] < 0.5 else 'black')
        plt.colorbar(im, ax=ax)

    fig.suptitle('Fig. C1: Cross-Domain GNN Benchmark — '
                 'SHM Damage Detection',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / '01_cross_domain_benchmark.png',
                bbox_inches='tight')
    print("  Saved %s" % (out_dir / '01_cross_domain_benchmark.png'))
    plt.close(fig)


# ====================================================================
# Main
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--n_dins', type=int, default=200,
                        help='Number of DINS-SHM samples')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--arch', default='sage')
    args = parser.parse_args()

    print("=" * 60)
    print("Cross-Domain GNN Benchmark")
    print("=" * 60)

    # 1. Load and convert datasets
    print("\n[1/4] Loading datasets...")
    datasets = {}

    dins_graphs = load_dins_graphs(n_samples=args.n_dins)
    if dins_graphs:
        n_split = int(0.8 * len(dins_graphs))
        datasets['DINS-SHM'] = {
            'train': dins_graphs[:n_split],
            'val': dins_graphs[n_split:],
            'in_channels': dins_graphs[0].x.size(1),
        }

    concept_graphs = load_concept_graphs()
    if concept_graphs:
        n_split = int(0.8 * len(concept_graphs))
        datasets['CONCEPT'] = {
            'train': concept_graphs[:n_split],
            'val': concept_graphs[n_split:],
            'in_channels': concept_graphs[0].x.size(1),
        }

    # Note: Fairing data has 34-dim features vs 9-dim for time-series graphs
    # Cross-domain requires same feature dim → we skip fairing↔external
    # direct transfer but report all in-domain results

    # 2. In-domain evaluation
    print("\n[2/4] In-domain evaluation...")
    results = {'in_domain': {}, 'cross_domain': {}}

    for name, data in datasets.items():
        print("\n  Training %s on %s..." % (args.arch.upper(), name))
        auroc, f1, model = train_eval_gnn(
            data['train'], data['val'],
            in_channels=data['in_channels'],
            arch=args.arch, epochs=args.epochs,
            device=args.device)
        results['in_domain'][name] = {'auroc': auroc, 'f1': f1}
        data['model'] = model
        print("    AUROC=%.4f  F1=%.4f" % (auroc, f1))

    # 3. Cross-domain evaluation (between same-format datasets)
    print("\n[3/4] Cross-domain evaluation...")
    ds_names = list(datasets.keys())
    for src_name in ds_names:
        for tgt_name in ds_names:
            if src_name == tgt_name:
                continue
            src_data = datasets[src_name]
            tgt_data = datasets[tgt_name]
            if src_data['in_channels'] != tgt_data['in_channels']:
                print("  %s→%s: SKIP (dim mismatch %d vs %d)" % (
                    src_name, tgt_name,
                    src_data['in_channels'], tgt_data['in_channels']))
                continue
            if 'model' not in src_data:
                continue
            auroc, f1 = evaluate_cross_domain(
                src_data['model'], tgt_data['val'], device=args.device)
            key = '%s→%s' % (src_name, tgt_name)
            results['cross_domain'][key] = {'auroc': auroc, 'f1': f1}
            print("  %s: AUROC=%.4f  F1=%.4f" % (key, auroc, f1))

    # 4. Visualization
    print("\n[4/4] Generating figures...")
    plot_results(results, OUT)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("\nIn-Domain:")
    for name, r in results['in_domain'].items():
        print("  %-12s AUROC=%.4f  F1=%.4f" % (name, r['auroc'], r['f1']))
    print("\nCross-Domain:")
    for key, r in results['cross_domain'].items():
        print("  %-20s AUROC=%.4f  F1=%.4f" % (key, r['auroc'], r['f1']))
    print("\nFigures saved to: %s" % OUT)

    # Save results
    torch.save(results, OUT / 'benchmark_results.pt')
    print("Results saved to: %s" % (OUT / 'benchmark_results.pt'))


if __name__ == '__main__':
    main()
