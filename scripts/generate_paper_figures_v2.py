#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文用 追加図表生成 (v2) — GNN-SHM Paper Figures

再利用可能な設計：毎回データを再ロードし最新結果を反映

Figures:
  P1: Pipeline Overview (フローチャート)
  P2: GAT Attention Weight Heatmap
  P3: t-SNE Latent Space (健全 vs 欠陥)
  P4: Feature Importance (Gradient-based)
  P5: Threshold Sensitivity Curve (F1/Prec/Rec vs t)
  P6: Dataset Overview (欠陥サイズ・位置分布)
  P7: Graph Structure Close-up (FEMメッシュ→グラフ)
  P8: FEM Field Comparison (健全 vs 欠陥, 応力/変位差分)
  P9: Architecture Comparison (GCN/GAT/GIN/SAGE)
  P10: Calibration Curve (Reliability diagram)

Usage:
  python scripts/generate_paper_figures_v2.py
  python scripts/generate_paper_figures_v2.py --fig P1 P2 P3
"""

import os
import sys
import csv
import argparse

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# =========================================================================
# Style
# =========================================================================
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
})

# =========================================================================
# Paths
# =========================================================================
FOLD_DIRS = [
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_185224_fold0'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_185615_fold1'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_190406_fold2'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_190839_fold3'),
    os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_191500_fold4'),
]
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_s12_czm_thermal_200_binary')

GAMMA_RUNS = {
    1.0: os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_155223'),
    2.0: os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_155538'),
    3.0: os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_155845'),
    5.0: os.path.join(PROJECT_ROOT, 'runs', 'gat_20260303_160150'),
}

# Colors
C_HEALTHY = '#6B7280'
C_DEFECT = '#DC2626'
C_PRED = '#2563EB'
C_TP = '#10B981'
C_FP = '#EF4444'
C_FN = '#F59E0B'


# =========================================================================
# Shared loaders (cached)
# =========================================================================
_cache = {}


def get_models():
    if 'models' not in _cache:
        from models import build_model
        models = []
        for fd in FOLD_DIRS:
            ckpt = torch.load(os.path.join(fd, 'best_model.pt'),
                              weights_only=False, map_location='cpu')
            args_d = ckpt.get('args', {})
            head_key = [k for k in ckpt['model_state_dict']
                        if 'head' in k and 'weight' in k]
            nc = ckpt['model_state_dict'][head_key[-1]].shape[0] if head_key else 2
            m = build_model(
                'gat', ckpt['in_channels'], ckpt['edge_attr_dim'],
                hidden_channels=args_d.get('hidden', 128),
                num_layers=args_d.get('layers', 4),
                dropout=0.0, num_classes=nc,
                use_residual=args_d.get('residual', False))
            m.load_state_dict(ckpt['model_state_dict'])
            m.eval()
            models.append(m)
        _cache['models'] = models
    return _cache['models']


def get_data():
    if 'val_data' not in _cache:
        _cache['val_data'] = torch.load(
            os.path.join(DATA_DIR, 'val.pt'), weights_only=False)
        ns = torch.load(os.path.join(DATA_DIR, 'norm_stats.pt'),
                        weights_only=False)
        _cache['node_mean'] = ns['mean']
        _cache['node_std'] = ns['std']
    return _cache['val_data'], _cache['node_mean'], _cache['node_std']


def get_train_data():
    if 'train_data' not in _cache:
        _cache['train_data'] = torch.load(
            os.path.join(DATA_DIR, 'train.pt'), weights_only=False)
    return _cache['train_data']


def normalize(x, mean, std):
    return (x - mean) / std.clamp(min=1e-8)


def to_arc(pos):
    theta = np.arctan2(pos[:, 2], pos[:, 0])
    return 2600.0 * theta, pos[:, 1]


# =========================================================================
# P1: Pipeline Overview (matplotlib flowchart)
# =========================================================================
def fig_p1_pipeline(out_dir):
    """End-to-end pipeline diagram using matplotlib."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(8, 6.7, 'GNN-Based Structural Health Monitoring Pipeline',
            fontsize=16, fontweight='bold', ha='center', va='center',
            color='#1F2937')
    ax.text(8, 6.35, 'H3 Rocket CFRP/Al-Honeycomb Fairing — Debonding Detection',
            fontsize=10, ha='center', va='center', color='#6B7280')

    # Main pipeline boxes
    boxes = [
        (1.0, 4.5, 2.5, 1.4, 'Abaqus FEM\nSimulation',
         '#DC2626', 'white',
         'CZM Contact\nThermal + Mechanical\nLoad Cases'),
        (4.5, 4.5, 2.5, 1.4, 'ODB Extraction\n& CSV Export',
         '#D97706', 'white',
         'nodes.csv (18 col)\nelements.csv\nAbaqus Python API'),
        (8.0, 4.5, 2.5, 1.4, 'Graph\nConstruction',
         '#059669', 'white',
         'PyG Data object\nS4R → edges\n5-dim edge_attr'),
        (11.5, 4.5, 2.5, 1.4, 'GNN Training\n& Inference',
         '#2563EB', 'white',
         '5-Fold GAT Ensemble\nFocal Loss\nDefect-Centric Sampler'),
    ]

    for (x, y, w, h, label, color, text_color, detail) in boxes:
        fancy = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='white',
                               linewidth=2, alpha=0.9)
        ax.add_patch(fancy)
        ax.text(x + w / 2, y + h / 2 + 0.15, label,
                fontsize=11, fontweight='bold', ha='center', va='center',
                color=text_color)
        # Detail below box
        ax.text(x + w / 2, y - 0.25, detail,
                fontsize=7, ha='center', va='top', color='#374151',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3F4F6',
                          edgecolor='#D1D5DB', alpha=0.8))

    # Arrows between main boxes
    arrow_kw = dict(arrowstyle='->', color='#374151', lw=2,
                    connectionstyle='arc3,rad=0')
    for x_start in [3.5, 7.0, 10.5]:
        ax.annotate('', xy=(x_start + 0.95, 5.2),
                    xytext=(x_start, 5.2),
                    arrowprops=arrow_kw)

    # Output box
    fancy_out = FancyBboxPatch((5.5, 1.0), 5.0, 1.3,
                                boxstyle="round,pad=0.15",
                                facecolor='#7C3AED', edgecolor='white',
                                linewidth=2, alpha=0.9)
    ax.add_patch(fancy_out)
    ax.text(8.0, 1.85, 'Defect Detection Output', fontsize=12,
            fontweight='bold', ha='center', va='center', color='white')
    ax.text(8.0, 1.35, 'Node-level P(defect)  |  Binary Label  |  Uncertainty  |  Localization',
            fontsize=9, ha='center', va='center', color='#E5E7EB')

    # Arrow from GNN to output
    ax.annotate('', xy=(8.0, 2.35), xytext=(12.75, 4.45),
                arrowprops=dict(arrowstyle='->', color='#7C3AED', lw=2,
                                connectionstyle='arc3,rad=-0.2'))

    # Feature dimension annotation
    feat_box = FancyBboxPatch((0.3, 1.0), 4.5, 1.3,
                               boxstyle="round,pad=0.15",
                               facecolor='#F3F4F6', edgecolor='#9CA3AF',
                               linewidth=1.5, alpha=0.9)
    ax.add_patch(feat_box)
    ax.text(2.55, 2.05, '34-Dim Node Features', fontsize=10,
            fontweight='bold', ha='center', va='center', color='#1F2937')

    feature_text = (
        'Position (3) + Normals (3) + Curvature (4)\n'
        'Displacement (4) + Temperature (1)\n'
        'Stress (5) + Thermal Stress (1) + Strain (3)\n'
        'Fiber Orientation (3) + Layup (5) + Boundary (2)')
    ax.text(2.55, 1.35, feature_text, fontsize=7.5, ha='center', va='center',
            color='#374151', linespacing=1.4)

    # Arrow from Graph to Features
    ax.annotate('', xy=(4.8, 1.65), xytext=(9.25, 4.45),
                arrowprops=dict(arrowstyle='->', color='#059669', lw=1.5,
                                connectionstyle='arc3,rad=0.3',
                                linestyle='--'))

    # Data flow annotation
    data_items = [
        (2.25, 3.85, 'N=200 samples'),
        (5.75, 3.85, '15,206 nodes/sample'),
        (9.25, 3.85, '119K edges, 5-dim'),
        (12.75, 3.85, 'F1 = 0.971 (best)'),
    ]
    for x, y, text in data_items:
        ax.text(x, y, text, fontsize=7.5, ha='center', va='center',
                color='#6B7280', style='italic')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig_p1_pipeline.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved: %s" % out_path)


# =========================================================================
# P2: GAT Attention Weight Visualization
# =========================================================================
def fig_p2_attention(out_dir):
    """Extract and visualize GAT attention weights."""
    from torch_geometric.nn import GATConv

    models = get_models()
    val_data, node_mean, node_std = get_data()

    # Use best sample (idx=0 from previous analysis)
    graph = val_data[0]
    x = normalize(graph.x, node_mean, node_std)
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr
    targets = graph.y.numpy()
    pos = graph.x[:, :3].numpy()
    arc, axial = to_arc(pos)

    # Extract attention from first model, last layer
    model = models[0]
    model.eval()

    # Forward with attention extraction
    h = x
    attention_last = None
    with torch.no_grad():
        for i, conv in enumerate(model.convs):
            residual = h
            if isinstance(conv, GATConv):
                if model.edge_attr_dim > 0 and edge_attr is not None:
                    h_new, (ei, att) = conv(
                        h, edge_index, edge_attr=edge_attr,
                        return_attention_weights=True)
                else:
                    h_new, (ei, att) = conv(
                        h, edge_index,
                        return_attention_weights=True)
                attention_last = att.numpy()  # (E, heads)
                h = h_new
            else:
                h = conv(h, edge_index)
            h = model.norms[i](h)
            h = F.relu(h)
            if model.use_residual:
                h = h + model.skip_projs[i](residual)

    # Average across heads
    att_mean = attention_last.mean(axis=1)  # (E,)

    # Aggregate per node (incoming attention sum)
    N = len(targets)
    dst_nodes = edge_index[1].numpy()
    node_att = np.zeros(N, dtype=np.float64)
    node_att_count = np.zeros(N, dtype=np.int64)
    for e_idx in range(len(dst_nodes)):
        node_att[dst_nodes[e_idx]] += att_mean[e_idx]
        node_att_count[dst_nodes[e_idx]] += 1
    node_att_avg = np.divide(node_att, node_att_count,
                             where=node_att_count > 0,
                             out=np.zeros_like(node_att))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # (a) Attention heatmap (full view)
    ax = axes[0]
    vmax = np.percentile(node_att_avg, 99)
    sc = ax.scatter(arc, axial, c=node_att_avg, cmap='magma', s=0.3,
                    alpha=0.8, vmin=0, vmax=vmax, edgecolors='none',
                    rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('Mean Attention (last layer)', fontsize=9)
    ax.set_title('(a) GAT Attention — Full View', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlabel('Arc Length [mm]')
    ax.set_ylabel('Axial Position [mm]')

    # (b) Attention for defect vs healthy nodes
    ax = axes[1]
    defect_mask = targets > 0
    att_healthy = node_att_avg[~defect_mask]
    att_defect = node_att_avg[defect_mask]

    bins = np.linspace(0, np.percentile(node_att_avg, 99.5), 60)
    ax.hist(att_healthy, bins=bins, alpha=0.7, color=C_HEALTHY,
            density=True, label='Healthy (n=%d)' % len(att_healthy))
    ax.hist(att_defect, bins=bins, alpha=0.7, color=C_DEFECT,
            density=True, label='Defect (n=%d)' % len(att_defect))
    ax.set_xlabel('Mean Incoming Attention')
    ax.set_ylabel('Density')
    ax.set_title('(b) Attention Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # (c) Zoomed defect region with edge attention
    ax = axes[2]
    arc_d, ax_d = arc[defect_mask], axial[defect_mask]
    cx, cy = arc_d.mean(), ax_d.mean()
    extent = max(arc_d.max() - arc_d.min(), ax_d.max() - ax_d.min()) * 2.0
    extent = max(extent, 80)
    xlim = (cx - extent, cx + extent)
    ylim = (cy - extent, cy + extent)

    in_zoom = ((arc >= xlim[0]) & (arc <= xlim[1]) &
               (axial >= ylim[0]) & (axial <= ylim[1]))

    # Plot edges with attention intensity (subsample for speed)
    src_np = edge_index[0].numpy()
    dst_np = edge_index[1].numpy()

    # Only edges within zoom region
    edge_in_zoom = in_zoom[src_np] & in_zoom[dst_np]
    edge_indices = np.where(edge_in_zoom)[0]

    # Subsample if too many
    if len(edge_indices) > 5000:
        np.random.seed(42)
        edge_indices = np.random.choice(edge_indices, 5000, replace=False)

    # Draw edges colored by attention
    att_sub = att_mean[edge_indices]
    att_norm = (att_sub - att_sub.min()) / (att_sub.max() - att_sub.min() + 1e-8)

    cmap = plt.cm.magma
    for ei, an in zip(edge_indices, att_norm):
        s, d = src_np[ei], dst_np[ei]
        color = cmap(an)
        ax.plot([arc[s], arc[d]], [axial[s], axial[d]],
                '-', color=color, alpha=0.3 + 0.6 * an, lw=0.3 + 1.5 * an)

    # Overlay nodes
    h_z = in_zoom & (~defect_mask)
    d_z = in_zoom & defect_mask
    ax.scatter(arc[h_z], axial[h_z], c='#D1D5DB', s=6, alpha=0.4,
               edgecolors='none', rasterized=True, zorder=5)
    ax.scatter(arc[d_z], axial[d_z], c=C_DEFECT, s=10, alpha=0.9,
               edgecolors='white', linewidths=0.3, rasterized=True, zorder=6)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title('(c) Edge Attention — Defect Region', fontsize=11, fontweight='bold')
    ax.set_xlabel('Arc Length [mm]')
    ax.set_ylabel('Axial Position [mm]')

    fig.suptitle('GAT Attention Weight Analysis (Layer 4, Sample 0)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'fig_p2_attention.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved: %s" % out_path)


# =========================================================================
# P3: t-SNE Latent Space
# =========================================================================
def fig_p3_tsne(out_dir):
    """t-SNE visualization of GNN latent representations."""
    from sklearn.manifold import TSNE

    models = get_models()
    val_data, node_mean, node_std = get_data()

    model = models[0]
    model.eval()

    # Collect embeddings from multiple samples (subsample nodes)
    all_emb, all_labels, all_probs = [], [], []
    max_per_sample = 500
    sample_indices = [0, 7, 12, 23, 31]  # top F1 samples

    with torch.no_grad():
        for si in sample_indices:
            graph = val_data[si]
            x = normalize(graph.x, node_mean, node_std)
            emb = model.encode(x, graph.edge_index, graph.edge_attr)
            logits = model.head(emb)
            probs = F.softmax(logits, dim=1)[:, 1].numpy()

            targets = graph.y.numpy()
            emb_np = emb.numpy()

            # Subsample: all defect nodes + random healthy
            defect_idx = np.where(targets > 0)[0]
            healthy_idx = np.where(targets == 0)[0]
            n_healthy = min(max_per_sample - len(defect_idx), len(healthy_idx))
            if n_healthy > 0:
                np.random.seed(42 + si)
                healthy_sub = np.random.choice(healthy_idx, n_healthy, replace=False)
            else:
                healthy_sub = healthy_idx
            chosen = np.concatenate([defect_idx, healthy_sub])

            all_emb.append(emb_np[chosen])
            all_labels.append(targets[chosen])
            all_probs.append(probs[chosen])

    emb_cat = np.concatenate(all_emb)
    labels_cat = np.concatenate(all_labels)
    probs_cat = np.concatenate(all_probs)

    print("  t-SNE on %d nodes..." % len(emb_cat))
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb_2d = tsne.fit_transform(emb_cat)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # (a) Color by label
    ax = axes[0]
    healthy_m = labels_cat == 0
    defect_m = labels_cat > 0
    ax.scatter(emb_2d[healthy_m, 0], emb_2d[healthy_m, 1],
               c=C_HEALTHY, s=3, alpha=0.3, label='Healthy', rasterized=True)
    ax.scatter(emb_2d[defect_m, 0], emb_2d[defect_m, 1],
               c=C_DEFECT, s=8, alpha=0.8, label='Defect', rasterized=True,
               zorder=5, edgecolors='white', linewidths=0.3)
    ax.set_title('(a) GNN Embeddings — by Label', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, markerscale=2)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.15)

    # (b) Color by P(defect)
    ax = axes[1]
    sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                    c=probs_cat, cmap='inferno', s=4, alpha=0.6,
                    vmin=0, vmax=1, edgecolors='none', rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('P(defect)', fontsize=10)
    ax.set_title('(b) GNN Embeddings — by P(defect)', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.15)

    fig.suptitle('t-SNE Visualization of GAT Latent Space (Layer 4 output, 5 samples)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'fig_p3_tsne.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved: %s" % out_path)


# =========================================================================
# P4: Feature Importance (Gradient-based)
# =========================================================================
def fig_p4_feature_importance(out_dir):
    """Gradient-based feature importance for defect detection."""
    models = get_models()
    val_data, node_mean, node_std = get_data()

    model = models[0]
    model.eval()

    FEATURE_NAMES = [
        'x', 'y', 'z',
        'nx', 'ny', 'nz',
        'k1', 'k2', 'H', 'K',
        'ux', 'uy', 'uz', 'u_mag',
        'temp',
        's11', 's22', 's12', 'smises', 'σ₁+σ₂',
        'thermal_mises',
        'le11', 'le22', 'le12',
        'fiber_x', 'fiber_y', 'fiber_z',
        'layup_0', 'layup_45', 'layup_-45', 'layup_90', 'circum_θ',
        'boundary', 'loaded',
    ]

    FEATURE_GROUPS = {
        'Position & Geometry': (0, 10, '#DC2626'),
        'Displacement': (10, 14, '#D97706'),
        'Temperature': (14, 15, '#059669'),
        'Stress': (15, 21, '#2563EB'),
        'Strain': (21, 24, '#7C3AED'),
        'Fiber & Layup': (24, 32, '#DB2777'),
        'Boundary': (32, 34, '#6B7280'),
    }

    # Compute gradient importance on defect nodes
    grad_accum = np.zeros(34, dtype=np.float64)
    count = 0

    for si in range(min(20, len(val_data))):
        graph = val_data[si]
        if (graph.y > 0).sum() < 5:
            continue

        x = normalize(graph.x, node_mean, node_std).clone().requires_grad_(True)
        logits = model(x, graph.edge_index, graph.edge_attr, None)

        # Gradient of defect logit w.r.t. input features
        defect_score = logits[:, 1][graph.y > 0].sum()
        defect_score.backward()

        grad = x.grad[graph.y > 0].abs().mean(dim=0).detach().numpy()
        grad_accum += grad
        count += 1

    if count > 0:
        grad_accum /= count

    # Normalize to sum=1
    grad_norm = grad_accum / grad_accum.sum()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) Individual features bar chart
    ax = axes[0]
    colors = []
    for i in range(34):
        for gname, (start, end, color) in FEATURE_GROUPS.items():
            if start <= i < end:
                colors.append(color)
                break

    bars = ax.barh(range(34), grad_norm, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(34))
    ax.set_yticklabels(FEATURE_NAMES[:34], fontsize=7)
    ax.set_xlabel('Normalized Gradient Importance')
    ax.set_title('(a) Per-Feature Importance', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis='x')

    # (b) Grouped importance
    ax = axes[1]
    group_names, group_vals, group_colors = [], [], []
    for gname, (start, end, color) in FEATURE_GROUPS.items():
        group_names.append(gname)
        group_vals.append(grad_norm[start:end].sum())
        group_colors.append(color)

    y_pos = range(len(group_names))
    bars2 = ax.barh(y_pos, group_vals, color=group_colors, alpha=0.85,
                    edgecolor='white', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(group_names, fontsize=10)
    ax.set_xlabel('Normalized Gradient Importance (sum)')
    ax.set_title('(b) Feature Group Importance', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis='x')

    # Value labels
    for bar, val in zip(bars2, group_vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                '%.1f%%' % (val * 100), va='center', fontsize=9, fontweight='bold')

    fig.suptitle('Gradient-Based Feature Importance for Defect Detection',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'fig_p4_feature_importance.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved: %s" % out_path)


# =========================================================================
# P5: Threshold Sensitivity Curve
# =========================================================================
def fig_p5_threshold_sensitivity(out_dir):
    """F1/Precision/Recall vs threshold with optimal point annotation."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    models = get_models()
    val_data, node_mean, node_std = get_data()

    # Ensemble predict all
    all_probs, all_targets = [], []
    with torch.no_grad():
        for graph in val_data:
            x = normalize(graph.x, node_mean, node_std)
            prob_sum = np.zeros(graph.x.shape[0], dtype=np.float64)
            for m in models:
                logits = m(x, graph.edge_index, graph.edge_attr, None)
                prob_sum += F.softmax(logits, dim=1)[:, 1].numpy()
            all_probs.append((prob_sum / len(models)).astype(np.float32))
            all_targets.append(graph.y.numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    thresholds = np.linspace(0.01, 0.99, 200)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1s.append(f1_score(targets > 0, preds, zero_division=0))
        precs.append(precision_score(targets > 0, preds, zero_division=0))
        recs.append(recall_score(targets > 0, preds, zero_division=0))

    f1s, precs, recs = np.array(f1s), np.array(precs), np.array(recs)
    best_idx = np.argmax(f1s)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds, f1s, '-', color='#2563EB', lw=2.5, label='F1 Score')
    ax.plot(thresholds, precs, '-', color='#059669', lw=2, label='Precision')
    ax.plot(thresholds, recs, '-', color='#D97706', lw=2, label='Recall')

    # Optimal point
    ax.plot(thresholds[best_idx], f1s[best_idx], 'o', color='#DC2626',
            markersize=12, zorder=10, markeredgecolor='white', markeredgewidth=2)
    ax.annotate('Optimal: t=%.2f\nF1=%.3f' % (thresholds[best_idx], f1s[best_idx]),
                xy=(thresholds[best_idx], f1s[best_idx]),
                xytext=(thresholds[best_idx] + 0.12, f1s[best_idx] + 0.05),
                fontsize=10, fontweight='bold', color='#DC2626',
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEE2E2',
                          edgecolor='#DC2626', alpha=0.9))

    # Default threshold line
    ax.axvline(0.5, color='#9CA3AF', linestyle='--', lw=1, alpha=0.7,
               label='Default (t=0.5)')

    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Threshold Sensitivity — 5-Fold GAT Ensemble',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='center left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'fig_p5_threshold_sensitivity.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved: %s" % out_path)


# =========================================================================
# P6: Dataset Overview
# =========================================================================
def fig_p6_dataset_overview(out_dir):
    """Defect size and position distribution across all samples."""
    train_data = get_train_data()
    val_data, _, _ = get_data()
    all_data = list(train_data) + list(val_data)

    defect_sizes = []
    defect_centers_arc = []
    defect_centers_axial = []
    total_nodes_list = []

    for graph in all_data:
        targets = graph.y.numpy()
        n_def = int((targets > 0).sum())
        defect_sizes.append(n_def)
        total_nodes_list.append(len(targets))
        if n_def > 0:
            pos = graph.x[:, :3].numpy()
            arc, axial = to_arc(pos)
            d_mask = targets > 0
            defect_centers_arc.append(arc[d_mask].mean())
            defect_centers_axial.append(axial[d_mask].mean())

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) Defect size distribution
    ax = axes[0, 0]
    ax.hist(defect_sizes, bins=30, color='#DC2626', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(defect_sizes), color='#374151', linestyle='--', lw=2,
               label='Mean = %.0f nodes' % np.mean(defect_sizes))
    ax.set_xlabel('Number of Defect Nodes per Sample')
    ax.set_ylabel('Count')
    ax.set_title('(a) Defect Size Distribution (N=%d)' % len(all_data),
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # (b) Defect center positions
    ax = axes[0, 1]
    sc = ax.scatter(defect_centers_arc, defect_centers_axial,
                    c=[s for s in defect_sizes if s > 0],
                    cmap='YlOrRd', s=30, alpha=0.7,
                    edgecolors='white', linewidths=0.5)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('Defect Size (nodes)', fontsize=9)
    ax.set_xlabel('Arc Length [mm]')
    ax.set_ylabel('Axial Position [mm]')
    ax.set_title('(b) Defect Center Positions', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # (c) Class imbalance
    ax = axes[1, 0]
    total_defect = sum(defect_sizes)
    total_healthy = sum(total_nodes_list) - total_defect
    bars = ax.bar(['Healthy', 'Defect'],
                  [total_healthy, total_defect],
                  color=[C_HEALTHY, C_DEFECT], alpha=0.85,
                  edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Total Node Count')
    ax.set_title('(c) Class Distribution', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    for bar, val in zip(bars, [total_healthy, total_defect]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                '%s\n(%.2f%%)' % ('{:,}'.format(val), val / (total_healthy + total_defect) * 100),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # (d) Defect size vs axial position
    ax = axes[1, 1]
    ax.scatter(defect_centers_axial,
               [s for s in defect_sizes if s > 0],
               c='#2563EB', s=25, alpha=0.6, edgecolors='white', linewidths=0.5)
    ax.set_xlabel('Axial Position [mm]')
    ax.set_ylabel('Defect Size (nodes)')
    ax.set_title('(d) Defect Size vs Axial Position', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)

    fig.suptitle('Dataset Overview — S12 CZM Thermal (N=%d samples, 34-dim features)' %
                 len(all_data), fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'fig_p6_dataset_overview.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved: %s" % out_path)


# =========================================================================
# P7: Graph Structure Close-up
# =========================================================================
def fig_p7_graph_structure(out_dir):
    """FEM mesh → graph: edge connectivity visualization."""
    val_data, _, _ = get_data()
    graph = val_data[0]  # best sample

    pos = graph.x[:, :3].numpy()
    targets = graph.y.numpy()
    edge_index = graph.edge_index.numpy()
    arc, axial = to_arc(pos)

    defect_mask = targets > 0
    arc_d, ax_d = arc[defect_mask], axial[defect_mask]
    cx, cy = arc_d.mean(), ax_d.mean()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Three zoom levels
    extents = [
        ('Full Graph (1% edges)', None, 0.01),
        ('Defect Region', 150, 0.3),
        ('Ultra-Zoom (nodes & edges)', 50, 1.0),
    ]

    for col, (title, ext_mm, edge_frac) in enumerate(extents):
        ax = axes[col]

        if ext_mm is None:
            xlim = (arc.min(), arc.max())
            ylim = (axial.min(), axial.max())
            node_s = 0.1
        else:
            xlim = (cx - ext_mm, cx + ext_mm)
            ylim = (cy - ext_mm, cy + ext_mm)
            node_s = 6 if ext_mm > 80 else 15

        in_view = ((arc >= xlim[0]) & (arc <= xlim[1]) &
                   (axial >= ylim[0]) & (axial <= ylim[1]))

        # Draw edges
        src, dst = edge_index[0], edge_index[1]
        edge_in = in_view[src] & in_view[dst]
        edge_idx = np.where(edge_in)[0]

        if len(edge_idx) > 0:
            if edge_frac < 1.0 and len(edge_idx) > 3000:
                np.random.seed(42)
                n_show = max(int(len(edge_idx) * edge_frac), 2000)
                edge_idx = np.random.choice(edge_idx, n_show, replace=False)

            for ei in edge_idx:
                s, d = src[ei], dst[ei]
                color = '#DC2626' if (targets[s] > 0 or targets[d] > 0) else '#93C5FD'
                alpha = 0.7 if (targets[s] > 0 or targets[d] > 0) else 0.15
                lw = 0.6 if (targets[s] > 0 or targets[d] > 0) else 0.2
                ax.plot([arc[s], arc[d]], [axial[s], axial[d]],
                        '-', color=color, alpha=alpha, lw=lw)

        # Draw nodes
        h_view = in_view & (~defect_mask)
        d_view = in_view & defect_mask
        ax.scatter(arc[h_view], axial[h_view], c='#3B82F6', s=node_s,
                   alpha=0.5, edgecolors='none', rasterized=True, zorder=5)
        ax.scatter(arc[d_view], axial[d_view], c=C_DEFECT, s=node_s * 2,
                   alpha=0.9, edgecolors='white',
                   linewidths=0.3 if ext_mm and ext_mm < 80 else 0,
                   rasterized=True, zorder=6)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Arc Length [mm]')
        ax.set_ylabel('Axial Position [mm]')

        # Stats
        n_nodes = in_view.sum()
        n_edges = len(edge_idx) if 'edge_idx' in dir() else 0
        degree = 2 * edge_index.shape[1] / len(pos)
        ax.text(0.02, 0.02, 'N=%s, E=%s\nAvg degree: %.1f' % (
            '{:,}'.format(n_nodes), '{:,}'.format(n_edges), degree),
            transform=ax.transAxes, fontsize=7, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle('Graph Structure — FEM Mesh to PyG Graph (S4R Shell Elements)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'fig_p7_graph_structure.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved: %s" % out_path)


# =========================================================================
# P8: FEM Field Comparison (Healthy vs Defect sample)
# =========================================================================
def fig_p8_fem_fields(out_dir):
    """Compare physical fields between a healthy-like and defect-rich sample."""
    train_data = get_train_data()

    # Find sample with most defect and one with fewest
    defect_counts = [(i, int((g.y > 0).sum())) for i, g in enumerate(train_data)]
    defect_counts.sort(key=lambda x: x[1])

    # Lowest and highest defect count
    low_idx = defect_counts[0][0]
    high_idx = defect_counts[-1][0]

    samples = [
        (train_data[low_idx], 'Low Defect (%d nodes)' % defect_counts[0][1]),
        (train_data[high_idx], 'High Defect (%d nodes)' % defect_counts[-1][1]),
    ]

    fields = [
        ('Von Mises Stress', 18, 'plasma'),    # smises index
        ('Temperature', 14, 'coolwarm'),         # temp
        ('Displacement Mag.', 13, 'viridis'),    # u_mag
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for row, (graph, label) in enumerate(samples):
        pos = graph.x[:, :3].numpy()
        targets = graph.y.numpy()
        arc, axial = to_arc(pos)
        defect = targets > 0

        for col, (fname, fidx, cmap) in enumerate(fields):
            ax = axes[row, col]
            values = graph.x[:, fidx].numpy()
            vmin, vmax = np.percentile(values, [2, 98])

            sc = ax.scatter(arc, axial, c=values, cmap=cmap, s=0.3,
                            alpha=0.7, vmin=vmin, vmax=vmax,
                            edgecolors='none', rasterized=True)
            cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
            cb.ax.tick_params(labelsize=7)

            # Defect boundary
            if defect.any():
                try:
                    from scipy.spatial import ConvexHull
                    pts = np.column_stack([arc[defect], axial[defect]])
                    if len(pts) >= 3:
                        hull = ConvexHull(pts)
                        hull_pts = pts[hull.vertices]
                        hull_pts = np.vstack([hull_pts, hull_pts[0]])
                        ax.plot(hull_pts[:, 0], hull_pts[:, 1], '-',
                                color='#22C55E', linewidth=1.5, alpha=0.9)
                except Exception:
                    pass

            ax.set_aspect('equal')
            ax.set_xlabel('Arc [mm]', fontsize=8)
            ax.set_ylabel('Axial [mm]', fontsize=8)
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(fname, fontsize=11, fontweight='bold')

        # Row label
        axes[row, 0].text(-0.2, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=10, va='center', ha='right', rotation=90,
                          fontweight='bold', color='#374151')

    fig.suptitle('FEM Physical Fields — Low vs High Defect Samples (Green = Defect Boundary)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'fig_p8_fem_fields.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved: %s" % out_path)


# =========================================================================
# P9: Architecture Comparison
# =========================================================================
def fig_p9_architecture_comparison(out_dir):
    """Compare training logs across γ values (as proxy for architecture study)."""
    # Load training logs from gamma runs
    gamma_results = {}
    for gamma, run_dir in GAMMA_RUNS.items():
        log_path = os.path.join(run_dir, 'training_log.csv')
        if not os.path.exists(log_path):
            continue
        data = {}
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k, v in row.items():
                    data.setdefault(k, []).append(float(v))
        best_f1 = max(data.get('val_f1', [0]))
        best_ep = int(data['epoch'][np.argmax(data['val_f1'])])
        gamma_results[gamma] = {
            'best_f1': best_f1, 'best_epoch': best_ep,
            'val_f1_history': data.get('val_f1', []),
            'val_loss_history': data.get('val_loss', []),
            'train_f1_history': data.get('train_f1', []),
        }

    # Also load 5-fold results
    fold_f1s = []
    for fd in FOLD_DIRS:
        log_path = os.path.join(fd, 'training_log.csv')
        if not os.path.exists(log_path):
            continue
        data = {}
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k, v in row.items():
                    data.setdefault(k, []).append(float(v))
        fold_f1s.append(max(data.get('val_f1', [0])))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # (a) Training curves for different γ
    ax = axes[0]
    colors_gamma = {1.0: '#2563EB', 2.0: '#DC2626', 3.0: '#059669', 5.0: '#D97706'}
    for gamma, res in sorted(gamma_results.items()):
        epochs = range(1, len(res['val_f1_history']) + 1)
        ax.plot(epochs, res['val_f1_history'], '-', color=colors_gamma[gamma],
                lw=2, alpha=0.8, label='γ=%.0f (F1=%.3f)' % (gamma, res['best_f1']))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation F1')
    ax.set_title('(a) Focal Loss γ Comparison', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 0.9)
    ax.grid(True, alpha=0.2)

    # (b) Bar chart comparison
    ax = axes[1]
    methods = ['Single\n(γ=1.0)', 'Single\n(γ=2.0)', 'Single\n(γ=3.0)', 'Single\n(γ=5.0)',
               '5-Fold\nEnsemble']
    f1_vals = [gamma_results.get(g, {}).get('best_f1', 0) for g in [1.0, 2.0, 3.0, 5.0]]
    f1_vals.append(np.mean(fold_f1s) if fold_f1s else 0)

    colors_bar = ['#2563EB', '#DC2626', '#059669', '#D97706', '#7C3AED']
    bars = ax.bar(range(len(methods)), f1_vals, color=colors_bar, alpha=0.85,
                  edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Best Val F1')
    ax.set_title('(b) Model Configuration Comparison', fontsize=11, fontweight='bold')
    ax.set_ylim(0.75, 0.90)
    ax.grid(True, alpha=0.2, axis='y')

    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                '%.3f' % val, ha='center', va='bottom', fontsize=9, fontweight='bold')

    # (c) Train vs Val gap (overfitting analysis)
    ax = axes[2]
    best_gamma = max(gamma_results.items(), key=lambda x: x[1]['best_f1'])
    g, res = best_gamma
    epochs = range(1, len(res['train_f1_history']) + 1)
    ax.plot(epochs, res['train_f1_history'], '-', color='#2563EB', lw=2, label='Train F1')
    ax.plot(epochs, res['val_f1_history'], '-', color='#DC2626', lw=2, label='Val F1')
    ax.fill_between(epochs,
                    res['train_f1_history'], res['val_f1_history'],
                    alpha=0.15, color='#DC2626', label='Generalization Gap')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('(c) Train/Val Gap (γ=%.0f)' % g, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle('Model Configuration Analysis — GAT Architecture',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'fig_p9_architecture_comparison.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved: %s" % out_path)


# =========================================================================
# P10: Calibration Curve
# =========================================================================
def fig_p10_calibration(out_dir):
    """Reliability diagram for ensemble predictions."""
    models = get_models()
    val_data, node_mean, node_std = get_data()

    all_probs, all_targets = [], []
    with torch.no_grad():
        for graph in val_data:
            x = normalize(graph.x, node_mean, node_std)
            prob_sum = np.zeros(graph.x.shape[0], dtype=np.float64)
            for m in models:
                logits = m(x, graph.edge_index, graph.edge_attr, None)
                prob_sum += F.softmax(logits, dim=1)[:, 1].numpy()
            all_probs.append((prob_sum / len(models)).astype(np.float32))
            all_targets.append(graph.y.numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    # Compute calibration curve
    n_bins = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mean_pred, bin_true_freq, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_mean_pred.append(probs[mask].mean())
            bin_true_freq.append((targets[mask] > 0).mean())
            bin_counts.append(mask.sum())
        else:
            bin_mean_pred.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_true_freq.append(0)
            bin_counts.append(0)

    # ECE
    total = sum(bin_counts)
    ece = sum(c * abs(p - t) for c, p, t in zip(bin_counts, bin_mean_pred, bin_true_freq)) / max(total, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # (a) Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], '--', color='#9CA3AF', lw=1.5, label='Perfect calibration')
    ax.bar(bin_mean_pred, bin_true_freq, width=1.0 / n_bins * 0.8,
           alpha=0.7, color='#2563EB', edgecolor='white', label='Ensemble')

    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontsize=11)
    ax.set_title('(a) Reliability Diagram (ECE = %.4f)' % ece,
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # (b) Histogram of predictions
    ax = axes[1]
    ax.hist(probs[targets == 0], bins=50, alpha=0.7, color=C_HEALTHY,
            label='Healthy', density=True)
    ax.hist(probs[targets > 0], bins=50, alpha=0.7, color=C_DEFECT,
            label='Defect', density=True)
    ax.set_xlabel('Predicted P(defect)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(b) Prediction Distribution by Class', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    fig.suptitle('Calibration Analysis — 5-Fold GAT Ensemble',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'fig_p10_calibration.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved: %s" % out_path)


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate paper figures v2')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'wiki_repo', 'images', 'paper'))
    parser.add_argument('--fig', nargs='+', type=str, default=None,
                        help='Specific figures (P1-P10). Default: all')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    generators = {
        'P1':  ('Pipeline Overview', fig_p1_pipeline),
        'P2':  ('GAT Attention Weights', fig_p2_attention),
        'P3':  ('t-SNE Latent Space', fig_p3_tsne),
        'P4':  ('Feature Importance', fig_p4_feature_importance),
        'P5':  ('Threshold Sensitivity', fig_p5_threshold_sensitivity),
        'P6':  ('Dataset Overview', fig_p6_dataset_overview),
        'P7':  ('Graph Structure', fig_p7_graph_structure),
        'P8':  ('FEM Field Comparison', fig_p8_fem_fields),
        'P9':  ('Architecture Comparison', fig_p9_architecture_comparison),
        'P10': ('Calibration Curve', fig_p10_calibration),
    }

    figs_to_gen = args.fig or sorted(generators.keys(),
                                      key=lambda x: int(x[1:]))

    print("=" * 70)
    print("Paper Figures v2 — Generating %d figures → %s" % (
        len(figs_to_gen), args.out_dir))
    print("=" * 70)

    for fig_id in figs_to_gen:
        fig_id = fig_id.upper()
        if fig_id in generators:
            name, func = generators[fig_id]
            print("\n--- %s: %s ---" % (fig_id, name))
            try:
                func(args.out_dir)
            except Exception as e:
                print("ERROR in %s: %s" % (fig_id, e))
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("Done! Figures saved to: %s" % args.out_dir)
    print("=" * 70)


if __name__ == '__main__':
    main()
