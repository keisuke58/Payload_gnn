#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAT Attention Visualization — Plot attention weights on FEM mesh.

Shows which nodes/edges the GNN focuses on for defect detection.
Generates:
  1. Attention heatmap on mesh (top-k attended nodes)
  2. Attention vs ground truth comparison
  3. Edge attention distribution

Usage:
  python3 scripts/plot_attention.py \
      --model runs/verify_realistic/gat_20260301_010732/best_model.pt \
      --data_dir data/processed_s12_thermal_700v2_5class \
      --output results/attention/
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from models import build_model

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')


def extract_attention(model, data, device='cpu'):
    """Forward pass with attention extraction from GAT layers.

    Directly calls GATConv with return_attention_weights=True.
    Works with BaseGNN (GATModel) from models.py.
    """
    model.eval()
    data = data.to(device)

    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    all_alphas = []

    with torch.no_grad():
        # Input projection (BaseGNN.lin_in)
        if hasattr(model, 'lin_in'):
            x = model.lin_in(x)

        # Process through conv layers
        for i, conv in enumerate(model.convs):
            conv_type = type(conv).__name__

            if 'GAT' in conv_type:
                # GATConv: call with return_attention_weights=True
                try:
                    if edge_attr is not None and hasattr(conv, 'lin_edge') and conv.lin_edge is not None:
                        out, (ei, alpha) = conv(x, edge_index, edge_attr=edge_attr,
                                                 return_attention_weights=True)
                    else:
                        out, (ei, alpha) = conv(x, edge_index,
                                                 return_attention_weights=True)
                    all_alphas.append(alpha.detach().cpu())
                    x = out
                except Exception as e:
                    print(f"  Layer {i} attention extraction failed: {e}")
                    x = conv(x, edge_index)
            else:
                x = conv(x, edge_index)

            # BatchNorm + activation
            if hasattr(model, 'bns') and i < len(model.bns):
                x = model.bns[i](x)
            x = F.elu(x)

    return all_alphas, data


def compute_node_attention(alphas, edge_index, n_nodes):
    """Aggregate edge attention to node-level attention scores."""
    node_attn = torch.zeros(n_nodes)
    for alpha in alphas:
        # alpha: (E, heads) or (E,)
        if alpha.dim() > 1:
            alpha_mean = alpha.mean(dim=1)  # average over heads
        else:
            alpha_mean = alpha

        # Sum incoming attention for each node
        dst = edge_index[1].cpu()
        for j in range(len(dst)):
            if dst[j] < n_nodes:
                node_attn[dst[j]] += alpha_mean[j].item()

    # Normalize
    node_attn = node_attn / (node_attn.max() + 1e-8)
    return node_attn


def plot_attention_mesh(data, node_attn, labels, output_path, title=''):
    """Plot attention heatmap on 2D mesh projection."""
    pos = data.pos.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(title or 'GAT Attention Analysis', fontsize=16, fontweight='bold')

    # 1. Attention heatmap
    ax = axes[0]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=node_attn.numpy(),
                    cmap='hot', s=1, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Attention Score', shrink=0.8)
    ax.set_title('(a) Attention Heatmap', fontsize=13)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_aspect('equal')

    # 2. Ground truth labels
    ax = axes[1]
    colors = ['#2196F3' if l == 0 else '#F44336' for l in labels.numpy()]
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=1, alpha=0.6)
    ax.set_title('(b) Ground Truth (blue=healthy, red=defect)', fontsize=13)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_aspect('equal')

    # 3. Attention on defect vs healthy
    ax = axes[2]
    defect_mask = labels > 0
    healthy_mask = labels == 0

    attn_healthy = node_attn[healthy_mask].numpy()
    attn_defect = node_attn[defect_mask].numpy()

    ax.hist(attn_healthy, bins=50, alpha=0.6, label='Healthy nodes',
            color='#2196F3', density=True)
    if len(attn_defect) > 0:
        ax.hist(attn_defect, bins=50, alpha=0.6, label='Defect nodes',
                color='#F44336', density=True)
    ax.set_title('(c) Attention Distribution', fontsize=13)
    ax.set_xlabel('Attention Score')
    ax.set_ylabel('Density')
    ax.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_attention_3d(data, node_attn, output_path):
    """3D scatter plot of attention on mesh."""
    pos = data.pos.cpu().numpy()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Top 5% attention nodes highlighted
    threshold = np.percentile(node_attn.numpy(), 95)
    high_mask = node_attn.numpy() > threshold
    low_mask = ~high_mask

    ax.scatter(pos[low_mask, 0], pos[low_mask, 2], pos[low_mask, 1],
               c='lightgray', s=0.5, alpha=0.2)
    sc = ax.scatter(pos[high_mask, 0], pos[high_mask, 2], pos[high_mask, 1],
                    c=node_attn[high_mask].numpy(), cmap='hot',
                    s=5, alpha=0.9, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='Attention', shrink=0.6)

    ax.set_title('Top 5% Attention Nodes (GAT)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Z [mm]')
    ax.set_zlabel('Y [mm]')

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_layer_attention(alphas, output_path):
    """Plot attention statistics per layer."""
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 4))
    if len(alphas) == 1:
        axes = [axes]

    fig.suptitle('Edge Attention per GAT Layer', fontsize=14, fontweight='bold')

    for i, alpha in enumerate(alphas):
        ax = axes[i]
        if alpha.dim() > 1:
            # Multiple heads
            for h in range(alpha.shape[1]):
                ax.hist(alpha[:, h].numpy(), bins=50, alpha=0.4,
                        label=f'Head {h}')
            ax.legend(fontsize=8)
        else:
            ax.hist(alpha.numpy(), bins=50, alpha=0.7, color='#2196F3')
        ax.set_title(f'Layer {i+1}', fontsize=12)
        ax.set_xlabel('Attention Weight')
        ax.set_ylabel('Count')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help='Path to best_model.pt')
    parser.add_argument('--data_dir',
                        default='data/processed_s12_thermal_700v2_5class')
    parser.add_argument('--output', default='results/attention')
    parser.add_argument('--sample_idx', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Find model
    if args.model is None:
        candidates = [
            'runs/verify_realistic/gat_20260301_010732/best_model.pt',
        ]
        for c in candidates:
            if os.path.exists(c):
                args.model = c
                break
    if args.model is None or not os.path.exists(args.model):
        print("ERROR: No model found. Specify --model path")
        return

    print("=" * 60)
    print("GAT Attention Visualization")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {args.model}")
    ckpt = torch.load(args.model, weights_only=False, map_location='cpu')
    in_ch = ckpt.get('in_channels', 34)
    ea_dim = ckpt.get('edge_attr_dim', 5)
    model_args = ckpt.get('args', {})
    hidden = model_args.get('hidden', 64)
    layers = model_args.get('layers', 3)

    # Detect num_classes from last head layer in checkpoint
    num_classes = 2
    for k, v in ckpt['model_state_dict'].items():
        if 'head' in k and 'weight' in k:
            num_classes = v.shape[0]  # last match = final layer

    model = build_model('gat', in_ch, ea_dim, hidden_channels=hidden,
                         num_layers=layers, dropout=0.0, num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Model: GAT h={hidden} L={layers} classes={num_classes}")

    # Load data
    print(f"\nLoading data: {args.data_dir}")
    val_data = torch.load(f'{args.data_dir}/val.pt',
                           weights_only=False, map_location='cpu')
    norm_path = f'{args.data_dir}/norm_stats.pt'
    if os.path.exists(norm_path):
        norm = torch.load(norm_path, weights_only=False, map_location='cpu')
        for d in val_data:
            d.x = (d.x - norm['mean']) / norm['std']
            if d.edge_attr is not None:
                ea_m = norm.get('edge_attr_mean', d.edge_attr.mean(0))
                ea_s = norm.get('edge_attr_std',
                                d.edge_attr.std(0).clamp(min=1e-6))
                d.edge_attr = (d.edge_attr - ea_m) / ea_s

    idx = min(args.sample_idx, len(val_data) - 1)
    data = val_data[idx]
    print(f"  Sample {idx}: {data.x.shape[0]} nodes, "
          f"{data.edge_index.shape[1]} edges")

    n_defect = (data.y > 0).sum().item()
    print(f"  Labels: {n_defect} defect / {data.x.shape[0]} total")

    # Extract attention
    print("\nExtracting attention weights...")
    alphas, data = extract_attention(model, data)
    print(f"  Captured {len(alphas)} layers of attention")

    if not alphas:
        print("  WARNING: No attention weights captured. "
              "Model may not use GATConv.")
        return

    # Compute node-level attention
    node_attn = compute_node_attention(alphas, data.edge_index, data.x.shape[0])
    print(f"  Node attention: mean={node_attn.mean():.4f}, "
          f"max={node_attn.max():.4f}")

    # Generate figures
    print("\nGenerating figures...")

    # Figure 1: 2D attention mesh + ground truth + distribution
    plot_attention_mesh(
        data, node_attn, data.y,
        os.path.join(args.output, 'fig_attention_mesh.png'),
        title='GAT Attention vs Ground Truth (Validation Sample)')

    # Figure 2: 3D attention
    if data.pos is not None and data.pos.shape[1] >= 3:
        plot_attention_3d(
            data, node_attn,
            os.path.join(args.output, 'fig_attention_3d.png'))

    # Figure 3: Layer-wise attention distribution
    plot_layer_attention(
        alphas,
        os.path.join(args.output, 'fig_attention_layers.png'))

    # Print attention stats for defect vs healthy
    defect_mask = data.y > 0
    if defect_mask.any():
        attn_defect = node_attn[defect_mask].mean().item()
        attn_healthy = node_attn[~defect_mask].mean().item()
        ratio = attn_defect / (attn_healthy + 1e-8)
        print(f"\n  Attention on defect nodes: {attn_defect:.4f}")
        print(f"  Attention on healthy nodes: {attn_healthy:.4f}")
        print(f"  Ratio (defect/healthy): {ratio:.2f}x")

    print("\nDone!")


if __name__ == '__main__':
    main()
